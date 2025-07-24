from model.unet import UNet
from model.transunet import TransUNet
import torch
from time import time
from data.dataset import BraTS2020Dataset
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as v2
from tqdm import tqdm
from losses.focal_loss import FocalLoss, reweight
from losses.dice_loss import DiceLoss, diceCoefficient

from torch.utils.tensorboard import SummaryWriter

'''
Computation Parameters
'''
TENSOR_CORES = True
NUM_EPOCHS = 10
BATCH_SIZE = 16  # Adjust based on GPU memory
CLEAN_DATA = True
MIN_ACTIVE_PIXELS = 0.2 # Keeps data with at least the portion of non-zero pixel values

'''
Loss Specific Parameters
'''
LOSS = "Dice" # CE, Focal, Dice, or Combo
COMBO_ALPHA = 0.65
LOG_COSH=True
COMPUTE_WEIGHTS = True
WEIGHTS_BETA = 0.5

'''
Model Type
'''
MODEL_TYPE = "TransUNet"  # UNet or TransUNet

# Ex. description: "Cleaned_0.2_WeightedCE_UNet"
RUN_DESC = f"--{f"Cleaned_{MIN_ACTIVE_PIXELS}" if CLEAN_DATA else "Uncleaned"}_ \
                {"Weighted" if COMPUTE_WEIGHTS else "Unweighted"} \
                {"LogCosh" if LOG_COSH and (LOSS=="Dice" or LOSS=="Combo") else ""} \
                {LOSS}_ \
                {MODEL_TYPE}"

def get_mean_std(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = torch.zeros(4)
    std = torch.zeros(4)
    for images, _, _ in tqdm(loader, desc="Computing statisics"):
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(dim=(0, 2, 3))
        std += images.std(dim=(0, 2, 3))

    mean /= len(loader)
    std /= len(loader)

    return mean, std

if TENSOR_CORES:

    writer = SummaryWriter(comment=RUN_DESC)


    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":

    generator1 = torch.Generator().manual_seed(42)

    # Pre-computed based on training dataset when generated with manual seed
    mean = torch.tensor([0.0017, 0.0019, 0.0020, 0.0011])
    std = torch.tensor([0.0056, 0.0067, 0.0068, 0.0042])
    norm_transform = v2.Normalize(mean, std)

    # Parameters
    transform = v2.Compose(
        [
            v2.RandomRotation(45),
            v2.RandomHorizontalFlip(0.15),
            v2.RandomVerticalFlip(0.15),
            v2.ElasticTransform(),
        ]
    )

    data = BraTS2020Dataset(normalizer=norm_transform, clean_data=CLEAN_DATA, min_active_pixels=MIN_ACTIVE_PIXELS)
    train, test, val = random_split(data, [0.7, 0.2, 0.1], generator1)

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=6)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_TYPE == 'UNet':
        net = UNet(in_channels=4, num_classes=3, padding=0, padding_mode="reflect").to(
            device=device, dtype=torch.float32
        )

    elif MODEL_TYPE == 'TransUNet':
        net = TransUNet(img_size=240, patch_size=2, in_channels=4, num_classes=3, padding=0, padding_mode="reflect", embed_dim=512, num_blocks=8).to(
            device=device, dtype=torch.float32
        )
    else:
        raise ValueError(f"Unsupported model_type: {MODEL_TYPE}")
    
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)  # momentum=0.99)

    weights = None
    if COMPUTE_WEIGHTS:
        weights = reweight(torch.tensor([3257699276, 8161996, 21302318, 7268410]), beta=WEIGHTS_BETA)

    if LOSS == "CE":
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights).to(device=device)
    elif LOSS == "Focal":
        loss_fn = FocalLoss(weight=weights, gamma=1, device=device)
    elif LOSS == "Dice":
        loss_fn = DiceLoss(n_classes=4, log_cosh=LOG_COSH, device=device)
    elif LOSS == "Combo":
        diceLoss = DiceLoss(n_classes=4, log_cosh=LOG_COSH, device=device)
        ceLoss = torch.nn.CrossEntropyLoss(weight=weights).to(device=device)

        loss_fn = lambda x,y: (COMBO_ALPHA * diceLoss(x,y)) + ((1-COMBO_ALPHA) * diceLoss(x,y))

    softmax_fn = torch.nn.Softmax(dim=1)
    reflection_pad_68_fn = torch.nn.ReflectionPad2d(68)
    reflection_pad_1_fn = torch.nn.ReflectionPad2d(1)

    x_tmp, _ = next(iter(train_dataloader))
    x_tmp = x_tmp.to(device=device, dtype=torch.float32)

    if MODEL_TYPE == 'UNet':
        x_tmp = reflection_pad_68_fn(x_tmp)
    elif MODEL_TYPE == 'TransUNet':
        x_tmp = reflection_pad_1_fn(x_tmp)

    writer.add_graph(net, x_tmp)

    upsample_fn = torch.nn.Upsample((370, 370), mode='bilinear')

    size = len(train_dataloader.dataset)
    for epoch in range(NUM_EPOCHS):

        net.train()

        print(f"Epoch: {epoch}")

        avg_train_loss = 0
        avg_train_dice = 0
        avg_val_dice = 0
        avg_val_loss = 0
        for batch, (X, y) in enumerate(train_dataloader):

            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device)

            if MODEL_TYPE == 'UNet':
                logits = net(upsample_fn(X))
            elif MODEL_TYPE == 'TransUNet':
                logits = net(reflection_pad_1_fn(X))

            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            pred = torch.argmax(softmax_fn(logits), dim=1)
            avg_train_dice += diceCoefficient(pred, y, n_classes=4)
            avg_train_loss += loss

            # if batch % 32 == 0:
            #     loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        avg_train_loss /= batch
        avg_train_dice /= batch

        writer.add_scalar("charts/train_dice", avg_train_dice, epoch)
        writer.add_scalar("charts/train_loss", avg_train_loss, epoch)

        net.eval()

        with torch.no_grad():

            for batch, (X, y) in enumerate(val_dataloader):
                X = X.to(device=device)
                y = y.to(device=device)

                if MODEL_TYPE == 'UNet':
                    logits = net(upsample_fn(X))
                elif MODEL_TYPE == 'TransUNet':
                    logits = net(reflection_pad_1_fn(X))

                avg_val_loss += loss_fn(logits, y)                
                pred = torch.argmax(softmax_fn(logits), dim=1)

                avg_val_dice += (diceCoefficient(pred, y, n_classes=4))

            avg_val_dice /= batch
            avg_val_loss /= batch
            # print(f"Avg. dice coeff. at epoch {epoch}: {avg_dice_score}")
            writer.add_scalar("charts/val_dice", avg_val_dice, epoch)
            writer.add_scalar("charts/val_loss", avg_val_loss, epoch)

    torch.save(net, "trained_unet.pth")


