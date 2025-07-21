from model.unet import UNet
import torch
from time import time
from data.dataset import BraTS2020Dataset, PyTMinMaxScalerVectorized
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as v2
from tqdm import tqdm
from losses.focal_loss import FocalLoss, reweight
import matplotlib.pyplot as plt

TENSOR_CORES = True
NUM_EPOCHS = 1
BATCH_SIZE = 8  # Adjust based on GPU memory

def debug(mask):
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].imshow(mask[0, 0].cpu().numpy())
    axs[0, 1].imshow(mask[0, 1].cpu().numpy())
    axs[1, 0].imshow(mask[0, 2].cpu().numpy())
    axs[1, 1].imshow(mask[0, 3].cpu().numpy())

    plt.tight_layout()

    fig.show()
    plt.waitforbuttonpress()
    plt.close(fig)


def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2.0 * intersection + epsilon) / (union + epsilon)

    return dice


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
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":

    generator1 = torch.Generator().manual_seed(80)
    data = BraTS2020Dataset()

    train, test, val = random_split(data, [0.7, 0.2, 0.1], generator1)

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    # mean, std = get_mean_std(train_dataloader)

    mean = torch.tensor([0.0017, 0.0019, 0.0020, 0.0011])
    std = torch.tensor([0.0056, 0.0067, 0.0068, 0.0042])
    norm_transform = v2.Normalize(mean, std)

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")

    # Parameters
    transform = v2.Compose(
        [
            v2.RandomRotation(45),
            v2.RandomHorizontalFlip(0.15),
            v2.RandomVerticalFlip(0.15),
            v2.ElasticTransform(),
        ]
    )

    data = BraTS2020Dataset(normalizer=norm_transform)
    train, test, val = random_split(data, [0.7, 0.2, 0.1], generator1)

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = UNet(in_channels=4, num_classes=3, padding=0, padding_mode="reflect").to(
        device=device, dtype=torch.float32
    )

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.99)

    softmax_fn = torch.nn.Softmax(dim=1)
    weights = reweight(torch.tensor([3257699276, 8161996, 21302318, 7268410]), beta=0.5)
    loss_fn = FocalLoss(weights, gamma=1, device=device)
    ce_fn = torch.nn.CrossEntropyLoss(weight=weights).to(device=device)
    reflection_pad_fn = torch.nn.ReflectionPad2d(68)
    

    size = len(train_dataloader.dataset)
    for epoch in range(NUM_EPOCHS):

        net.train()
        print(f"Epoch: {epoch}")
        for batch, (X, y) in enumerate(train_dataloader):

            X = X.to(device=device, dtype=torch.float32)
            y = y.to(device=device)

            logits = net(reflection_pad_fn(X))
            pred = torch.argmax(logits, dim=1).float()
            loss = ce_fn(logits, y)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=.1)
            optimizer.step()
            optimizer.zero_grad()

            if batch % 32 == 0:
                loss, current = loss.item(), batch * BATCH_SIZE + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        net.eval()

        with torch.no_grad():

            avg_dice_score = 0

            for batch, (X, y) in enumerate(val_dataloader):
                X = X.to(device=device)
                y = y.to(device=device)

                logits = net(reflection_pad_fn(X))
                # pred = softmax_fn(logits).long()
                pred = torch.argmax(logits, dim=1)

                avg_dice_score += dice_coefficient(pred, y)

            avg_dice_score /= batch
            print(f"Avg. dice coeff. at epoch {epoch}: {avg_dice_score}")

    torch.save(net, "trained_unet.pth")


