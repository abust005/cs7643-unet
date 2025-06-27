from model.unet import UNet
import torch
from time import time
from data.dataset import BraTS2020Dataset
from torch.utils.data import DataLoader, random_split

TENSOR_CORES = True
NUM_EPOCHS = 5

def dice_coefficient(prediction, target, epsilon=1e-07):
    prediction_copy = prediction.clone()

    prediction_copy[prediction_copy < 0] = 0
    prediction_copy[prediction_copy > 0] = 1

    intersection = abs(torch.sum(prediction_copy * target))
    union = abs(torch.sum(prediction_copy) + torch.sum(target))
    dice = (2. * intersection + epsilon) / (union + epsilon)
    
    return dice

if TENSOR_CORES:
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True

    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

if __name__ == '__main__':

    generator1 = torch.Generator().manual_seed(42)
    data = BraTS2020Dataset()

    d = data.__getitem__(0)

    train, test, val = random_split(data, [0.7, 0.2, 0.1], generator1)

    # Split into training and validation sets
    # train_files, val_files = train_test_split(h5_files, test_size=0.2, random_state=42)

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train)}")
    print(f"Validation samples: {len(val)}")

    # Parameters
    batch_size = 32  # Adjust based on GPU memory
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = UNet(in_channels=4, num_classes=3, padding=0, padding_mode='reflect').to(device=device, dtype=torch.float32)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.99)

    softmax_fn = torch.nn.Softmax(dim=1)
    loss_fn = torch.nn.CrossEntropyLoss()
    reflection_pad_fn = torch.nn.ReflectionPad2d(68)

    size = len(train_dataloader.dataset)
    for epoch in range(NUM_EPOCHS):

        net.train()
        print(f'Epoch: {epoch}')
        for batch, (X, y) in enumerate(train_dataloader):

            X = X.to(device=device)
            y = y.to(device=device)

            logits = net(reflection_pad_fn(X))
            pred = softmax_fn(logits)
            loss = loss_fn(y, pred)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 32 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        net.eval()

        with torch.no_grad():

            avg_dice_score = 0

            for batch, (X, y) in enumerate(val_dataloader):
                X = X.to(device=device)
                y = X.to(device=device)

                logits = net(reflection_pad_fn(X))
                pred = softmax_fn(logits)

                avg_dice_score += dice_coefficient(pred, y)

            avg_dice_score /= batch
            print(f'Avg. dice coeff. at epoch {epoch}: {avg_dice_score}')

    torch.save(net, 'trained_unet.pth')



