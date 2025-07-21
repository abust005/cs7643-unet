import torch
from data.dataset import BraTS2020Dataset
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import random

if __name__ == "__main__":
    net = torch.load("trained_unet.pth", weights_only=False)
    net = net.to(device="cpu")
    generator1 = torch.Generator().manual_seed(42)
    data = BraTS2020Dataset()
    train, test, val = random_split(data, [0.7, 0.1, 0.2], generator1)
    test_dataloader = DataLoader(test, batch_size=8, shuffle=False)
    sfmax = torch.nn.Softmax(dim=1)

    net.eval()

    for x, y in tqdm(test_dataloader):
        logits = net(x)

        pred = torch.argmax(sfmax(logits), dim=1)
        # pred = torch.round(pred)

        # pred_mask = torch.empty((y.shape[2], y.shape[3]))
        # y_mask = torch.empty((y.shape[2], y.shape[3]))

        # for c in range(pred.shape[1]):
        #     pred[:, c] *= c
        #     y[:, c] *= c

        # pred_mask = torch.sum(pred, dim=1, keepdim=False)
        # y_mask = torch.sum(y, dim=1, keepdim=False)

        for i in range(y.shape[0]):

            if not (pred[i].max() > 0):
            # if random() < 0.7:
                continue 

            fig, axs = plt.subplots(1, 2)

            axs[0].imshow(pred[i].detach())
            axs[1].imshow(y[i].detach())

            axs[0].set_title("Predicted Mask")
            axs[1].set_title("Actual Mask")

            plt.tight_layout()

            fig.show()
            plt.waitforbuttonpress()
            plt.close(fig)
