import torch
from data.dataset import BraTS2020Dataset, PyTMinMaxScalerVectorized
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2
from torchvision.transforms.functional import center_crop
from matplotlib.colors import BoundaryNorm

# set SHOW_EMPTY_MASK=True when you want to see images even when there is no
# applicable segmentation mask (i.e., a clean image)
SHOW_EMPTY_MASK = False

if __name__ == "__main__":

    train_files = None
    val_files = None
    data_dir = None
    batch_size = 8
    image_dim = (240, 240)
    composed_transform = v2.Compose([v2.RandomRotation(80),
                         v2.RandomHorizontalFlip(0.15),
                         v2.RandomVerticalFlip(0.15),
                         v2.ElasticTransform()])
    
    training_dataloader = DataLoader(
        BraTS2020Dataset(transform=composed_transform), batch_size=batch_size, shuffle=False
    )
    # training_dataloader = DataLoader(BraTS2020Dataset(transform=PyTMinMaxScalerVectorized), batch_size=batch_size, shuffle=False)

    b_norm = BoundaryNorm(torch.arange(0, 5), ncolors=4)
    reflection_pad_fn = torch.nn.ReflectionPad1d(68)

    for b in range(0, 10):
        trainX, trainY = next(iter(training_dataloader))
        for i in range(0, batch_size):
            if (torch.max(trainY) >= 1) or (SHOW_EMPTY_MASK):

                fig, axs = plt.subplots(3, 2, figsize=(10, 10))

                axs[0, 0].imshow(
                    trainY[i],
                    cmap=plt.get_cmap("tab10"),
                    norm=b_norm,
                )

                axs[1, 0].imshow(trainX[i, 0, :, :])
                axs[1, 1].imshow(trainX[i, 1, :, :])
                axs[2, 0].imshow(trainX[i, 2, :, :])
                axs[2, 1].imshow(trainX[i, 3, :, :])

                axs[0, 0].set_title("Mask")
                axs[0, 1].set_title("Split Mask")
                axs[1, 0].set_title("T1")
                axs[1, 1].set_title("T1Gd")
                axs[2, 0].set_title("T2")
                axs[2, 1].set_title("T2-FLAIR")

                plt.tight_layout()

                fig.show()
                plt.waitforbuttonpress()
                plt.close(fig)
