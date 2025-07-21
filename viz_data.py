import torch
from data.dataset import BraTS2020Dataset, PyTMinMaxScalerVectorized
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms.functional import center_crop
from torchvision.transforms import Compose, v2
from matplotlib.colors import BoundaryNorm

# set SHOW_EMPTY_MASK=True when you want to see images even when there is no
# applicable segmentation mask (i.e., a clean image)
SHOW_EMPTY_MASK = True

if __name__ == "__main__":

    train_files = None
    val_files = None
    data_dir = None
    batch_size = 8
    image_dim = (240, 240)
    transform = Compose([v2.RandomRotation(45),
                         v2.RandomHorizontalFlip(0.15),
                         v2.RandomVerticalFlip(0.15),
                         v2.ElasticTransform()])
    clean_data = True
    MIN_ACTIVE_PIXELS = 0.2
    
    training_dataloader = DataLoader(
        BraTS2020Dataset(transform=transform, clean_data=clean_data, min_active_pixels=MIN_ACTIVE_PIXELS), batch_size=batch_size, shuffle=False
    )
    # training_dataloader = DataLoader(BraTS2020Dataset(transform=PyTMinMaxScalerVectorized), batch_size=batch_size, shuffle=False)

    b_norm = BoundaryNorm(torch.arange(0, 5), ncolors=4)
    reflection_pad_fn = torch.nn.ReflectionPad1d(68)

    for b in range(0, 10):
        trainX, trainY, singleY = next(iter(training_dataloader))
        for i in range(0, batch_size):
            if (torch.max(trainY[i]) > 0) or (SHOW_EMPTY_MASK):

                fig, axs = plt.subplots(3, 2, figsize=(10, 10))
                #   fig.delaxes(axs[2,1])
                axs[0, 0].imshow(
                    torch.moveaxis(singleY[i], 0, -1),
                    cmap=plt.get_cmap("tab10"),
                    norm=b_norm,
                )
                axs[0, 1].imshow(
                    torch.moveaxis(trainY[i,1:4], 0, -1),
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
