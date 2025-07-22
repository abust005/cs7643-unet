# Import necessary libraries
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import kagglehub
import tifffile as tif
from torchvision.io import decode_image
import matplotlib.pyplot as plt

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """

    def __call__(self, tensor):
        dist = tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0]
        dist[dist == 0.0] = 1.0
        scale = 1.0 / dist
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor

class BraTS2020Dataset(Dataset):
    def __init__(self, transform=None, normalizer=None, clean_data=False, min_active_pixels=0.2):

        self.data_dir = kagglehub.dataset_download(
            "adrielbustamante/brats2020-training-set-tiff"
        )
        self.data_dir += "/brats2020/training"
        self.x = [f for f in os.listdir(f"{self.data_dir}/x") if f.endswith(".tif")]
        self.y = [f for f in os.listdir(f"{self.data_dir}/y") if f.endswith(".tif")]
        self.transform = transform
        self.normalizer = normalizer

        if clean_data:
            self.x, self.y = self.clean_data(self.x, self.y, min_active_pixels)
        

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = f"{self.data_dir}/x/{self.x[idx]}"
        mask_path = f"{self.data_dir}/y/{self.y[idx]}"

        image = torch.Tensor(tif.imread(img_path)) / 65535
        mask = torch.Tensor(tif.imread(mask_path))
        mask = mask.to(dtype=torch.long)
        mask[mask > 2] = 3

        mask = torch.moveaxis(mask, -1, 0)
        image = torch.moveaxis(image, (-1, -2), (0, 1))

        if self.normalizer != None:
            image = self.normalizer(image)

        if self.transform != None:
            image, mask = self.transform(image, mask)

        return image, mask
    
    def clean_data(self, x:list, y:list, min_active=0.2) -> tuple[list, list]:
        """Cleans data, removing sparse or blank x images

        Args:
            x (list): list of paths to x input .tif files
            y (list): segmantation mask paths
            min_active (float, optional): Minimum portion of x pixels with a non-zero value. Defaults to 0.2.

        Returns:
            tuple[list, list]: cleaned_x, cleaned_y
        """
        out_x = []
        out_y = []
        removed = 0
        original_total = len(x)

        assert len(x) == len(y)

        print(f"Removing all x dataset entries with less than {int(min_active * 100)}% active pixels")

        for ix in range(len(x)):
            img_path = f"{self.data_dir}/x/{self.x[ix]}"
            # mask_path = f"{self.data_dir}/y/{self.y[ix]}"
            x_image = tif.imread(img_path)
            active_pixels = (x_image > 0).sum()
            total_pixels = x_image.size
            active_ratio = active_pixels / total_pixels
            if active_ratio > min_active:
                out_x.append(self.x[ix])
                out_y.append(self.y[ix])
            else:
                removed += 1

        print(f"Removed {removed} images out of {original_total} from the dataset.")

        return (out_x, out_y)

