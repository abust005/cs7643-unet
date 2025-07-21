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
    def __init__(self, transform=None, normalizer=None):

        self.data_dir = kagglehub.dataset_download(
            "adrielbustamante/brats2020-training-set-tiff"
        )
        self.data_dir += "/brats2020/training"
        self.x = [f for f in os.listdir(f"{self.data_dir}/x") if f.endswith(".tif")]
        self.y = [f for f in os.listdir(f"{self.data_dir}/y") if f.endswith(".tif")]
        self.transform = transform
        self.normalizer = normalizer

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
