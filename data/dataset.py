# Import necessary libraries
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import kagglehub
import tifffile as tif

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
    def __init__(self, transform=None, target_transform=None):

        # self.data_dir = kagglehub.dataset_download("awsaf49/brats2020-training-data")
        self.data_dir = kagglehub.dataset_download("adrielbustamante/brats2020-training-set-tiff")
        self.data_dir += "/brats2020/training"
        self.x = [f for f in os.listdir(f'{self.data_dir}/x') if f.endswith(".tif")]
        self.y = [f for f in os.listdir(f'{self.data_dir}/y') if f.endswith(".tif")]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = f'{self.data_dir}/x/{self.x[idx]}'
        mask_path = f'{self.data_dir}/y/{self.y[idx]}'

        # if "image" in hf.keys() and "mask" in hf.keys():
        #     image = torch.Tensor(np.array(hf["image"]))  # Expected shape: (H, W, 4)
        #     mask = torch.Tensor(
        #         np.array(hf["mask"][:])
        #     )  # Expected shape: (H, W) or (H, W, C)

        image = torch.Tensor(tif.imread(img_path)) / 65535
        single_mask = torch.Tensor(tif.imread(mask_path))
        single_mask = single_mask.to( dtype=torch.uint8)
        single_mask[single_mask > 2] = 3

        split_mask = torch.zeros((4, single_mask.shape[0], single_mask.shape[1]))

        for v in np.unique(single_mask):
            
            v = int(v)
            split_mask[v] = torch.where(single_mask == v, 1, 0).to(torch.uint8)

        split_mask = torch.moveaxis(split_mask, -1, 1)
        image = torch.moveaxis(image, (-1, -2), (0, 1))

        if self.transform != None:
            image = self.transform(image)

        # single_mask = torch.einsum(
        #     "kij, k->ij",
        #     split_mask,
        #     torch.arange(split_mask.shape[0], dtype=split_mask.dtype),
        # )

        return image, split_mask, single_mask
