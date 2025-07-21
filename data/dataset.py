# Import necessary libraries
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import kagglehub
import tifffile as tif
from torchvision.io import decode_image

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
        single_mask = torch.Tensor(tif.imread(mask_path))
        single_mask = single_mask.to(dtype=torch.uint8)
        single_mask[single_mask > 2] = 3

        split_mask = torch.zeros((4, single_mask.shape[0], single_mask.shape[1]))

        for v in np.unique(single_mask):

            v = int(v)
            split_mask[v] = torch.where(single_mask == v, 1, 0).to(torch.uint8)

        single_mask = torch.moveaxis(single_mask, -1, 0)
        split_mask = torch.moveaxis(split_mask, -1, 1)
        image = torch.moveaxis(image, (-1, -2), (0, 1))

        if self.normalizer != None:
            image = self.normalizer(image)

        if self.transform != None:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            split_mask = self.transform(split_mask)
            torch.set_rng_state(state)
            single_mask = self.transform(single_mask.unsqueeze(dim=0))

            split_mask[0, :, :] = torch.where(single_mask < 1, 1, 0)

        # val, idxs = split_mask.max(dim=0)
        # split_mask = torch.where(split_mask[idxs] < 1, 1, split_mask[idxs])

        return image, split_mask, single_mask.squeeze(dim=0)
