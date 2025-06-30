# Import necessary libraries
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import kagglehub

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """
    def __call__(self, tensor):
        dist = (tensor.max(dim=1, keepdim=True)[0] - tensor.min(dim=1, keepdim=True)[0])
        dist[dist==0.] = 1.
        scale = 1.0 /  dist
        tensor.mul_(scale).sub_(tensor.min(dim=1, keepdim=True)[0])
        return tensor

class BraTS2020Dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):

        self.data_dir = kagglehub.dataset_download("awsaf49/brats2020-training-data")
        self.data_dir += '/BraTS2020_training_data/content/data/'
        self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.h5')]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.data_files[idx])
        hf = h5py.File(img_path, 'r')
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)

        if 'image' in hf.keys() and 'mask' in hf.keys():
            image = torch.Tensor(np.array(hf['image']))  # Expected shape: (H, W, 4)
            mask = torch.Tensor(np.array(hf['mask'][:]))    # Expected shape: (H, W) or (H, W, C)

        new_mask = torch.empty((mask.shape[0], mask.shape[1], mask.shape[2] + 1))
        new_mask[:, :, 0] = (torch.ones(mask.shape[0], mask.shape[1]) - mask.sum(dim=-1))
        new_mask[:, :, 1:mask.shape[2]+1] = mask

        image[image < 0] = 0

        split_mask = torch.moveaxis(new_mask, -1, 0)
        image = torch.moveaxis(image, (-1, -2), (0, 1))
        if self.transform != None:
            image = self.transform()(image)

        single_mask = torch.einsum('kij, k->ij', split_mask, torch.arange(split_mask.shape[0], dtype=split_mask.dtype))

        return image, split_mask, single_mask

