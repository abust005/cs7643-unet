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
# -------------------------------
# 4. Prepare the Data
# -------------------------------

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

        # If mask has multiple channels, convert to single-channel
        if mask.ndim > 2:
            mask = torch.mean(mask, dim=-1)  # Example: Convert RGB to grayscale

        image = torch.moveaxis(image, -1, 0)

        return image, mask

