import torch
from torchvision.transforms.functional import center_crop
import torch.nn as nn
import torch.nn.init as init

"""
  U-Net modules as described in https://arxiv.org/abs/1505.04597
  See Section 2: Network Architecture
"""


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, padding="same", padding_mode="reflect"
    ):
        super().__init__()

        self.module_list = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
                stride=1,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=padding,
                padding_mode=padding_mode,
                stride=1,
            ),
            nn.ReLU(),
        )

        for layer in self.module_list:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(self, x):
        return self.module_list.forward(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding, padding_mode):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels, padding, padding_mode)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.dropout = nn.Dropout2d(0.3)

    def forward(self, x):

        x_crop = self.conv(x)
        x_out = self.pool(x_crop)
        x_out = self.dropout(x_out)

        return x_crop, x_out


class UpConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.module_list = nn.ModuleDict(
            {
                "up_conv": nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=in_channels // 2,
                    kernel_size=2,
                    stride=2,
                ),
                "conv_block": ConvBlock(in_channels, out_channels=in_channels // 2),
                "dropout": nn.Dropout2d(0.3),
            }
        )

    def forward(self, x1, x2):
        """
        x1 : output of previous layer
        x2 : output from corresponding down-conv
             to be cropped and concatenated
        """

        x1 = self.module_list["up_conv"](x1)
        x1_img_shape = [x1.shape[-2], x1.shape[-1]]

        x3 = torch.cat((x1, center_crop(x2, x1_img_shape)), dim=1)

        return self.module_list["dropout"](self.module_list["conv_block"](x3))
