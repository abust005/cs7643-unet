import torch.nn as nn
from model.unet_blocks import *

class UNet(nn.Module):
  
  def __init__(self, in_channels=3, down_padding = None, down_padding_mode = None):
    super().__init__()

    self.module_list = nn.ModuleDict(
      {
        'down_conv_1' : DownConv(in_channels, 64, padding=down_padding, padding_mode=down_padding_mode),
        'down_conv_2' : DownConv(64, 128, padding=down_padding, padding_mode=down_padding_mode),
        'down_conv_3' : DownConv(128, 256, padding=down_padding, padding_mode=down_padding_mode),
        'down_conv_4' : DownConv(256, 512, padding=down_padding, padding_mode=down_padding_mode),
        'conv_1' : ConvBlock(512, 1024),
        'up_conv_1' : UpConv(1024),
        'up_conv_2' : UpConv(512),
        'up_conv_3' : UpConv(256),
        'up_conv_4' : UpConv(128),
        'conv_2' : nn.Conv2d(64, 2, kernel_size=1)
      }
    )

  def forward(self, x):
    
    if type(x) != torch.Tensor:
      x = torch.Tensor(x)

    x1_crop, x1 = self.module_list['down_conv_1'](x)
    x2_crop, x2 = self.module_list['down_conv_2'](x1)
    x3_crop, x3 = self.module_list['down_conv_3'](x2)
    x4_crop, x4 = self.module_list['down_conv_4'](x3)
    x5 = self.module_list['conv_1'](x4)
    x6 = self.module_list['up_conv_1'](x5, x4_crop)
    x7 = self.module_list['up_conv_2'](x6, x3_crop)
    x8 = self.module_list['up_conv_3'](x7, x2_crop)
    x9 = self.module_list['up_conv_4'](x8, x1_crop)
    return self.module_list['conv_2'](x9)