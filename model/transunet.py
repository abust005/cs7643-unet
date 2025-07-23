import torch.nn as nn
from model.transunet_blocks import *
from model.unet_blocks import *


class TransUNet(nn.Module):
    def __init__(self, img_size=128, patch_size=2, in_channels=3, num_classes=1,  padding=0, padding_mode="zeros", embed_dim=1024, num_blocks=8):
        super().__init__()
        self.grid_size = img_size // (patch_size * 8)  # due to 3 stride-2 convs

        # CNN encoder stem to generate skip features
        self.encoder_stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=padding, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=padding, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=padding, padding_mode=padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=padding, padding_mode=padding_mode),
            nn.ReLU(inplace=True)
        )

        # Patch Embedding for ViT
        self.patch_embed = PatchEmbedding(512, embed_dim, patch_size=patch_size)
        self.pos_embed = PositionalEncoding(embed_dim)
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(embed_dim) for _ in range(num_blocks)
        ])

        # Decoder
        #self.decoder1 = DecoderBlock(1024, 512, 512)
        self.decoder2 = DecoderBlock(512, 256, 256)
        self.decoder3 = DecoderBlock(256, 128, 128)
        self.decoder4 = DecoderBlock(128, 64, 64)

        #self.final_conv = nn.Conv2d(64, num_classes + 1, kernel_size=1)
        self.final_conv = nn.Conv2d(64, 16, kernel_size=3, padding=1)
        self.output_conv = nn.Conv2d(16, num_classes + 1, kernel_size=1)

    def forward(self, x):
        # CNN Encoder for skip connections
        x1 = self.encoder_stem[0](x)        # 64
        x2 = self.encoder_stem[1](x1)
        x3 = self.encoder_stem[2](x2)       # 128
        x4 = self.encoder_stem[3](x3)
        x5 = self.encoder_stem[4](x4)       # 256
        x6 = self.encoder_stem[5](x5)
        x7 = self.encoder_stem[6](x6)       # 512
        x8 = self.encoder_stem[7](x7)

        # ViT Encoder
        x_patch, (H, W), _ = self.patch_embed(x8)
        x_patch = self.pos_embed(x_patch)
        x_patch = self.transformer_blocks(x_patch)

        # Reshape transformer output
        x_trans = x_patch.transpose(1, 2).reshape(x.shape[0], 512, H, W)

        # Decoder
        #x = self.decoder1(x_trans, x7)  # 1024 + 512 -> 512
        x = self.decoder2(x_trans, x5)  # 512 + 256 -> 256
        x = self.decoder3(x, x3)        # 256 + 128 -> 128
        x = self.decoder4(x, x1)        # 128 + 64 -> 64

        #out = self.final_conv(x)
        x = self.final_conv(x)
        out = self.output_conv(x)
        return out
