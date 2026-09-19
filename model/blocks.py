"""
model/blocks.py — Reusable building blocks for the hourglass autoencoder.

ConvBlock       : Conv2d + BatchNorm2d + ReLU  (used in encoder)
ConvTransposeBlock: ConvTranspose2d + BatchNorm2d + ReLU  (used in decoder)
"""
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Downsampling block: Conv2d (stride=2) → BatchNorm2d → ReLU.
    Halves spatial dimensions; doubles channels (typically).

    Args:
        in_ch : input channels
        out_ch: output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """
    Upsampling block: ConvTranspose2d (stride=2) → BatchNorm2d → ReLU.
    Doubles spatial dimensions; halves channels (typically).

    Args:
        in_ch : input channels
        out_ch: output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)
