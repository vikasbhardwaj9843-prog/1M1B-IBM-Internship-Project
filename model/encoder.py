"""
model/encoder.py — CNN Encoder.

Spatial downsampling: 1024 → 512 → 256 → 128 → 64 → 32
Channel expansion  : in_ch → 16 → 32 → 64 → 128 → 256

No skip connections — all information must pass through the bottleneck.
"""
import torch.nn as nn
from model.blocks import ConvBlock


class Encoder(nn.Module):
    """
    Compresses a (B, in_ch, 1024, 1024) image to a (B, 256, 32, 32) latent tensor.

    Args:
        in_channels  : Number of input image channels (1=grayscale, 3=RGB).
        base_channels: Number of channels after the first conv (default 16).
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16):
        super().__init__()
        bc = base_channels
        self.layers = nn.Sequential(
            ConvBlock(in_channels, bc),        # 1024 → 512,  ch: in → 16
            ConvBlock(bc,      bc * 2),        # 512  → 256,  ch: 16 → 32
            ConvBlock(bc * 2,  bc * 4),        # 256  → 128,  ch: 32 → 64
            ConvBlock(bc * 4,  bc * 8),        # 128  → 64,   ch: 64 → 128
            ConvBlock(bc * 8,  bc * 16),       # 64   → 32,   ch: 128 → 256
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_ch, 1024, 1024)
        Returns:
            z: (B, 256, 32, 32)  — the bottleneck latent tensor
        """
        return self.layers(x)
