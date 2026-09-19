"""
model/decoder.py — CNN Decoder (mirror of Encoder).

Spatial upsampling  : 32 → 64 → 128 → 256 → 512 → 1024
Channel contraction : 256 → 128 → 64 → 32 → 16 → out_ch

Final activation is Sigmoid to map outputs to [0, 1].
"""
import torch.nn as nn
from model.blocks import ConvTransposeBlock


class Decoder(nn.Module):
    """
    Reconstructs a (B, in_ch, 1024, 1024) image from a (B, 256, 32, 32) latent tensor.

    Args:
        out_channels : Number of output image channels (must match Encoder in_channels).
        base_channels: Must match the value used in Encoder (default 16).
    """

    def __init__(self, out_channels: int = 1, base_channels: int = 16):
        super().__init__()
        bc = base_channels
        self.layers = nn.Sequential(
            ConvTransposeBlock(bc * 16, bc * 8),   # 32 → 64,   ch: 256 → 128
            ConvTransposeBlock(bc * 8,  bc * 4),   # 64 → 128,  ch: 128 → 64
            ConvTransposeBlock(bc * 4,  bc * 2),   # 128 → 256, ch: 64  → 32
            ConvTransposeBlock(bc * 2,  bc),        # 256 → 512, ch: 32  → 16
            # Final layer: upsample to 1024, no BN, Sigmoid activation
            nn.ConvTranspose2d(bc, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        Args:
            z: (B, 256, 32, 32) — bottleneck latent tensor
        Returns:
            x_hat: (B, out_ch, 1024, 1024) — reconstructed image in [0, 1]
        """
        return self.layers(z)
