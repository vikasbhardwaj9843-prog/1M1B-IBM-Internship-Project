"""
model/autoencoder.py — Composed Autoencoder (Encoder + Decoder).

Exposes:
    encode(x)  → latent z
    decode(z)  → reconstructed image x_hat
    forward(x) → x_hat  (standard nn.Module forward)
"""
import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Autoencoder(nn.Module):
    """
    CNN Hourglass Autoencoder for 1024×1024 image compression.

    Args:
        in_channels      : Image channels (1=grayscale, 3=RGB).
        base_channels    : Base channel width (default 16).
        bottleneck_channels: Not used as a hyperparameter here (fixed at base*16=256)
                            but kept for documentation / future extension.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        bottleneck_channels: int = 256,  # informational; derived from base_channels
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        self.encoder = Encoder(in_channels=in_channels, base_channels=base_channels)
        self.decoder = Decoder(out_channels=in_channels, base_channels=base_channels)

        self._log_param_count()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress image to bottleneck latent tensor.

        Args:
            x: (B, C, 1024, 1024) normalised to [0, 1]
        Returns:
            z: (B, 256, 32, 32)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct image from bottleneck latent tensor.

        Args:
            z: (B, 256, 32, 32)
        Returns:
            x_hat: (B, C, 1024, 1024) in [0, 1]
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode → decode pass (used during training).

        Args:
            x: (B, C, 1024, 1024)
        Returns:
            x_hat: (B, C, 1024, 1024)
        """
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_param_count(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        print(
            f"[Autoencoder] Parameters — "
            f"Encoder: {enc:,}  Decoder: {dec:,}  Total: {total:,}"
        )
