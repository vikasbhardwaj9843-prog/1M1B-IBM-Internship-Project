"""
train/loss.py — Combined MSE + SSIM reconstruction loss.

Loss = alpha * MSE(x, x_hat) + (1 - alpha) * SSIM_loss(x, x_hat)

MSE  : per-pixel fidelity
SSIM : structural / perceptual quality (critical for medical/scientific imagery)

kornia.losses.ssim_loss returns 1 − SSIM (so minimising it maximises structural similarity).
"""
import torch
import torch.nn as nn
import kornia.losses as KL


class ReconstructionLoss(nn.Module):
    """
    Combined MSE + SSIM loss for autoencoder reconstruction.

    Args:
        alpha      : Weight for MSE component (default 0.5).
                     alpha=1.0 → pure MSE; alpha=0.0 → pure SSIM loss.
        window_size: SSIM sliding window size (default 11).
    """

    def __init__(self, alpha: float = 0.5, window_size: int = 11):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.window_size = window_size
        self.mse = nn.MSELoss()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            pred  : Reconstructed image tensor (B, C, H, W) in [0, 1]
            target: Original image tensor      (B, C, H, W) in [0, 1]

        Returns:
            loss  : Scalar combined loss
            components: dict with 'mse' and 'ssim_loss' values for logging
        """
        mse_val = self.mse(pred, target)
        # kornia ssim_loss: 1 - SSIM, averaged over batch
        ssim_val = KL.ssim_loss(pred, target, window_size=self.window_size)

        loss = self.alpha * mse_val + (1.0 - self.alpha) * ssim_val
        return loss, {"mse": mse_val.item(), "ssim_loss": ssim_val.item()}
