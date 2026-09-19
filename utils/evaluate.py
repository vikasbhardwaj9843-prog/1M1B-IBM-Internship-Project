"""
utils/evaluate.py — Compute PSNR, SSIM, and compression ratio metrics.
"""
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from skimage.metrics import structural_similarity as ski_ssim


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.
    Assumes both tensors are in [0, 1], shape (B, C, H, W) or (C, H, W).
    """
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Structural Similarity Index (SSIM), averaged over batch.
    Converts tensors to numpy for scikit-image computation.
    """
    pred_np = pred.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()

    scores = []
    for p, t in zip(pred_np, tgt_np):
        # p, t: (C, H, W) in [0, 1]
        p_img = np.transpose(p, (1, 2, 0))  # → (H, W, C)
        t_img = np.transpose(t, (1, 2, 0))
        channel_axis = 2 if p_img.shape[2] > 1 else None
        if channel_axis is None:
            p_img = p_img[:, :, 0]
            t_img = t_img[:, :, 0]
        score = ski_ssim(
            p_img, t_img,
            data_range=1.0,
            channel_axis=channel_axis,
        )
        scores.append(score)
    return float(np.mean(scores))


def evaluate_batch(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> dict:
    """
    Compute mean PSNR and SSIM for a batch.

    Args:
        preds  : (B, C, H, W) reconstructed images in [0, 1]
        targets: (B, C, H, W) original images in [0, 1]

    Returns:
        dict with keys 'psnr_db' and 'ssim'
    """
    return {
        "psnr_db": psnr(preds, targets),
        "ssim": ssim(preds, targets),
    }


def compression_ratio(original_paths: Sequence[Path], latent_paths: Sequence[Path]) -> float:
    """
    Compute mean compression ratio: original_bytes / latent_bytes.

    Args:
        original_paths: Paths to original images.
        latent_paths  : Paths to corresponding .latent.pt files.
    """
    ratios = []
    for orig, lat in zip(original_paths, latent_paths):
        o = Path(orig).stat().st_size
        l_ = Path(lat).stat().st_size
        if l_ > 0:
            ratios.append(o / l_)
    return float(np.mean(ratios)) if ratios else 0.0
