"""
utils/visualise.py — Side-by-side original vs reconstructed comparison plots.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_comparison(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    path: str | Path,
    title: str = "",
) -> None:
    """
    Save a side-by-side comparison of original and reconstructed images.

    Args:
        original     : (1, C, H, W) or (C, H, W) tensor in [0, 1]
        reconstructed: same shape as original
        path         : Output file path (.png)
        title        : Optional super-title string
    """
    orig = original.squeeze(0) if original.ndim == 4 else original
    recon = reconstructed.squeeze(0) if reconstructed.ndim == 4 else reconstructed

    orig_np = orig.detach().cpu().permute(1, 2, 0).numpy()
    recon_np = recon.detach().cpu().permute(1, 2, 0).numpy()

    cmap = "gray" if orig_np.shape[2] == 1 else None
    if orig_np.shape[2] == 1:
        orig_np = orig_np[:, :, 0]
        recon_np = recon_np[:, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(orig_np, cmap=cmap, vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(recon_np, cmap=cmap, vmin=0, vmax=1)
    axes[1].set_title("Reconstructed")
    axes[1].axis("off")

    if title:
        fig.suptitle(title, fontsize=13)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(path), dpi=100, bbox_inches="tight")
    plt.close(fig)


def save_batch_comparisons(
    originals: torch.Tensor,
    reconstructeds: torch.Tensor,
    output_dir: str | Path,
    prefix: str = "sample",
) -> None:
    """
    Save comparison plots for each image in a batch.

    Args:
        originals     : (B, C, H, W)
        reconstructeds: (B, C, H, W)
        output_dir    : Directory to save plots.
        prefix        : Filename prefix.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(originals.size(0)):
        save_comparison(
            originals[i],
            reconstructeds[i],
            out_dir / f"{prefix}_{i:04d}.png",
            title=f"{prefix} {i}",
        )
