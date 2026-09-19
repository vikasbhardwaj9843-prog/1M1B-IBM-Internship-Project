"""
train/test_loss.py — Unit tests for ReconstructionLoss.

Run: python -m pytest train/test_loss.py -v
  or: python train/test_loss.py
"""
import torch
from train.loss import ReconstructionLoss


def test_identical_inputs_give_near_zero_loss():
    """Identical pred and target → loss ≈ 0."""
    loss_fn = ReconstructionLoss(alpha=0.5)
    x = torch.rand(2, 1, 64, 64)
    loss, components = loss_fn(x, x)
    assert loss.item() < 1e-4, f"Expected ~0, got {loss.item()}"
    assert components["mse"] < 1e-6


def test_zeros_vs_ones_give_large_loss():
    """All-zero pred vs all-one target → MSE = 1.0, loss should be large."""
    loss_fn = ReconstructionLoss(alpha=0.5)
    pred = torch.zeros(2, 1, 64, 64)
    target = torch.ones(2, 1, 64, 64)
    loss, components = loss_fn(pred, target)
    assert components["mse"] > 0.9, f"Expected MSE ~1, got {components['mse']}"
    assert loss.item() > 0.4


def test_alpha_zero_uses_only_ssim():
    """alpha=0 → loss equals ssim_loss component."""
    loss_fn = ReconstructionLoss(alpha=0.0)
    x = torch.rand(2, 1, 64, 64)
    y = torch.rand(2, 1, 64, 64)
    loss, components = loss_fn(x, y)
    assert abs(loss.item() - components["ssim_loss"]) < 1e-5


def test_alpha_one_uses_only_mse():
    """alpha=1 → loss equals mse component."""
    loss_fn = ReconstructionLoss(alpha=1.0)
    x = torch.rand(2, 1, 64, 64)
    y = torch.rand(2, 1, 64, 64)
    loss, components = loss_fn(x, y)
    assert abs(loss.item() - components["mse"]) < 1e-5


if __name__ == "__main__":
    test_identical_inputs_give_near_zero_loss()
    test_zeros_vs_ones_give_large_loss()
    test_alpha_zero_uses_only_ssim()
    test_alpha_one_uses_only_mse()
    print("All loss tests passed.")
