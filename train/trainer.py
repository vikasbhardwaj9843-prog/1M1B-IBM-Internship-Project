"""
train/trainer.py — Trainer class: training loop, validation loop, checkpointing.
"""
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from train.loss import ReconstructionLoss


def _psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Peak Signal-to-Noise Ratio in dB. Assumes values in [0, 1]."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10 * math.log10(1.0 / mse)


class Trainer:
    """
    Manages training and validation for the Autoencoder.

    Args:
        model       : Autoencoder instance.
        optimizer   : PyTorch optimiser.
        loss_fn     : ReconstructionLoss instance.
        train_loader: Training DataLoader.
        val_loader  : Validation DataLoader.
        config      : Loaded config dict.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: ReconstructionLoss,
        train_loader,
        val_loader,
        config: dict,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.checkpoint_dir = Path(config.get("checkpoint_dir", "outputs/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = Path(config.get("best_model_path", "outputs/best_model.pt"))
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)

        self.log_path = Path(config.get("training_log", "outputs/training_log.csv"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._best_val_loss = float("inf")
        self._init_log()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """Run the full training loop for `config['epochs']` epochs."""
        epochs = self.config.get("epochs", 50)
        print(f"[Trainer] Training on {self.device} for {epochs} epochs.")
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(epoch, epochs)
            val_loss, val_psnr = self._val_epoch()
            self._log_epoch(epoch, train_loss, val_loss, val_psnr)
            self._save_checkpoint(epoch, val_loss)
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.5f}  "
                f"val_loss={val_loss:.5f}  "
                f"val_psnr={val_psnr:.2f} dB"
            )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int, total: int) -> float:
        self.model.train()
        running = 0.0
        bar = tqdm(self.train_loader, desc=f"Train {epoch}/{total}", leave=False)
        for x, _ in bar:
            x = x.to(self.device)
            self.optimizer.zero_grad()
            x_hat = self.model(x)
            loss, _ = self.loss_fn(x_hat, x)
            loss.backward()
            self.optimizer.step()
            running += loss.item()
            bar.set_postfix(loss=f"{loss.item():.5f}")
        return running / len(self.train_loader)

    def _val_epoch(self) -> tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_psnr = 0.0
        with torch.no_grad():
            for x, _ in tqdm(self.val_loader, desc="Val  ", leave=False):
                x = x.to(self.device)
                x_hat = self.model(x)
                loss, _ = self.loss_fn(x_hat, x)
                running_loss += loss.item()
                running_psnr += _psnr(x_hat, x)
        n = len(self.val_loader)
        return running_loss / n, running_psnr / n

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
        }
        torch.save(ckpt, self.checkpoint_dir / f"epoch_{epoch:03d}.pt")
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            torch.save(ckpt, self.best_model_path)
            print(f"  ✓ New best model saved (val_loss={val_loss:.5f})")

    def _init_log(self) -> None:
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_psnr_db"])

    def _log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, val_psnr: float
    ) -> None:
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{val_psnr:.4f}"])
