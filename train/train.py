"""
train/train.py — Entry-point for training NeuralZip.

Usage:
    python -m train.train
    python -m train.train --config config.yaml
"""
import argparse

import torch

from utils.config import load_config
from utils.dataloader import get_dataloaders
from model.autoencoder import Autoencoder
from train.loss import ReconstructionLoss
from train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the NeuralZip autoencoder.")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config YAML (default: config.yaml)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"[train] Config loaded from '{args.config}'")
    print(f"[train] Image size : {config['image_size']}×{config['image_size']}")
    print(f"[train] Channels   : {config['in_channels']}")
    print(f"[train] Epochs     : {config['epochs']}")
    print(f"[train] Batch size : {config['batch_size']}")

    # --- Model ---
    model = Autoencoder(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        bottleneck_channels=config["bottleneck_channels"],
    )

    # --- Optimiser ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # --- Loss ---
    loss_fn = ReconstructionLoss(alpha=config.get("alpha", 0.5))

    # --- Data ---
    train_loader, val_loader = get_dataloaders(config)
    print(
        f"[train] Dataset     : {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val images"
    )

    # --- Train ---
    trainer = Trainer(model, optimizer, loss_fn, train_loader, val_loader, config)
    trainer.fit()

    print("[train] Training complete.")
    print(f"[train] Best model  : {config.get('best_model_path', 'outputs/best_model.pt')}")
    print(f"[train] Training log: {config.get('training_log', 'outputs/training_log.csv')}")


if __name__ == "__main__":
    main()
