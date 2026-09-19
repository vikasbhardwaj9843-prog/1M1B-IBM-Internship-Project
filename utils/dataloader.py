"""
utils/dataloader.py — Build train and validation DataLoaders from config.
"""
import torch
from torch.utils.data import DataLoader, random_split

from utils.config import load_config
from utils.dataset import ImageDataset


def get_dataloaders(config: dict | None = None):
    """
    Build train and validation DataLoaders.

    Args:
        config: Loaded config dict. If None, loads from default config.yaml.

    Returns:
        (train_loader, val_loader)
    """
    if config is None:
        config = load_config()

    dataset = ImageDataset(
        root_dir=config["data_dir"],
        image_size=config["image_size"],
        in_channels=config["in_channels"],
    )

    val_size = int(len(dataset) * config.get("val_split", 0.2))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
