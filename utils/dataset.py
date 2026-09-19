"""
utils/dataset.py — ImageDataset: returns (tensor, tensor) pairs for autoencoder training.

The same image is used as both input and reconstruction target.
Images are resized to image_size × image_size and normalised to [0, 1].
"""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """
    Loads all images from root_dir (recursively, common formats only).

    Args:
        root_dir  : Directory containing preprocessed images.
        image_size: Spatial size (square) for resizing.
        in_channels: 1 for grayscale, 3 for RGB.
        transform : Optional additional transforms applied after base preprocessing.
    """

    _EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        root_dir: str | Path,
        image_size: int = 1024,
        in_channels: int = 1,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.in_channels = in_channels
        self.transform = transform

        self.paths = sorted(
            p for p in self.root_dir.rglob("*") if p.suffix.lower() in self._EXTENSIONS
        )
        if len(self.paths) == 0:
            raise FileNotFoundError(
                f"No images found in '{self.root_dir}'. "
                "Run `python -m utils.preprocess` to populate data/processed/."
            )

        mode = "L" if in_channels == 1 else "RGB"
        self.base_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),           # → [0, 1] float32, shape (C, H, W)
            ]
        )
        self._pil_mode = mode

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert(self._pil_mode)
        tensor = self.base_transform(img)
        if self.transform is not None:
            tensor = self.transform(tensor)
        # Input == target (autoencoder)
        return tensor, tensor
