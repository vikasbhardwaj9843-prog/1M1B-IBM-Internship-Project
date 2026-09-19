"""
utils/preprocess.py — Resize raw images to image_size × image_size and save to data/processed/.

Usage:
    python -m utils.preprocess                  # uses config.yaml defaults
    python -m utils.preprocess --src data/raw --dst data/processed --size 1024
"""
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from utils.config import load_config

_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def preprocess(src_dir: Path, dst_dir: Path, size: int, in_channels: int) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    pil_mode = "L" if in_channels == 1 else "RGB"

    paths = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in _EXTENSIONS)
    if not paths:
        print(f"No images found in '{src_dir}'. Nothing to preprocess.")
        return

    print(f"Preprocessing {len(paths)} images → {dst_dir} ({size}×{size}, mode={pil_mode})")
    for src_path in tqdm(paths, unit="img"):
        img = Image.open(src_path).convert(pil_mode).resize((size, size), Image.LANCZOS)
        # Preserve relative sub-directory structure
        rel = src_path.relative_to(src_dir)
        dst_path = dst_dir / rel.with_suffix(".png")
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, format="PNG")

    print("Done.")


def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(description="Preprocess images for NeuralZip training.")
    parser.add_argument("--src", default=cfg.get("data_dir", "data/raw"))
    parser.add_argument("--dst", default="data/processed")
    parser.add_argument("--size", type=int, default=cfg.get("image_size", 1024))
    parser.add_argument("--channels", type=int, default=cfg.get("in_channels", 1))
    args = parser.parse_args()

    preprocess(Path(args.src), Path(args.dst), args.size, args.channels)


if __name__ == "__main__":
    main()
