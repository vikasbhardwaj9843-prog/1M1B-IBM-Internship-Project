"""
compress/compress.py — CLI: encode an image to a latent tensor (and optional residual).

Usage:
    # Near-lossless
    python -m compress.compress --input image.png --model outputs/best_model.pt --output compressed/

    # Lossless (saves residual map alongside latent)
    python -m compress.compress --input image.png --model outputs/best_model.pt --output compressed/ --lossless
"""
import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from utils.config import load_config
from model.autoencoder import Autoencoder


def load_model(model_path: str, config: dict, device: torch.device) -> Autoencoder:
    model = Autoencoder(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        bottleneck_channels=config["bottleneck_channels"],
    )
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)  # support both raw state_dict and checkpoint dicts
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_image(image_path: str, image_size: int, in_channels: int) -> torch.Tensor:
    """Load and preprocess image → (1, C, H, W) tensor in [0, 1]."""
    pil_mode = "L" if in_channels == 1 else "RGB"
    img = Image.open(image_path).convert(pil_mode)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # (1, C, H, W)


def compress(
    input_path: str,
    model_path: str,
    output_dir: str,
    lossless: bool = False,
    config_path: str = "config.yaml",
) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(input_path).stem

    # Load model and image
    model = load_model(model_path, config, device)
    x = load_image(input_path, config["image_size"], config["in_channels"]).to(device)

    with torch.no_grad():
        z = model.encode(x)                # (1, 256, 32, 32)
        x_hat = model.decode(z).clamp(0, 1)

    # Save latent
    latent_path = out_dir / f"{stem}.latent.pt"
    torch.save(z.cpu(), latent_path)

    # Compression ratio
    original_bytes = os.path.getsize(input_path)
    latent_bytes = latent_path.stat().st_size
    ratio = original_bytes / latent_bytes if latent_bytes > 0 else float("nan")

    print(f"[compress] Input     : {input_path}  ({original_bytes / 1024:.1f} KB)")
    print(f"[compress] Latent    : {latent_path}  ({latent_bytes / 1024:.1f} KB)")
    print(f"[compress] Ratio     : {ratio:.2f}×  (original / latent)")

    if lossless:
        residual = (x - x_hat).cpu()        # may be negative; preserve sign
        residual_path = out_dir / f"{stem}.residual.pt"
        torch.save(residual, residual_path)
        res_bytes = residual_path.stat().st_size
        total_bytes = latent_bytes + res_bytes
        lossless_ratio = original_bytes / total_bytes if total_bytes > 0 else float("nan")
        print(f"[compress] Residual  : {residual_path}  ({res_bytes / 1024:.1f} KB)")
        print(f"[compress] Lossless ratio: {lossless_ratio:.2f}×  (original / latent+residual)")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralZip — Compress an image.")
    parser.add_argument("--input",   required=True, help="Path to input image")
    parser.add_argument("--model",   required=True, help="Path to trained model .pt file")
    parser.add_argument("--output",  required=True, help="Output directory for latent/residual files")
    parser.add_argument("--lossless", action="store_true", help="Also save residual correction map")
    parser.add_argument("--config",  default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    compress(args.input, args.model, args.output, args.lossless, args.config)


if __name__ == "__main__":
    main()
