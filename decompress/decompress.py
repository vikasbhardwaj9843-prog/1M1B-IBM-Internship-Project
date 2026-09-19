"""
decompress/decompress.py — CLI: reconstruct an image from a latent tensor.

Usage:
    # Near-lossless reconstruction
    python -m decompress.decompress --latent compressed/image.latent.pt \\
        --model outputs/best_model.pt --output reconstructed/

    # Lossless (bit-exact) reconstruction
    python -m decompress.decompress --latent compressed/image.latent.pt \\
        --residual compressed/image.residual.pt \\
        --model outputs/best_model.pt --output reconstructed/
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import save_image

from utils.config import load_config
from model.autoencoder import Autoencoder


def load_model(model_path: str, config: dict, device: torch.device) -> Autoencoder:
    model = Autoencoder(
        in_channels=config["in_channels"],
        base_channels=config["base_channels"],
        bottleneck_channels=config["bottleneck_channels"],
    )
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def decompress(
    latent_path: str,
    model_path: str,
    output_dir: str,
    residual_path: str | None = None,
    config_path: str = "config.yaml",
) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(latent_path).name.replace(".latent.pt", "")

    model = load_model(model_path, config, device)

    # Load latent
    z = torch.load(latent_path, map_location=device)   # (1, 256, 32, 32)

    with torch.no_grad():
        x_hat = model.decode(z).clamp(0, 1)           # (1, C, H, W) in [0, 1]

    # Lossless mode: add residual correction
    if residual_path is not None:
        residual = torch.load(residual_path, map_location=device)  # (1, C, H, W)
        x_hat = (x_hat + residual).clamp(0, 1)
        mode = "lossless"
    else:
        mode = "near-lossless"

    out_path = out_dir / f"{stem}_reconstructed.png"
    save_image(x_hat, out_path)

    print(f"[decompress] Mode        : {mode}")
    print(f"[decompress] Latent      : {latent_path}")
    if residual_path:
        print(f"[decompress] Residual    : {residual_path}")
    print(f"[decompress] Reconstructed: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralZip — Decompress a latent to an image.")
    parser.add_argument("--latent",   required=True, help="Path to .latent.pt file")
    parser.add_argument("--model",    required=True, help="Path to trained model .pt file")
    parser.add_argument("--output",   required=True, help="Output directory for reconstructed image")
    parser.add_argument("--residual", default=None,  help="Path to .residual.pt for lossless mode")
    parser.add_argument("--config",   default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    decompress(args.latent, args.model, args.output, args.residual, args.config)


if __name__ == "__main__":
    main()
