"""
eval.py — Entry-point: evaluate the best model on the validation set.

Outputs:
  - Mean PSNR and SSIM printed to stdout
  - Side-by-side comparison images saved to outputs/samples/
  - Benchmark numbers written to docs/evaluation_results.md

Usage:
    python eval.py
    python eval.py --config config.yaml --model outputs/best_model.pt
"""
import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from utils.config import load_config
from utils.dataloader import get_dataloaders
from model.autoencoder import Autoencoder
from utils.evaluate import evaluate_batch
from utils.visualise import save_batch_comparisons


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


def write_results(results: dict, config: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Results\n",
        "| Dataset | Mode | PSNR (dB) | SSIM | Notes |\n",
        "|---|---|---|---|---|\n",
        f"| {config.get('dataset', 'unknown')} "
        f"| near-lossless "
        f"| {results['psnr_db']:.2f} "
        f"| {results['ssim']:.4f} "
        f"| Validation set, best model |\n",
    ]
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"[eval] Results written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralZip — Evaluate reconstruction quality.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model",  default=None, help="Override model path from config")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of comparison images to save")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = args.model or config.get("best_model_path", "outputs/best_model.pt")
    print(f"[eval] Loading model from '{model_path}'")
    model = load_model(model_path, config, device)

    _, val_loader = get_dataloaders(config)

    all_psnr, all_ssim, n_batches = 0.0, 0.0, 0
    samples_saved = False

    for x, _ in tqdm(val_loader, desc="Evaluating"):
        x = x.to(device)
        with torch.no_grad():
            x_hat = model(x).clamp(0, 1)

        metrics = evaluate_batch(x_hat, x)
        all_psnr += metrics["psnr_db"]
        all_ssim += metrics["ssim"]
        n_batches += 1

        # Save first batch as visual comparison samples
        if not samples_saved:
            n = min(args.n_samples, x.size(0))
            save_batch_comparisons(
                x[:n].cpu(), x_hat[:n].cpu(),
                config.get("sample_dir", "outputs/samples"),
                prefix="val",
            )
            samples_saved = True

    mean_psnr = all_psnr / n_batches
    mean_ssim = all_ssim / n_batches

    print(f"\n[eval] ── Validation Results ──────────────────")
    print(f"[eval] Mean PSNR : {mean_psnr:.2f} dB")
    print(f"[eval] Mean SSIM : {mean_ssim:.4f}")
    print(f"[eval] Samples   : {config.get('sample_dir', 'outputs/samples')}/")

    results = {"psnr_db": mean_psnr, "ssim": mean_ssim}
    write_results(results, config, Path("docs/evaluation_results.md"))


if __name__ == "__main__":
    main()
