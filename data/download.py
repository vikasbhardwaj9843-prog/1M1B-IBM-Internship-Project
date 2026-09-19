"""
data/download.py — Stream and download open datasets into data/raw/, then preprocess.

Supported datasets
──────────────────
  eurosat       EuroSAT RGB (Zenodo, ~90 MB, no authentication required)
                https://zenodo.org/records/7711810
  nih_chest_xray NIH Chest X-ray14 via Kaggle API (requires kaggle.json credentials)
                https://www.kaggle.com/datasets/nih-chest-xrays/data

Usage
─────
  # EuroSAT — fully automatic, no credentials
  python -m data.download --dataset eurosat

  # NIH Chest X-ray (needs ~/.kaggle/kaggle.json)
  python -m data.download --dataset nih_chest_xray

  # Download only (no preprocessing step)
  python -m data.download --dataset eurosat --no-preprocess

  # Override destination and image size
  python -m data.download --dataset eurosat --raw-dir data/raw --processed-dir data/processed --size 1024

The script streams each file in chunks so memory usage stays flat regardless of
archive size. A tqdm progress bar shows live download speed and ETA.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

from utils.config import load_config
from utils.preprocess import preprocess

# ── Dataset registry ──────────────────────────────────────────────────────────

# EuroSAT: official Zenodo record for the RGB 64×64 jpg archive.
# We download the RGB version and upsample to 1024×1024 in the preprocess step.
_EUROSAT_URL = (
    "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
)

_DATASETS: dict[str, dict] = {
    "eurosat": {
        "url": _EUROSAT_URL,
        "archive_type": "zip",   # "zip" | "tar.gz" | "tar"
        "description": "EuroSAT RGB satellite imagery (Zenodo, ~90 MB, no auth)",
        "in_channels": 3,
    },
    "nih_chest_xray": {
        "url": None,             # downloaded via Kaggle API
        "kaggle_dataset": "nih-chest-xrays/data",
        "archive_type": "zip",
        "description": "NIH Chest X-ray14 (Kaggle, requires ~/.kaggle/kaggle.json)",
        "in_channels": 1,
    },
}

_CHUNK_SIZE = 1024 * 64   # 64 KB chunks — keeps memory flat, good for streaming


# ── Streaming download ─────────────────────────────────────────────────────────

def _stream_download(url: str, dest: Path) -> None:
    """
    Stream a URL to dest, showing a live tqdm progress bar.
    Resumes partial downloads if dest already exists and the server supports
    the Range header.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing_bytes = dest.stat().st_size if dest.exists() else 0

    headers = {}
    if existing_bytes:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"[download] Resuming from {existing_bytes / 1024**2:.1f} MB")

    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("Content-Length", 0)) + existing_bytes
    mode = "ab" if existing_bytes else "wb"

    desc = dest.name[:40]
    with (
        open(dest, mode) as f,
        tqdm(
            total=total if total > 0 else None,
            initial=existing_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
            dynamic_ncols=True,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=_CHUNK_SIZE):
            f.write(chunk)
            bar.update(len(chunk))


# ── Archive extraction ─────────────────────────────────────────────────────────

def _extract(archive_path: Path, dest_dir: Path, archive_type: str) -> None:
    """Extract zip or tar archive with a tqdm member-count progress bar."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[extract] {archive_path.name} → {dest_dir}")

    if archive_type == "zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            members = zf.namelist()
            for member in tqdm(members, desc="Extracting", unit="file", dynamic_ncols=True):
                zf.extract(member, dest_dir)

    elif archive_type in ("tar.gz", "tar"):
        mode = "r:gz" if archive_type == "tar.gz" else "r"
        with tarfile.open(archive_path, mode) as tf:
            members = tf.getmembers()
            for member in tqdm(members, desc="Extracting", unit="file", dynamic_ncols=True):
                tf.extract(member, dest_dir, filter="data")

    else:
        raise ValueError(f"Unknown archive type: {archive_type!r}")


# ── Kaggle downloader ──────────────────────────────────────────────────────────

def _download_kaggle(kaggle_dataset: str, dest_dir: Path) -> None:
    """
    Download a Kaggle dataset using the official kaggle Python package.
    Requires ~/.kaggle/kaggle.json (or KAGGLE_USERNAME + KAGGLE_KEY env vars).
    """
    try:
        import kaggle  # noqa: F401 — just check it's installed
    except ImportError:
        print(
            "[error] The 'kaggle' package is not installed.\n"
            "        Install it with:  pip install kaggle\n"
            "        Then place your API credentials at ~/.kaggle/kaggle.json\n"
            "        See: https://github.com/Kaggle/kaggle-api#api-credentials"
        )
        sys.exit(1)

    creds = Path.home() / ".kaggle" / "kaggle.json"
    if not creds.exists():
        env_user = os.environ.get("KAGGLE_USERNAME")
        env_key = os.environ.get("KAGGLE_KEY")
        if not (env_user and env_key):
            print(
                "[error] Kaggle credentials not found.\n"
                f"        Expected: {creds}\n"
                "        Or set env vars: KAGGLE_USERNAME and KAGGLE_KEY\n"
                "        See: https://github.com/Kaggle/kaggle-api#api-credentials"
            )
            sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] Downloading dataset '{kaggle_dataset}' → {dest_dir}")
    print("[kaggle] This may take several minutes (NIH dataset is ~45 GB total).")
    print("[kaggle] Tip: download only images_001.zip for a smaller sample:")
    print(f"         kaggle datasets download nih-chest-xrays/data -f images_001.zip -p {dest_dir}")

    # Use the kaggle CLI via subprocess so tqdm output is preserved
    import subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            kaggle_dataset,
            "--path", str(dest_dir),
            "--unzip",
        ],
        check=False,
    )
    if result.returncode != 0:
        print("[error] Kaggle download failed. Check credentials and dataset name.")
        sys.exit(1)


# ── Flatten extracted directory ────────────────────────────────────────────────

def _flatten_images(src_dir: Path, raw_dir: Path, extensions: set[str]) -> None:
    """
    Move all image files from src_dir (recursively) into raw_dir, preserving
    relative paths. Used to normalise the extracted directory structure.
    """
    found = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in extensions)
    if not found:
        print(f"[flatten] No images found under {src_dir}")
        return
    print(f"[flatten] Moving {len(found)} images into {raw_dir}")
    for src in tqdm(found, desc="Moving", unit="img", dynamic_ncols=True):
        rel = src.relative_to(src_dir)
        dst = raw_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))


# ── Main download + preprocess pipeline ───────────────────────────────────────

def download_and_prepare(
    dataset: str,
    raw_dir: Path,
    processed_dir: Path,
    image_size: int,
    do_preprocess: bool = True,
    keep_archive: bool = False,
) -> None:
    """
    Full pipeline: stream download → extract → flatten → preprocess.

    Args:
        dataset      : Dataset key ("eurosat" or "nih_chest_xray").
        raw_dir      : Destination for raw downloaded images.
        processed_dir: Destination for preprocessed images.
        image_size   : Target spatial resolution for preprocessing.
        do_preprocess: If False, stop after download+extract.
        keep_archive : If True, keep the downloaded archive file.
    """
    if dataset not in _DATASETS:
        print(f"[error] Unknown dataset '{dataset}'. Choose from: {list(_DATASETS)}")
        sys.exit(1)

    meta = _DATASETS[dataset]
    in_channels = meta["in_channels"]
    print(f"\n{'─'*60}")
    print(f" NeuralZip Data Downloader")
    print(f" Dataset    : {dataset}")
    print(f" Description: {meta['description']}")
    print(f" Raw dir    : {raw_dir}")
    print(f" Processed  : {processed_dir}")
    print(f"{'─'*60}\n")

    raw_dir.mkdir(parents=True, exist_ok=True)

    if dataset == "nih_chest_xray":
        # ── NIH Chest X-ray via Kaggle ──────────────────────────────────────
        _download_kaggle(meta["kaggle_dataset"], raw_dir)
        # Kaggle --unzip already extracts; images land directly in raw_dir

    else:
        # ── Direct URL streaming download ───────────────────────────────────
        archive_name = _EUROSAT_URL.split("/")[-1].split("?")[0]
        archive_path = raw_dir.parent / archive_name

        if archive_path.exists():
            print(f"[download] Archive already exists: {archive_path}. Skipping download.")
        else:
            print(f"[download] Streaming from:\n  {meta['url']}\n")
            _stream_download(meta["url"], archive_path)

        # Extract archive
        extract_tmp = raw_dir.parent / "_extract_tmp"
        _extract(archive_path, extract_tmp, meta["archive_type"])

        # Flatten all images into raw_dir
        _extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        _flatten_images(extract_tmp, raw_dir, _extensions)

        # Clean up temp extraction directory
        shutil.rmtree(extract_tmp, ignore_errors=True)
        print(f"[download] Temporary extraction dir removed.")

        if not keep_archive:
            archive_path.unlink(missing_ok=True)
            print(f"[download] Archive removed (use --keep-archive to retain).")

    # ── Preprocess ───────────────────────────────────────────────────────────
    if do_preprocess:
        print(f"\n[preprocess] Resizing all images to {image_size}×{image_size} …")
        preprocess(raw_dir, processed_dir, image_size, in_channels)
    else:
        print("[preprocess] Skipped (--no-preprocess flag set).")

    print(f"\n✓ Done! Processed images in: {processed_dir}")
    print(f"  Next step: python -m train.train\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        description="NeuralZip — Stream-download a dataset and preprocess for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m data.download --dataset eurosat
  python -m data.download --dataset nih_chest_xray
  python -m data.download --dataset eurosat --no-preprocess --keep-archive
  python -m data.download --dataset eurosat --size 512 --channels 3
        """,
    )
    parser.add_argument(
        "--dataset",
        choices=list(_DATASETS),
        default=cfg.get("dataset", "eurosat"),
        help="Dataset to download (default: from config.yaml)",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory to stream raw files into (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Directory for preprocessed images (default: data/processed)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=cfg.get("image_size", 1024),
        help="Output image size in pixels (default: from config.yaml)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=None,
        help="Override in_channels (1=grayscale, 3=RGB). Default: per-dataset setting.",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Download and extract only; skip the preprocess step.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded archive file after extraction.",
    )
    args = parser.parse_args()

    # channels override
    if args.channels is not None:
        _DATASETS[args.dataset]["in_channels"] = args.channels

    download_and_prepare(
        dataset=args.dataset,
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        image_size=args.size,
        do_preprocess=not args.no_preprocess,
        keep_archive=args.keep_archive,
    )


if __name__ == "__main__":
    main()
