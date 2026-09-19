# Dataset Download Instructions

## Automated Download (recommended)

The fastest way to get data is the built-in streaming downloader:

```bash
# EuroSAT — no credentials required, ~90 MB archive
python -m data.download --dataset eurosat

# NIH Chest X-ray — requires Kaggle credentials (see Option B below)
python -m data.download --dataset nih_chest_xray
```

This single command will: stream the archive → extract → flatten → preprocess
all in one step. See `python -m data.download --help` for all options.

---

## Manual Download

NeuralZip requires a dataset of images resized to 1024×1024 pixels.
Two recommended sources are provided below.

---

## Option A — NIH Chest X-ray (grayscale, `in_channels: 1`)

1. Go to: https://www.kaggle.com/datasets/nih-chest-xrays/data
2. Download `images_001.tar.gz` through `images_012.tar.gz` (or a subset for testing).
3. Extract all `.png` files into `data/raw/`:
   ```
   data/raw/00000001_000.png
   data/raw/00000002_000.png
   ...
   ```
4. Run the preprocessor:
   ```bash
   python -m utils.preprocess --src data/raw --dst data/processed --size 1024 --channels 1
   ```

> **Note:** Set `in_channels: 1` in `config.yaml` for grayscale images.

---

## Option B — EuroSAT (RGB satellite imagery, `in_channels: 3`)

1. Go to: https://github.com/phelber/EuroSAT
2. Download the RGB dataset (EuroSAT.zip, ~90MB).
3. Extract into `data/raw/`:
   ```
   data/raw/AnnualCrop/AnnualCrop_00001.jpg
   data/raw/Forest/Forest_00001.jpg
   ...
   ```
4. Run the preprocessor:
   ```bash
   python -m utils.preprocess --src data/raw --dst data/processed --size 1024 --channels 3
   ```

> **Note:** Set `in_channels: 3` in `config.yaml` for RGB images.

---

## Directory Structure After Preprocessing

```
data/
├── raw/            ← original downloaded images (gitignored)
├── processed/      ← resized 1024×1024 PNGs ready for training (gitignored)
└── README.md       ← this file
```

Both `raw/` and `processed/` are gitignored — they must be recreated locally from
the original data source.
