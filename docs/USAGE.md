# Usage Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended (CPU works but is slow for 1024×1024 images)

---

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Prepare Data

### Option A — Automated streaming download (recommended)

The downloader streams the archive in chunks, extracts it, and runs preprocessing
in a single command. No manual steps required.

```bash
# EuroSAT RGB — no credentials, ~90 MB download (default dataset)
python -m data.download --dataset eurosat

# NIH Chest X-ray — requires ~/.kaggle/kaggle.json  (see Option B)
python -m data.download --dataset nih_chest_xray

# All available flags:
python -m data.download --help
```

### Option B — Manual download

See [`data/README.md`](../data/README.md) for step-by-step instructions.
After placing raw images in `data/raw/`, run the preprocessor separately:

```bash
# EuroSAT (RGB)
python -m utils.preprocess --src data/raw --dst data/processed --size 1024 --channels 3

# NIH Chest X-ray (grayscale)
python -m utils.preprocess --src data/raw --dst data/processed --size 1024 --channels 1
```

---

## 3. Configure

Edit [`config.yaml`](../config.yaml) to match your dataset and hardware:

```yaml
in_channels: 1       # 1 for X-ray, 3 for RGB
batch_size: 8        # Reduce to 4 if GPU OOM on 1024×1024
epochs: 50
lr: 1.0e-4
alpha: 0.5           # MSE vs SSIM balance
```

---

## 4. Train

```bash
python -m train.train
```

**Expected output:**
```
[Autoencoder] Parameters — Encoder: 1,312,896  Decoder: 1,312,577  Total: 2,625,473
[train] Dataset     : 7200 train / 1800 val images
[Trainer] Training on cuda for 50 epochs.
Epoch   1/50  train_loss=0.04231  val_loss=0.03987  val_psnr=32.14 dB
...
  ✓ New best model saved (val_loss=0.00812)
```

Outputs written to:
- `outputs/checkpoints/epoch_NNN.pt` — per-epoch checkpoints
- `outputs/best_model.pt` — best validation loss checkpoint
- `outputs/training_log.csv` — epoch-by-epoch metrics

---

## 5. Compress an Image

### Near-lossless mode

```bash
python -m compress.compress \
    --input  path/to/image.png \
    --model  outputs/best_model.pt \
    --output compressed/
```

Output:
```
[compress] Input  : path/to/image.png  (512.4 KB)
[compress] Latent : compressed/image.latent.pt  (1024.0 KB)
[compress] Ratio  : 0.50×  (original / latent)
```

> **Note on ratio:** Raw PNG may already be smaller than 1 MB; compare against
> uncompressed TIFF for a fair ratio benchmark.

### Lossless mode (adds residual)

```bash
python -m compress.compress \
    --input   path/to/image.png \
    --model   outputs/best_model.pt \
    --output  compressed/ \
    --lossless
```

Output also includes:
```
[compress] Residual      : compressed/image.residual.pt  (48.2 KB)
[compress] Lossless ratio: 0.48×
```

---

## 6. Decompress

### Near-lossless reconstruction

```bash
python -m decompress.decompress \
    --latent  compressed/image.latent.pt \
    --model   outputs/best_model.pt \
    --output  reconstructed/
```

### Lossless (bit-exact) reconstruction

```bash
python -m decompress.decompress \
    --latent   compressed/image.latent.pt \
    --residual compressed/image.residual.pt \
    --model    outputs/best_model.pt \
    --output   reconstructed/
```

Output:
```
[decompress] Mode         : lossless
[decompress] Reconstructed: reconstructed/image_reconstructed.png
```

---

## 7. Evaluate Quality

```bash
python eval.py
```

Outputs:
- Mean PSNR and SSIM on the validation set
- Comparison images in `outputs/samples/`
- Updated benchmark table in `docs/evaluation_results.md`

```
[eval] Mean PSNR : 42.31 dB
[eval] Mean SSIM : 0.9812
```

---

## File Reference

| Command | Purpose |
|---|---|
| `python -m utils.preprocess` | Resize raw images to 1024×1024 |
| `python -m train.train` | Train the autoencoder |
| `python -m compress.compress` | Encode image → latent (+ residual) |
| `python -m decompress.decompress` | Decode latent → reconstructed image |
| `python eval.py` | Evaluate PSNR/SSIM on validation set |
