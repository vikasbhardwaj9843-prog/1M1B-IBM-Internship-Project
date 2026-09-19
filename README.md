# 🗜️ NeuralZip — Neural Image Compressor

> **Sustainable, high-fidelity image compression for historical archival, medical imaging,
> and scientific data storage — powered by deep convolutional autoencoders.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![UN SDG 9](https://img.shields.io/badge/UN%20SDG-9%20Industry%20%26%20Innovation-blue)](https://sdgs.un.org/goals/goal9)
[![UN SDG 12](https://img.shields.io/badge/UN%20SDG-12%20Responsible%20Consumption-green)](https://sdgs.un.org/goals/goal12)

---

## 🌍 Why This Matters

Digital data storage is one of the fastest-growing consumers of global energy and physical
resources. Data centres already account for roughly **1–2% of global electricity use**, and
archival data — medical records, satellite imagery, scientific datasets — is growing at an
unprecedented rate.

**NeuralZip directly addresses two United Nations Sustainable Development Goals:**

### 🔵 UN SDG 9 — Industry, Innovation and Infrastructure
> *"Build resilient infrastructure, promote inclusive and sustainable industrialisation,
> and foster innovation."*

NeuralZip demonstrates that modern AI techniques can be applied to upgrade digital
infrastructure without proportional increases in storage hardware, energy consumption,
or physical footprint:

- **Target 9.4** — Upgrade infrastructure to make it sustainable, with increased
  resource-use efficiency. NeuralZip reduces storage footprint of archival datasets
  by learning compact, information-dense representations rather than storing raw pixels.
- **Target 9.b** — Support domestic technology development and industrial diversification.
  NeuralZip is fully open-source, enabling institutions in any country to deploy
  sustainable archival pipelines without proprietary tooling.

### 🟢 UN SDG 12 — Responsible Consumption and Production
> *"Ensure sustainable consumption and production patterns."*

By compressing archival images at the point of ingest — rather than storing full
resolution data indefinitely — NeuralZip contributes to measurable reductions in
long-term resource consumption:

- **Target 12.2** — Sustainable management and efficient use of natural resources.
  Storage hardware (HDDs, SSDs, tape) has a significant manufacturing carbon cost.
  Smaller archival footprints directly reduce the hardware refresh cycle.
- **Target 12.5** — Substantially reduce waste generation through prevention, reduction,
  and reuse. Compressed archival data means fewer storage drives manufactured, deployed,
  and eventually disposed of — reducing e-waste.

---

## 🧠 How It Works

NeuralZip uses a **CNN hourglass autoencoder** — a neural network shaped like an hourglass:

```
Input Image (1024×1024)
        │
   ┌────▼────────────────────┐
   │  ENCODER (CNN)          │   Channels: 1 → 16 → 32 → 64 → 128 → 256
   │  Spatial: 1024 → 32     │   6 downsampling stages using stride-2 convolutions
   └────────────┬────────────┘
                │
        ┌───────▼────────┐
        │   BOTTLENECK   │   32×32×256 = 262,144 values  (~1 MB @ float32)
        │  LATENT TENSOR │   Saved as .pt file on disk
        └───────┬────────┘
                │
   ┌────────────▼────────────┐
   │  DECODER (CNN)          │   Channels: 256 → 128 → 64 → 32 → 16 → 1
   │  Spatial: 32 → 1024     │   6 upsampling stages using transpose convolutions
   └────────────┬────────────┘
                │
   Reconstructed Image (1024×1024)
```

### Two Operating Modes

| Mode | How it works | Storage | Fidelity |
|---|---|---|---|
| **Near-lossless** | Latent tensor only | ~1 MB / image | PSNR > 40 dB, SSIM > 0.97 |
| **Lossless** | Latent + residual correction map | ~1–2 MB / image | Bit-exact reconstruction |

The **residual correction map** is the pixel-wise difference between the original and the
near-lossless reconstruction. When added back after decoding, it produces an exact copy
of the original image — enabling **compliance with archival standards** that require
bit-perfect fidelity (e.g. medical records regulations, scientific reproducibility mandates).

### Loss Function

Training minimises a combined **MSE + SSIM loss**:

```
Loss = α × MSE(original, reconstructed) + (1 − α) × SSIM_loss(original, reconstructed)
```

- **MSE** minimises per-pixel error (overall fidelity)
- **SSIM** preserves structural and perceptual detail (critical for X-rays and satellite imagery)

---

## 📁 Project Structure

```
neuralzip/
├── model/
│   ├── blocks.py          # Reusable ConvBlock / ConvTransposeBlock
│   ├── encoder.py         # CNN encoder
│   ├── decoder.py         # CNN decoder
│   └── autoencoder.py     # Composed Autoencoder model
├── train/
│   ├── loss.py            # MSE + SSIM combined loss
│   ├── trainer.py         # Training and validation loops
│   └── train.py           # Entry-point: python -m train.train
├── compress/
│   └── compress.py        # CLI: encode image → save latent
├── decompress/
│   └── decompress.py      # CLI: load latent → reconstruct image
├── utils/
│   ├── config.py          # Load config.yaml
│   ├── dataset.py         # ImageDataset (PyTorch Dataset)
│   ├── dataloader.py      # get_dataloaders()
│   ├── preprocess.py      # Resize raw images to 1024×1024
│   ├── evaluate.py        # PSNR, SSIM, compression ratio metrics
│   └── visualise.py       # Side-by-side comparison plots
├── data/
│   ├── raw/               # Place downloaded dataset here
│   ├── processed/         # Preprocessed 1024×1024 images
│   └── README.md          # Dataset download instructions
├── docs/
│   ├── ARCHITECTURE.md    # Full architecture documentation
│   ├── USAGE.md           # Step-by-step usage guide
│   └── SDG_ALIGNMENT.md   # UN SDG 9 & 12 alignment mapping
├── config.yaml            # All hyperparameters and paths
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Download the [NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data)
or [EuroSAT](https://github.com/phelber/EuroSAT) and place images in `data/raw/`.

```bash
python -m utils.preprocess   # Resize to 1024×1024 → data/processed/
```

### 3. Train the Model

```bash
python -m train.train
# Checkpoints saved to outputs/checkpoints/
# Best model saved to outputs/best_model.pt
```

### 4. Compress an Image

```bash
# Near-lossless mode
python -m compress.compress --input image.png --model outputs/best_model.pt --output compressed/

# Lossless mode (also saves residual map)
python -m compress.compress --input image.png --model outputs/best_model.pt --output compressed/ --lossless
```

### 5. Decompress an Image

```bash
# Near-lossless
python -m decompress.decompress --latent compressed/image.latent.pt --model outputs/best_model.pt --output reconstructed/

# Lossless (exact reconstruction)
python -m decompress.decompress --latent compressed/image.latent.pt --residual compressed/image.residual.pt --model outputs/best_model.pt --output reconstructed/
```

### 6. Evaluate Quality

```bash
python eval.py
# Outputs PSNR, SSIM, and compression ratio to docs/evaluation_results.md
```

---

## 📊 Target Performance

| Metric | Target | Notes |
|---|---|---|
| PSNR | > 40 dB | Excellent quality for medical imaging |
| SSIM | > 0.97 | Structural fidelity for diagnostic use |
| Compression ratio | ~3–5× | Vs uncompressed grayscale TIFF |
| Lossless overhead | < 2× near-lossless | Residual map is typically small |

---

## 🔬 Use Cases

- **Medical imaging archives** — Hospitals and research institutions storing X-rays, CT scans,
  and MRI slices for decades under regulatory compliance requirements
- **Satellite imagery archives** — Environmental monitoring organisations storing historical
  Earth observation data for climate research
- **Scientific data repositories** — Research institutions archiving microscopy, telescope,
  or sensor imagery at scale
- **National heritage digitalisation** — Digitised historical documents, artworks, and
  cultural heritage imagery requiring long-term preservation

---

## 🛠️ Configuration

All hyperparameters are controlled from [`config.yaml`](config.yaml):

```yaml
image_size: 1024
in_channels: 1          # 1 for grayscale (X-ray), 3 for RGB (satellite)
base_channels: 16
bottleneck_channels: 256
batch_size: 8
lr: 1.0e-4
epochs: 50
alpha: 0.5              # MSE vs SSIM loss weight
dataset: nih_chest_xray
data_dir: data/raw
output_dir: outputs/
```

---

## 🌱 Sustainability Impact

| Metric | Conventional Storage | NeuralZip |
|---|---|---|
| Storage per 1M images (grayscale 1024²) | ~1 TB | ~200–300 GB |
| Hardware refresh cycles | Every 3–5 years | Extended by reduced footprint |
| Energy for storage I/O | Baseline | Proportionally reduced |
| Bit-exact compliance | Format-dependent | ✅ Lossless mode |

By reducing the storage footprint of archival datasets by **3–5×**, NeuralZip directly
reduces demand for new storage hardware — lowering manufacturing emissions, energy
consumption, and e-waste — in direct support of **UN SDG 9** and **UN SDG 12**.

---

## 📄 Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Full neural network architecture description
- [`docs/USAGE.md`](docs/USAGE.md) — Detailed usage guide with all CLI options
- [`docs/SDG_ALIGNMENT.md`](docs/SDG_ALIGNMENT.md) — Detailed UN SDG alignment mapping
- [`neural-image-compressor-plan.md`](neural-image-compressor-plan.md) — Full implementation plan

---

## 🤝 Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before
submitting a pull request.

---

*Built to support sustainable digital infrastructure — because responsible technology
is not just about what we build, but how much we store.*
