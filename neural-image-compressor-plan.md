# Neural Image Compressor â€” Plan

## Top-Level Overview

Build a PyTorch-based image compression and decompression system for **1024Ã—1024 images**
(primary targets: medical imaging such as NIH Chest X-rays, and satellite imagery such as EuroSAT).

The system uses a **CNN hourglass autoencoder** whose bottleneck forces the network to learn
compact feature representations. Two operating modes are supported:

- **Near-lossless mode** â€” encoder + decoder only; loss measured with MSE + SSIM.
- **Lossless mode** â€” near-lossless reconstruction plus a stored residual correction map;
  combining the two gives bit-exact reconstruction.

Compressed artifacts are stored as PyTorch `.pt` or NumPy `.npz` files â€” no custom I/O
format is required.

The project is framed around sustainable digital archival storage, aligning with
**UN SDG 9** (Industry, Innovation and Infrastructure) and **UN SDG 12**
(Responsible Consumption and Production).

---

## Architecture Diagram (described in text â€” render in Mermaid outside this file)

```
Input 1024Ã—1024
       â”‚
  [Encoder â€” CNN downsampling blocks]
  1024 â†’ 512 â†’ 256 â†’ 128 â†’ 64 â†’ 32  (spatial)
  channels: 1/3 â†’ 16 â†’ 32 â†’ 64 â†’ 128 â†’ 256
       â”‚
  [Bottleneck latent tensor]  â† saved as .pt / .npz
       â”‚
  [Decoder â€” CNN upsampling blocks]
  32 â†’ 64 â†’ 128 â†’ 256 â†’ 512 â†’ 1024  (spatial)
  channels: 256 â†’ 128 â†’ 64 â†’ 32 â†’ 16 â†’ 1/3
       â”‚
  Reconstructed 1024Ã—1024
       â”‚
  [Residual = Original âˆ’ Reconstructed]  â† saved alongside latent (lossless mode)
```

---

## Sub-Tasks

---

### Sub-Task 1 â€” Project Scaffolding

**Status:** `[x] done`

**Intent**
Create the repository directory structure, dependency files, and configuration so every
subsequent sub-task has a stable foundation to build on.

**Expected Outcomes**
- Directory tree with `model/`, `train/`, `compress/`, `decompress/`, `utils/`, `data/`, `docs/`
- `requirements.txt` listing all Python dependencies
- `config.yaml` holding all hyper-parameters (image size, channels, bottleneck dim, batch size,
  learning rate, epochs, dataset path, output paths)
- `.gitignore` appropriate for Python / PyTorch projects

**Todo List**
1. Create top-level directories: `model/`, `train/`, `compress/`, `decompress/`, `utils/`, `data/raw/`, `data/processed/`, `docs/`
2. Add `__init__.py` stubs in each Python package directory (`model`, `train`, `compress`, `decompress`, `utils`)
3. Write `requirements.txt` â€” pin: `torch`, `torchvision`, `numpy`, `Pillow`, `scikit-image`, `pyyaml`, `tqdm`, `matplotlib`
4. Write `config.yaml` â€” keys: `image_size: 1024`, `in_channels: 1`, `base_channels: 16`, `bottleneck_channels: 256`, `batch_size: 8`, `lr: 1e-4`, `epochs: 50`, `dataset: nih_chest_xray`, `data_dir: data/raw`, `output_dir: outputs/`
5. Write `.gitignore` â€” exclude `__pycache__`, `*.pt`, `*.npz`, `data/raw/`, `outputs/`

**Relevant Context**
- Fresh repository â€” no existing code to integrate with.
- `config.yaml` will be loaded by every other module via a `utils/config.py` helper.

---

### Sub-Task 2 â€” Data Pipeline

**Status:** `[x] done`

**Intent**
Build a reproducible data loading pipeline that feeds 1024Ã—1024 grayscale (or RGB) image
tensors to the trainer. Handles download instructions, preprocessing, and train/val split.

**Expected Outcomes**
- `utils/config.py` â€” loads and exposes `config.yaml` as a Python dict
- `data/README.md` â€” instructions for downloading NIH Chest X-ray dataset from Kaggle or
  EuroSAT from the official source
- `utils/dataset.py` â€” `ImageDataset` class (PyTorch `Dataset`) that:
  - reads images from `data/processed/`
  - resizes to 1024Ã—1024
  - normalises pixel values to [0, 1]
  - returns `(tensor, tensor)` pairs (same image used as both input and target)
- `utils/dataloader.py` â€” `get_dataloaders(config)` returning train and val `DataLoader`s

**Todo List**
1. Write `utils/config.py` â€” `load_config(path="config.yaml")` function using `pyyaml`
2. Write `data/README.md` â€” step-by-step download and placement instructions for NIH Chest X-ray and EuroSAT
3. Write `utils/dataset.py` â€” `ImageDataset(root_dir, transform=None)` using `torchvision.transforms`
4. Write `utils/dataloader.py` â€” `get_dataloaders(config)` with 80/20 train-val split, `num_workers=4`
5. Add a `utils/preprocess.py` script that resizes raw images to 1024Ã—1024 and saves them to `data/processed/`

**Relevant Context**
- `config.yaml` from Sub-Task 1 drives all paths and sizes.
- Medical images (X-rays) are typically single-channel (grayscale); satellite images are RGB.
  `in_channels` in `config.yaml` controls this.

---

### Sub-Task 3 â€” Model Architecture

**Status:** `[x] done`

**Intent**
Implement the CNN hourglass autoencoder in PyTorch. The encoder compresses spatial
dimensions while expanding channels; the decoder mirrors this. The bottleneck is the
narrowest point and is the representation that gets saved.

**Expected Outcomes**
- `model/encoder.py` â€” `Encoder` class
- `model/decoder.py` â€” `Decoder` class
- `model/autoencoder.py` â€” `Autoencoder` class composing encoder and decoder, exposing
  `encode(x)` and `decode(z)` and `forward(x)` methods
- `model/__init__.py` â€” exports `Autoencoder`
- Model summary printable via `python -m model` showing parameter count at each layer

**Todo List**
1. Write `model/blocks.py` â€” reusable `ConvBlock(in_ch, out_ch, stride)` (Conv2d + BN + ReLU)
   and `ConvTransposeBlock(in_ch, out_ch, stride)` (ConvTranspose2d + BN + ReLU)
2. Write `model/encoder.py` â€” `Encoder` downsampling from 1024 to 32 spatial, channels
   `in_ch â†’ 16 â†’ 32 â†’ 64 â†’ 128 â†’ 256`
3. Write `model/decoder.py` â€” `Decoder` upsampling mirror of encoder, final layer uses
   `Sigmoid` activation to output values in [0, 1]
4. Write `model/autoencoder.py` â€” `Autoencoder` with `encode`, `decode`, `forward`
5. Write `model/__init__.py` exporting `Autoencoder`
6. Verify parameter count is logged when the model is instantiated

**Relevant Context**
- Bottleneck spatial size: 32Ã—32 with 256 channels = **262 144 values** per image.
  At float32 this is ~1 MB; with quantisation or float16 this halves to ~512 KB.
- Use `stride=2` convolutions (not MaxPool) so the decoder transpose convolutions
  are a clean mirror.
- No skip connections (unlike U-Net) â€” the bottleneck must encode everything,
  which is the compression constraint.

---

### Sub-Task 4 â€” Loss Function

**Status:** `[x] done`

**Intent**
Define a combined loss that drives the network toward high-fidelity reconstruction.
Pure MSE tends to produce blurry outputs; adding SSIM preserves structural detail,
which is critical for medical and scientific imagery.

**Expected Outcomes**
- `train/loss.py` â€” `ReconstructionLoss` class with configurable `alpha` weight
  between MSE and SSIM components
- Loss value is logged per epoch during training

**Todo List**
1. Write `train/loss.py` â€” `ReconstructionLoss(alpha=0.5)`:
   - MSE component: `torch.nn.MSELoss`
   - SSIM component: use `kornia.losses.ssim_loss` or implement a simple SSIM via
     `skimage.metrics.structural_similarity` wrapped in a differentiable form
   - Combined: `loss = alpha * mse + (1 - alpha) * ssim_loss`
2. Add `kornia` to `requirements.txt` (provides differentiable SSIM in PyTorch)
3. Write a short unit test in `train/test_loss.py` that passes a known identical pair
   and a zeroed pair, asserting loss behaves correctly

**Relevant Context**
- `alpha` is exposed in `config.yaml` so it can be tuned without code changes.
- SSIM is particularly important for X-ray images where structural edges are diagnostically critical.

---

### Sub-Task 5 â€” Training Loop

**Status:** `[x] done`

**Intent**
Implement the training script that loads data, runs forward/backward passes, saves
checkpoints, and logs metrics.

**Expected Outcomes**
- `train/trainer.py` â€” `Trainer` class handling training and validation loops
- `train/train.py` â€” entry-point script runnable as `python -m train.train`
- Checkpoints saved to `outputs/checkpoints/` after each epoch
- Best model (lowest val loss) saved as `outputs/best_model.pt`
- Per-epoch loss logged to `outputs/training_log.csv`

**Todo List**
1. Write `train/trainer.py` â€” `Trainer(model, optimizer, loss_fn, train_loader, val_loader, config)`:
   - `train_epoch()` â€” forward pass, loss, backward, optimizer step
   - `val_epoch()` â€” no grad, compute val loss and PSNR
   - `fit()` â€” outer loop calling both, saving checkpoints and best model
2. Write `train/train.py` â€” loads config, instantiates model / optimizer / loss / dataloaders,
   creates `Trainer`, calls `fit()`
3. Add PSNR metric alongside loss in validation logging
4. Add `tqdm` progress bars for epoch and batch loops

**Relevant Context**
- Use `torch.optim.Adam` with `lr` from `config.yaml`.
- `outputs/` is gitignored; users create it on first run.
- Checkpoint format: `{"epoch": n, "model_state": ..., "optimizer_state": ..., "val_loss": ...}`

---

### Sub-Task 6 â€” Compression (Encode + Save)

**Status:** `[x] done`

**Intent**
Build the CLI tool that takes an input image, runs it through the trained encoder,
and saves the latent tensor (and optionally the residual) to disk.

**Expected Outcomes**
- `compress/compress.py` â€” script runnable as:
  `python -m compress.compress --input image.png --model outputs/best_model.pt --output compressed/`
- Saves `<name>.latent.pt` (latent tensor) to the output directory
- In `--lossless` mode also saves `<name>.residual.pt` (original minus reconstructed)
- Prints compression ratio to stdout

**Todo List**
1. Write `compress/compress.py` â€” argument parser for `--input`, `--model`, `--output`, `--lossless`
2. Load image â†’ preprocess â†’ encode â†’ save latent as `.pt`
3. In lossless mode: decode latent â†’ compute residual = original âˆ’ reconstructed â†’ save residual as `.pt`
4. Compute and print compression ratio: `original_bytes / latent_bytes`
5. Write `compress/__init__.py`

**Relevant Context**
- Latent tensor shape: `[1, 256, 32, 32]` â†’ 262 144 float32 values â†’ ~1 MB.
- A 1024Ã—1024 grayscale PNG is typically 0.5â€“2 MB; RGB uncompressed TIFF is ~3 MB.
  The compression ratio depends on the original file format.
- For archival use, saving latents as `float16` halves storage with minimal quality impact.

---

### Sub-Task 7 â€” Decompression (Load + Decode)

**Status:** `[x] done`

**Intent**
Build the CLI tool that loads a saved latent (and optionally residual) and reconstructs
the original image.

**Expected Outcomes**
- `decompress/decompress.py` â€” script runnable as:
  `python -m decompress.decompress --latent compressed/image.latent.pt --model outputs/best_model.pt --output reconstructed/`
- In `--lossless` mode, also accepts `--residual compressed/image.residual.pt`
- Saves reconstructed image as `.png`

**Todo List**
1. Write `decompress/decompress.py` â€” argument parser for `--latent`, `--residual`, `--model`, `--output`
2. Load latent â†’ decode â†’ (if residual provided) add residual â†’ clamp to [0, 1]
3. Denormalise and save as PNG using `torchvision.utils.save_image`
4. Write `decompress/__init__.py`

**Relevant Context**
- Residual addition must be done in the same dtype / range as training (normalised [0, 1]).
- Round-trip validation: `original == round(decode(encode(original)) + residual)` should hold for lossless mode.

---

### Sub-Task 8 â€” Evaluation & Visualisation

**Status:** `[x] done`

**Intent**
Provide tooling to measure reconstruction quality and produce visual comparisons,
enabling reproducible benchmarking.

**Expected Outcomes**
- `utils/evaluate.py` â€” functions computing PSNR, SSIM, and compression ratio for a
  batch of image pairs
- `utils/visualise.py` â€” saves side-by-side original vs reconstructed plots to `outputs/samples/`
- `docs/evaluation_results.md` â€” template for recording benchmark numbers

**Todo List**
1. Write `utils/evaluate.py` â€” `evaluate_batch(originals, reconstructeds)` returning
   mean PSNR and SSIM using `skimage.metrics`
2. Write `utils/visualise.py` â€” `save_comparison(original, reconstructed, path)` using `matplotlib`
3. Write `docs/evaluation_results.md` â€” table template with columns:
   Dataset | Mode | PSNR (dB) | SSIM | Compression Ratio | Notes
4. Add an `eval.py` entry-point script that loads the best model, runs on val set, and fills the table

**Relevant Context**
- PSNR > 40 dB is generally considered excellent for medical imaging.
- Results feed directly into the README's claims about quality and sustainability.

---

### Sub-Task 9 â€” Documentation

**Status:** `[x] done`

**Intent**
Produce the technical documentation file that records the full system design,
rationale, and usage â€” the durable reference for future contributors.

**Expected Outcomes**
- `docs/ARCHITECTURE.md` â€” full architecture description, design decisions, and
  loss function rationale
- `docs/USAGE.md` â€” step-by-step: install â†’ data prep â†’ train â†’ compress â†’ decompress â†’ evaluate
- `docs/SDG_ALIGNMENT.md` â€” detailed mapping of the project to UN SDG 9 and SDG 12

**Todo List**
1. Write `docs/ARCHITECTURE.md` â€” diagram (ASCII), layer-by-layer description,
   bottleneck analysis, lossless residual design
2. Write `docs/USAGE.md` â€” full command examples with expected outputs
3. Write `docs/SDG_ALIGNMENT.md` â€” narrative + table mapping project features to
   SDG 9 targets (9.4, 9.b) and SDG 12 targets (12.2, 12.5)

**Relevant Context**
- SDG 9.4: Upgrade infrastructure for sustainability.
- SDG 9.b: Support domestic technology development.
- SDG 12.2: Sustainable management of natural resources (digital storage = energy).
- SDG 12.5: Substantially reduce waste generation.

---

## Implementation Order

```
Sub-Task 1 (Scaffolding)
      â”‚
Sub-Task 2 (Data Pipeline)
      â”‚
Sub-Task 3 (Model Architecture)
      â”‚
Sub-Task 4 (Loss Function)
      â”‚
Sub-Task 5 (Training Loop)
      â”‚
Sub-Task 6 (Compression CLI)
      â”‚
Sub-Task 7 (Decompression CLI)
      â”‚
Sub-Task 8 (Evaluation)
      â”‚
Sub-Task 9 (Documentation)
```

Each sub-task depends on the one above it. Sub-tasks 6 and 7 can be developed in parallel
after Sub-Task 5 is complete.

---

## Key Design Decisions (Recorded for Future Reference)

| Decision | Choice | Rationale |
|---|---|---|
| Architecture | CNN hourglass, no skip connections | Bottleneck must encode all information; skip connections would bypass compression |
| Loss | MSE + SSIM (alpha=0.5) | MSE minimises pixel error; SSIM preserves structural quality for medical/scientific use |
| Lossless strategy | Residual correction map | Avoids re-training; residual is tiny after good near-lossless reconstruction |
| Artifact format | `.pt` / `.npz` | Standard, no custom I/O; easy to load in any PyTorch environment |
| Compression ratio | Fixed single target | Simpler model; one optimised point is sufficient for archival use |
| Dataset | NIH Chest X-ray / EuroSAT | Publicly available, domain-relevant, widely cited benchmarks |
| Channels | Configurable via `in_channels` | Same codebase supports grayscale (X-ray) and RGB (satellite) |

