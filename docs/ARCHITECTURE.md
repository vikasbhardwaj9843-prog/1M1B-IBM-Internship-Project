# Architecture

## Overview

NeuralZip uses a **CNN hourglass autoencoder** — a symmetric encoder–decoder pair with
no skip connections. The absence of skip connections is the key design choice: unlike
U-Net, all information must pass through the narrow bottleneck, which forces the network
to learn the most compact possible representation of the input image. This is the
compression constraint.

---

## Encoder

```
Input (B, C, 1024, 1024)
  │
  ├─ ConvBlock(C → 16)      Conv2d(k=4, s=2, p=1) + BN + ReLU   →  (B,  16, 512, 512)
  ├─ ConvBlock(16 → 32)                                           →  (B,  32, 256, 256)
  ├─ ConvBlock(32 → 64)                                           →  (B,  64, 128, 128)
  ├─ ConvBlock(64 → 128)                                          →  (B, 128,  64,  64)
  └─ ConvBlock(128 → 256)                                         →  (B, 256,  32,  32)
```

Each `ConvBlock` uses:
- `Conv2d(kernel_size=4, stride=2, padding=1)` — halves spatial dimensions exactly
- `BatchNorm2d` — stabilises training
- `ReLU(inplace=True)` — non-linearity

The encoder compresses a 1024×1024 image to a **32×32×256 latent tensor**
(262,144 values per image).

---

## Bottleneck

The bottleneck latent tensor `z` is the compressed representation:

- Shape: `(B, 256, 32, 32)`
- Size: 262,144 float32 values ≈ **1 MB per image** at full precision
- Size: 262,144 float16 values ≈ **0.5 MB per image** (optional quantisation)

The bottleneck is saved to disk as a `.latent.pt` file via `torch.save`.

---

## Decoder

```
Latent (B, 256, 32, 32)
  │
  ├─ ConvTransposeBlock(256 → 128)   ConvTranspose2d(k=4,s=2,p=1)+BN+ReLU → (B,128,  64,  64)
  ├─ ConvTransposeBlock(128 → 64)                                            → (B, 64, 128, 128)
  ├─ ConvTransposeBlock(64 → 32)                                             → (B, 32, 256, 256)
  ├─ ConvTransposeBlock(32 → 16)                                             → (B, 16, 512, 512)
  └─ ConvTranspose2d(16 → C, k=4, s=2, p=1) + Sigmoid                      → (B,  C,1024,1024)
```

The final layer uses `Sigmoid` to map all outputs to `[0, 1]`, matching the normalised
input range. No `BatchNorm` on the final layer to avoid constraining output statistics.

---

## Lossless Residual Layer

After near-lossless reconstruction, a **residual correction map** is computed:

```
residual = original − reconstructed        # element-wise, signed float
```

Stored alongside the latent as `<name>.residual.pt`.

**Reconstruction**:
```
exact_original = clamp(decode(z) + residual, 0, 1)
```

The residual is typically very small (high PSNR reconstruction → small residuals),
so the combined latent + residual storage is only modestly larger than near-lossless.

---

## Loss Function

```
Loss = α × MSE(x, x̂) + (1 − α) × SSIM_loss(x, x̂)
```

- **MSE** (`torch.nn.MSELoss`): minimises per-pixel L2 error — overall fidelity.
- **SSIM_loss** (`kornia.losses.ssim_loss`): `1 − SSIM`, differentiable, minimising it
  maximises structural similarity — critical for preserving diagnostic edges in X-rays.
- **α** (default 0.5): configurable in `config.yaml`.

---

## Parameter Count

With `base_channels=16` (default), `in_channels=1` (grayscale):

| Component | Parameters |
|---|---|
| Encoder | ~1.3 M |
| Decoder | ~1.3 M |
| **Total** | **~2.6 M** |

This is intentionally small — the bottleneck compression constraint, not model scale,
drives reconstruction quality.

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| No skip connections | ✓ | Bottleneck must encode everything; skip connections bypass compression |
| Stride-2 Conv (not MaxPool) | ✓ | Clean mirror symmetry with ConvTranspose2d in decoder |
| BatchNorm in all layers except last | ✓ | Training stability; removed on output to avoid constraining output range |
| Sigmoid on decoder output | ✓ | Enforces [0,1] output range matching normalised input |
| Residual lossless strategy | ✓ | No re-training needed; leverages near-lossless quality |
