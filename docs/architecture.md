# SPAI Architecture

**SPAI (Spectral AI-Generated Image Detector)** is the official implementation of the CVPR 2025 paper *"Any-Resolution AI-Generated Image Detection by Spectral Learning"*. It detects AI-generated images by learning the spectral distribution of real images through self-supervised masked spectral learning, then identifying AI-generated images as out-of-distribution samples.

---

## High-Level Overview

SPAI operates in two phases:

1. **Pre-training (MFM):** A Vision Transformer is pre-trained using Masked Frequency Modeling — a self-supervised task where masked frequency components of an image are reconstructed from corrupted inputs.
2. **Fine-tuning (SID):** The pre-trained backbone is used to extract features from the original image and its spectral components (low-frequency and high-frequency). A Frequency Restoration Estimator computes spectral reconstruction similarity scores, and a Spectral Context Attention mechanism aggregates patch-level features for any-resolution classification.

The core insight is that real images exhibit consistent spectral reconstruction patterns, while AI-generated images diverge — making them detectable as out-of-distribution samples.

---

## Project Structure

```
spai/
├── configs/
│   └── spai.yaml                 # Default training/inference configuration
├── data/
│   ├── fake_*.csv                # CSV metadata for AI-generated image sources
│   └── real_*.csv                # CSV metadata for real image sources
├── docs/
│   ├── architecture.md           # This file
│   ├── architecture.svg          # Architecture diagram
│   ├── data.md                   # Data downloading instructions
│   └── overview.svg              # Paper overview diagram
├── spai/                         # Main Python package
│   ├── __main__.py               # CLI entry point (train/test/infer/export)
│   ├── config.py                 # YACS-based configuration system
│   ├── data/                     # Data loading and augmentation
│   │   ├── data_finetune.py      # CSVDataset and data loaders for fine-tuning/inference
│   │   ├── data_mfm.py           # Data loaders for MFM pre-training
│   │   ├── readers.py            # FileSystem and LMDB data readers
│   │   ├── filestorage.py        # LMDB file storage abstraction
│   │   ├── blur_kernels.py       # Blur kernel generation utilities
│   │   └── random_degradations.py# Random image degradation transforms
│   ├── models/                   # Model architectures
│   │   ├── build.py              # Model factory (build_model, build_cls_model)
│   │   ├── vision_transformer.py # ViT backbone implementation (based on BEiT)
│   │   ├── swin_transformer.py   # Swin Transformer backbone
│   │   ├── backbones.py          # CLIP and DINOv2 backbone wrappers
│   │   ├── mfm.py                # Masked Frequency Modeling pre-training model
│   │   ├── sid.py                # SID models: MFViT, PatchBasedMFViT, FRE, classifiers
│   │   ├── filters.py            # Spectral filtering (FFT-based frequency separation)
│   │   ├── losses.py             # Loss functions (BCE, SupCon, Triplet)
│   │   ├── frequency_loss.py     # Focal frequency loss for MFM pre-training
│   │   └── utils.py              # Image patchification, positional embeddings, etc.
│   ├── data_utils.py             # CSV read/write utilities
│   ├── logger.py                 # Logging setup
│   ├── lr_scheduler.py           # Learning rate scheduler builder
│   ├── metrics.py                # Evaluation metrics (AUC, AP, Accuracy, F1)
│   ├── onnx.py                   # ONNX export validation utilities
│   ├── optimizer.py              # Optimizer builder with layer-wise decay
│   ├── tsne.py                   # t-SNE visualization of feature embeddings
│   ├── utils.py                  # Checkpoint loading/saving, gradient utilities
│   └── tools/                    # Standalone data preparation scripts
│       ├── augment_dataset.py
│       ├── create_dir_csv.py
│       ├── create_dmid_ldm_train_val_csv.py
│       ├── create_synthbuster_csv.py
│       └── reduce_csv_column.py
├── tests/                        # Unit tests
│   ├── data/
│   │   └── test_data_finetune.py
│   └── models/
│       ├── test_backbones.py
│       ├── test_filters.py
│       ├── test_sid.py
│       └── test_utils.py
├── requirements.txt
├── LICENSE                       # Apache 2.0
└── README.md
```

---

## Core Architecture Components

### 1. Spectral Filtering (`models/filters.py`)

The spectral filtering module decomposes an input image into low-frequency and high-frequency components using 2D FFT:

```
Input Image → FFT → Shift → Apply Circular Mask → IFFT → Low-Frequency Image
                                                        → High-Frequency Image (residual)
```

- `filter_image_frequencies()`: Splits an image into low-freq (within mask radius) and high-freq (outside mask radius) components via 2D DFT.
- `generate_circular_mask()`: Creates a binary circular mask in the frequency domain with a configurable radius (default: 16).

### 2. Vision Transformer Backbone (`models/vision_transformer.py`)

A standard ViT-B/16 implementation (based on BEiT) with key extensions:

- **Intermediate layer extraction:** Returns features from all 12 transformer blocks (layers 0–11), not just the final layer.
- **Absolute and relative positional embeddings** support.
- **Mean pooling** over patch tokens.
- **Return features mode:** When enabled, stacks intermediate layer outputs as a `B × N × L × D` tensor (Batch × Layers × Patches × Dim).

Alternative backbones are also supported via `models/backbones.py`:
- **CLIP ViT-B/16** — uses hooks to extract intermediate representations.
- **DINOv2 (ViT-B/14, ViT-L/14, ViT-G/14)** — uses the `get_intermediate_layers()` API.

### 3. Masked Frequency Modeling — MFM (`models/mfm.py`)

The self-supervised pre-training stage. An encoder–decoder framework that:

1. Corrupts the input by masking a band of frequencies in the 2D DFT spectrum.
2. Feeds the corrupted image through a ViT encoder.
3. Reconstructs the original image via a lightweight decoder (Conv2d + PixelShuffle).
4. Minimizes a **Focal Frequency Loss** between the reconstruction and the original in the frequency domain.

Supports Swin Transformer, ViT, and ResNet encoders. Training uses distributed data parallelism (SLURM/PyTorch).

### 4. Frequency Restoration Estimator — FRE (`models/sid.py :: FrequencyRestorationEstimator`)

The core detection mechanism. Given an image, FRE:

1. Decomposes it into **original**, **low-frequency**, and **high-frequency** versions.
2. Passes all three through the frozen (or fine-tuned) ViT backbone to extract intermediate features.
3. Projects the features via per-layer MLP projectors (`FeatureSpecificProjector`).
4. Computes **Spectral Reconstruction Similarity** — pairwise cosine similarities between the three feature sets across all intermediate layers:
   - `sim(original, low_freq)` — mean and std
   - `sim(original, hi_freq)` — mean and std
   - `sim(low_freq, hi_freq)` — mean and std
5. Optionally extracts a **Spectral Context Vector** from the original features via `FeatureImportanceProjector` — a learned weighted aggregation across layers.
6. Concatenates all into a feature vector of dimension `6N + D` (where `N` = number of intermediate layers, `D` = projection dim).

### 5. Spectral Context Attention — SCA (`models/sid.py :: PatchBasedMFViT`)

Enables **any-resolution** image processing:

1. The input image is split into overlapping 224×224 patches (with a configurable stride).
2. Each patch is independently processed by the MFViT encoder (FRE applied per-patch).
3. A **learnable query vector** attends to all patch features via multi-head cross-attention:
   ```
   Patches → [K, V projection] → Cross-Attention(learnable_query, K, V) → Aggregated Feature
   ```
4. The aggregated feature is passed through a LayerNorm and a classification head.

For small images (fewer patches than `MINIMUM_PATCHES`), a five-crop fallback is used.

### 6. Classification Head (`models/sid.py :: ClassificationHead`)

A 3-layer MLP:
```
Linear(D, D×R) → ReLU → Dropout → Linear(D×R, D×R) → ReLU → Dropout → Linear(D×R, 1)
```
Outputs a single logit; sigmoid is applied during inference to produce a probability score.

---

## Data Pipeline

### Dataset Format

All data is described via CSV files with columns: `image` (path), `split` (train/val/test), `class` (0=real, 1=fake).

### Training Augmentations (`data/data_finetune.py`)

Uses **Albumentations** for the augmentation pipeline:
- Random resized crop (224×224)
- Horizontal/vertical flip, rotation
- Gaussian blur, Gaussian noise
- JPEG/WEBP compression
- Color jitter, sharpening
- Normalization to [0, 1] range (default config `positive_0_1`; ImageNet normalization is used when configured)

Multiple augmented views per image are supported for contrastive or multi-view training.

### Inference Transforms

- Optional test-time perturbations (blur, noise, compression, scaling)
- Center crop or original resolution (with padding to minimum 224×224)
- Normalization to [0, 1]

### Data Readers

- **FileSystemReader:** Reads images from disk.
- **LMDBFileStorageReader:** Reads images from an LMDB database for faster I/O.

---

## Training Pipeline

### Pre-training (MFM)

```
python spai/main_mfm.py --cfg <config> --data-path <imagenet_path>
```

- Distributed training via SLURM or PyTorch DDP.
- Encoder: ViT-B/16 (or Swin/ResNet).
- Objective: Frequency reconstruction loss in the spectral domain.
- Produces `mfm_pretrain_vit_base.pth`.

### Fine-tuning (SID/SPAI)

```
python -m spai train --cfg ./configs/spai.yaml --data-path <csv> --pretrained <mfm_ckpt>
```

- Loads MFM pre-trained ViT weights into the backbone.
- Backbone is initially frozen; only the FRE, SCA, and classification head are trained.
- Loss: **BCEWithLogitsLoss** (binary cross-entropy).
- Optimizer: AdamW with layer-wise learning rate decay (0.8).
- Scheduler: Cosine annealing with warmup (5 epochs warmup, 35 total epochs).
- Mixed precision training via NVIDIA Apex (O2 level).
- Checkpoints saved when validation loss decreases.
- Logging: Neptune + TensorBoard.

### Evaluation Metrics

- **AUC** (Area Under the ROC Curve)
- **AP** (Average Precision)
- **Accuracy** (at 0.5 threshold)

---

## Inference Pipeline

```
python -m spai infer --input <dir_or_csv> --output <output_dir>
```

1. Images are loaded and optionally padded to meet the minimum patch size.
2. Each image is patchified into 224×224 patches.
3. Each patch undergoes spectral decomposition → backbone feature extraction → FRE processing.
4. Spectral Context Attention aggregates patch-level features.
5. Classification head produces a logit → sigmoid → probability score.
6. Results are written to a CSV file in the output directory.

---

## CLI Commands

The entry point is `python -m spai <command>`:

| Command | Description |
|---|---|
| `train` | Fine-tune SPAI on a labeled dataset |
| `test` | Evaluate a trained model on test CSVs |
| `infer` | Run inference on images (directory or CSV) |
| `tsne` | Generate t-SNE visualizations of embeddings |
| `export-onnx` | Export model to ONNX format (patch encoder + patch aggregator) |
| `validate-onnx` | Validate ONNX export against PyTorch model |

---

## Configuration System

SPAI uses **YACS** for hierarchical configuration. Key config groups:

| Group | Purpose |
|---|---|
| `MODEL` | Architecture type, backbone params, FRE settings, resolution mode |
| `MODEL.VIT` | ViT hyperparameters (depth, heads, embed dim, intermediate layers) |
| `MODEL.FRE` | Frequency Restoration Estimator (masking radius, projection settings) |
| `MODEL.PATCH_VIT` | Spectral Context Attention (stride, attention heads, min patches) |
| `DATA` | Dataset paths, image size, batch size, workers, augmentation views |
| `TRAIN` | Epochs, learning rates, optimizer, loss function, layer decay |
| `AUG` | Training augmentation probabilities and parameters |
| `TEST` | Test-time perturbations, resolution handling, view generation |

---

## Key Design Decisions

- **Spectral domain analysis:** Rather than analyzing pixel-level artifacts, SPAI operates on frequency-domain features, making it robust to common post-processing.
- **Self-supervised pre-training:** The MFM backbone learns general spectral reconstruction ability from real images only, avoiding overfitting to specific generators.
- **Any-resolution support:** The patch-based architecture with SCA enables processing images at their native resolution, capturing subtle spectral inconsistencies that would be lost with resizing.
- **OOD detection paradigm:** AI-generated images are detected as outliers of the learned real-image spectral distribution, providing generalization to unseen generators.

---

## Dependencies

Core libraries: **PyTorch**, **timm** (0.4.12), **albumentations**, **YACS**, **OpenCV**, **einops**, **CLIP**, **Neptune**, **torchmetrics**, **NVIDIA Apex** (optional, required only for mixed-precision training).
