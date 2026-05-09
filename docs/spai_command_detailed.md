# SPAI `__main__.py` — Documentation

## Overview

`__main__.py` is the **CLI entry point and orchestration layer** for the SPAI (Spectral AI-Generated Image Detector) system. It is built using the [Click](https://click.palletsprojects.com/) library and exposes all major operations — training, testing, inference, visualization, and ONNX export — as subcommands under a single CLI group.

It contains **no model logic itself**. Instead, it wires together all subsystems defined in the `spai/` package: configuration, data loading, model construction, optimization, loss functions, checkpoint management, and experiment logging.

---

## Entry Point

```bash
python -m spai <command> [OPTIONS]
```

---

## CLI Commands

### `train`

Fine-tunes the SPAI model on a labeled dataset.

**What it does:**
1. Parses CLI options and builds a YACS config via `get_config()`.
2. Sets random seeds for reproducibility (`torch`, `numpy`).
3. Optionally scales the learning rate linearly with batch size.
4. Initializes TensorBoard and Neptune logging.
5. Builds train/val dataloaders via `build_loader()`.
6. Constructs the model with `build_cls_model()` and moves it to GPU.
7. Sets up the AdamW optimizer, cosine LR scheduler, and loss function.
8. Loads MFM pre-trained weights into the ViT backbone if `--pretrained` is provided; otherwise unfreezes the backbone for full training.
9. Optionally builds test dataloaders if `--test-csv` is provided.
10. Calls `train_model()` to run the full training loop.

**Key options:**

| Option | Description |
|---|---|
| `--cfg` | Path to YAML config file (required) |
| `--data-path` | Path to training CSV file (required) |
| `--pretrained` | Path to MFM pre-trained weights |
| `--batch-size` | Batch size per GPU |
| `--learning-rate` | Override base learning rate |
| `--accumulation-steps` | Gradient accumulation steps |
| `--amp-opt-level` | Mixed precision level (O0/O1/O2) |
| `--test-csv` | Optional test CSVs for per-epoch evaluation |
| `--save-all` | Save checkpoints every epoch (not just on loss improvement) |
| `--output` | Output directory |
| `--tag` | Experiment tag |

---

### `test`

Evaluates one or more saved model checkpoints against test CSV datasets.

**What it does:**
1. Builds the config and sets up logging and Neptune.
2. Loads test dataloaders from provided CSVs.
3. Discovers all checkpoint files via `find_pretrained_checkpoints()`.
4. For each checkpoint, loads the model, runs `validate()`, and logs ACC, AP, AUC, and loss.
5. Optionally writes per-sample prediction scores back into the CSV under a new column (`--update-csv`).
6. Also writes attention mask paths if available.

**Key options:**

| Option | Description |
|---|---|
| `--test-csv` | One or more CSV files for testing (required) |
| `--model` | Path to trained model weights |
| `--split` | Data split to evaluate (`test` by default) |
| `--update-csv` | Write predicted scores into the CSV |
| `--resize-to` | Resize images so their largest dimension ≤ this value |

---

### `infer`

Runs inference on a directory of images or a CSV file. Does not require ground-truth labels.

**What it does:**
1. Accepts image directories or CSV files as `--input`.
2. Creates dummy CSV metadata if input is a directory (no labels).
3. Loads the model and runs `validate()` with `return_predictions=True`.
4. If ground-truth labels exist (CSV input), logs ACC, AP, and AUC.
5. Writes prediction scores to a CSV in the `--output` directory.
6. Also writes attention mask paths if the model produces them.

**Key options:**

| Option | Description |
|---|---|
| `--input` | Directory or CSV file(s) to run inference on (required, repeatable) |
| `--model` | Path to model weights (default: `./weights/spai.pth`) |
| `--output` | Directory where results CSV is written (default: `./output`) |
| `--tag` | Column name prefix for predictions in output CSV |
| `--resize-to` | Resize images before inference |

---

### `tsne`

Generates t-SNE visualizations of the model's feature embeddings.

**What it does:**
1. Loads the model and test dataloaders (batch size is forced to 1 for correct per-sample embedding extraction).
2. Calls `tsne_utils.visualize_tsne()` for each dataset.
3. Logs results to Neptune and TensorBoard.

**Key options:** Same structure as `test`, without `--update-csv`.

---

### `export-onnx`

Exports the trained PyTorch model to ONNX format.

**What it does:**
1. Loads the model checkpoint.
2. Moves the model to CPU and sets it to eval mode.
3. Calls `model.export_onnx()` to produce two separate ONNX files:
   - `patch_encoder.onnx` — the per-patch feature extractor (FRE).
   - `patch_aggregator.onnx` — the Spectral Context Attention aggregator and classifier.
4. Saves the files to `<output>/onnx/`.

**Key options:**

| Option | Description |
|---|---|
| `--exclude-preprocessing` | Export encoder without FFT preprocessing (accepts pre-filtered inputs) |

---

### `validate-onnx`

Numerically validates the exported ONNX model against the original PyTorch model.

**What it does:**
1. Loads both the PyTorch model and the ONNX files from `<output>/onnx/`.
2. Runs `compare_pytorch_onnx_models()` to check that outputs match within tolerance.
3. Must be run after `export-onnx`.

---

## Core Training Functions

### `train_model()`

The **epoch-level training loop**.

```
for epoch in range(start_epoch, total_epochs):
    train_one_epoch(...)       # Forward + backward pass over all batches
    validate(...)              # Compute val loss, ACC, AP, AUC
    save_checkpoint(...)       # Save if val loss improved (or save_all=True)
    validate(...) per test CSV # Optional per-epoch test evaluation
    neptune_run.sync()
```

- Tracks best epoch for each metric (loss, ACC, AP, AUC) and logs them after every epoch.
- Computes and logs total training time at the end.

---

### `train_one_epoch()`

The **batch-level training loop**.

- Supports two training modes:
  - **Standard:** `(samples, targets, _)` batch — supports multiple augmented views per image.
  - **Triplet:** `(anchor, positive, negative)` batch — for `TripletMarginLoss`.
- Handles **gradient accumulation** (`ACCUMULATION_STEPS > 1`).
- Handles **mixed precision training** via NVIDIA Apex AMP.
- Handles **gradient clipping** (`CLIP_GRAD`).
- Logs loss, gradient norm, and learning rate to TensorBoard and Neptune every `ACCUMULATION_STEPS` batches.
- Prints progress every `PRINT_FREQ` steps with ETA and memory usage.

---

### `validate()`

The **shared evaluation function**, used by training, testing, and inference.

```
for batch in data_loader:
    output = model(images)
    loss = criterion(output, targets)
    output = sigmoid(output)
    update metrics (AUC, AP, Accuracy)
    optionally store per-sample predictions
```

- Handles three image input formats:
  - **List of tensors** — arbitrary-resolution inputs (patch-based model).
  - **4D tensor with views** — multi-crop test-time augmentation; reduces via `max` or `mean`.
  - **Standard 4D tensor** — single-view fixed-resolution inputs.
- Returns `(acc, ap, auc, avg_loss)`, or optionally `(acc, ap, auc, avg_loss, predictions_dict)` when `return_predictions=True`.
- `predictions_dict` maps dataset sample indices to `(score, attention_mask)` tuples.

---

## Data Flow Summary

```
CLI args
   └─► get_config()
          └─► build_loader() / build_loader_test()
                 └─► CSVDataset + DataLoader
                        └─► train_one_epoch() / validate()
                               └─► model(images)
                                      └─► criterion(output, targets)
                                             └─► loss.backward() + optimizer.step()

                               └─► metrics.compute() → AUC, AP, ACC
                                      └─► Neptune + TensorBoard logging
                                             └─► save_checkpoint()
```

---

## Logging & Experiment Tracking

| System | What is logged |
|---|---|
| **Neptune** | Train/val/test loss, ACC, AP, AUC, LR, grad norm, epoch time, total time |
| **TensorBoard** | Train loss, grad norm, LR (epoch_1000x scale) |
| **File logger** | Per-epoch summaries, best epoch tracking, model parameter counts |
| **Config JSON** | Full YACS config snapshot saved to output directory |

---

## Dependencies (used in this file)

| Import | Purpose |
|---|---|
| `click` | CLI framework |
| `torch`, `torch.nn` | Model training and inference |
| `neptune` | Experiment tracking |
| `tensorboard` | Local training visualization |
| `apex.amp` | Mixed precision training (optional) |
| `timm.utils.AverageMeter` | Running average for loss/time metrics |
| `spai.config.get_config` | YACS configuration builder |
| `spai.models.build_cls_model` | Model factory |
| `spai.data.build_loader` / `build_loader_test` | DataLoader builders |
| `spai.optimizer.build_optimizer` | AdamW with layer-wise decay |
| `spai.lr_scheduler.build_scheduler` | Cosine scheduler with warmup |
| `spai.models.losses.build_loss` | Loss function factory (BCE, SupCon, Triplet) |
| `spai.utils` | Checkpoint save/load, gradient norm, checkpoint discovery |
| `spai.metrics.Metrics` | AUC, AP, Accuracy computation |