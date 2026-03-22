# SPAI CLI Commands

## `train`

```bash
python -m spai train \
  --cfg <path>               # Path to YAML config file (required)
  --data-path <path>         # Path to training CSV file (required)
  --pretrained <path>        # Path to MFM pre-trained weights
  --batch-size <int>         # Batch size per GPU
  --learning-rate <float>    # Override base learning rate
  --csv-root-dir <path>      # Root dir for relative paths in CSV
  --lmdb <path>              # Path to LMDB file storage
  --resume                   # Resume from checkpoint (flag)
  --accumulation-steps <int> # Gradient accumulation steps (default: 1)
  --use-checkpoint           # Enable gradient checkpointing (flag)
  --amp-opt-level <O0|O1|O2> # Mixed precision level (default: O1)
  --output <path>            # Output directory
  --tag <str>                # Experiment tag
  --local_rank <int>         # Local rank for distributed training (default: 0)
  --test-csv <path>          # Test CSV for per-epoch evaluation (repeatable)
  --test-csv-root-dir <path> # Root dir for test CSV paths (repeatable)
  --data-workers <int>       # Number of data loading workers
  --disable-pin-memory       # Disable pinned memory (flag)
  --data-prefetch-factor <int>
  --save-all                 # Save checkpoint every epoch (flag)
  --opt <key> <value>        # Extra config overrides (repeatable)
```

---

## `test`

```bash
python -m spai test \
  --cfg <path>               # Path to YAML config file (required)
  --test-csv <path>          # Test CSV file(s) (repeatable)
  --test-csv-root-dir <path> # Root dir for test CSV paths (repeatable)
  --model <path>             # Path to trained model weights
  --batch-size <int>         # Batch size
  --split <str>              # Data split to evaluate (default: "test")
  --lmdb <path>              # Path to LMDB file storage
  --output <path>            # Output directory
  --tag <str>                # Experiment tag
  --resize-to <int>          # Resize images to max dimension
  --update-csv               # Write predicted scores into the CSV (flag)
  --opt <key> <value>        # Extra config overrides (repeatable)
```

---

## `infer`

```bash
python -m spai infer \
  --input <path>                # Image directory or CSV file (required, repeatable)
  --model <path>                # Path to model weights (default: ./weights/spai.pth)
  --cfg <path>                  # Path to YAML config (default: ./configs/spai.yaml)
  --batch-size <int>            # Inference batch size (default: 1)
  --input-csv-root-dir <path>   # Root dir for input CSV paths (repeatable)
  --split <str>                 # Data split to use (default: "test")
  --lmdb <path>                 # Path to LMDB file storage
  --output <path>               # Output directory (default: ./output)
  --tag <str>                   # Column name prefix for predictions (default: "spai")
  --resize-to <int>             # Resize images to max dimension
  --opt <key> <value>           # Extra config overrides (repeatable)
```

---

## `tsne`

```bash
python -m spai tsne \
  --cfg <path>               # Path to YAML config file (required)
  --test-csv <path>          # Test CSV file(s) (repeatable)
  --test-csv-root-dir <path> # Root dir for test CSV paths (repeatable)
  --model <path>             # Path to trained model weights
  --lmdb <path>              # Path to LMDB file storage
  --output <path>            # Output directory
  --tag <str>                # Experiment tag
  --resize-to <int>          # Resize images to max dimension
  --opt <key> <value>        # Extra config overrides (repeatable)
```

---

## `export-onnx`

```bash
python -m spai export-onnx \
  --cfg <path>                 # Path to YAML config file (required)
  --model <path>               # Path to trained model weights
  --output <path>              # Output directory
  --tag <str>                  # Experiment tag
  --exclude-preprocessing      # Exclude FFT preprocessing from encoder (flag)
  --opt <key> <value>          # Extra config overrides (repeatable)
```

---

## `validate-onnx`

```bash
python -m spai validate-onnx \
  --cfg <path>                 # Path to YAML config file (required)
  --test-csv <path>            # Test CSV file(s) (repeatable)
  --test-csv-root-dir <path>   # Root dir for test CSV paths (repeatable)
  --model <path>               # Path to trained model weights
  --batch-size <int>           # Batch size
  --split <str>                # Data split to evaluate (default: "test")
  --lmdb <path>                # Path to LMDB file storage
  --output <path>              # Output directory
  --tag <str>                  # Experiment tag
  --device <str>               # Device to run on (default: "cpu")
  --exclude-preprocessing      # Match encoder exported without preprocessing (flag)
  --opt <key> <value>          # Extra config overrides (repeatable)
```