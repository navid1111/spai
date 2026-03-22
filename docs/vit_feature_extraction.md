# ViT Backbone Feature Extraction on New Images

This guide explains how to extract feature vectors from the ViT backbone using a trained SPAI checkpoint.

## What you get

- Backbone features from the internal ViT (not only fake/real scores).
- One `.npy` file per image, or all features in memory for custom processing.

## Important behavior in this repo

- The model is built through `build_cls_model` and checkpoint loading via `load_pretrained`.
- The ViT backbone is exposed through `model.get_vision_transformer()`.
- Inputs to the backbone should be in `[0, 1]`, then normalized with SPAI backbone normalization.
- In the default `configs/spai.yaml`, `MODEL.VIT.USE_INTERMEDIATE_LAYERS: True`.
  - This means the ViT output can be token-level intermediate features, not a single vector.
  - You may want to apply pooling yourself.

## 1. Environment setup

From project root:

```powershell
pip install -r requirements.txt
```

## 2. Save this script as tools/extract_vit_features.py

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from spai.config import get_config
from spai.models import build_cls_model
from spai.utils import load_pretrained


class _Logger:
    @staticmethod
    def info(msg: str) -> None:
        print(msg)


def _collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    return sorted([p for p in input_path.rglob("*") if p.suffix.lower() in exts])


def _pool_features(feats: torch.Tensor, pooling: str) -> torch.Tensor:
    # Keep batch dim and reduce all non-batch dims if needed.
    if pooling == "none":
        return feats
    if feats.ndim == 2:
        return feats
    if pooling == "mean":
        reduce_dims = tuple(range(1, feats.ndim))
        return feats.mean(dim=reduce_dims)
    if pooling == "max":
        x = feats
        while x.ndim > 2:
            x = x.max(dim=1).values
        return x
    raise ValueError(f"Unsupported pooling: {pooling}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ViT backbone features from images")
    parser.add_argument("--cfg", type=str, default="configs/spai.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True, help="Image file or directory")
    parser.add_argument("--output", type=str, default="output/vit_features")
    parser.add_argument("--pooling", type=str, default="mean", choices=["none", "mean", "max"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logger = _Logger()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config({
        "cfg": args.cfg,
        "pretrained": args.checkpoint,
        "opts": (),
    })

    model = build_cls_model(config)
    model = model.to(args.device).eval()

    load_pretrained(
        config,
        model,
        logger,
        checkpoint_path=Path(args.checkpoint),
        verbose=False,
    )

    vit = model.get_vision_transformer().to(args.device).eval()

    # Input should be [0,1] before SPAI normalization.
    to_tensor = transforms.ToTensor()

    image_paths = _collect_images(input_path)
    if not image_paths:
        raise RuntimeError(f"No images found in: {input_path}")

    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        x = to_tensor(img).unsqueeze(0).to(args.device)

        # Match internal SPAI normalization used before ViT.
        if hasattr(model, "mfvit") and hasattr(model.mfvit, "backbone_norm"):
            x = model.mfvit.backbone_norm(x)

        feats = vit(x)
        feats = _pool_features(feats, args.pooling)

        feats_np = feats.detach().cpu().numpy()

        out_file = output_dir / f"{img_path.stem}.npy"
        np.save(out_file, feats_np)
        print(f"Saved {out_file} shape={tuple(feats_np.shape)}")


if __name__ == "__main__":
    main()
```

## 3. Run feature extraction

Single image:

```powershell
python tools/extract_vit_features.py `
  --checkpoint path/to/your_model.pth `
  --input path/to/image.jpg `
  --output output/vit_features `
  --pooling mean
```

Folder of images:

```powershell
python tools/extract_vit_features.py `
  --checkpoint path/to/your_model.pth `
  --input path/to/image_folder `
  --output output/vit_features `
  --pooling mean
```

## 4. Understanding output shape

- With `--pooling none`: raw ViT output tensor is saved (shape depends on config).
- With `--pooling mean` or `--pooling max`: one vector per image (`[1, D]` or `[1]` depending on model output).

If you want a stable embedding vector for retrieval or clustering, `--pooling mean` is usually the best starting point.

## 5. Notes and troubleshooting

- If you get CUDA errors, run with `--device cpu`.
- If output looks unexpected, verify your checkpoint and config match.
- The normal `python -m spai infer ...` command returns prediction scores, not backbone feature vectors.
- Keep preprocessing consistent with training/inference (`ToTensor` then SPAI normalization).
