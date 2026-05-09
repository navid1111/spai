"""
SPAI Image Predictor
====================
Detects whether an image is AI-generated or real.

Supports two backends — auto-detected from the file extension:

  PyTorch  (.pth)   — full model, single file
  ONNX     (.onnx)  — requires TWO files:
                        patch_encoder.onnx      (ViT + frequency features)
                        patch_aggregator.onnx   (cross-attention + classifier)

Usage:
    # PyTorch backend
    python predict_image.py image.jpg spai/weights/spai.pth

    # ONNX backend (pass encoder; aggregator must be in the same folder)
    python predict_image.py image.jpg weights/exported/patch_encoder.onnx

    # ONNX backend (pass either file; the other is inferred)
    python predict_image.py image.jpg weights/exported/patch_aggregator.onnx

    # Explicit paths for both ONNX files
    python predict_image.py image.jpg --encoder enc.onnx --aggregator agg.onnx

    # Process a folder of images
    python predict_image.py images/ spai/weights/spai.pth --ext jpg png webp
"""

import argparse
import logging
import os
import pathlib
import sys
import time
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
log = logging.getLogger("spai.predict")


# ═══════════════════════════════════════════════════════════════════════════════
#  Shared preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMG_SIZE      = 224


def load_image(path: str) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess_for_pytorch(img: np.ndarray, config=None) -> torch.Tensor:
    """
    Convert image to a float32 tensor in [0, 1] at ORIGINAL resolution.

    Critical: PatchBasedMFViT.forward() internally:
      1. Patchifies the full-res image into 224x224 patches
      2. Runs FFT frequency split per patch
      3. Normalises with ImageNet stats
    So we must NOT resize, crop, or normalise here — just to-tensor.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),   # → float32 [0, 1], keeps original resolution
    ])
    return transform(img).unsqueeze(0)   # 1 x 3 x H x W (full resolution)


def preprocess_for_onnx(img: np.ndarray) -> torch.Tensor:
    """
    Simple resize + to-tensor for ONNX path.
    Returns 1 x 3 x 224 x 224 float32 tensor with pixels in [0, 1].
    NOTE: normalisation is applied inside fft_preprocess, not here.
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),          # → [0, 1]
    ])
    return transform(img).unsqueeze(0)  # 1 x 3 x 224 x 224


# ═══════════════════════════════════════════════════════════════════════════════
#  PyTorch backend
# ═══════════════════════════════════════════════════════════════════════════════

class PytorchPredictor:
    def __init__(self, model_path: str, config_path: str = "configs/spai.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"[PyTorch] Device: {self.device}")

        # Config
        from spai.config import get_config
        self.config = get_config({
            "cfg": config_path,
            "pretrained": model_path,
            "opts": [],
            "batch_size": None, "data_path": None, "resume": "",
            "accumulation_steps": None, "use_checkpoint": None,
            "amp_opt_level": None, "output": ".", "tag": "predict",
            "eval": False, "throughput": False, "local_rank": 0,
        })

        # Build + load
        log.info("[PyTorch] Building model…")
        from spai.models.build import build_cls_model
        from spai.utils import load_pretrained
        from spai.models.sid import build_mf_vit, PatchBasedMFViT
        from spai.models.mfm import MFM

        model = build_cls_model(self.config)
        if isinstance(model, MFM):
            log.info("[PyTorch] Switching to PatchBasedMFViT…")
            model = build_mf_vit(self.config)
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt))
            model.load_state_dict(state, strict=False)
        else:
            _logger = logging.getLogger("spai.load")
            _logger.propagate = False   # prevent double-printing via root logger
            if not _logger.handlers:
                _logger.addHandler(logging.StreamHandler())
                _logger.setLevel(logging.INFO)
            load_pretrained(self.config, model, _logger)

        self.model = model.to(self.device).eval()

    def predict(self, img: np.ndarray) -> float:
        """Returns probability of being AI-generated (0–1)."""
        # Keep original resolution — PatchBasedMFViT patchifies internally.
        # Pass as list[Tensor] so it routes to forward_arbitrary_resolution_batch.
        # Each tensor in the list is 1 x 3 x H x W in [0, 1], unnormalised.
        tensor = preprocess_for_pytorch(img, self.config).to(self.device)

        feat_batch = getattr(self.config.MODEL, "FEATURE_EXTRACTION_BATCH", 400)

        with torch.no_grad():
            # list input → forward_arbitrary_resolution_batch path
            out = self.model(
                [tensor],
                feature_extraction_batch_size=feat_batch
            )

        # out is B x 1 logit (BCE head), apply sigmoid
        logits = out[0] if isinstance(out, (tuple, list)) else out
        prob = torch.sigmoid(logits)[0, 0].item()
        return prob


# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX backend
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ort_session(onnx_path: str):
    """Create an ONNXRuntime session preferring CUDA then CPU."""
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    used = sess.get_providers()[0]
    log.info(f"  ORT provider: {used}  ({pathlib.Path(onnx_path).name})")
    return sess


def _fft_preprocess_numpy(
    image_tensor: torch.Tensor,
    freq_mask: torch.Tensor,
    backbone_norm,
) -> tuple:
    """
    FFT frequency split + normalisation in PyTorch (not in ONNX).
    Returns (x, x_low, x_hi) as float32 numpy arrays ready for ORT.
    """
    from spai.models import filters
    image_f = image_tensor.float()
    freq_mask_dev = freq_mask.to(image_f.device)

    low_freq, hi_freq = filters.filter_image_frequencies(image_f, freq_mask_dev)
    low_freq = torch.clamp(low_freq, 0., 1.).to(image_tensor.dtype)
    hi_freq  = torch.clamp(hi_freq,  0., 1.).to(image_tensor.dtype)

    x     = backbone_norm(image_tensor).cpu().numpy().astype(np.float32)
    x_low = backbone_norm(low_freq).cpu().numpy().astype(np.float32)
    x_hi  = backbone_norm(hi_freq).cpu().numpy().astype(np.float32)
    return x, x_low, x_hi


class OnnxPredictor:
    """
    Two-stage ONNX inference:
      Stage 1: patch_encoder.onnx    — ViT + frequency features (3 inputs, post-FFT)
      Stage 2: patch_aggregator.onnx — cross-attention + classifier
    """
    def __init__(
        self,
        encoder_path: str,
        aggregator_path: str,
        config_path: str = "configs/spai.yaml",
        masking_radius: int = 16,
    ):
        import onnxruntime  # noqa — validate installed
        self.enc_sess = _make_ort_session(encoder_path)
        self.agg_sess = _make_ort_session(aggregator_path)

        # We need the frequency mask + backbone normalisation from the SPAI config.
        # Build a minimal MFViT just to get these — no checkpoint needed.
        from spai.config import get_config
        from spai.models.sid import build_mf_vit
        from spai.models.mfm import MFM

        cfg = get_config({
            "cfg": config_path, "pretrained": "", "opts": [],
            "batch_size": None, "data_path": None, "resume": "",
            "accumulation_steps": None, "use_checkpoint": None,
            "amp_opt_level": None, "output": ".", "tag": "onnx_predict",
            "eval": False, "throughput": False, "local_rank": 0,
        })
        mf_model = build_mf_vit(cfg)

        # PatchBasedMFViT → .mfvit (MFViT) holds the mask and norm
        inner = mf_model.mfvit if hasattr(mf_model, "mfvit") else mf_model
        self.freq_mask     = inner.frequencies_mask.data.clone().cpu()
        self.backbone_norm = inner.backbone_norm

        # Patchification params — must match training config exactly
        self.patch_size   = cfg.DATA.IMG_SIZE                      # 224
        self.patch_stride = cfg.MODEL.PATCH_VIT.PATCH_STRIDE       # e.g. 224
        self.min_patches  = cfg.MODEL.PATCH_VIT.MINIMUM_PATCHES    # e.g. 4

        log.info(f"[ONNX] Predictor ready. "
                 f"patch_size={self.patch_size} stride={self.patch_stride} "
                 f"min_patches={self.min_patches}")

    def predict(self, img: np.ndarray) -> float:
        """
        Replicates PatchBasedMFViT.forward_arbitrary_resolution_batch exactly:
          1. Convert full-res image to [0,1] tensor (no resize, no normalize)
          2. Patchify into 224x224 patches (same stride as training)
          3. FFT-split + normalize each patch in PyTorch
          4. Run each patch through Stage 1 ONNX encoder → [L, D]
          5. Run [1, L, D] through Stage 2 ONNX aggregator → [1, 1] logit
          6. Sigmoid → probability
        """
        from spai.models.utils import patchify_image
        from torchvision.transforms.functional import five_crop

        # 1. Full-res to [0,1] tensor — NO resize, NO normalize
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        raw_tensor = transform(img).unsqueeze(0)  # 1 x 3 x H x W in [0,1]

        # 2. Patchify exactly as PatchBasedMFViT does
        patch_size   = self.patch_size    # 224
        patch_stride = self.patch_stride  # from config (e.g. 224 non-overlap)
        min_patches  = self.min_patches   # e.g. 4

        patched = patchify_image(
            raw_tensor,
            (patch_size, patch_size),
            (patch_stride, patch_stride),
        )  # 1 x L x 3 x 224 x 224

        # Fall back to five_crop if too few patches (matches model logic)
        if patched.size(1) < min_patches:
            crops = five_crop(raw_tensor.squeeze(0), [patch_size, patch_size])
            patched = torch.stack(crops, dim=0).unsqueeze(0)  # 1 x 5 x 3 x 224 x 224

        L = patched.size(1)
        patches = patched.squeeze(0)  # L x 3 x 224 x 224

        # 3+4. FFT-split + encode each patch → collect [L, D]
        all_features = []
        for i in range(L):
            patch = patches[i].unsqueeze(0)  # 1 x 3 x 224 x 224

            x, x_low, x_hi = _fft_preprocess_numpy(
                patch, self.freq_mask, self.backbone_norm
            )
            feat = self.enc_sess.run(
                None, {"x": x, "x_low": x_low, "x_hi": x_hi}
            )[0]  # 1 x D
            all_features.append(feat)

        # Stack → [1, L, D]
        patch_features = np.concatenate(all_features, axis=0)[np.newaxis]  # 1 x L x D

        # 5. Aggregator → [1, 1]
        logit = self.agg_sess.run(
            None, {"patch_features": patch_features.astype(np.float32)}
        )[0]

        # 6. Sigmoid
        prob = float(1.0 / (1.0 + np.exp(-logit.flatten()[0])))
        return prob


# ═══════════════════════════════════════════════════════════════════════════════
#  Predictor factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_predictor(
    model_path: str,
    encoder_path: Optional[str] = None,
    aggregator_path: Optional[str] = None,
    config_path: str = "configs/spai.yaml",
) -> Union[PytorchPredictor, OnnxPredictor]:
    """
    Auto-detect backend from file extension and build the right predictor.

    Rules:
      - model_path ends with .pth  → PyTorch
      - model_path ends with .onnx → ONNX (infer sibling file automatically)
      - --encoder / --aggregator   → ONNX explicit
    """
    # Explicit ONNX paths
    if encoder_path and aggregator_path:
        log.info("[Auto] Backend: ONNX (explicit paths)")
        return OnnxPredictor(encoder_path, aggregator_path, config_path)

    p = pathlib.Path(model_path)

    if p.suffix == ".pth":
        log.info("[Auto] Backend: PyTorch")
        return PytorchPredictor(model_path, config_path)

    if p.suffix == ".onnx":
        log.info("[Auto] Backend: ONNX")
        # Infer the sibling file
        name = p.stem
        folder = p.parent
        if "encoder" in name:
            enc = str(p)
            agg = str(folder / p.name.replace("encoder", "aggregator"))
        elif "aggregator" in name:
            agg = str(p)
            enc = str(folder / p.name.replace("aggregator", "encoder"))
        else:
            raise ValueError(
                "Could not infer encoder/aggregator pair from filename. "
                "Use --encoder and --aggregator explicitly."
            )
        if not pathlib.Path(enc).exists():
            raise FileNotFoundError(f"Encoder ONNX not found: {enc}")
        if not pathlib.Path(agg).exists():
            raise FileNotFoundError(f"Aggregator ONNX not found: {agg}")
        return OnnxPredictor(enc, agg, config_path)

    raise ValueError(
        f"Unrecognised model file extension '{p.suffix}'. "
        "Expected .pth or .onnx"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Result display
# ═══════════════════════════════════════════════════════════════════════════════

def print_result(image_path: str, prob: float, elapsed: float) -> None:
    label = "AI Generated ⚠" if prob > 0.5 else "Real Image ✓"
    bar_len = 30
    filled = int(bar_len * prob)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'─'*50}")
    print(f"  Image   : {image_path}")
    print(f"  AI prob : [{bar}] {prob*100:.1f}%")
    print(f"  Result  : {label}")
    print(f"  Time    : {elapsed*1000:.0f} ms")
    print(f"{'─'*50}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SPAI predictor — PyTorch or ONNX backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input",
        help="Image file or directory of images")
    p.add_argument("model", nargs="?", default=None,
        help=".pth checkpoint  OR  one of the two .onnx files "
             "(sibling file auto-inferred)")
    p.add_argument("--encoder",     default=None,
        help="Explicit path to patch_encoder.onnx")
    p.add_argument("--aggregator",  default=None,
        help="Explicit path to patch_aggregator.onnx")
    p.add_argument("--cfg", default="configs/spai.yaml",
        help="Path to spai.yaml config (default: configs/spai.yaml)")
    p.add_argument("--ext", nargs="+",
        default=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        help="Image extensions to scan when input is a directory")
    p.add_argument("--threshold", type=float, default=0.5,
        help="Decision threshold (default: 0.5)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.model is None and args.encoder is None:
        print("ERROR: Provide a model path (positional) or --encoder/--aggregator.")
        sys.exit(1)

    # Build predictor once
    predictor = build_predictor(
        model_path=args.model or "",
        encoder_path=args.encoder,
        aggregator_path=args.aggregator,
        config_path=args.cfg,
    )

    # Collect image paths
    input_path = pathlib.Path(args.input)
    if input_path.is_dir():
        image_paths = []
        for ext in args.ext:
            image_paths.extend(input_path.glob(f"*.{ext}"))
            image_paths.extend(input_path.glob(f"*.{ext.upper()}"))
        image_paths = sorted(set(image_paths))
        if not image_paths:
            print(f"No images found in {input_path}")
            sys.exit(1)
        print(f"Found {len(image_paths)} image(s) in {input_path}")
    else:
        image_paths = [input_path]

    # Run predictions
    results = []
    for img_path in image_paths:
        try:
            img = load_image(str(img_path))
            t0 = time.perf_counter()
            prob = predictor.predict(img)
            elapsed = time.perf_counter() - t0
            print_result(str(img_path), prob, elapsed)
            results.append({
                "path": str(img_path),
                "prob": prob,
                "label": "AI Generated" if prob > args.threshold else "Real",
            })
        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")

    # Summary for batch runs
    if len(results) > 1:
        ai_count = sum(1 for r in results if r["label"] == "AI Generated")
        print(f"\n{'═'*50}")
        print(f"  Summary: {ai_count}/{len(results)} images classified as AI Generated")
        print(f"{'═'*50}")


if __name__ == "__main__":
    main()