"""
SPAI Model Export: ONNX + TensorRT
====================================
Exports the two SPAI model stages:
  Stage 1 — Patch Encoder  (MFViT):            [B, 3, 224, 224] -> [B, D]
  Stage 2 — Patch Aggregator (SCAClassifier):  [B, L, D]        -> [B, 1]

Usage:
    python export_spai.py \
        --cfg    configs/spai.yaml \
        --model  spai/weights/spai.pth \
        --output ./weights/exported

Requirements (install once):
    pip install onnx onnxruntime onnxsim
    # For TensorRT (Linux / WSL2 only):
    pip install tensorrt pycuda
"""

import argparse
import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# ── Attempt TensorRT import (optional) ─────────────────────────────────────────
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    warnings.warn(
        "TensorRT / pycuda not found. ONNX export will still work; "
        "TRT export will be skipped. To enable: pip install tensorrt pycuda"
    )

# ── Silence Albumentations update noise ────────────────────────────────────────
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"


# ═══════════════════════════════════════════════════════════════════════════════
#  Exportable wrapper classes  (module-level so Dynamo / TorchScript can trace)
# ═══════════════════════════════════════════════════════════════════════════════

# ── FFT preprocessing helper (pure PyTorch, NOT exported to ONNX) ─────────────
def fft_preprocess(
    image: torch.Tensor,
    freq_mask: torch.Tensor,
    backbone_norm,
) -> tuple:
    """
    Split image into low/high frequency components and normalise all three.
    Run this in PyTorch BEFORE feeding into the ONNX patch encoder.

    Args:
        image      : B x 3 x 224 x 224, float32, pixels in [0, 1]
        freq_mask  : circular mask tensor (on same device as image)
        backbone_norm: torchvision Normalize transform

    Returns:
        (x_norm, x_low_norm, x_hi_norm) — each B x 3 x 224 x 224
    """
    from spai.models import filters
    low_freq, hi_freq = filters.filter_image_frequencies(
        image.float(), freq_mask.to(image.device)
    )
    low_freq = torch.clamp(low_freq, min=0., max=1.).to(image.dtype)
    hi_freq  = torch.clamp(hi_freq,  min=0., max=1.).to(image.dtype)
    x_norm     = backbone_norm(image)
    x_low_norm = backbone_norm(low_freq)
    x_hi_norm  = backbone_norm(hi_freq)
    return x_norm, x_low_norm, x_hi_norm


class ExportablePatchEncoder(nn.Module):
    """
    Stage 1b — post-FFT ViT encoder exported to ONNX.

    WHY SPLIT: aten::fft_fft2 is not in the ONNX opset. The FFT frequency
    split must be done in PyTorch (fft_preprocess above) before calling
    this ONNX model.

    Inputs:  x      B x 3 x 224 x 224  (normalised original image)
             x_low  B x 3 x 224 x 224  (normalised low-freq component)
             x_hi   B x 3 x 224 x 224  (normalised high-freq component)
    Output:  B x D  (frequency-restoration similarity features)
    """
    def __init__(self, mfvit_instance):
        super().__init__()
        self.mfvit = mfvit_instance
        # Stash freq mask and norm for fft_preprocess convenience
        self.register_buffer(
            "frequencies_mask",
            mfvit_instance.frequencies_mask.data.clone()
        )
        self.backbone_norm = mfvit_instance.backbone_norm

    def _vit_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ViT backbone, bypassing any MFM masking wrapper."""
        vit = self.mfvit.vit
        # Direct VisionTransformer (when MFM was not used)
        if hasattr(vit, "forward_features"):
            return vit.forward_features(x)
        # MFM wraps a VisionTransformer as .encoder
        if hasattr(vit, "encoder"):
            enc = vit.encoder
            if hasattr(enc, "forward_features"):
                return enc.forward_features(x)
            return enc(x)
        return vit(x)

    def forward(
        self,
        x: torch.Tensor,
        x_low: torch.Tensor,
        x_hi: torch.Tensor,
    ) -> torch.Tensor:
        """Post-FFT forward: inputs are already normalised frequency bands."""
        x     = self._vit_forward(x)
        x_low = self._vit_forward(x_low)
        x_hi  = self._vit_forward(x_hi)
        return self.mfvit.features_processor.exportable_forward(x, x_low, x_hi)


class ExportablePatchAggregator(nn.Module):
    """
    Stage 2 — wraps the spectral-context attention + norm + cls_head.
    Input:  B x L x D  (patch feature vectors from Stage 1)
    Output: B x 1      (raw logit; apply sigmoid for probability)
    """
    def __init__(self, patch_based_mfvit_instance):
        super().__init__()
        self.model = patch_based_mfvit_instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.patches_attention(x)   # B x D
        x = self.model.norm(x)                # B x D
        x = self.model.cls_head(x)            # B x 1
        return x


# ═══════════════════════════════════════════════════════════════════════════════
#  ONNX helpers
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_onnx(
    model: nn.Module,
    dummy_input,                    # tensor or tuple of tensors
    output_path: pathlib.Path,
    input_names: list,
    output_names: list,
    dynamic_axes: dict,
    opset: int = 17,
) -> pathlib.Path:
    """Export a PyTorch model to ONNX using the stable legacy exporter.

    Always exports on CPU — ONNX constant folding fails when model buffers
    and inputs span multiple devices (cuda + cpu). The resulting .onnx file
    runs on any device at inference time via ONNXRuntime.
    """
    # Move everything to CPU for the export pass
    original_device = next(model.parameters()).device
    model_cpu = model.cpu().eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dummy_input, tuple):
        args_cpu = tuple(t.cpu() for t in dummy_input)
    else:
        args_cpu = (dummy_input.cpu(),)

    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            args_cpu,
            str(output_path),
            opset_version=opset,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,   # avoids cross-device cat during shape inference
            export_params=True,
        )

    # Restore model to original device
    model.to(original_device)
    print(f"  [✓] ONNX saved → {output_path}")
    return output_path


def simplify_onnx(onnx_path: pathlib.Path) -> pathlib.Path:
    """Run onnx-simplifier to fold constants and clean up the graph."""
    try:
        import onnxsim
    except ImportError:
        print("  [!] onnxsim not installed; skipping simplification. "
              "pip install onnxsim")
        return onnx_path

    model = onnx.load(str(onnx_path))
    model_simplified, ok = onnxsim.simplify(model)
    if ok:
        sim_path = onnx_path.with_stem(onnx_path.stem + "_simplified")
        onnx.save(model_simplified, str(sim_path))
        print(f"  [✓] Simplified ONNX → {sim_path}")
        return sim_path
    else:
        print("  [!] Simplification failed; using original.")
        return onnx_path


def validate_onnx(onnx_path: pathlib.Path, sample_input: np.ndarray,
                  input_name: str) -> None:
    """Quick sanity-check: run one forward pass through ONNXRuntime."""
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    outputs = sess.run(None, {input_name: sample_input})
    print(f"  [✓] ORT validation passed. Output shape: {outputs[0].shape}")


def validate_onnx_multi(onnx_path: pathlib.Path, feed: dict) -> None:
    """Sanity-check for models with multiple inputs."""
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    outputs = sess.run(None, {k: v.astype(np.float32) for k, v in feed.items()})
    print(f"  [✓] ORT validation passed. Output shape: {outputs[0].shape}")


# ═══════════════════════════════════════════════════════════════════════════════
#  TensorRT helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_trt_engine(
    onnx_path: pathlib.Path,
    trt_path: pathlib.Path,
    input_name: str,
    min_shape: tuple,
    opt_shape: tuple,
    max_shape: tuple,
    fp16: bool = True,
    workspace_gb: int = 4,
) -> Optional[pathlib.Path]:
    """
    Build a TensorRT engine from an ONNX file.

    min/opt/max_shape: full tensor shapes including batch dim,
                       e.g. (1, 3, 224, 224) for the patch encoder.
    """
    if not TRT_AVAILABLE:
        print("  [!] TensorRT not available; skipping TRT build.")
        return None

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"    TRT parse error {i}: {parser.get_error(i)}")
            raise RuntimeError("TensorRT ONNX parsing failed.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  [i] FP16 mode enabled.")

    # Dynamic shape profile
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"  [~] Building TRT engine (this may take several minutes)…")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build returned None.")

    trt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(trt_path), "wb") as f:
        f.write(serialized)
    print(f"  [✓] TRT engine saved → {trt_path}")
    return trt_path


# ═══════════════════════════════════════════════════════════════════════════════
#  Main export pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run_export(args):
    # ── 1. Bootstrap SPAI config & model ──────────────────────────────────────
    # Inline the config loading that SPAI's __main__.py does so we don't
    # depend on the CLI entry-point.
    sys.path.insert(0, str(pathlib.Path(__file__).parent))  # make spai importable

    import logging
    from spai.config import get_config
    from spai import models
    from spai.utils import load_pretrained

    # Build a real logger — load_pretrained calls logger.info() unconditionally
    logger = logging.getLogger("spai.export")
    if not logger.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(_h)
        logger.setLevel(logging.INFO)

    # SPAI's get_config expects a dict, not an argparse.Namespace
    args_dict = {
        "cfg":  args.cfg,
        "opts": args.opts if args.opts else [],
        # Fields SPAI's update_config may probe — provide safe defaults
        "batch_size":        None,
        "data_path":         None,
        "pretrained":        args.model,
        "resume":            "",
        "accumulation_steps": None,
        "use_checkpoint":    None,
        "amp_opt_level":     None,
        "output":            args.output,
        "tag":               "export",
        "eval":              False,
        "throughput":        False,
        "local_rank":        0,
    }
    cfg = get_config(args_dict)
    cfg.defrost()
    cfg.MODEL.RESUME = ""          # don't auto-resume training ckpt
    cfg.PRETRAINED   = args.model  # make sure checkpoint path is set
    cfg.freeze()

    print(f"\n[1/5] Building model ({cfg.MODEL.TYPE}/{cfg.MODEL.NAME})…")

    # build_model with cfg.MODEL.NAME="finetune" builds PatchBasedMFViT / MFViT.
    # If it builds MFM instead (pre-training wrapper), we force the finetune build
    # and load the checkpoint manually with strict=False.
    from spai.models.sid import build_mf_vit as _build_mf_vit, PatchBasedMFViT as _PBMFViT
    from spai.models.mfm import MFM as _MFM

    model = models.build_model(cfg)

    if isinstance(model, _MFM):
        print("  [!] build_model returned MFM (pre-training); "
              "rebuilding as PatchBasedMFViT and loading checkpoint manually.")
        model = _build_mf_vit(cfg)

        ckpt = torch.load(args.model, map_location="cpu")
        # Checkpoint may be nested under 'model' or 'state_dict'
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [i] Missing keys  ({len(missing)}): {missing[:3]}…")
        if unexpected:
            print(f"  [i] Unexpected keys ({len(unexpected)}): {unexpected[:3]}…")
        print("  [✓] Checkpoint loaded into PatchBasedMFViT.")
    else:
        load_pretrained(cfg, model, logger=logger)

    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"      Device: {device}")

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 2. Stage 1: Patch Encoder ─────────────────────────────────────────────
    print("\n[2/5] Exporting Stage 1 — Patch Encoder…")

    # Resolve the MFViT instance
    from spai.models.sid import MFViT as _MFViT
    if isinstance(model, _MFViT):
        inner_mfvit = model
    elif hasattr(model, "mfvit") and isinstance(model.mfvit, _MFViT):
        inner_mfvit = model.mfvit
    else:
        raise RuntimeError(
            f"Cannot locate MFViT in model of type {type(model)}. "
            f"Top-level attrs: {list(model._modules.keys())}"
        )
    print(f"      inner_mfvit type : {type(inner_mfvit).__name__}")
    print(f"      inner_mfvit.vit  : {type(inner_mfvit.vit).__name__}")

    encoder_wrapper = ExportablePatchEncoder(inner_mfvit).to(device).eval()

    # aten::fft_fft2 is NOT in ONNX — the encoder takes 3 pre-split tensors.
    # At inference time, call fft_preprocess() first, then feed into the ONNX model.
    print("  [i] FFT is excluded from ONNX (not in opset). "
          "Use fft_preprocess() at inference time before calling the ONNX model.")
    print("  [i] ONNX encoder inputs: (x, x_low, x_hi) — each B x 3 x 224 x 224")

    dummy_x    = torch.rand(2, 3, 224, 224, device=device)
    dummy_xlow = torch.rand(2, 3, 224, 224, device=device)
    dummy_xhi  = torch.rand(2, 3, 224, 224, device=device)

    enc_onnx = export_to_onnx(
        encoder_wrapper,
        (dummy_x, dummy_xlow, dummy_xhi),
        out_dir / "patch_encoder.onnx",
        input_names=["x", "x_low", "x_hi"],
        output_names=["features"],
        dynamic_axes={
            "x":        {0: "batch_size"},
            "x_low":    {0: "batch_size"},
            "x_hi":     {0: "batch_size"},
            "features": {0: "batch_size"},
        },
    )

    # ── 3. Stage 2: Patch Aggregator ──────────────────────────────────────────
    print("\n[3/5] Exporting Stage 2 — Patch Aggregator…")

    # Infer D (cls_vector_dim) from the model
    cls_vector_dim: int = model.cls_vector_dim
    agg_wrapper = ExportablePatchAggregator(model).to(device).eval()

    # B=2 images, L=4 patches each
    dummy_agg = torch.rand(2, 4, cls_vector_dim, device=device)

    agg_onnx = export_to_onnx(
        agg_wrapper,
        dummy_agg,
        out_dir / "patch_aggregator.onnx",
        input_names=["patch_features"],
        output_names=["logits"],
        dynamic_axes={
            "patch_features": {0: "batch_size", 1: "num_patches"},
            "logits": {0: "batch_size"},
        },
    )

    # ── 4. Simplify + Validate ─────────────────────────────────────────────────
    print("\n[4/5] Simplifying & validating ONNX graphs…")

    enc_onnx  = simplify_onnx(enc_onnx)
    agg_onnx  = simplify_onnx(agg_onnx)

    validate_onnx_multi(
        enc_onnx,
        {
            "x":     dummy_x.cpu().numpy(),
            "x_low": dummy_xlow.cpu().numpy(),
            "x_hi":  dummy_xhi.cpu().numpy(),
        }
    )
    validate_onnx(
        agg_onnx,
        dummy_agg.cpu().numpy(),
        input_name="patch_features"
    )

    # ── 5. TensorRT ───────────────────────────────────────────────────────────
    print("\n[5/5] Building TensorRT engines…")

    if TRT_AVAILABLE:
        # Patch Encoder engine
        # min: batch=1  |  opt: batch=4  |  max: batch=16
        build_trt_engine(
            enc_onnx,
            out_dir / "patch_encoder.trt",
            input_name="image",
            min_shape=(1,  3, 224, 224),
            opt_shape=(4,  3, 224, 224),
            max_shape=(16, 3, 224, 224),
            fp16=True,
        )

        # Patch Aggregator engine
        # min: 1 image,  1 patch   |  opt: 2 images, 4 patches  |  max: 8 images, 64 patches
        build_trt_engine(
            agg_onnx,
            out_dir / "patch_aggregator.trt",
            input_name="patch_features",
            min_shape=(1, 1,  cls_vector_dim),
            opt_shape=(2, 4,  cls_vector_dim),
            max_shape=(8, 64, cls_vector_dim),
            fp16=True,
        )
    else:
        print("  [!] TensorRT unavailable — skipping. "
              "Install on Linux/WSL2: pip install tensorrt pycuda")

    print("\n✅  Export complete. Files in:", out_dir)
    for f in sorted(out_dir.iterdir()):
        print(f"    {f.name}  ({f.stat().st_size / 1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════════
#  TensorRT inference helper (standalone, for later use)
# ═══════════════════════════════════════════════════════════════════════════════

class TRTInferenceSession:
    """
    Minimal helper to run a TRT engine.

    Usage:
        enc = TRTInferenceSession("patch_encoder.trt")
        features = enc.infer(image_np)   # numpy [B, 3, 224, 224]

        agg = TRTInferenceSession("patch_aggregator.trt")
        logits  = agg.infer(features)    # numpy [B, L, D]
        probs   = 1 / (1 + np.exp(-logits))  # sigmoid
    """

    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT is not installed.")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def infer(self, input_np: np.ndarray) -> np.ndarray:
        """Run synchronous inference. Input must be float32."""
        input_np = input_np.astype(np.float32)

        # Allocate I/O buffers
        bindings = []
        outputs = []
        stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            shape = tuple(self.context.get_binding_shape(i))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

            device_mem = cuda.mem_alloc(nbytes)
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(i):
                cuda.memcpy_htod_async(device_mem, input_np, stream)
            else:
                host_mem = cuda.pagelocked_empty(shape, dtype)
                outputs.append((host_mem, device_mem, shape))

        self.context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

        results = []
        for host_mem, device_mem, shape in outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, stream)
            results.append(host_mem.reshape(shape))
        stream.synchronize()

        return results[0] if len(results) == 1 else results


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Export SPAI to ONNX + TensorRT")
    p.add_argument("--cfg",    required=True,  help="Path to spai.yaml config")
    p.add_argument("--model",  required=True,  help="Path to spai.pth checkpoint")
    p.add_argument("--output", required=True,  help="Output directory")
    p.add_argument(
        "--opts", nargs="*", default=[],
        help="Extra config overrides: KEY VALUE KEY VALUE …"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_export(args)