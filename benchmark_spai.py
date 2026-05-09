"""
SPAI Backend Benchmark + Parity Checker
========================================
Compares PyTorch / ONNX / TensorRT backends on:
  - Numerical parity (do they agree on the same image?)
  - Latency (ms per image)
  - Throughput (images/sec)
  - Accuracy on a folder of images (optional, if you have labels)

Usage:
    # Full benchmark: parity check + speed on a folder
    python benchmark_spai.py \\
        --image    data/synthbuster/dalle3/1.png \\
        --pth      spai/weights/spai.pth \\
        --encoder  weights/exported/patch_encoder_simplified.onnx \\
        --aggregator weights/exported/patch_aggregator_simplified.onnx \\
        --folder   data/synthbuster/dalle3 \\
        --runs     5

    # Parity check only (diagnose ONNX vs PyTorch disagreement)
    python benchmark_spai.py \\
        --image  data/synthbuster/dalle3/1.png \\
        --pth    spai/weights/spai.pth \\
        --encoder weights/exported/patch_encoder.onnx \\
        --aggregator weights/exported/patch_aggregator.onnx \\
        --parity-only
"""

import argparse
import pathlib
import sys
import time
import os
import warnings

os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# ── Import predictor classes from predict_image.py ───────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from predict_image import (
    PytorchPredictor,
    OnnxPredictor,
    load_image,
    _fft_preprocess_numpy,
)


# ═══════════════════════════════════════════════════════════════════════════════
#  Numerical parity checker
# ═══════════════════════════════════════════════════════════════════════════════

def parity_check(
    pth_path: str,
    encoder_path: str,
    aggregator_path: str,
    image_path: str,
    config_path: str = "configs/spai.yaml",
    atol: float = 1e-3,
):
    """
    Deep numerical parity check between PyTorch and ONNX.

    Compares at THREE levels:
      1. Single-patch encoder output  (ViT features)
      2. Full image prediction        (end-to-end probability)
      3. Per-patch feature cosine similarity

    If encoder outputs differ by >atol, the ONNX export is corrupted.
    """
    print("\n" + "═"*60)
    print("  PARITY CHECK: PyTorch vs ONNX")
    print("═"*60)

    # ── Build both predictors ─────────────────────────────────────────────────
    print("\n[1/3] Loading predictors…")
    pt_pred   = PytorchPredictor(pth_path, config_path)
    ort_pred  = OnnxPredictor(encoder_path, aggregator_path, config_path)

    img = load_image(image_path)
    print(f"      Image: {image_path}  ({img.shape[1]}×{img.shape[0]})")

    # ── Level 1: Single patch encoder output ─────────────────────────────────
    print("\n[2/3] Comparing single-patch encoder output…")

    to_tensor = transforms.ToTensor()
    raw = to_tensor(img).unsqueeze(0)  # 1 x 3 x H x W

    # Crop a 224×224 patch from the top-left
    patch = raw[:, :, :224, :224]  # 1 x 3 x 224 x 224

    # PyTorch encoder pass
    inner_mfvit = pt_pred.model.mfvit
    with torch.no_grad():
        patch_dev = patch.to(pt_pred.device)
        pt_feat = inner_mfvit(patch_dev)  # 1 x D
    pt_feat_np = pt_feat.cpu().numpy()

    # ONNX encoder pass
    x_np, x_low_np, x_hi_np = _fft_preprocess_numpy(
        patch, ort_pred.freq_mask, ort_pred.backbone_norm
    )
    ort_feat_np = ort_pred.enc_sess.run(
        None, {"x": x_np, "x_low": x_low_np, "x_hi": x_hi_np}
    )[0]  # 1 x D

    max_diff   = float(np.abs(pt_feat_np - ort_feat_np).max())
    mean_diff  = float(np.abs(pt_feat_np - ort_feat_np).mean())
    cos_sim    = float(
        np.dot(pt_feat_np.flatten(), ort_feat_np.flatten()) /
        (np.linalg.norm(pt_feat_np) * np.linalg.norm(ort_feat_np) + 1e-9)
    )

    status = "✓ PASS" if max_diff < atol else "✗ FAIL — export may be corrupted"
    print(f"      Max |diff|  : {max_diff:.6f}  {status}")
    print(f"      Mean |diff| : {mean_diff:.6f}")
    print(f"      Cosine sim  : {cos_sim:.6f}  (1.0 = identical)")

    if max_diff > 0.1:
        print("\n  ⚠  Large encoder discrepancy detected!")
        print("     Likely cause: do_constant_folding=False + onnxsim dropped weights.")
        print("     Fix: re-export with do_constant_folding=True on CPU-only model,")
        print("     or skip onnxsim and use the raw (non-simplified) .onnx file.")

    # ── Level 2: End-to-end probability ──────────────────────────────────────
    print("\n[3/3] End-to-end prediction comparison…")

    with torch.no_grad():
        pt_prob = pt_pred.predict(img)
    ort_prob = ort_pred.predict(img)

    prob_diff = abs(pt_prob - ort_prob)
    print(f"      PyTorch prob : {pt_prob*100:.2f}%")
    print(f"      ONNX prob    : {ort_prob*100:.2f}%")
    print(f"      |Difference| : {prob_diff*100:.2f}pp")

    if prob_diff > 0.1:
        print("  ⚠  Probabilities differ by >10pp — backends are NOT equivalent.")
    else:
        print("  ✓  Backends agree within 10pp.")

    print("\n" + "═"*60)
    return {
        "max_feat_diff": max_diff,
        "cos_sim": cos_sim,
        "pt_prob": pt_prob,
        "ort_prob": ort_prob,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Speed benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_predictor(predictor, images: list, name: str, runs: int = 3) -> dict:
    """
    Warm up then time `runs` passes over the image list.
    Returns per-image latency stats in milliseconds.
    """
    print(f"\n  Benchmarking [{name}]…")

    # Warmup (1 pass, not timed)
    _ = predictor.predict(images[0])

    latencies = []
    for run in range(runs):
        run_times = []
        for img in images:
            t0 = time.perf_counter()
            predictor.predict(img)
            run_times.append((time.perf_counter() - t0) * 1000)
        latencies.extend(run_times)
        mean_ms = np.mean(run_times)
        print(f"    Run {run+1}/{runs}: {mean_ms:.0f} ms/img  "
              f"({1000/mean_ms:.2f} img/s)")

    stats = {
        "name":       name,
        "mean_ms":    float(np.mean(latencies)),
        "median_ms":  float(np.median(latencies)),
        "p95_ms":     float(np.percentile(latencies, 95)),
        "min_ms":     float(np.min(latencies)),
        "max_ms":     float(np.max(latencies)),
        "throughput": 1000.0 / float(np.mean(latencies)),
        "n_images":   len(images),
        "n_runs":     runs,
    }
    return stats


def print_benchmark_table(all_stats: list[dict], baseline_name: str = "PyTorch") -> None:
    baseline = next((s for s in all_stats if s["name"] == baseline_name), all_stats[0])

    print("\n" + "═"*70)
    print("  BENCHMARK RESULTS")
    print("═"*70)
    header = f"{'Backend':<20} {'Mean':>8} {'Median':>8} {'P95':>8} {'img/s':>8} {'Speedup':>8}"
    print(header)
    print("─"*70)
    for s in all_stats:
        speedup = baseline["mean_ms"] / s["mean_ms"]
        speedup_str = f"{speedup:.2f}x" if s["name"] != baseline_name else "1.00x"
        print(
            f"  {s['name']:<18} "
            f"{s['mean_ms']:>7.0f}ms "
            f"{s['median_ms']:>7.0f}ms "
            f"{s['p95_ms']:>7.0f}ms "
            f"{s['throughput']:>7.1f}  "
            f"{speedup_str:>8}"
        )
    print("═"*70)


# ═══════════════════════════════════════════════════════════════════════════════
#  TensorRT export (Linux/WSL2 only)
# ═══════════════════════════════════════════════════════════════════════════════

def build_trt_engines(
    encoder_onnx: str,
    aggregator_onnx: str,
    out_dir: pathlib.Path,
    cls_vector_dim: int = 1096,   # 6*12 + 1024 = 1096 for default config
    fp16: bool = True,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Build TRT engines from ONNX files. Linux/WSL2 only."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa
    except ImportError:
        raise RuntimeError(
            "TensorRT/pycuda not found. "
            "Install on Linux/WSL2: pip install tensorrt pycuda"
        )

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def _build(onnx_path, trt_path, input_profiles):
        builder  = trt.Builder(TRT_LOGGER)
        network  = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser   = trt.OnnxParser(network, TRT_LOGGER)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"TRT parse failed: {errs}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        for name, (mn, opt, mx) in input_profiles.items():
            profile.set_shape(name, mn, opt, mx)
        config.add_optimization_profile(profile)

        print(f"  [~] Building {trt_path.name}…  (may take several minutes)")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT build returned None")
        trt_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(trt_path), "wb") as f:
            f.write(serialized)
        print(f"  [✓] {trt_path}")

    enc_trt = out_dir / "patch_encoder.trt"
    agg_trt = out_dir / "patch_aggregator.trt"

    _build(encoder_onnx, enc_trt, {
        "x":     ((1,3,224,224), (4,3,224,224),  (16,3,224,224)),
        "x_low": ((1,3,224,224), (4,3,224,224),  (16,3,224,224)),
        "x_hi":  ((1,3,224,224), (4,3,224,224),  (16,3,224,224)),
    })
    _build(aggregator_onnx, agg_trt, {
        "patch_features": (
            (1,  1, cls_vector_dim),
            (1,  8, cls_vector_dim),
            (1, 64, cls_vector_dim),
        ),
    })
    return enc_trt, agg_trt


class TRTPredictor:
    """TensorRT inference — mirrors OnnxPredictor exactly but uses TRT engines."""

    def __init__(
        self,
        encoder_trt: str,
        aggregator_trt: str,
        config_path: str = "configs/spai.yaml",
    ):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa
            self._trt = trt
            self._cuda = cuda
        except ImportError:
            raise RuntimeError("TensorRT/pycuda not installed.")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        def _load(path):
            with open(path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            return engine, engine.create_execution_context()

        self.enc_engine,  self.enc_ctx  = _load(encoder_trt)
        self.agg_engine,  self.agg_ctx  = _load(aggregator_trt)

        # Freq mask + norm — same as OnnxPredictor
        from spai.config import get_config
        from spai.models.sid import build_mf_vit

        cfg = get_config({
            "cfg": config_path, "pretrained": "", "opts": [],
            "batch_size": None, "data_path": None, "resume": "",
            "accumulation_steps": None, "use_checkpoint": None,
            "amp_opt_level": None, "output": ".", "tag": "trt_predict",
            "eval": False, "throughput": False, "local_rank": 0,
        })
        mf_model = build_mf_vit(cfg)
        inner = mf_model.mfvit if hasattr(mf_model, "mfvit") else mf_model
        self.freq_mask     = inner.frequencies_mask.data.clone().cpu()
        self.backbone_norm = inner.backbone_norm
        self.patch_size    = cfg.DATA.IMG_SIZE
        self.patch_stride  = cfg.MODEL.PATCH_VIT.PATCH_STRIDE
        self.min_patches   = cfg.MODEL.PATCH_VIT.MINIMUM_PATCHES

    def _infer(self, engine, ctx, feed: dict) -> np.ndarray:
        import pycuda.driver as cuda
        trt = self._trt
        stream = cuda.Stream()
        bindings = []
        output_bufs = []

        for i in range(engine.num_bindings):
            dtype = trt.nptype(engine.get_binding_dtype(i))
            shape = tuple(ctx.get_binding_shape(i))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            dev_mem = cuda.mem_alloc(nbytes)
            bindings.append(int(dev_mem))
            if engine.binding_is_input(i):
                name = engine.get_binding_name(i)
                arr = feed[name].astype(dtype)
                cuda.memcpy_htod_async(dev_mem, arr, stream)
            else:
                host_mem = cuda.pagelocked_empty(shape, dtype)
                output_bufs.append((host_mem, dev_mem))

        ctx.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        results = []
        for host_mem, dev_mem in output_bufs:
            cuda.memcpy_dtoh_async(host_mem, dev_mem, stream)
            results.append(host_mem.copy())
        stream.synchronize()
        return results[0] if len(results) == 1 else results

    def predict(self, img: np.ndarray) -> float:
        from spai.models.utils import patchify_image
        from torchvision.transforms.functional import five_crop

        to_tensor = transforms.ToTensor()
        raw = to_tensor(img).unsqueeze(0)

        patched = patchify_image(
            raw,
            (self.patch_size, self.patch_size),
            (self.patch_stride, self.patch_stride),
        )
        if patched.size(1) < self.min_patches:
            crops = five_crop(raw.squeeze(0), [self.patch_size, self.patch_size])
            patched = torch.stack(crops, dim=0).unsqueeze(0)

        L = patched.size(1)
        patches = patched.squeeze(0)

        # Set dynamic input shapes for encoder
        self.enc_ctx.set_binding_shape(0, (1, 3, 224, 224))
        self.enc_ctx.set_binding_shape(1, (1, 3, 224, 224))
        self.enc_ctx.set_binding_shape(2, (1, 3, 224, 224))

        all_features = []
        for i in range(L):
            patch = patches[i].unsqueeze(0)
            x, x_low, x_hi = _fft_preprocess_numpy(
                patch, self.freq_mask, self.backbone_norm
            )
            feat = self._infer(self.enc_engine, self.enc_ctx,
                               {"x": x, "x_low": x_low, "x_hi": x_hi})
            all_features.append(feat)

        patch_features = np.concatenate(all_features, axis=0)[np.newaxis].astype(np.float32)

        # Set aggregator shape
        self.agg_ctx.set_binding_shape(0, patch_features.shape)
        logit = self._infer(self.agg_engine, self.agg_ctx,
                            {"patch_features": patch_features})

        return float(1.0 / (1.0 + np.exp(-logit.flatten()[0])))


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="SPAI benchmark + parity check")
    p.add_argument("--image",       required=True,  help="Single image for parity check")
    p.add_argument("--pth",         default=None,   help="PyTorch .pth checkpoint")
    p.add_argument("--encoder",     default=None,   help="patch_encoder.onnx")
    p.add_argument("--aggregator",  default=None,   help="patch_aggregator.onnx")
    p.add_argument("--enc-trt",     default=None,   help="patch_encoder.trt (optional)")
    p.add_argument("--agg-trt",     default=None,   help="patch_aggregator.trt (optional)")
    p.add_argument("--folder",      default=None,
                   help="Folder of images for speed benchmark")
    p.add_argument("--n-images",    type=int, default=20,
                   help="Max images to use from folder for benchmark (default 20)")
    p.add_argument("--runs",        type=int, default=3,
                   help="Timing runs per backend (default 3)")
    p.add_argument("--cfg",         default="configs/spai.yaml")
    p.add_argument("--parity-only", action="store_true",
                   help="Only run parity check, skip speed benchmark")
    p.add_argument("--build-trt",   action="store_true",
                   help="Build TRT engines from the ONNX files before benchmarking")
    p.add_argument("--trt-out",     default="weights/exported",
                   help="Output dir for built TRT engines")
    p.add_argument("--atol",        type=float, default=1e-3,
                   help="Absolute tolerance for parity check (default 1e-3)")
    return p.parse_args()


def main():
    args = parse_args()
    all_stats = []

    # ── Optional: build TRT engines first ────────────────────────────────────
    if args.build_trt:
        if not args.encoder or not args.aggregator:
            print("ERROR: --build-trt requires --encoder and --aggregator")
            sys.exit(1)
        print("\n[TRT] Building TensorRT engines…")
        enc_trt, agg_trt = build_trt_engines(
            args.encoder, args.aggregator,
            pathlib.Path(args.trt_out)
        )
        args.enc_trt = str(enc_trt)
        args.agg_trt = str(agg_trt)

    # ── Parity check ─────────────────────────────────────────────────────────
    if args.pth and (args.encoder or args.aggregator):
        if not (args.encoder and args.aggregator):
            print("ERROR: Parity check needs both --encoder and --aggregator")
            sys.exit(1)
        parity_check(
            args.pth, args.encoder, args.aggregator,
            args.image, args.cfg, args.atol
        )

    if args.parity_only:
        return

    # ── Load images for benchmark ─────────────────────────────────────────────
    bench_images = []
    if args.folder:
        folder = pathlib.Path(args.folder)
        exts = ["jpg","jpeg","png","webp","bmp"]
        paths = []
        for ext in exts:
            paths.extend(folder.glob(f"*.{ext}"))
            paths.extend(folder.glob(f"*.{ext.upper()}"))
        paths = sorted(set(paths))[:args.n_images]
        print(f"\nLoading {len(paths)} images from {folder}…")
        for p in paths:
            try:
                bench_images.append(load_image(str(p)))
            except Exception as e:
                print(f"  Skip {p.name}: {e}")
    else:
        # Single image repeated
        bench_images = [load_image(args.image)] * 5
        print(f"\nUsing single image ×{len(bench_images)} for benchmark")

    if not bench_images:
        print("No images to benchmark.")
        return

    print(f"\n{'═'*60}")
    print(f"  SPEED BENCHMARK  ({len(bench_images)} images × {args.runs} runs)")
    print(f"{'═'*60}")

    # ── PyTorch ───────────────────────────────────────────────────────────────
    if args.pth:
        pt = PytorchPredictor(args.pth, args.cfg)
        all_stats.append(benchmark_predictor(pt, bench_images, "PyTorch", args.runs))
        del pt

    # ── ONNX ─────────────────────────────────────────────────────────────────
    if args.encoder and args.aggregator:
        ort = OnnxPredictor(args.encoder, args.aggregator, args.cfg)
        all_stats.append(benchmark_predictor(ort, bench_images, "ONNX-CPU", args.runs))
        del ort

    # ── TensorRT ─────────────────────────────────────────────────────────────
    if args.enc_trt and args.agg_trt:
        try:
            trt_pred = TRTPredictor(args.enc_trt, args.agg_trt, args.cfg)
            all_stats.append(benchmark_predictor(trt_pred, bench_images, "TensorRT", args.runs))
            del trt_pred
        except RuntimeError as e:
            print(f"  [!] TRT skipped: {e}")

    # ── Print table ───────────────────────────────────────────────────────────
    if all_stats:
        print_benchmark_table(all_stats, baseline_name=all_stats[0]["name"])

    # ── Per-image probability comparison table ────────────────────────────────
    if len(all_stats) > 1 and args.folder:
        print("\n  Per-image probability comparison (first 10):")
        print(f"  {'Image':<30} " + "  ".join(f"{s['name']:>10}" for s in all_stats))
        print("  " + "─"*60)

        # Re-run quickly for comparison table
        predictors = []
        if args.pth:
            predictors.append(("PyTorch",  PytorchPredictor(args.pth, args.cfg)))
        if args.encoder and args.aggregator:
            predictors.append(("ONNX",     OnnxPredictor(args.encoder, args.aggregator, args.cfg)))

        folder = pathlib.Path(args.folder)
        sample_paths = sorted(
            p for ext in ["png","jpg","jpeg"]
            for p in folder.glob(f"*.{ext}")
        )[:10]

        for img_path in sample_paths:
            img = load_image(str(img_path))
            probs = []
            for _, pred in predictors:
                probs.append(pred.predict(img))
            prob_strs = "  ".join(f"{p*100:>9.1f}%" for p in probs)
            print(f"  {img_path.name:<30} {prob_strs}")


if __name__ == "__main__":
    main()