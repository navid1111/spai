# SPAI on Medical Images — Domain Gap Analysis
### Applied to the OpenI Chest X-Ray Dataset

Based on CVPR 2025: *"Any-Resolution AI-Generated Image Detection by Spectral Learning"*

---

## Background: What is SPAI?

SPAI operates in two phases:

1. **Pre-training (MFM):** A Vision Transformer learns to reconstruct masked spectral bands of natural images in a self-supervised manner.
2. **Fine-tuning (SID):** FRE computes spectral similarity scores; SCA aggregates patch-level features for any-resolution classification.

**Core assumption:** Real images have consistent spectral reconstruction patterns. AI-generated images are detected as out-of-distribution (OOD) in frequency space.

| Module | Role |
|---|---|
| `filters.py` | FFT-based frequency split |
| `mfm.py` | Self-supervised pre-training |
| `sid.py` (FRE) | Spectral similarity scores |
| `sid.py` (SCA) | Any-res patch aggregation |
| `vision_transformer.py` | ViT-B/16 backbone |

> **Training Data:** ImageNet only — natural RGB images.

---

## The OpenI Dataset

OpenI is a public chest X-ray dataset from Indiana University, distributed via the NIH. It contains ~7,000 frontal and lateral chest radiographs paired with radiology reports.

**Format as distributed:** PNG images converted from original DICOM acquisitions, grayscale, typically 8-bit, resolution varies (~2000×2500 px down to ~800×1000 px depending on acquisition).

**What is lost in the DICOM → PNG conversion:**
- Pixel spacing (mm/pixel) — physical scale metadata dropped
- Bit depth — original 12-bit acquisitions are windowed and clipped to 8-bit PNG
- Rescale slope / intercept — HU-equivalent calibration removed
- Modality tag — no longer machine-readable
- Patient orientation metadata — discarded

This means for OpenI, the input to SPAI is already a converted grayscale PNG — but the domain gap problems described below still apply.

---

## Natural vs. Medical Images — The Domain Gap

| Property | Natural Images | OpenI / Chest X-Ray |
|---|---|---|
| Format | 2D RGB, 8-bit | Grayscale PNG, 8-bit (converted from 12-bit DICOM) |
| Pixel Intensity | Perceptual only | Encodes tissue opacity (attenuation) |
| Object Location | Irrelevant | Clinically meaningful (lesion position matters) |
| Object Scale | Irrelevant | Diagnostically critical (nodule size in mm) |
| Orientation | Canonical (upright) | PA vs lateral; no rotation standard |
| Noise Profile | Camera sensor noise | Quantum noise, grid lines, scatter |
| Compression | JPEG / PNG | Lossless PNG (from DICOM export) |
| Data Availability | Millions (web-scraped) | ~7,000 images, expert-labelled reports |

> **Key implication:** Every statistical assumption SPAI makes is calibrated for natural images. OpenI chest X-rays violate almost all of them simultaneously.

---

## Why SPAI Fails on OpenI

### Reason 1 — Wrong Spectral Prior from MFM Pre-training

MFM is pre-trained on ImageNet only. The ViT backbone's learned "normal" spectral reconstruction pattern does not transfer to chest X-ray data.

When a real OpenI chest X-ray passes through SPAI, its spectral reconstruction diverges from the natural-image prior — so SPAI classifies it as OOD even though it is a genuine scan. This results in a **high false-positive rate on real chest X-rays**.

Chest X-rays have fundamentally different frequency energy distributions compared to natural photos — dominated by bone and soft-tissue boundaries rather than colour textures and object edges.

---

### Reason 2 — Normalization Discards Radiographic Intensity

`data_finetune.py` normalizes all images to `[0, 1]` with optional ImageNet mean/std subtraction.

For OpenI PNG images, pixel intensity encodes tissue attenuation — brighter regions correspond to denser structures (bone, consolidation), darker regions to air-filled lung. This is not perceptual; it is physically meaningful.

Normalization collapses these differences, making AI-hallucinated lung texture look spectrally identical to real tissue.

| | Natural Images | OpenI Chest X-Ray |
|---|---|---|
| Intensity meaning | Perceptual only | Encodes tissue density |
| Absolute values | Irrelevant | Carry diagnostic signal |
| Normalization | Safe | Removes discriminative signal |

Even though OpenI PNG images have already lost true HU values during DICOM export, the relative intensity structure (lung fields vs. mediastinum vs. bone) is still meaningful and should not be collapsed by ImageNet-style normalization.

---

### Reason 3 — RGB Pipeline vs. Grayscale Input

SPAI's `filters.py` FFT operates on 3-channel 8-bit images. OpenI images are single-channel grayscale PNGs.

**What goes wrong with OpenI:**
- Grayscale images are forced into the 3-channel pipeline by replicating the single channel — producing three identical channels with no additional information.
- The circular frequency mask (radius = 16) was tuned for the spatial statistics of RGB natural image patches, not chest radiograph anatomy.
- Colour jitter and saturation augmentations are applied to channels that carry no colour information — these transforms are entirely meaningless for X-rays.

---

### Reason 4 — Fixed Frequency Mask Radius Misaligned

`generate_circular_mask()` applies a fixed `radius=16` to separate low- and high-frequency bands, empirically tuned for 224×224 natural image patches.

For OpenI chest X-rays, the clinically relevant structures operate at different spatial frequencies. Fine pulmonary vessel detail, subtle interstitial markings, and early nodules exist at high-frequency bands that may not align with the fixed radius=16 boundary. This leads to incorrect FRE cosine similarity scores and unreliable predictions.

> The optimal frequency boundary varies by radiographic content — a fixed radius cannot correctly isolate clinically relevant bands.

---

### Reason 5 — Scale Invariance Assumption Breaks Down

SPAI warps all patches to 224×224 and discards scale information. For OpenI, this creates two problems:

1. **Physical scale is lost:** OpenI images have varying acquisition resolutions. A 5 mm nodule appearing in a high-resolution scan and a low-resolution scan will be represented by different numbers of pixels — but SPAI has no way to distinguish these after resizing.
2. **AI generators replicate texture but fail at physically accurate sizing** — this is a detectable signal that SPAI discards by being scale-agnostic.

Since the original DICOM PixelSpacing is not preserved in the OpenI PNG distribution, this information is unavailable at inference time unless re-derived from acquisition metadata or report context.

---

### Reason 6 — No Orientation Invariance for Radiographs

OpenI contains both PA (posteroanterior) and lateral projections. Within PA views, patient positioning varies. SPAI's ViT backbone has no architectural rotation invariance.

The same real lung lesion at slightly different patient rotations produces different FRE similarity scores — adding spurious variance to the detection signal that is unrelated to whether the image is AI-generated.

---

### Reason 7 — Mismatched Augmentation Pipeline

SPAI's augmentations in `data_finetune.py` are derived entirely from natural photography artifacts. For OpenI, the actual degradation patterns present in real chest X-rays are completely different:

| Remove (photography) | Add (chest X-ray) |
|---|---|
| JPEG / WEBP compression | Quantum noise |
| Colour jitter | Grid line artifacts |
| Saturation shift | Scatter / beam hardening |
| Natural camera blur | Receptor noise patterns |
| — | Exposure variation |

Medical scanner artifacts alter FRE cosine similarity scores in ways the model has never been trained on — making detection outputs uncalibrated for clinical data.

---

## Summary — Why SPAI Fails on OpenI

| SPAI Design Choice | Natural Images | OpenI Chest X-Ray |
|---|---|---|
| MFM pre-trained on ImageNet | Correct spectral prior | Wrong spectral prior |
| Normalize to [0, 1] | Safe | Destroys attenuation signal |
| RGB 3-channel 8-bit input | Correct format | Grayscale replicated to 3ch |
| JPEG / colour augmentations | Realistic artifacts | Wrong artifact types |
| Fixed circular mask (r=16) | Tuned for natural imgs | Wrong frequency boundary |
| 224×224 + scale discard | Scale-invariant OK | Scale carries diagnostic signal |
| No rotation invariance | Not needed | PA/lateral variance adds noise |
| OOD from natural distribution | Well-calibrated | Real X-rays appear OOD |

---

## Proposed Solutions for OpenI

The fixes are split into two parts based on where they need to be applied. **Part A** covers limitations that are baked into the pre-trained ViT backbone — these require re-running MFM pre-training and are computationally expensive, so they are documented here as known limitations and future work. **Part B** covers fixes that can be applied during fine-tuning on OpenI with the existing backbone, at no additional pre-training cost.

---

## Part A — Backbone-Level Limitations (Require Pre-training)

These issues stem from the ViT backbone weights themselves. They are documented here for completeness and to explain why the fine-tuning fixes in Part B have a ceiling — the backbone's spectral prior will remain misaligned until re-pre-training is feasible.

---

### Limitation A1 — Wrong Spectral Prior in the Backbone

The ViT backbone was pre-trained via MFM on ImageNet only. Its learned "normal" spectral reconstruction pattern is calibrated entirely for natural RGB images and does not transfer to chest X-ray data. This is the root limitation — everything downstream in the FRE and SCA is affected by it.

Ideally, the backbone would be re-pre-trained on large chest X-ray datasets such as CheXpert (224K images) or MIMIC-CXR (227K images), using a two-stage approach: ImageNet pre-training for general spectral features, followed by chest X-ray domain adaptation.

This is the highest-impact fix but also the most computationally expensive — it requires running `main_mfm.py` from scratch on medical data.

---

### Limitation A2 — RGB Patch Embedding

The ViT's patch embedding is a Conv2d with `in_channels=3`, learned during MFM. The backbone was never trained on grayscale data. At fine-tuning time, OpenI images are force-replicated to 3 identical channels, which adds no information and wastes model capacity. Fixing this properly requires retraining the patch embedding layer during pre-training.

---

### Limitation A3 — Fixed Frequency Mask Radius Baked into Pre-training

The circular mask radius=16 is used not just at fine-tuning but during MFM pre-training itself — it defines which frequency bands the ViT learns to reconstruct. Because the backbone was pre-trained with this boundary, it has never learned to attend to the frequency ranges most relevant to chest radiographs. A multi-scale mask bank (radii 8, 16, 32) would need to be introduced at pre-training time.

---

### Limitation A4 — No Rotation Invariance in the Backbone

The ViT architecture has no built-in rotation equivariance. Replacing patch convolutions with G-Conv layers (equivariant to 90° rotations and flips) would need to happen before pre-training, since it is an architectural change. The TTA workaround in Part B partially mitigates this at inference time.

---

## Part B — Fine-tuning Fixes (Applicable Now)

These fixes can all be applied when running `python -m spai train` on the OpenI dataset. They do not require touching the pre-trained backbone.

---

### Fix B1 — OpenI-Appropriate Normalization

Replace ImageNet mean/std normalization with statistics computed from the OpenI training split. Even though OpenI PNG images have lost true HU values during DICOM export, the relative intensity structure (lung fields vs. mediastinum vs. bone) is still meaningful and should not be collapsed by ImageNet colour statistics.

```yaml
# spai.yaml — fine-tuning config
DATA.NORM: dataset_specific
DATA.MEAN: [0.482]          # computed from OpenI train set
DATA.STD:  [0.237]          # computed from OpenI train set
DATA.GRAYSCALE: true
```

**Files to modify:** `data/data_finetune.py`, `config.py`

---

### Fix B2 — Disable Colour Augmentations for Grayscale Input

Colour jitter, saturation shifts, and hue transforms have zero effect on grayscale images and add spurious training noise. These should be disabled entirely for OpenI regardless of any other changes.

```yaml
# spai.yaml — fine-tuning config
AUG.COLOR_JITTER: 0.0
AUG.GRAYSCALE_PROB: 0.0
AUG.SATURATION: 0.0
```

**Files to modify:** `spai.yaml`, `data/data_finetune.py`

---

### Fix B3 — Chest X-Ray Augmentation Pipeline

Replace photography-specific Albumentations transforms with radiograph-appropriate augmentations. Medical scanner artifacts alter FRE cosine similarity scores in ways the model has never been trained on — calibrating against real radiograph degradation patterns improves score reliability.

| Remove | Add |
|---|---|
| JPEG / WEBP compression | Quantum noise simulation |
| Colour jitter | Exposure variation |
| Saturation shift | Random grid-line artifacts |
| Natural camera blur | Scanner-specific PSF blur |
| — | Random contrast window shifts |

**Library recommendation:** Replace Albumentations with **TorchIO** or **MONAI** for medical-appropriate transforms.

**Files to modify:** `data/data_finetune.py`, `data/blur_kernels.py`

---

### Fix B4 — Resolution-Based Scale Embedding in FRE

Since OpenI PNG images do not carry DICOM PixelSpacing, original image resolution (height × width before patch resizing) is the best available proxy for physical scale. This can be appended to the FRE output vector as a learned scale embedding during fine-tuning — no backbone changes required.

```
FRE vector: 6N + D  (current)
         →  6N + D + S  (proposed)
```

where `S` is a learned projection of the (height, width) tuple normalized by dataset statistics.

**Files to modify:** `models/sid.py`, `config.py`

---

### Fix B5 — Test-Time Augmentation for Orientation Robustness

Since the backbone cannot be made architecturally rotation-equivariant without re-pre-training, TTA is the practical workaround. At inference time, average predictions across 8 orientations (0°, 90°, 180°, 270° + horizontal flips). This reduces the spurious variance introduced by patient positioning differences in PA and lateral views.

This is a purely inference-time change — no retraining required.

**Files to modify:** `__main__.py` (`validate()` function), `config.py`

---

## Summary

### Part A — Backbone Limitations (Future Work)

| Limitation | Root Cause | Files That Would Need Changing |
|---|---|---|
| A1 — Wrong spectral prior | MFM pre-trained on ImageNet only | `main_mfm.py`, `spai.yaml` |
| A2 — RGB patch embedding | Patch Conv2d learned on 3-channel data | `models/vision_transformer.py` |
| A3 — Fixed mask radius in pre-training | Radius=16 baked into MFM objective | `models/filters.py`, `models/mfm.py` |
| A4 — No rotation invariance | ViT architecture | `models/vision_transformer.py` |

### Part B — Fine-tuning Fixes (Actionable Now)

| Fix | Files Affected | Impact |
|---|---|---|
| B1 — OpenI-specific normalization | `data/data_finetune.py`, `config.py` | High |
| B2 — Disable colour augmentations | `spai.yaml`, `data/data_finetune.py` | High |
| B3 — Chest X-ray augmentation pipeline | `data/data_finetune.py`, `data/blur_kernels.py` | Medium |
| B4 — Resolution-based scale embedding | `models/sid.py`, `config.py` | Medium |
| B5 — TTA for orientation robustness | `__main__.py`, `config.py` | Low |

---

## Conclusion

SPAI's difficulties on OpenI all stem from a single root cause: every component — the spectral prior, normalization, input format, mask radius, and augmentations — was designed exclusively for natural RGB images. Chest X-rays, even after DICOM-to-PNG conversion, violate these assumptions at every level.

The Part B fine-tuning fixes address the data pipeline issues and can meaningfully improve performance on OpenI without any pre-training cost. However, there is a ceiling — as long as the backbone's spectral prior remains calibrated for natural images, real chest X-rays will continue to look partially OOD to the model regardless of how well the fine-tuning is configured.

The Part A limitations are documented not as immediate action items but as an explanation of why that ceiling exists. Re-pre-training on chest X-ray data (CheXpert, MIMIC-CXR) would be the path to fully resolving the domain gap, when computational resources allow.

**The OOD detection paradigm is sound — only the domain of the prior needs to change. SPAI's architecture does not need to be rebuilt from scratch.**