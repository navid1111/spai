# Using CNN Weights for SPAI Inference (Without Fine-Tuning)

This guide focuses only on inference with CNN checkpoints.

If you only need predicted scores and do not care about training, you can avoid optimizer and training-loop changes. You still need a small model-construction and checkpoint-loading update, because current inference assumes the transformer path.

## What is currently hardcoded

The current inference command uses:

- `spai/__main__.py` -> `infer(...)` -> `build_cls_model(config)` and `load_pretrained(...)`
- `spai/models/build.py` -> SID model creation only for `MODEL.TYPE == "vit"`
- `spai/utils.py::load_pretrained()` -> key remapping only for `vit` and `swin`

This means a CNN checkpoint will not run out of the box.

## Minimal inference-only migration

You only need four parts:

1. Add a CNN backbone wrapper.
2. Allow SID model build for `MODEL.TYPE: cnn`.
3. Allow checkpoint loading for `MODEL.TYPE: cnn`.
4. Use a CNN inference config.

No optimizer changes are required for inference-only usage.

## Step 1: Add CNN config options

Update `spai/config.py`:

```python
_C.MODEL.CNN = CN()
_C.MODEL.CNN.NAME = "resnet50"
_C.MODEL.CNN.OUT_INDICES = [1, 2, 3, 4]
_C.MODEL.CNN.IN_CHANS = 3
_C.MODEL.CNN.FEATURE_DIM = 768
_C.MODEL.CNN.PRETRAINED = False
```

## Step 2: Add a CNN backbone wrapper

Add in `spai/models/backbones.py`:

```python
import timm

class CNNBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "resnet50",
        out_indices: tuple[int, ...] = (1, 2, 3, 4),
        feature_dim: int = 768,
        in_chans: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()
        self.cnn = timm.create_model(
            model_name,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_chans,
            pretrained=pretrained,
        )
        channels = self.cnn.feature_info.channels()
        self.proj = nn.ModuleList([nn.Conv2d(c, feature_dim, kernel_size=1) for c in channels])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.cnn(x)  # list of B x C x H x W

        # Make all stages share the same spatial size before tokenization.
        target_h, target_w = feats[-1].shape[-2], feats[-1].shape[-1]
        tokens: list[torch.Tensor] = []
        for f, p in zip(feats, self.proj):
            f = torch.nn.functional.interpolate(
                f, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            z = p(f)  # B x D x H x W
            z = z.flatten(2).transpose(1, 2)  # B x L x D
            tokens.append(z)

        return torch.stack(tokens, dim=1)  # B x N x L x D
```

## Step 3: Plug CNN into SID construction

Update `spai/models/sid.py` in the builder (currently `build_mf_vit`):

```python
elif config.MODEL.TYPE == "cnn":
    vit = backbones.CNNBackbone(
        model_name=config.MODEL.CNN.NAME,
        out_indices=tuple(config.MODEL.CNN.OUT_INDICES),
        feature_dim=config.MODEL.CNN.FEATURE_DIM,
        in_chans=config.MODEL.CNN.IN_CHANS,
        pretrained=config.MODEL.CNN.PRETRAINED,
    )
    initialization_scope = "local"
```

Also set FRE input width from `MODEL.CNN.FEATURE_DIM` when `MODEL.TYPE == "cnn"`.

In type checks inside SID modules, include `backbones.CNNBackbone` where backbone class is validated.

## Step 4: Allow model build for cnn type

Update `spai/models/build.py` so inference can instantiate SID when `MODEL.TYPE == "cnn"`:

```python
if task_type == "freq_restoration" and model_type in ["vit", "cnn"]:
    model = build_mf_vit(config)
else:
    raise NotImplementedError(...)
```

## Step 5: Allow checkpoint loading for cnn type

Update `spai/utils.py::load_pretrained()`:

```python
if config.MODEL.TYPE == "swin":
    checkpoint_model = remap_pretrained_keys_swin(model, checkpoint_model, logger)
elif config.MODEL.TYPE == "vit":
    checkpoint_model = remap_pretrained_keys_vit(model, checkpoint_model, logger)
elif config.MODEL.TYPE == "cnn":
    # No ViT/Swin remap. Optional key-prefix cleanup for third-party checkpoints.
    pass
else:
    raise NotImplementedError

msg = model.load_state_dict(checkpoint_model, strict=False)
```

If needed, strip common prefixes first: `module.`, `backbone.`, `encoder.`.

## Step 6: Inference config

Create `configs/spai_cnn_infer.yaml`:

```yaml
MODEL:
  TYPE: cnn
  SID_APPROACH: "freq_restoration"
  NUM_CLASSES: 2
  RESOLUTION_MODE: "arbitrary"
  REQUIRED_NORMALIZATION: "imagenet"
  CNN:
    NAME: "resnet50"
    OUT_INDICES: [1, 2, 3, 4]
    IN_CHANS: 3
    FEATURE_DIM: 768
    PRETRAINED: false
```

Then run inference:

```bash
python -m spai infer \
  --cfg "./configs/spai_cnn_infer.yaml" \
  --model "./weights/your_cnn_checkpoint.pth" \
  --input "./path/to/images_or_csv" \
  --output "./output/cnn_infer"
```

## Compatibility checklist

Before full inference, verify:

1. Model forward works for one batch.
2. Log from `load_state_dict` has mostly expected matches.
3. CNN outputs are transformed to `B x N x L x D`.
4. `MODEL.CNN.FEATURE_DIM` matches FRE expectations.

## When to choose a simpler route

If your checkpoint is already a complete SPAI model checkpoint (`checkpoint["model"]` from this repo), you do not need these changes.

If your checkpoint is a generic CNN encoder from another project, you need the migration above.
