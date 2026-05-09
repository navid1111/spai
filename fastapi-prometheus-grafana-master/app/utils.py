"""Image processing utilities for detection."""

import numpy as np
from PIL import Image

def preprocess(img_rgb: np.ndarray, input_hw: tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess image for model inference."""
    h, w = input_hw
    pil = Image.fromarray(img_rgb)
    resized = pil.resize((w, h))
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr
