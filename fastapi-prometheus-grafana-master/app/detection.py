"""Detection service using SPAI."""

import time
import sys
from pathlib import Path
import numpy as np

# Need to import OnnxPredictor from predict_image.py in the root directory.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from predict_image import OnnxPredictor

class DetectionService:
    """Service for running ONNX model inference for SPAI fake detection."""
    
    def __init__(self):
        self.predictor = None
        self.model_name = "spai-onnx"
        self.input_hw = (224, 224)

    def initialize(self, model_path: str = "") -> None:
        """Initialize the ONNX model."""
        base_dir = Path(__file__).resolve().parent.parent.parent
        encoder_path = str(base_dir / "weights/exported/patch_encoder.onnx")
        aggregator_path = str(base_dir / "weights/exported/patch_aggregator.onnx")
        config_path = str(base_dir / "configs/spai.yaml")
        
        self.predictor = OnnxPredictor(
            encoder_path=encoder_path,
            aggregator_path=aggregator_path,
            config_path=config_path
        )

    def is_ready(self) -> bool:
        """Check if the model is initialized."""
        return self.predictor is not None

    def infer(self, image_np: np.ndarray, **kwargs) -> tuple[float, float]:
        """
        Run inference on the image.
        
        Returns:
            Tuple of (fake_probability, inference_time_ms)
        """
        if not self.is_ready():
            raise RuntimeError("Model is not initialized")

        start = time.perf_counter()
        prob = self.predictor.predict(image_np)
        inference_ms = (time.perf_counter() - start) * 1000.0

        return prob, inference_ms
