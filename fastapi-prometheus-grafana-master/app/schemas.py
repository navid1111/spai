"""Pydantic models and response schemas."""

from pydantic import BaseModel

class PredictionOut(BaseModel):
    """Prediction response schema."""
    id: int
    created_at: str
    model_name: str
    inference_ms: float
    image_url: str
    fake_probability: float
    ground_truth: dict | None
