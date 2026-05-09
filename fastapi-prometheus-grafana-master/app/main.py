"""FastAPI application for fake image detection."""

import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from prometheus_fastapi_instrumentator import Instrumentator

from .config import UPLOAD_DIR
from .database import init_db
from .db_service import get_prediction, get_predictions, save_prediction
from .detection import DetectionService
from .schemas import PredictionOut

# Initialize FastAPI app
app = FastAPI(title="SPAI Fake Detection API")

# Initialize detection service (global)
detection_service = DetectionService()


@app.on_event("startup")
def startup_event() -> None:
    """Initialize database and model on startup."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    init_db()
    detection_service.initialize()


@app.get("/")
def home() -> dict:
    """Health check endpoint."""
    return {"message": "SPAI Fake Detection API is running"}


@app.get("/health")
def health() -> dict:
    """Detailed health check."""
    return {
        "status": "ok",
        "model": detection_service.model_name,
        "input_hw": detection_service.input_hw,
    }


@app.post("/predict", response_model=PredictionOut)
async def predict(
    image: UploadFile = File(...),
    ground_truth: str | None = Form(None),
) -> PredictionOut:
    """
    Run inference on an uploaded image.
    """
    if not detection_service.is_ready():
        raise HTTPException(status_code=500, detail="Model is not initialized")

    try:
        image_bytes = await image.read()
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_np = np.array(pil_img)

    # Save original image
    original_name = f"{uuid.uuid4().hex}_{image.filename or 'image.jpg'}"
    original_path = UPLOAD_DIR / original_name
    pil_img.save(original_path)

    # Run inference
    prob, inference_ms = detection_service.infer(image_np)

    # Record Prometheus metrics
    from .metrics import (
        prediction_counter,
        inference_time_histogram,
        fake_probability_histogram,
        avg_fake_probability_gauge,
        high_fake_probability_counter
    )
    
    model_name = detection_service.model_name
    prediction_counter.labels(model_name=model_name).inc()
    inference_time_histogram.labels(model_name=model_name).observe(inference_ms)
    
    fake_probability_histogram.labels(model_name=model_name).observe(prob)
    avg_fake_probability_gauge.labels(model_name=model_name).set(prob)
    
    if prob >= 0.5:
        high_fake_probability_counter.labels(model_name=model_name).inc()

    # Save to database
    prediction = save_prediction(
        image_path=str(original_path),
        model_name=model_name,
        inference_ms=inference_ms,
        fake_probability=prob,
        ground_truth_json=ground_truth,
    )

    return get_prediction(prediction.id)


@app.get("/predictions", response_model=list[PredictionOut])
def list_predictions(limit: int = 20) -> list[PredictionOut]:
    """Get recent predictions."""
    return get_predictions(limit)


@app.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction_detail(prediction_id: int) -> PredictionOut:
    """Get a specific prediction by ID."""
    result = get_prediction(prediction_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result


# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Add Prometheus instrumentation
Instrumentator().instrument(app).expose(app)
