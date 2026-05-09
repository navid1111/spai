"""Database service for predictions and detections."""

import json
from pathlib import Path

from .database import Prediction, SessionLocal
from .schemas import PredictionOut


def save_prediction(
    image_path: str,
    model_name: str,
    inference_ms: float,
    fake_probability: float,
    ground_truth_json: str | None = None,
) -> Prediction:
    """Save prediction and detections to the database."""
    session = SessionLocal()
    try:
        prediction = Prediction(
            image_path=image_path,
            model_name=model_name,
            inference_ms=inference_ms,
            fake_probability=fake_probability,
            ground_truth_json=ground_truth_json,
        )
        session.add(prediction)
        session.commit()
        session.refresh(prediction)
        return prediction
    finally:
        session.close()


def get_predictions(limit: int = 20) -> list[PredictionOut]:
    """Get recent predictions from the database."""
    session = SessionLocal()
    try:
        rows = (
            session.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
        return [_prediction_to_response(row) for row in rows]
    finally:
        session.close()


def get_prediction(prediction_id: int) -> PredictionOut | None:
    """Get a specific prediction by ID."""
    session = SessionLocal()
    try:
        row = session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if row is None:
            return None
        return _prediction_to_response(row)
    finally:
        session.close()


def _prediction_to_response(pred: Prediction) -> PredictionOut:
    """Convert a Prediction object to a PredictionOut response."""
    ground_truth = None
    if pred.ground_truth_json:
        try:
            ground_truth = json.loads(pred.ground_truth_json)
        except json.JSONDecodeError:
            ground_truth = {"raw": pred.ground_truth_json}

    return PredictionOut(
        id=pred.id,
        created_at=pred.created_at.isoformat(),
        model_name=pred.model_name,
        inference_ms=pred.inference_ms,
        image_url=f"/uploads/{Path(pred.image_path).name}",
        fake_probability=pred.fake_probability,
        ground_truth=ground_truth,
    )
