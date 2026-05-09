import json
import os
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from time import sleep

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://app_user:app_password@mysql:3306/predictions_db",
)
MODEL_PATH = os.getenv("MODEL_PATH", "/models/best.onnx")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/app/uploads"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
DB_WAIT_SECONDS = int(os.getenv("DB_WAIT_SECONDS", "240"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    image_path: Mapped[str] = mapped_column(String(512))
    annotated_image_path: Mapped[str] = mapped_column(String(512))
    model_name: Mapped[str] = mapped_column(String(255))
    inference_ms: Mapped[float] = mapped_column(Float)
    ground_truth_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    class_id: Mapped[int] = mapped_column(Integer)
    class_name: Mapped[str] = mapped_column(String(255))
    confidence: Mapped[float] = mapped_column(Float)
    x1: Mapped[float] = mapped_column(Float)
    y1: Mapped[float] = mapped_column(Float)
    x2: Mapped[float] = mapped_column(Float)
    y2: Mapped[float] = mapped_column(Float)

    prediction: Mapped[Prediction] = relationship(back_populates="detections")


class DetectionOut(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]


class PredictionOut(BaseModel):
    id: int
    created_at: str
    model_name: str
    inference_ms: float
    image_url: str
    annotated_image_url: str
    ground_truth: dict | None
    detections: list[DetectionOut]


app = FastAPI(title="Retail Detection API")
onnx_session: ort.InferenceSession | None = None
input_name: str | None = None
input_hw: tuple[int, int] = (640, 640)

# Custom Prometheus metrics for ML observability
prediction_counter = Counter(
    "ml_predictions_total",
    "Total number of predictions made",
    ["model_name"]
)
detection_counter = Counter(
    "ml_detections_total", 
    "Total number of objects detected",
    ["class_name", "model_name"]
)
confidence_histogram = Histogram(
    "ml_detection_confidence",
    "Distribution of detection confidence scores",
    ["class_name"],
    buckets=(0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0)
)
inference_time_histogram = Histogram(
    "ml_inference_duration_ms",
    "Model inference duration in milliseconds",
    ["model_name"],
    buckets=(10, 25, 50, 100, 200, 500, 1000, 2000, 5000)
)
detections_per_image = Histogram(
    "ml_detections_per_image",
    "Number of detections per image",
    ["model_name"],
    buckets=(0, 1, 2, 5, 10, 20, 50, 100)
)
avg_confidence_gauge = Gauge(
    "ml_avg_confidence",
    "Average confidence score of recent predictions",
    ["model_name"]
)
low_confidence_counter = Counter(
    "ml_low_confidence_detections_total",
    "Count of low confidence detections (< 0.5)",
    ["class_name", "model_name"]
)


class DetectionResult:
    def __init__(self, class_id: int, confidence: float, box_xyxy: list[float]):
        self.class_id = class_id
        self.confidence = confidence
        self.box_xyxy = box_xyxy


def _ensure_database_ready() -> None:
    attempts = max(1, DB_WAIT_SECONDS // 2)
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            Base.metadata.create_all(bind=engine)
            return
        except Exception as exc:
            last_error = exc
            sleep(2)

    raise RuntimeError(f"MySQL is not ready after waiting {DB_WAIT_SECONDS}s: {last_error}")


def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
    out = np.zeros_like(boxes_xywh)
    out[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    out[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    out[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    out[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    return out


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    intersection = inter_w * inter_h

    area_box = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - intersection + 1e-9
    return intersection / union


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = np.argsort(scores)[::-1]
    keep: list[int] = []

    while order.size > 0:
        idx = int(order[0])
        keep.append(idx)
        if order.size == 1:
            break

        rest = order[1:]
        ious = _compute_iou(boxes[idx], boxes[rest])
        order = rest[ious <= iou_threshold]

    return keep


def _preprocess(img_rgb: np.ndarray) -> np.ndarray:
    h, w = input_hw
    pil = Image.fromarray(img_rgb)
    resized = pil.resize((w, h))
    arr = np.array(resized).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def _postprocess(output: np.ndarray, orig_w: int, orig_h: int) -> list[DetectionResult]:
    preds = np.squeeze(output)

    if preds.ndim == 2 and preds.shape[0] < preds.shape[1]:
        preds = preds.T

    if preds.ndim != 2 or preds.shape[1] < 6:
        return []

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    mask = scores >= CONF_THRESHOLD
    if not np.any(mask):
        return []

    boxes_xywh = boxes_xywh[mask]
    class_ids = class_ids[mask]
    scores = scores[mask]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    scale_x = orig_w / float(input_hw[1])
    scale_y = orig_h / float(input_hw[0])
    boxes_xyxy[:, [0, 2]] *= scale_x
    boxes_xyxy[:, [1, 3]] *= scale_y

    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    keep = _nms(boxes_xyxy, scores, IOU_THRESHOLD)

    results: list[DetectionResult] = []
    for idx in keep:
        box = boxes_xyxy[idx].tolist()
        results.append(
            DetectionResult(
                class_id=int(class_ids[idx]),
                confidence=float(scores[idx]),
                box_xyxy=[float(v) for v in box],
            )
        )
    return results


def _draw_boxes(img_rgb: np.ndarray, detections: list[DetectionResult]) -> np.ndarray:
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    for det in detections:
        x1, y1, x2, y2 = det.box_xyxy
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
        draw.text((x1 + 2, max(0, y1 - 12)), f"id:{det.class_id} {det.confidence:.2f}", fill=(255, 0, 0))
    return np.array(img)


def _class_name(class_id: int) -> str:
    return f"class_{class_id}"


def _prediction_to_response(pred: Prediction) -> PredictionOut:
    ground_truth = None
    if pred.ground_truth_json:
        try:
            ground_truth = json.loads(pred.ground_truth_json)
        except json.JSONDecodeError:
            ground_truth = {"raw": pred.ground_truth_json}

    detections = [
        DetectionOut(
            class_id=d.class_id,
            class_name=d.class_name,
            confidence=d.confidence,
            bbox_xyxy=[d.x1, d.y1, d.x2, d.y2],
        )
        for d in pred.detections
    ]

    return PredictionOut(
        id=pred.id,
        created_at=pred.created_at.isoformat(),
        model_name=pred.model_name,
        inference_ms=pred.inference_ms,
        image_url=f"/uploads/{Path(pred.image_path).name}",
        annotated_image_url=f"/uploads/{Path(pred.annotated_image_path).name}",
        ground_truth=ground_truth,
        detections=detections,
    )


@app.on_event("startup")
def startup_event() -> None:
    global onnx_session, input_name, input_hw

    _ensure_database_ready()

    if not Path(MODEL_PATH).exists():
        raise RuntimeError(f"ONNX model not found at: {MODEL_PATH}")

    onnx_session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_info = onnx_session.get_inputs()[0]
    input_name = input_info.name

    shape = input_info.shape
    if isinstance(shape, list) and len(shape) >= 4:
        h = shape[2] if isinstance(shape[2], int) else 640
        w = shape[3] if isinstance(shape[3], int) else 640
        input_hw = (int(h), int(w))


@app.get("/")
def home() -> dict:
    return {"message": "Retail Detection API is running"}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH, "input_hw": input_hw}


@app.post("/predict", response_model=PredictionOut)
async def predict(
    image: UploadFile = File(...),
    ground_truth: str | None = Form(None),
) -> PredictionOut:
    if onnx_session is None or input_name is None:
        raise HTTPException(status_code=500, detail="Model is not initialized")

    try:
        image_bytes = await image.read()
        pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    image_np = np.array(pil_img)
    orig_h, orig_w = image_np.shape[:2]

    original_name = f"{uuid.uuid4().hex}_{image.filename or 'image.jpg'}"
    original_path = UPLOAD_DIR / original_name
    pil_img.save(original_path)

    model_input = _preprocess(image_np)

    start = time.perf_counter()
    outputs = onnx_session.run(None, {input_name: model_input})
    inference_ms = (time.perf_counter() - start) * 1000.0

    detections = _postprocess(outputs[0], orig_w=orig_w, orig_h=orig_h)
    annotated = _draw_boxes(image_np, detections)

    annotated_name = f"annotated_{original_name}"
    annotated_path = UPLOAD_DIR / annotated_name
    Image.fromarray(annotated).save(annotated_path)

    # Record Prometheus metrics
    model_name = Path(MODEL_PATH).name
    prediction_counter.labels(model_name=model_name).inc()
    inference_time_histogram.labels(model_name=model_name).observe(inference_ms)
    detections_per_image.labels(model_name=model_name).observe(len(detections))
    
    # Calculate and record confidence metrics
    if detections:
        confidences = [det.confidence for det in detections]
        avg_conf = sum(confidences) / len(confidences)
        avg_confidence_gauge.labels(model_name=model_name).set(avg_conf)
        
        for det in detections:
            class_name = _class_name(det.class_id)
            detection_counter.labels(class_name=class_name, model_name=model_name).inc()
            confidence_histogram.labels(class_name=class_name).observe(det.confidence)
            
            if det.confidence < 0.5:
                low_confidence_counter.labels(class_name=class_name, model_name=model_name).inc()

    session = SessionLocal()
    try:
        prediction = Prediction(
            image_path=str(original_path),
            annotated_image_path=str(annotated_path),
            model_name=Path(MODEL_PATH).name,
            inference_ms=inference_ms,
            ground_truth_json=ground_truth,
        )
        session.add(prediction)
        session.flush()

        for det in detections:
            x1, y1, x2, y2 = det.box_xyxy
            session.add(
                Detection(
                    prediction_id=prediction.id,
                    class_id=det.class_id,
                    class_name=_class_name(det.class_id),
                    confidence=det.confidence,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        session.commit()
        session.refresh(prediction)
        _ = prediction.detections
        return _prediction_to_response(prediction)
    finally:
        session.close()


@app.get("/predictions", response_model=list[PredictionOut])
def list_predictions(limit: int = 20) -> list[PredictionOut]:
    session = SessionLocal()
    try:
        rows = (
            session.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
            .all()
        )
        for row in rows:
            _ = row.detections
        return [_prediction_to_response(row) for row in rows]
    finally:
        session.close()


@app.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction(prediction_id: int) -> PredictionOut:
    session = SessionLocal()
    try:
        row = session.query(Prediction).filter(Prediction.id == prediction_id).first()
        if row is None:
            raise HTTPException(status_code=404, detail="Prediction not found")
        _ = row.detections
        return _prediction_to_response(row)
    finally:
        session.close()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
Instrumentator().instrument(app).expose(app)
