# FastAPI App Structure Refactoring

## Overview
The FastAPI application has been refactored from a monolithic `main.py` into a well-organized, modular structure following best practices for maintainability and scalability.

## Previous Structure
- Single `main.py` file containing:
  - Database models
  - Pydantic schemas
  - All business logic
  - All routes
  - Configuration
  - Prometheus metrics
  - Image processing utilities

## New Structure

```
app/
├── __init__.py              # Package initialization
├── main.py                  # FastAPI app and route definitions
├── config.py                # Configuration and environment variables
├── database.py              # SQLAlchemy models and database setup
├── schemas.py               # Pydantic schemas for API responses
├── detection.py             # ONNX model inference service
├── db_service.py            # Database operations (CRUD)
├── metrics.py               # Prometheus metrics definitions
├── utils.py                 # Image processing utilities (preprocessing, postprocessing, NMS)
├── Dockerfile               # Docker configuration (unchanged)
├── requirements.txt         # Dependencies (unchanged)
└── uploads/                 # Image storage directory
```

## Module Breakdown

### `config.py`
**Purpose:** Centralized configuration management
**Contains:**
- Database URL and connection settings
- Model path and inference thresholds
- Upload directory configuration
- Input dimensions for the ONNX model

### `database.py`
**Purpose:** Database layer with SQLAlchemy models
**Contains:**
- `Base` declarative base class
- `Prediction` model (ML predictions)
- `Detection` model (detected objects)
- Database engine and session factory
- `init_db()` function to initialize database on startup

### `schemas.py`
**Purpose:** API request/response models using Pydantic
**Contains:**
- `DetectionOut` - Detection response schema
- `PredictionOut` - Full prediction response schema

### `metrics.py`
**Purpose:** Prometheus metrics for ML observability
**Contains:**
- `prediction_counter` - Total predictions made
- `detection_counter` - Objects detected
- `confidence_histogram` - Detection confidence distribution
- `inference_time_histogram` - Model inference duration
- `detections_per_image` - Detections per image histogram
- `avg_confidence_gauge` - Average confidence gauge
- `low_confidence_counter` - Low confidence detections

### `detection.py`
**Purpose:** ONNX model inference service
**Contains:**
- `DetectionService` class
  - `initialize()` - Load ONNX model
  - `is_ready()` - Check model status
  - `infer()` - Run inference on image

### `utils.py`
**Purpose:** Image processing utilities
**Contains:**
- `DetectionResult` class - Detection data structure
- `preprocess()` - Image preprocessing for ONNX
- `postprocess()` - Parse ONNX outputs
- `xywh_to_xyxy()` - Coordinate format conversion
- `compute_iou()` - Intersection over Union calculation
- `nms()` - Non-Maximum Suppression
- `draw_boxes()` - Visualize detections
- `get_class_name()` - Class ID to name mapping

### `db_service.py`
**Purpose:** Database operations layer
**Contains:**
- `save_prediction()` - Save prediction and detections
- `get_predictions()` - Retrieve recent predictions
- `get_prediction()` - Get specific prediction by ID
- `_prediction_to_response()` - Convert ORM objects to API responses

### `main.py`
**Purpose:** FastAPI application setup and route definitions
**Contains:**
- FastAPI app initialization
- Global `detection_service` instance
- Startup event handler
- API endpoints:
  - `GET /` - Health check
  - `GET /health` - Detailed status
  - `POST /predict` - Run inference
  - `GET /predictions` - List recent predictions
  - `GET /predictions/{id}` - Get specific prediction
- Middleware configuration (CORS)
- Static file mounting for uploads
- Prometheus instrumentation

## Benefits of the New Structure

1. **Separation of Concerns**: Each module has a single responsibility
2. **Reusability**: Components like `DetectionService` and utilities can be reused
3. **Testability**: Smaller, focused modules are easier to unit test
4. **Maintainability**: Bug fixes and features are isolated to relevant modules
5. **Scalability**: Easy to add new routes, services, or utilities
6. **Configuration Management**: Centralized configuration in one place
7. **Code Clarity**: Clearer imports and dependencies between modules

## Breaking Changes
- All functionality remains the same
- API endpoints and parameters are unchanged
- Database schema is unchanged
- Requires updating Dockerfile if it imports from main.py specifically

## Migration Notes
- The old `main.py` has been backed up to `main_old.py`
- All imports in the refactored code use relative imports (e.g., `from .config import ...`)
- The FastAPI app is exported in `__init__.py` for easy import: `from app import app`
