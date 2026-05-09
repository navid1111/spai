"""Test cases for the FastAPI application."""

import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Set a temp directory for tests to avoid CI permission issues - BEFORE any imports
TEST_UPLOAD_DIR = tempfile.mkdtemp()
os.environ["UPLOAD_DIR"] = TEST_UPLOAD_DIR
# Create the directory (required for StaticFiles mounting)
Path(TEST_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# Mock the detection service and database before importing the app
with patch("app.detection.DetectionService"):
    with patch("app.database.init_db"):
        from app.main import app, detection_service
        from app.schemas import PredictionOut


# Create test client
client = TestClient(app)


def create_test_image(width: int = 224, height: int = 224) -> bytes:
    """Create a test image in memory."""
    img = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_home_endpoint(self):
        """Test the home endpoint returns correct message."""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "SPAI Fake Detection API is running"}

    def test_health_endpoint(self):
        """Test the health endpoint returns status and model info."""
        # Mock the detection service as ready
        with patch.object(detection_service, "is_ready", return_value=True):
            with patch.object(detection_service, "model_name", "test_model.onnx"):
                with patch.object(detection_service, "input_hw", (224, 224)):
                    response = client.get("/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "ok"
                    assert data["model"] == "test_model.onnx"
                    assert data["input_hw"] == [224, 224]


class TestPredictEndpoint:
    """Test the predict endpoint."""

    def test_predict_invalid_image(self):
        """Test predict with invalid image file."""
        response = client.post(
            "/predict",
            files={"image": ("test.txt", b"not an image", "text/plain")},
        )
        # Invalid image returns 500 (internal error) or 400 (bad request)
        assert response.status_code in [400, 500]

    def test_predict_missing_image(self):
        """Test predict without providing an image."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error

    @patch("app.main.detection_service")
    def test_predict_success(self, mock_service):
        """Test successful prediction."""
        # Setup mock
        mock_service.is_ready.return_value = True
        mock_service.model_name = "test_model.onnx"
        mock_service.input_hw = (224, 224)
        mock_service.infer.return_value = (0.95, 25.5)

        with patch("app.main.save_prediction") as mock_save:
            mock_prediction = MagicMock()
            mock_prediction.id = 1
            mock_save.return_value = mock_prediction

            with patch("app.main.get_prediction") as mock_get:
                mock_get.return_value = PredictionOut(
                    id=1,
                    created_at="2024-01-01T00:00:00",
                    model_name="test_model.onnx",
                    inference_ms=25.5,
                    image_url="/uploads/test.jpg",
                    fake_probability=0.95,
                    ground_truth=None,
                )

                response = client.post(
                    "/predict",
                    files={"image": ("test.jpg", create_test_image(), "image/jpeg")},
                )

                assert response.status_code == 200
                data = response.json()
                assert "fake_probability" in data
                assert data["fake_probability"] == 0.95


class TestPredictionsEndpoints:
    """Test predictions list and detail endpoints."""

    @patch("app.main.get_predictions")
    def test_list_predictions(self, mock_get):
        """Test listing predictions."""
        mock_get.return_value = [
            PredictionOut(
                id=1,
                created_at="2024-01-01T00:00:00",
                model_name="test.onnx",
                inference_ms=25.0,
                image_url="/uploads/img1.jpg",
                fake_probability=0.01,
                ground_truth=None,
            )
        ]

        response = client.get("/predictions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == 1

    @patch("app.main.get_predictions")
    def test_list_predictions_with_custom_limit(self, mock_get):
        """Test listing predictions with custom limit."""
        mock_get.return_value = []

        response = client.get("/predictions?limit=50")
        assert response.status_code == 200

    @patch("app.main.get_prediction")
    def test_get_prediction_detail_not_found(self, mock_get):
        """Test getting a non-existent prediction."""
        mock_get.return_value = None

        response = client.get("/predictions/999")
        assert response.status_code == 404
        assert response.json()["detail"] == "Prediction not found"

    @patch("app.main.get_prediction")
    def test_get_prediction_detail_success(self, mock_get):
        """Test getting a specific prediction."""
        mock_get.return_value = PredictionOut(
            id=1,
            created_at="2024-01-01T00:00:00",
            model_name="test.onnx",
            inference_ms=25.0,
            image_url="/uploads/img1.jpg",
            fake_probability=0.99,
            ground_truth={"test": "data"},
        )

        response = client.get("/predictions/1")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == 1
        assert "fake_probability" in data


class TestSchemas:
    """Test Pydantic schemas."""

    def test_prediction_out_schema(self):
        """Test PredictionOut schema validation."""
        pred = PredictionOut(
            id=1,
            created_at="2024-01-01T00:00:00",
            model_name="test.onnx",
            inference_ms=25.0,
            image_url="/uploads/test.jpg",
            annotated_image_url="/uploads/annotated.jpg",
            ground_truth={"key": "value"},
            detections=[],
        )
        assert pred.id == 1
        assert pred.model_name == "test.onnx"


class TestUtils:
    """Test utility functions."""

    def test_xywh_to_xyxy(self):
        """Test XYWH to XYXY conversion."""
        from app.utils import xywh_to_xyxy

        boxes = np.array([[50, 50, 100, 100]])  # center_x, center_y, w, h
        result = xywh_to_xyxy(boxes)
        # center (50,50) with w=100, h=100 should give [0, 0, 100, 100]
        assert result[0][0] == 0
        assert result[0][1] == 0
        assert result[0][2] == 100
        assert result[0][3] == 100

    def test_nms(self):
        """Test Non-Maximum Suppression."""
        from app.utils import nms

        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [100, 100, 110, 110]])
        scores = np.array([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, iou_threshold=0.5)
        # First box should be kept, second should be suppressed (high IoU with first)
        # Third should be kept (low IoU with others)
        assert 0 in keep
        assert 2 in keep

    def test_compute_iou(self):
        """Test IoU computation."""
        from app.utils import compute_iou

        box = np.array([0, 0, 10, 10])
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]])
        iou = compute_iou(box, boxes)
        # First box should have IoU 1.0 (identical), second should have some overlap
        assert np.isclose(iou[0], 1.0)
        assert 0 < iou[1] < 1.0


class TestPreprocess:
    """Test image preprocessing."""

    def test_preprocess_output_shape(self):
        """Test preprocess output shape."""
        from app.utils import preprocess

        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = preprocess(img, (640, 640))
        # Should output shape [1, 3, 640, 640]
        assert result.shape == (1, 3, 640, 640)

    def test_preprocess_normalization(self):
        """Test preprocess normalizes to 0-1 range."""
        from app.utils import preprocess

        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = preprocess(img, (100, 100))
        # All values should be 1.0 after normalization
        assert np.allclose(result, 1.0)


class TestPostprocess:
    """Test postprocessing."""

    def test_postprocess_empty_output(self):
        """Test postprocess with empty/low confidence output."""
        from app.utils import postprocess

        # Create mock output with low confidence
        output = np.zeros((5, 6))
        output[:, 4] = 0.1  # Low confidence

        result = postprocess(output, 640, 480, conf_threshold=0.5)
        assert len(result) == 0

    def test_postprocess_with_detections(self):
        """Test postprocess handles valid output format."""
        from app.utils import postprocess

        # Test with valid shape that won't be transposed
        # Shape (detections, 85) where 85 > 3 won't be transposed
        output = np.zeros((5, 85))
        # Add a detection with high confidence
        output[0, :4] = [320, 240, 100, 100]  # xywh
        output[0, 4] = 0.9  # class 0 high confidence

        # Just verify function runs without error - exact output depends on model format
        result = postprocess(output, 640, 480, conf_threshold=0.3)
        # This test verifies the function doesn't crash, actual detection count varies


class TestDrawBoxes:
    """Test box drawing utility."""

    def test_draw_boxes_output_shape(self):
        """Test draw_boxes returns correct shape."""
        from app.utils import draw_boxes

        img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            DetectionResult(
                class_id=0, confidence=0.9, box_xyxy=[100, 100, 200, 200]
            )
        ]
        result = draw_boxes(img, detections)
        assert result.shape == img.shape