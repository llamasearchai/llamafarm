"""
Integration tests for the LlamaFarmAI API.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from fastapi.testclient import TestClient

from api.app import app


class TestLlamaFarmAPI(unittest.TestCase):
    """Integration tests for the LlamaFarmAI API."""

    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

        # Create a test image
        self.test_image = self._create_test_image()

    def _create_test_image(self):
        """Create a test image file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # Create a simple test image (green 100x100)
            img = Image.new("RGB", (100, 100), color=(0, 255, 0))
            img.save(f.name)
            return f.name

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, "test_image") and os.path.exists(self.test_image):
            os.unlink(self.test_image)

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "0.1.0")
        self.assertTrue("timestamp" in data)

    @patch("api.app.get_crop_classifier")
    def test_classify_crop(self, mock_classifier):
        """Test the crop classification endpoint."""
        # Mock the classifier
        mock_classifier_instance = MagicMock()
        mock_classifier_instance.predict.return_value = {
            "class_id": 0,
            "class_name": "corn",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat(),
        }
        mock_classifier.return_value = mock_classifier_instance

        # Make request
        with open(self.test_image, "rb") as f:
            response = self.client.post(
                "/api/v1/crop/classify",
                files={"image": ("test.jpg", f, "image/jpeg")},
                data={"backend": "mlx", "device": "auto"},
            )

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["class_name"], "corn")
        self.assertEqual(data["class_id"], 0)
        self.assertAlmostEqual(data["confidence"], 0.95, places=2)
        self.assertTrue("processing_time" in data)

        # Verify mocks
        mock_classifier.assert_called_once_with(backend="mlx", device="auto")
        mock_classifier_instance.predict.assert_called_once()

    @patch("api.app.get_disease_detector")
    def test_detect_disease(self, mock_detector):
        """Test the disease detection endpoint."""
        # Mock the detector
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect.return_value = {
            "timestamp": datetime.now().isoformat(),
            "detections": [
                {
                    "id": 0,
                    "class_id": 1,
                    "class_name": "bacterial_blight",
                    "confidence": 0.87,
                    "bbox": [10, 20, 50, 60],
                }
            ],
            "count": 1,
        }
        mock_detector.return_value = mock_detector_instance

        # Make request
        with open(self.test_image, "rb") as f:
            response = self.client.post(
                "/api/v1/disease/detect",
                files={"image": ("test.jpg", f, "image/jpeg")},
                data={"backend": "yolo", "device": "auto"},
            )

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["detections"]), 1)
        self.assertEqual(data["detections"][0]["class_name"], "bacterial_blight")
        self.assertEqual(data["count"], 1)
        self.assertTrue("processing_time" in data)

        # Verify mocks
        mock_detector.assert_called_once_with(backend="yolo", device="auto")
        mock_detector_instance.detect.assert_called_once()

    @patch("api.app.get_weed_detector")
    def test_detect_weeds(self, mock_detector):
        """Test the weed detection endpoint."""
        # Mock the detector
        mock_detector_instance = MagicMock()
        mock_detector_instance.detect.return_value = {
            "timestamp": datetime.now().isoformat(),
            "detections": [
                {
                    "id": 0,
                    "class_name": "weed",
                    "confidence": 0.92,
                    "bbox": [30, 40, 70, 80],
                }
            ],
            "count": 1,
            "weed_coverage_percent": 15.5,
        }
        mock_detector.return_value = mock_detector_instance

        # Make request
        with open(self.test_image, "rb") as f:
            response = self.client.post(
                "/api/v1/weed/detect",
                files={"image": ("test.jpg", f, "image/jpeg")},
                data={"backend": "yolo", "device": "auto"},
            )

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["detections"]), 1)
        self.assertEqual(data["detections"][0]["class_name"], "weed")
        self.assertEqual(data["count"], 1)
        self.assertAlmostEqual(data["weed_coverage_percent"], 15.5, places=1)
        self.assertTrue("processing_time" in data)

        # Verify mocks
        mock_detector.assert_called_once_with(backend="yolo", device="auto")
        mock_detector_instance.detect.assert_called_once()

    @patch("api.app.get_llm")
    def test_query_llm(self, mock_llm):
        """Test the LLM query endpoint."""
        # Mock the LLM
        mock_llm_instance = MagicMock()
        mock_llm_instance.ask.return_value = {
            "text": "This is a response about agriculture.",
            "model": "gpt-4",
            "type": "openai",
        }
        mock_llm.return_value = mock_llm_instance

        # Request data
        request_data = {
            "query": "What crops grow well in sandy soil?",
            "model_type": "openai",
            "model_name": "gpt-4",
        }

        # Make request
        response = self.client.post("/api/v1/llm/query", json=request_data)

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["text"], "This is a response about agriculture.")
        self.assertEqual(data["model"], "gpt-4")
        self.assertEqual(data["type"], "openai")
        self.assertTrue("processing_time" in data)

        # Verify mocks
        mock_llm.assert_called_once_with(model_type="openai", model_name="gpt-4")
        mock_llm_instance.ask.assert_called_once_with(
            query="What crops grow well in sandy soil?",
            prompt_template=None,
            context=None,
        )

    @patch("api.app.get_irrigation_optimizer")
    def test_optimize_irrigation(self, mock_optimizer):
        """Test the irrigation optimization endpoint."""
        # Mock the optimizer
        mock_optimizer_instance = MagicMock()
        mock_optimizer_instance.optimize_schedule.return_value = {
            "daily_schedule": [
                {
                    "day": 1,
                    "date": "2023-06-01",
                    "water_volume_m3": 50.5,
                    "duration_minutes": 45,
                    "start_time": "06:00",
                },
                {
                    "day": 3,
                    "date": "2023-06-03",
                    "water_volume_m3": 60.0,
                    "duration_minutes": 50,
                    "start_time": "05:30",
                },
            ]
        }
        mock_optimizer.return_value = mock_optimizer_instance

        # Request data
        request_data = {
            "field_data": {
                "crop_type": "corn",
                "growth_stage": "mid-season",
                "area_hectares": 5.0,
                "soil_type": "loam",
                "current_moisture": 45.0,
            },
            "weather_forecast": [
                {
                    "date": "2023-06-01",
                    "temp_max_c": 28.0,
                    "temp_min_c": 18.0,
                    "precipitation_mm": 0.0,
                    "solar_radiation_mj_m2": 25.0,
                }
            ],
            "constraints": {
                "max_daily_water_m3": 100.0,
                "total_available_water_m3": 500.0,
                "min_moisture_pct": 40.0,
                "max_moisture_pct": 80.0,
            },
            "planning_horizon": 7,
        }

        # Make request
        response = self.client.post("/api/v1/irrigation/optimize", json=request_data)

        # Check response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["schedule"]), 2)
        self.assertEqual(data["schedule"][0]["water_volume_m3"], 50.5)
        self.assertEqual(data["schedule"][1]["water_volume_m3"], 60.0)
        self.assertEqual(data["water_usage"], 110.5)
        self.assertTrue("processing_time" in data)

        # Verify mocks
        mock_optimizer.assert_called_once()
        mock_optimizer_instance.optimize_schedule.assert_called_once_with(
            field_data=request_data["field_data"],
            weather_forecast=request_data["weather_forecast"],
            constraints=request_data["constraints"],
            planning_horizon=7,
        )


if __name__ == "__main__":
    unittest.main()
