"""
Unit tests for MLX integration module.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Import with conditional handling to handle the case where MLX isn't installed
try:
    import mlx.core as mx
    import mlx.nn as nn

    from core.ai.mlx_integration import (
        MLXAgricultureModelFactory,
        MLXModelWrapper,
        optimize_for_mlx,
    )

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False


# Create a simple MLX model for testing
@unittest.skipIf(not MLX_AVAILABLE, "MLX not available for testing")
class SimpleMLXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm(16)
        self.fc = nn.Linear(16, 10)

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.relu(x)
        x = mx.mean(x, axis=(2, 3))
        x = self.fc(x)
        return x


@unittest.skipIf(not MLX_AVAILABLE, "MLX not available for testing")
class TestMLXModelWrapper(unittest.TestCase):
    """Test the MLXModelWrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        if MLX_AVAILABLE:
            self.model = SimpleMLXModel()
            self.wrapper = MLXModelWrapper(self.model, model_name="test_model")

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.wrapper.model_name, "test_model")
        self.assertEqual(self.wrapper.device, "gpu")
        self.assertTrue(self.wrapper.is_initialized)

    def test_save_load_weights(self):
        """Test saving and loading weights."""
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            # Save weights
            save_path = tmp.name
            result = self.wrapper.save_weights(save_path)
            self.assertTrue(result)
            self.assertTrue(os.path.exists(save_path))

            # Check that config is also saved
            config_path = os.path.splitext(save_path)[0] + ".json"
            self.assertTrue(os.path.exists(config_path))

            # Load weights
            new_wrapper = MLXModelWrapper(SimpleMLXModel(), model_name="test_model_2")
            result = new_wrapper.load_weights(save_path)
            self.assertTrue(result)

            # Test with invalid path
            result = new_wrapper.load_weights("/invalid/path.npz")
            self.assertFalse(result)

    def test_predict(self):
        """Test prediction."""
        # Create dummy input
        input_data = mx.random.normal((1, 3, 32, 32))

        # Run prediction
        output = self.wrapper.predict(input_data)

        # Check output shape
        self.assertEqual(output.shape, (1, 10))

        # Test with batch size larger than input
        output = self.wrapper.predict(input_data, batch_size=10)
        self.assertEqual(output.shape, (1, 10))

        # Test with multiple batches
        input_data = mx.random.normal((5, 3, 32, 32))
        output = self.wrapper.predict(input_data, batch_size=2)
        self.assertEqual(output.shape, (5, 10))


@unittest.skipIf(not MLX_AVAILABLE, "MLX not available for testing")
class TestOptimizeForMLX(unittest.TestCase):
    """Test the optimize_for_mlx function."""

    def setUp(self):
        """Set up test fixtures."""
        if MLX_AVAILABLE:
            self.model = SimpleMLXModel()

    def test_optimize_vision_model(self):
        """Test optimizing a vision model."""
        optimized_model = optimize_for_mlx(self.model, model_type="vision")
        self.assertIsNotNone(optimized_model)

        # Test with quantization
        optimized_model = optimize_for_mlx(
            self.model, model_type="vision", quantize=True
        )
        self.assertIsNotNone(optimized_model)

    def test_optimize_nlp_model(self):
        """Test optimizing an NLP model."""
        optimized_model = optimize_for_mlx(self.model, model_type="nlp")
        self.assertIsNotNone(optimized_model)

    def test_optimize_timeseries_model(self):
        """Test optimizing a timeseries model."""
        optimized_model = optimize_for_mlx(self.model, model_type="timeseries")
        self.assertIsNotNone(optimized_model)


@unittest.skipIf(not MLX_AVAILABLE, "MLX not available for testing")
class TestMLXAgricultureModelFactory(unittest.TestCase):
    """Test the MLXAgricultureModelFactory class."""

    def test_create_crop_classifier(self):
        """Test creating a crop classifier."""
        model_wrapper = MLXAgricultureModelFactory.create_crop_classifier(num_classes=5)
        self.assertIsInstance(model_wrapper, MLXModelWrapper)
        self.assertEqual(model_wrapper.model_name, "crop_classifier")

        # Check model architecture
        model = model_wrapper.model
        self.assertEqual(model.fc2.weight.shape[1], 5)  # Output classes

        # Test with pretrained=False
        model_wrapper = MLXAgricultureModelFactory.create_crop_classifier(
            num_classes=10, pretrained=False
        )
        self.assertIsInstance(model_wrapper, MLXModelWrapper)

    def test_create_weed_detector(self):
        """Test creating a weed detector."""
        model_wrapper = MLXAgricultureModelFactory.create_weed_detector()
        self.assertIsInstance(model_wrapper, MLXModelWrapper)
        self.assertEqual(model_wrapper.model_name, "weed_detector")

    def test_create_yield_predictor(self):
        """Test creating a yield predictor."""
        model_wrapper = MLXAgricultureModelFactory.create_yield_predictor()
        self.assertIsInstance(model_wrapper, MLXModelWrapper)
        self.assertEqual(model_wrapper.model_name, "yield_predictor")


# Mock classes for testing without MLX
class TestWithoutMLX(unittest.TestCase):
    """Test behavior when MLX is not available."""

    @patch("core.ai.mlx_integration.MLX_AVAILABLE", False)
    @patch("core.ai.mlx_integration.logging")
    def test_optimize_without_mlx(self, mock_logging):
        """Test optimize_for_mlx when MLX is not available."""
        # Create a mock model
        model = MagicMock()

        # Import the function again with the patch
        from core.ai.mlx_integration import optimize_for_mlx

        # Call the function
        result = optimize_for_mlx(model)

        # Check that it returns the original model
        self.assertEqual(result, model)

        # Check that it logs a warning
        mock_logging.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
