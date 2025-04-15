"""
Unit tests for the MLX integration module.

These tests verify the functionality of the MLX integration module,
including model wrapping, optimization, and conversion utilities.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Skip tests if MLX is not available
try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Import the module to test
from llamafarms.core.ai.mlx_integration import (
    MLXAgricultureModelFactory,
    MLXModelWrapper,
    optimize_for_mlx,
)


@unittest.skipIf(not MLX_AVAILABLE, "MLX not available")
class TestMLXModelWrapper(unittest.TestCase):
    """Test the MLXModelWrapper class."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a simple MLX model for testing
        class SimpleModel(nn.Module):
            """Simple MLX model for testing."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm(16)
                self.fc = nn.Linear(16, 10)

            def __call__(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = mx.mean(x, axis=(2, 3))  # Global average pooling
                return self.fc(x)

        self.model = SimpleModel()
        self.wrapper = MLXModelWrapper(
            model=self.model,
            model_name="test_model",
            device="gpu",
            config={"test_param": True},
        )

    def test_initialization(self):
        """Test initialization of the wrapper."""
        self.assertEqual(self.wrapper.model_name, "test_model")
        self.assertEqual(self.wrapper.device, "gpu")
        self.assertEqual(self.wrapper.config, {"test_param": True})
        self.assertTrue(self.wrapper.is_initialized)

    @patch("mlx.save")
    def test_save_weights(self, mock_save):
        """Test saving model weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.mlx")
            result = self.wrapper.save_weights(save_path)

            # Check that mx.save was called
            mock_save.assert_called_once()

            # Check that config was saved
            config_path = os.path.join(tmpdir, "model.json")
            self.assertTrue(os.path.exists(config_path))

            # Check return value
            self.assertTrue(result)

    @patch("mlx.load")
    def test_load_weights(self, mock_load):
        """Test loading model weights."""
        # Mock the load function to return a dict
        mock_load.return_value = {"conv.weight": mx.zeros((16, 3, 3, 3))}

        result = self.wrapper.load_weights("dummy_path.mlx")

        # Check that mx.load was called
        mock_load.assert_called_once_with("dummy_path.mlx")

        # Check return value
        self.assertTrue(result)

    def test_predict(self):
        """Test model prediction."""
        # Create a dummy input
        input_data = mx.random.normal((1, 224, 224, 3))

        # Run prediction
        output = self.wrapper.predict(input_data)

        # Check output shape
        self.assertEqual(output.shape, (1, 10))


@unittest.skipIf(not MLX_AVAILABLE, "MLX not available")
class TestOptimizeForMLX(unittest.TestCase):
    """Test the optimize_for_mlx function."""

    def test_optimize_vision_model(self):
        """Test optimizing a vision model."""

        # Create a dummy model
        class DummyModel:
            def __init__(self):
                self.optimized = False

        model = DummyModel()

        # Mock the optimization process
        with patch("llamafarms.core.ai.mlx_integration.logger") as mock_logger:
            result = optimize_for_mlx(model, model_type="vision", quantize=True)

            # Check that logging happened
            mock_logger.info.assert_any_call("Optimizing vision model for MLX")
            mock_logger.info.assert_any_call("Applying quantization")
            mock_logger.info.assert_any_call("Model optimization complete")

            # Check that the original model was returned
            self.assertEqual(result, model)


@unittest.skipIf(not MLX_AVAILABLE, "MLX not available")
class TestMLXAgricultureModelFactory(unittest.TestCase):
    """Test the MLXAgricultureModelFactory class."""

    def test_create_crop_classifier(self):
        """Test creating a crop classifier."""
        # Create a crop classifier
        model_wrapper = MLXAgricultureModelFactory.create_crop_classifier(
            num_classes=5, pretrained=False
        )

        # Check that a model wrapper was returned
        self.assertIsInstance(model_wrapper, MLXModelWrapper)
        self.assertEqual(model_wrapper.model_name, "crop_classifier")

        # Check that the model has the correct architecture
        model = model_wrapper.model
        self.assertIsInstance(model, nn.Module)

        # Test with a dummy input
        input_data = mx.random.normal((1, 224, 224, 3))
        output = model(input_data)

        # Check output shape
        self.assertEqual(output.shape, (1, 5))


# Mock classes for testing without MLX
class MockMLXModelWrapper:
    """Mock MLXModelWrapper for testing without MLX."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model_name = kwargs.get("model_name", "mock_model")


@unittest.skipIf(MLX_AVAILABLE, "MLX is available, skipping mock tests")
class TestWithoutMLX(unittest.TestCase):
    """Test behavior when MLX is not available."""

    def test_raises_import_error(self):
        """Test that ImportError is raised when MLX is not available."""
        # Patch the MLX_AVAILABLE flag
        with patch("llamafarms.core.ai.mlx_integration.MLX_AVAILABLE", False):
            # Test MLXModelWrapper
            with self.assertRaises(ImportError):
                MLXModelWrapper(model=None, model_name="test")

            # Test optimize_for_mlx
            with patch("llamafarms.core.ai.mlx_integration.logger") as mock_logger:
                model = MagicMock()
                result = optimize_for_mlx(model)
                mock_logger.warning.assert_called_with(
                    "MLX not available, returning original model"
                )
                self.assertEqual(result, model)


if __name__ == "__main__":
    unittest.main()
