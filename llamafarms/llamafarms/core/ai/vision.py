"""
Vision Module for Agricultural Applications

This module provides computer vision capabilities for agricultural applications,
including crop classification, disease detection, and weed detection.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Configure logging
logger = logging.getLogger("llamafarms.core.ai.vision")

# Try to import optional dependencies
try:
    import torch
    import torchvision

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Some vision features will be limited.")

try:
    import mlx.core as mx
    import mlx.nn as nn

    from ..mlx_integration import MLXModelWrapper, optimize_for_mlx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Hardware acceleration will be limited.")

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Object detection will be limited.")

try:
    from segment_anything import SamPredictor, sam_model_registry

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    logger.warning(
        "Segment Anything Model not available. Segmentation will be limited."
    )


class CropClassifier:
    """Classifier for identifying crop types in images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_classes: int = 10,
        backend: str = "mlx",  # 'mlx' or 'torch'
        device: str = "auto",
    ):
        """
        Initialize a crop classifier.

        Args:
            model_path: Path to model weights
            num_classes: Number of crop classes
            backend: ML backend to use
            device: Device to run on
        """
        self.backend = backend
        self.device = self._resolve_device(device)
        self.num_classes = num_classes
        self.model_path = model_path
        self.class_names = self._get_default_class_names()

        # Initialize model based on backend
        if backend == "mlx":
            if not MLX_AVAILABLE:
                raise ImportError("MLX backend requested but MLX is not available")
            self._init_mlx_model()
        elif backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch backend requested but PyTorch is not available"
                )
            self._init_torch_model()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        logger.info(
            f"Initialized crop classifier with {backend} backend on {self.device}"
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use based on availability."""
        if device != "auto":
            return device

        if self.backend == "mlx":
            # MLX automatically uses the best available device
            return "gpu"
        elif self.backend == "torch":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        return "cpu"

    def _get_default_class_names(self) -> List[str]:
        """Get default class names for crops."""
        return [
            "corn",
            "wheat",
            "rice",
            "barley",
            "soybean",
            "cotton",
            "sugarcane",
            "potato",
            "tomato",
            "other",
        ]

    def _init_mlx_model(self):
        """Initialize MLX model."""
        from ..mlx_integration import MLXAgricultureModelFactory

        # Create model using factory
        model_wrapper = MLXAgricultureModelFactory.create_crop_classifier(
            num_classes=self.num_classes, pretrained=True
        )

        self.model = model_wrapper
        logger.info("Initialized MLX crop classifier model")

    def _init_torch_model(self):
        """Initialize PyTorch model."""
        # For demonstration, we'll use a pre-trained ResNet and modify the final layer
        model = torchvision.models.resnet50(pretrained=True)

        # Replace final layer to match our number of classes
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, self.num_classes)

        # Move to appropriate device
        device = torch.device(self.device)
        model = model.to(device)
        model.eval()

        self.model = model
        logger.info(f"Initialized PyTorch crop classifier model on {self.device}")

    def preprocess_image(self, image: np.ndarray) -> Union[mx.array, torch.Tensor]:
        """
        Preprocess image for model input.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Processed image in the format expected by the model
        """
        # Basic preprocessing - resize and normalize
        from PIL import Image

        # Resize to expected input size
        input_size = (224, 224)
        if image.shape[0] != input_size[0] or image.shape[1] != input_size[1]:
            pil_image = Image.fromarray(image)
            pil_image = pil_image.resize(input_size)
            image = np.array(pil_image)

        # Normalize
        image = image.astype(np.float32) / 255.0

        if self.backend == "mlx":
            # Convert to MLX array with shape (1, H, W, C)
            return mx.array(image)
        else:
            # Convert to PyTorch tensor with shape (1, C, H, W)
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            image = np.expand_dims(image, 0)  # Add batch dimension
            return torch.tensor(image, device=self.device)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify crop in an image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Dictionary with classification results
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)

        # Run inference
        if self.backend == "mlx":
            # MLX inference
            outputs = self.model.predict(processed_image)

            # Get predictions
            if isinstance(outputs, mx.array):
                probs = nn.softmax(outputs)[0]
                class_id = int(mx.argmax(probs).item())
                confidence = float(probs[class_id].item())
            else:
                # Handle case where model returns a dict or other structure
                class_id = 0  # Placeholder
                confidence = 0.8  # Placeholder
        else:
            # PyTorch inference
            with torch.no_grad():
                outputs = self.model(processed_image)
                probs = torch.softmax(outputs, dim=1)[0]
                class_id = int(torch.argmax(probs).item())
                confidence = float(probs[class_id].item())

        # Get class name
        class_name = (
            self.class_names[class_id]
            if class_id < len(self.class_names)
            else "unknown"
        )

        # Return results
        return {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": confidence,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }


class PlantDiseaseDetector:
    """Detector for identifying plant diseases in images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "yolo",  # 'yolo', 'mlx', 'torch'
        device: str = "auto",
    ):
        """
        Initialize a plant disease detector.

        Args:
            model_path: Path to model weights
            backend: ML backend to use
            device: Device to run on
        """
        self.backend = backend
        self.device = self._resolve_device(device)
        self.model_path = model_path
        self.disease_classes = self._get_disease_classes()

        # Initialize model based on backend
        if backend == "yolo":
            if not YOLO_AVAILABLE:
                raise ImportError("YOLO backend requested but YOLO is not available")
            self._init_yolo_model()
        elif backend == "mlx":
            if not MLX_AVAILABLE:
                raise ImportError("MLX backend requested but MLX is not available")
            self._init_mlx_model()
        elif backend == "torch":
            if not TORCH_AVAILABLE:
                raise ImportError(
                    "PyTorch backend requested but PyTorch is not available"
                )
            self._init_torch_model()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        logger.info(
            f"Initialized plant disease detector with {backend} backend on {self.device}"
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use based on availability."""
        if device != "auto":
            return device

        if self.backend == "mlx":
            # MLX automatically uses the best available device
            return "gpu"
        elif self.backend in ["torch", "yolo"]:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        return "cpu"

    def _get_disease_classes(self) -> List[str]:
        """Get disease class names."""
        return [
            "healthy",
            "bacterial_blight",
            "bacterial_spot",
            "black_rot",
            "cercospora_leaf_spot",
            "common_rust",
            "early_blight",
            "fusarium_wilt",
            "late_blight",
            "leaf_mold",
            "northern_leaf_blight",
            "powdery_mildew",
            "septoria_leaf_spot",
            "southern_corn_rust",
            "target_spot",
            "yellow_leaf_curl_virus",
        ]

    def _init_yolo_model(self):
        """Initialize YOLO model."""
        # Use a pre-trained YOLO model or a custom one if path is provided
        if self.model_path:
            self.model = YOLO(self.model_path)
        else:
            # Use a pre-trained model
            self.model = YOLO("yolov8n.pt")

        logger.info("Initialized YOLO plant disease detector model")

    def _init_mlx_model(self):
        """Initialize MLX model."""
        # Placeholder for MLX model initialization
        logger.warning("MLX plant disease detector not fully implemented")
        self.model = None

    def _init_torch_model(self):
        """Initialize PyTorch model."""
        # Placeholder for PyTorch model initialization
        logger.warning("PyTorch plant disease detector not fully implemented")
        self.model = None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect diseases in a plant image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Dictionary with detection results
        """
        if self.backend == "yolo" and self.model:
            # Run YOLO detection
            results = self.model(image, verbose=False)

            # Process results
            detections = []
            for i, result in enumerate(results):
                boxes = result.boxes
                for j, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Get class and confidence
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Map to disease class
                    if cls_id < len(self.disease_classes):
                        cls_name = self.disease_classes[cls_id]
                    else:
                        cls_name = f"class_{cls_id}"

                    # Add to detections
                    detections.append(
                        {
                            "id": j,
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detections": detections,
                "count": len(detections),
            }
        else:
            # Placeholder for other backends or if model is not initialized
            logger.warning(f"Using placeholder detection with {self.backend} backend")

            # Return a placeholder detection
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detections": [
                    {
                        "id": 0,
                        "class_id": 1,
                        "class_name": "bacterial_blight",
                        "confidence": 0.85,
                        "bbox": [50, 50, 150, 150],
                    }
                ],
                "count": 1,
            }


class WeedDetector:
    """Detector for identifying weeds in field images."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        backend: str = "sam",  # 'sam', 'yolo', 'mlx'
        device: str = "auto",
    ):
        """
        Initialize a weed detector.

        Args:
            model_path: Path to model weights
            backend: ML backend to use
            device: Device to run on
        """
        self.backend = backend
        self.device = self._resolve_device(device)
        self.model_path = model_path

        # Initialize model based on backend
        if backend == "sam":
            if not SAM_AVAILABLE:
                raise ImportError(
                    "SAM backend requested but Segment Anything is not available"
                )
            self._init_sam_model()
        elif backend == "yolo":
            if not YOLO_AVAILABLE:
                raise ImportError("YOLO backend requested but YOLO is not available")
            self._init_yolo_model()
        elif backend == "mlx":
            if not MLX_AVAILABLE:
                raise ImportError("MLX backend requested but MLX is not available")
            self._init_mlx_model()
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        logger.info(
            f"Initialized weed detector with {backend} backend on {self.device}"
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve the device to use based on availability."""
        if device != "auto":
            return device

        if self.backend == "mlx":
            # MLX automatically uses the best available device
            return "gpu"
        elif self.backend in ["sam", "yolo"]:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"

        return "cpu"

    def _init_sam_model(self):
        """Initialize SAM model."""
        # Placeholder for SAM model initialization
        logger.warning("SAM weed detector not fully implemented")
        self.model = None

    def _init_yolo_model(self):
        """Initialize YOLO model."""
        # Use a pre-trained YOLO model or a custom one if path is provided
        if self.model_path:
            self.model = YOLO(self.model_path)
        else:
            # Use a pre-trained model
            self.model = YOLO("yolov8n-seg.pt")  # Segmentation model

        logger.info("Initialized YOLO weed detector model")

    def _init_mlx_model(self):
        """Initialize MLX model."""
        # Placeholder for MLX model initialization
        logger.warning("MLX weed detector not fully implemented")
        self.model = None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect weeds in a field image.

        Args:
            image: Input image as numpy array (H, W, C)

        Returns:
            Dictionary with detection results
        """
        if self.backend == "yolo" and self.model:
            # Run YOLO detection
            results = self.model(image, verbose=False)

            # Process results
            detections = []
            for i, result in enumerate(results):
                boxes = result.boxes
                for j, box in enumerate(boxes):
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    # Get class and confidence
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Add to detections
                    detections.append(
                        {
                            "id": j,
                            "class_name": "weed",
                            "confidence": conf,
                            "bbox": [x1, y1, x2, y2],
                        }
                    )

            # Estimate weed coverage
            weed_coverage = self._estimate_coverage(detections, image.shape)

            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "detections": detections,
                "count": len(detections),
                "weed_coverage_percent": weed_coverage,
            }
        else:
            # Placeholder for other backends or if model is not initialized
            logger.warning(f"Using placeholder detection with {self.backend} backend")

            # Return a placeholder detection
            return {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
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

    def _estimate_coverage(
        self, detections: List[Dict[str, Any]], image_shape: Tuple[int, int, int]
    ) -> float:
        """
        Estimate weed coverage percentage based on detections.

        Args:
            detections: List of detection results
            image_shape: Shape of the image (H, W, C)

        Returns:
            Estimated weed coverage percentage
        """
        if not detections:
            return 0.0

        # Calculate total image area
        image_area = image_shape[0] * image_shape[1]

        # Calculate total weed area
        weed_area = 0
        for detection in detections:
            bbox = detection["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            weed_area += width * height

        # Calculate coverage percentage
        coverage_percent = (weed_area / image_area) * 100

        # Cap at 100%
        return min(coverage_percent, 100.0)
