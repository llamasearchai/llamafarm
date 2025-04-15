"""
LlamaFarms Core AI Module

This module provides AI and ML capabilities for the LlamaFarms platform using state-of-the-art
models and frameworks including MLX for accelerated performance on Apple Silicon.
"""

from .mlx_integration import MLXModelWrapper, optimize_for_mlx
from .model_registry import get_model, register_model
from .vision import CropClassifier, PlantDiseaseDetector, WeedDetector

__version__ = "0.1.0"
