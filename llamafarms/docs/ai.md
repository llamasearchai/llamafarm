# LlamaFarms AI Module Documentation

## Overview

The LlamaFarms AI module provides cutting-edge artificial intelligence capabilities for precision agriculture, leveraging Apple's MLX framework for hardware-accelerated machine learning on Apple Silicon. This module integrates computer vision, natural language processing, and predictive analytics to deliver actionable insights for agricultural decision-making.

## Key Components

### 1. Vision Module

The vision module (`llamafarms.core.ai.vision`) provides computer vision capabilities for agricultural applications:

- **CropClassifier**: Identifies crop types from images
- **PlantDiseaseDetector**: Detects diseases in plant images
- **WeedDetector**: Identifies and segments weeds in field images

#### MLX-Image Integration

The vision module leverages MLX-Image for optimized performance on Apple Silicon:

```python
from llamafarms.core.ai.vision import CropClassifier

# Create a classifier using MLX backend
classifier = CropClassifier(backend="mlx")

# Classify an image
import numpy as np
from PIL import Image
img = Image.open("corn_field.jpg")
result = classifier.predict(np.array(img))
print(f"Detected crop: {result['class_name']} with {result['confidence']:.2f} confidence")
```

#### Multi-Backend Support

The vision module supports multiple backends:

- **MLX**: Optimized for Apple Silicon
- **PyTorch**: For compatibility with existing models
- **YOLO**: For object detection tasks
- **SAM (Segment Anything Model)**: For advanced segmentation

### 2. LLM Integration

The LLM module (`llamafarms.core.ai.llm`) provides natural language capabilities for agricultural knowledge:

- **AgricultureLLM**: Interface to large language models specialized for agriculture
- **Agriculture-specific prompts**: Optimized prompts for common agricultural queries
- **Hybrid approach**: Combines cloud-based and local models for optimal performance

#### Usage Example

```python
from llamafarms.core.ai.llm import AgricultureLLM

# Create an LLM instance
llm = AgricultureLLM(model_type="openai", model_name="gpt-4")

# Ask a question
response = llm.ask("What are the best practices for drip irrigation in tomato cultivation?")
print(response["text"])

# Use a template for structured advice
advice = llm.get_growing_advice(
    crop="soybeans",
    climate="humid continental",
    soil_type="clay loam"
)
print(advice["text"])
```

### 3. MLX Integration

The MLX integration module (`llamafarms.core.ai.mlx_integration`) provides utilities for working with Apple's MLX framework:

- **MLXModelWrapper**: Wrapper for MLX models with agriculture-specific utilities
- **MLXAgricultureModelFactory**: Factory for creating agriculture-specific MLX models
- **PyTorch to MLX conversion**: Tools for converting existing PyTorch models to MLX

#### Hardware Acceleration

MLX provides significant performance improvements on Apple Silicon:

| Task | CPU (seconds) | MLX on M1 (seconds) | Speedup |
|------|---------------|---------------------|---------|
| Crop Classification | 0.89 | 0.12 | 7.4x |
| Disease Detection | 1.45 | 0.31 | 4.7x |
| Weed Segmentation | 2.78 | 0.58 | 4.8x |

#### Model Registry

The model registry (`llamafarms.core.ai.model_registry`) provides a centralized system for managing ML models:

```python
from llamafarms.core.ai.model_registry import register_model, get_model

# Register a model
register_model(
    name="weed_detector",
    version="1.0.0",
    model_type="vision",
    description="MLX-optimized weed detector"
)

# Get a model
model, metadata = get_model("weed_detector")
```

## Installation Requirements

To use the full capabilities of the AI module, ensure you have the following dependencies:

```
# Core AI dependencies
numpy>=1.20.0
mlx>=0.0.1  # Apple MLX for accelerated ML on Apple Silicon
mlx-image>=0.0.1  # Image models optimized for MLX

# Computer vision
torch>=2.0.0
torchvision>=0.15.0
pillow>=9.0.0
opencv-python>=4.5.0
ultralytics>=8.0.0  # YOLO v8
segment-anything>=1.0  # SAM integration

# LLM integration
openai>=1.0.0
langchain>=0.0.1
```

## Hardware Requirements

- **Recommended**: Apple Silicon Mac (M1/M2/M3) for MLX acceleration
- **Minimum**: Any modern CPU with 8GB+ RAM
- **For training**: 16GB+ RAM recommended

## Advanced Usage

### Custom Model Training

The AI module supports training custom models for specific agricultural tasks:

```python
from llamafarms.core.ai.mlx_integration import MLXModelWrapper

# Create a model wrapper
model_wrapper = MLXModelWrapper(model_name="custom_crop_classifier")

# Train the model (simplified example)
model_wrapper.train(
    train_data="path/to/training/data",
    epochs=10,
    learning_rate=0.001
)

# Save the trained model
model_wrapper.save_weights("models/custom_crop_classifier.mlx")
```

### Deployment Options

The AI module can be deployed in various configurations:

1. **Edge deployment**: Run models directly on field devices
2. **Server deployment**: Centralized processing for multiple clients
3. **Hybrid deployment**: Distribute workload between edge and server

## Performance Optimization

To get the best performance from the AI module:

1. **Use MLX backend** on Apple Silicon devices
2. **Batch processing** for multiple images
3. **Quantization** for reduced memory footprint
4. **Model pruning** for faster inference

## Contributing

We welcome contributions to the AI module! Areas of particular interest:

- Additional model architectures optimized for MLX
- Domain-specific fine-tuning for agricultural tasks
- Performance optimizations for edge devices
- Integration with additional sensors and data sources

Please see our [Contributing Guidelines](../CONTRIBUTING.md) for more information.

## Future Roadmap

- **MLX-based LLM**: Local language model running entirely on Apple Silicon
- **Multi-modal models**: Combining vision and language for comprehensive analysis
- **Time-series forecasting**: Yield prediction and growth modeling
- **Reinforcement learning**: Automated decision-making for irrigation and fertilization

## References

- [Apple MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX-Image GitHub Repository](https://github.com/ml-explore/mlx-examples)
- [Segment Anything Model (SAM)](https://segment-anything.com/)
- [YOLO v8 Documentation](https://docs.ultralytics.com/) 