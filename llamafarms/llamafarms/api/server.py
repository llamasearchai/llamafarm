"""
LlamaFarms API Server

This module provides a FastAPI application that exposes the capabilities of LlamaFarms
through a modern REST API, enabling integration with various clients and services.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn

# FastAPI and web dependencies
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles

# Import from parent package
from .. import logger

# Initialize FastAPI app
app = FastAPI(
    title="LlamaFarms API",
    description="API for LlamaFarms - Precision Agriculture Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OAuth2 password bearer for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)


# Dependency for optional authentication
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user from the token."""
    if token is None:
        return None

    # In a real application, validate the token
    # This is a simplified version
    return {"username": "demo_user"}


@app.get("/api/v1/health")
async def health_check():
    """Check API health."""
    return {"status": "ok", "version": "0.1.0", "timestamp": datetime.now().isoformat()}


@app.get("/api/v1/models")
async def list_models(
    type: Optional[str] = None, current_user: Dict = Depends(get_current_user)
):
    """List available models."""
    try:
        from ..core.ai.model_registry import list_models as registry_list_models

        models = registry_list_models(type)
        return {"models": models, "count": len(models)}

    except ImportError:
        logger.error("Model registry not available")
        return {"models": [], "count": 0, "error": "Model registry not available"}


@app.post("/api/v1/vision/classify")
async def classify_crop(
    image: UploadFile = File(...),
    backend: str = Query("mlx", description="ML backend to use (mlx or torch)"),
    current_user: Dict = Depends(get_current_user),
):
    """Classify crop type in an uploaded image."""
    try:
        # Import vision module
        from ..core.ai.vision import CropClassifier

        # Check file format
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        import io

        import numpy as np
        from PIL import Image

        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        # Create classifier
        classifier = CropClassifier(backend=backend)

        # Run classification
        import time

        start_time = time.time()
        result = classifier.predict(img_array)

        # Add processing time and timestamp
        result["processing_time"] = time.time() - start_time
        result["timestamp"] = datetime.now().isoformat()

        return result

    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        raise HTTPException(
            status_code=500, detail=f"Required module not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error classifying crop: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/vision/disease")
async def detect_disease(
    image: UploadFile = File(...),
    backend: str = Query("yolo", description="ML backend to use (yolo, mlx)"),
    current_user: Dict = Depends(get_current_user),
):
    """Detect diseases in an uploaded plant image."""
    try:
        # Import vision module
        from ..core.ai.vision import PlantDiseaseDetector

        # Check file format
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        import io

        import numpy as np
        from PIL import Image

        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        # Create detector
        detector = PlantDiseaseDetector(backend=backend)

        # Run detection
        import time

        start_time = time.time()
        result = detector.detect(img_array)

        # Add processing time
        result["processing_time"] = time.time() - start_time

        return result

    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        raise HTTPException(
            status_code=500, detail=f"Required module not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error detecting disease: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/vision/weed")
async def detect_weeds(
    image: UploadFile = File(...),
    backend: str = Query("yolo", description="ML backend to use (sam, yolo, mlx)"),
    current_user: Dict = Depends(get_current_user),
):
    """Detect weeds in an uploaded field image."""
    try:
        # Import vision module
        from ..core.ai.vision import WeedDetector

        # Check file format
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read image data
        import io

        import numpy as np
        from PIL import Image

        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        # Create detector
        detector = WeedDetector(backend=backend)

        # Run detection
        import time

        start_time = time.time()
        result = detector.detect(img_array)

        # Add processing time
        result["processing_time"] = time.time() - start_time

        return result

    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        raise HTTPException(
            status_code=500, detail=f"Required module not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error detecting weeds: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/llm/ask")
async def ask_llm(
    query: str = Query(..., description="Question to ask"),
    model_type: str = Query("openai", description="Model type (openai, mlx, hybrid)"),
    model_name: str = Query("gpt-4", description="Model name"),
    current_user: Dict = Depends(get_current_user),
):
    """Query the agricultural LLM."""
    try:
        # Import LLM module
        from ..core.ai.llm import AgricultureLLM

        # Create LLM
        llm = AgricultureLLM(model_type=model_type, model_name=model_name)

        # Generate response
        import time

        start_time = time.time()
        response = llm.ask(query=query)

        # Extract text if response is a dict
        if isinstance(response, dict):
            result = response
        else:
            result = {"text": response, "model": model_name, "type": model_type}

        # Add processing time
        result["processing_time"] = time.time() - start_time

        return result

    except ImportError as e:
        logger.error(f"Required module not found: {e}")
        raise HTTPException(
            status_code=500, detail=f"Required module not available: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing LLM query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    # Import necessary core modules to ensure they're available
    try:
        from ..core.ai import mlx_integration

        mlx_integration.register_available_models()
        logger.info("Registered MLX models in the model registry")
    except ImportError:
        logger.warning("Could not import MLX integration module")

    # Start the server
    uvicorn.run(
        "llamafarms.api.server:app",
        host=host,
        port=port,
        reload=os.environ.get("LLAMAFARMS_ENV") == "development",
    )
