"""
LlamaFarmAI REST API

This module provides a FastAPI application that exposes the capabilities of LlamaFarmAI
through a modern REST API, enabling integration with various clients and services.
"""

import io
import json
import logging
import os

# Import core modules
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Image processing
import numpy as np
import uvicorn

# FastAPI and web dependencies
from fastapi import (
    BackgroundTasks,
    Body,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.irrigation.schedule_optimizer import IrrigationOptimizer
from core.satellite.sentinel_processor import SatelliteManager
from core.soil.moisture_predictor import SoilAnalyzer

from core.ai.llm import AgricultureLLM, FarmAssistant
from core.ai.vision import CropClassifier, PlantDiseaseDetector, WeedDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api_server.log")],
)
logger = logging.getLogger("llamafarm.api")


# Initialize FastAPI app
app = FastAPI(
    title="LlamaFarmAI API",
    description="API for LlamaFarmAI - Precision Agriculture Platform",
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
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Models for request/response
class TokenResponse(BaseModel):
    access_token: str
    token_type: str


class CropClassificationRequest(BaseModel):
    backend: str = "mlx"  # 'mlx' or 'torch'
    device: str = "auto"


class CropClassificationResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    timestamp: str
    processing_time: float


class DiseaseDetectionRequest(BaseModel):
    backend: str = "yolo"  # 'yolo', 'mlx', 'torch'
    device: str = "auto"


class DiseaseDetectionResponse(BaseModel):
    timestamp: str
    detections: List[Dict[str, Any]]
    count: int
    processing_time: float


class WeedDetectionRequest(BaseModel):
    backend: str = "yolo"  # 'sam', 'yolo', 'mlx'
    device: str = "auto"


class WeedDetectionResponse(BaseModel):
    timestamp: str
    detections: List[Dict[str, Any]]
    count: int
    weed_coverage_percent: float
    processing_time: float


class LLMQueryRequest(BaseModel):
    query: str
    prompt_template: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    model_type: str = "openai"  # 'openai', 'mlx', 'hybrid'
    model_name: str = "gpt-4"


class LLMQueryResponse(BaseModel):
    text: str
    model: str
    type: str
    processing_time: float


class IrrigationRequest(BaseModel):
    field_data: Dict[str, Any]
    weather_forecast: List[Dict[str, Any]]
    constraints: Dict[str, Any]
    planning_horizon: int = 7


class IrrigationResponse(BaseModel):
    schedule: List[Dict[str, Any]]
    water_usage: float
    processing_time: float


# Dependencies
async def get_current_user(token: str = Depends(oauth2_scheme)):
    """Get the current user from the token."""
    # In a real application, validate the token
    # This is a simplified version
    return {"username": "demo_user"}


def get_crop_classifier(backend: str = "mlx", device: str = "auto"):
    """Get a crop classifier instance."""
    return CropClassifier(
        model_path=None,  # Would use a real model path in production
        backend=backend,
        device=device,
    )


def get_disease_detector(backend: str = "yolo", device: str = "auto"):
    """Get a disease detector instance."""
    return PlantDiseaseDetector(
        model_path=None,  # Would use a real model path in production
        backend=backend,
        device=device,
    )


def get_weed_detector(backend: str = "yolo", device: str = "auto"):
    """Get a weed detector instance."""
    return WeedDetector(
        model_path=None,  # Would use a real model path in production
        backend=backend,
        device=device,
    )


def get_llm(model_type: str = "openai", model_name: str = "gpt-4"):
    """Get an agriculture LLM instance."""
    # In a real application, you'd use an actual API key
    api_key = os.environ.get("OPENAI_API_KEY", "dummy_key")

    return AgricultureLLM(model_type=model_type, model_name=model_name, api_key=api_key)


def get_irrigation_optimizer():
    """Get an irrigation optimizer instance."""
    return IrrigationOptimizer()


# API Routes
@app.post("/token", response_model=TokenResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Get an access token."""
    # In a real application, validate username and password
    # This is a simplified version
    if form_data.username != "demo" or form_data.password != "demo":
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Generate token (simplified)
    access_token = f"demo_token_{int(time.time())}"

    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/api/v1/crop/classify", response_model=CropClassificationResponse)
async def classify_crop(
    image: UploadFile = File(...),
    request: CropClassificationRequest = Depends(),
    current_user: Dict = Depends(get_current_user),
):
    """Classify crop type in an uploaded image."""
    start_time = time.time()

    try:
        # Read and process the image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        # Get classifier
        classifier = get_crop_classifier(backend=request.backend, device=request.device)

        # Run classification
        result = classifier.predict(img_array)

        # Add processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        return result

    except Exception as e:
        logger.error(f"Error in crop classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/disease/detect", response_model=DiseaseDetectionResponse)
async def detect_disease(
    image: UploadFile = File(...),
    request: DiseaseDetectionRequest = Depends(),
    current_user: Dict = Depends(get_current_user),
):
    """Detect diseases in an uploaded plant image."""
    start_time = time.time()

    try:
        # Read and process the image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        # Get detector
        detector = get_disease_detector(backend=request.backend, device=request.device)

        # Run detection
        result = detector.detect(img_array)

        # Add processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        return result

    except Exception as e:
        logger.error(f"Error in disease detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/weed/detect", response_model=WeedDetectionResponse)
async def detect_weeds(
    image: UploadFile = File(...),
    request: WeedDetectionRequest = Depends(),
    current_user: Dict = Depends(get_current_user),
):
    """Detect weeds in an uploaded field image."""
    start_time = time.time()

    try:
        # Read and process the image
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        img_array = np.array(img)

        # Get detector
        detector = get_weed_detector(backend=request.backend, device=request.device)

        # Run detection
        result = detector.detect(img_array)

        # Add processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        return result

    except Exception as e:
        logger.error(f"Error in weed detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/api/v1/llm/query", response_model=LLMQueryResponse)
async def query_llm(
    request: LLMQueryRequest, current_user: Dict = Depends(get_current_user)
):
    """Query the agricultural LLM."""
    start_time = time.time()

    try:
        # Get LLM instance
        llm = get_llm(model_type=request.model_type, model_name=request.model_name)

        # Generate response
        response = llm.ask(
            query=request.query,
            prompt_template=request.prompt_template,
            context=request.context,
        )

        # Extract text if response is a dict
        if isinstance(response, dict):
            result = response
        else:
            result = {
                "text": response,
                "model": request.model_name,
                "type": request.model_type,
            }

        # Add processing time
        processing_time = time.time() - start_time
        result["processing_time"] = processing_time

        return result

    except Exception as e:
        logger.error(f"Error in LLM query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/v1/irrigation/optimize", response_model=IrrigationResponse)
async def optimize_irrigation(
    request: IrrigationRequest, current_user: Dict = Depends(get_current_user)
):
    """Optimize irrigation schedule."""
    start_time = time.time()

    try:
        # Get optimizer instance
        optimizer = get_irrigation_optimizer()

        # Generate schedule
        schedule = optimizer.optimize_schedule(
            field_data=request.field_data,
            weather_forecast=request.weather_forecast,
            constraints=request.constraints,
            planning_horizon=request.planning_horizon,
        )

        # Calculate total water usage
        water_usage = sum(
            day.get("water_volume_m3", 0) for day in schedule.get("daily_schedule", [])
        )

        # Add processing time
        processing_time = time.time() - start_time

        return {
            "schedule": schedule.get("daily_schedule", []),
            "water_usage": water_usage,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Error in irrigation optimization: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error optimizing irrigation: {str(e)}"
        )


@app.get("/api/v1/health")
async def health_check():
    """Check API health."""
    return {"status": "ok", "version": "0.1.0", "timestamp": datetime.now().isoformat()}


def start_api_server():
    """Start the API server."""
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start_api_server()
