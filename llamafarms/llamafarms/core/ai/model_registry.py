"""
Model Registry Module

This module provides a registry for managing and accessing ML models
used throughout the LlamaFarms platform.
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("llamafarms.core.ai.model_registry")

# Global model registry
_MODEL_REGISTRY = {}


class ModelMetadata:
    """Metadata for registered models."""

    def __init__(
        self,
        name: str,
        version: str,
        model_type: str,
        description: str = "",
        is_loaded: bool = False,
        model_instance: Any = None,
        load_fn: Optional[Callable] = None,
        unload_fn: Optional[Callable] = None,
        model_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model metadata.

        Args:
            name: Model name (unique identifier)
            version: Model version
            model_type: Type of model (vision, llm, etc.)
            description: Description of the model
            is_loaded: Whether the model is currently loaded in memory
            model_instance: Reference to the loaded model instance
            load_fn: Function to load the model
            unload_fn: Function to unload the model
            model_path: Path to model weights
            params: Additional parameters
        """
        self.name = name
        self.version = version
        self.model_type = model_type
        self.description = description
        self.is_loaded = is_loaded
        self.model_instance = model_instance
        self.load_fn = load_fn
        self.unload_fn = unload_fn
        self.model_path = model_path
        self.params = params or {}
        self.last_used = None

    def __repr__(self) -> str:
        return f"<ModelMetadata: {self.name} v{self.version} ({self.model_type})>"

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "description": self.description,
            "is_loaded": self.is_loaded,
            "model_path": self.model_path,
            "params": self.params,
            "last_used": self.last_used,
        }


def register_model(
    name: str,
    version: str,
    model_type: str,
    description: str = "",
    load_fn: Optional[Callable] = None,
    unload_fn: Optional[Callable] = None,
    model_path: Optional[str] = None,
    params: Optional[Dict[str, Any]] = None,
    overwrite: bool = False,
) -> bool:
    """
    Register a model in the model registry.

    Args:
        name: Model name (unique identifier)
        version: Model version
        model_type: Type of model (vision, llm, etc.)
        description: Description of the model
        load_fn: Function to load the model
        unload_fn: Function to unload the model
        model_path: Path to model weights
        params: Additional parameters
        overwrite: Whether to overwrite existing model

    Returns:
        bool: Success status
    """
    model_id = f"{name}_{version}"

    if model_id in _MODEL_REGISTRY and not overwrite:
        logger.warning(
            f"Model {model_id} already exists in registry. Use overwrite=True to replace."
        )
        return False

    _MODEL_REGISTRY[model_id] = ModelMetadata(
        name=name,
        version=version,
        model_type=model_type,
        description=description,
        load_fn=load_fn,
        unload_fn=unload_fn,
        model_path=model_path,
        params=params,
    )

    logger.info(f"Registered model: {model_id} ({model_type})")
    return True


def get_model(
    name: str, version: Optional[str] = None, load_if_needed: bool = True, **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Get a model from the registry, optionally loading it if not already loaded.

    Args:
        name: Model name
        version: Model version (or None for latest)
        load_if_needed: Whether to load the model if not already loaded
        **kwargs: Additional parameters to pass to the load function

    Returns:
        Tuple of (model_instance, metadata_dict)
    """
    # Find the model(s) matching the name
    matches = [m for m_id, m in _MODEL_REGISTRY.items() if m.name == name]

    if not matches:
        logger.error(f"Model {name} not found in registry")
        return None, {"error": f"Model {name} not found"}

    if version:
        # Find specific version
        model_meta = next((m for m in matches if m.version == version), None)
        if not model_meta:
            logger.error(f"Model {name} version {version} not found")
            return None, {"error": f"Model version {version} not found"}
    else:
        # Use the latest version
        model_meta = sorted(matches, key=lambda m: m.version, reverse=True)[0]

    # If not loaded and we want to load it
    if not model_meta.is_loaded and load_if_needed and model_meta.load_fn:
        try:
            model_instance = model_meta.load_fn(
                model_path=model_meta.model_path, **{**model_meta.params, **kwargs}
            )
            model_meta.model_instance = model_instance
            model_meta.is_loaded = True
            logger.info(f"Loaded model: {model_meta.name} v{model_meta.version}")
        except Exception as e:
            logger.error(f"Failed to load model {model_meta.name}: {str(e)}")
            return None, {"error": f"Failed to load model: {str(e)}"}

    return model_meta.model_instance, model_meta.to_dict()


def list_models(model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all registered models, optionally filtered by type.

    Args:
        model_type: Optional filter by model type

    Returns:
        List of model metadata dictionaries
    """
    if model_type:
        return [
            m.to_dict() for m in _MODEL_REGISTRY.values() if m.model_type == model_type
        ]
    else:
        return [m.to_dict() for m in _MODEL_REGISTRY.values()]


def unload_model(name: str, version: Optional[str] = None) -> bool:
    """
    Unload a model from memory.

    Args:
        name: Model name
        version: Model version (or None for latest)

    Returns:
        bool: Success status
    """
    model_instance, metadata = get_model(name, version, load_if_needed=False)

    if "error" in metadata:
        return False

    model_id = f"{name}_{metadata['version']}"
    model_meta = _MODEL_REGISTRY[model_id]

    if model_meta.is_loaded and model_meta.unload_fn:
        try:
            model_meta.unload_fn(model_meta.model_instance)
            model_meta.is_loaded = False
            model_meta.model_instance = None
            logger.info(f"Unloaded model: {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {str(e)}")
            return False

    # If no unload function is specified, just set to None
    if model_meta.is_loaded:
        model_meta.is_loaded = False
        model_meta.model_instance = None
        logger.info(f"Unloaded model: {model_id}")
        return True

    return False


def save_registry(filepath: str) -> bool:
    """
    Save the current registry state to a file.

    Args:
        filepath: Path to save the registry

    Returns:
        bool: Success status
    """
    try:
        # Convert to serializable format
        registry_dict = {
            model_id: meta.to_dict() for model_id, meta in _MODEL_REGISTRY.items()
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(registry_dict, f, indent=2)

        logger.info(f"Saved model registry to {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to save registry: {str(e)}")
        return False


def load_registry(filepath: str) -> bool:
    """
    Load registry state from a file.

    Args:
        filepath: Path to the registry file

    Returns:
        bool: Success status
    """
    try:
        if not os.path.exists(filepath):
            logger.warning(f"Registry file not found: {filepath}")
            return False

        with open(filepath, "r") as f:
            registry_dict = json.load(f)

        # Clear current registry
        _MODEL_REGISTRY.clear()

        # Load from file (note: we can't serialize functions, so load_fn and unload_fn will be None)
        for model_id, meta_dict in registry_dict.items():
            _MODEL_REGISTRY[model_id] = ModelMetadata(
                name=meta_dict["name"],
                version=meta_dict["version"],
                model_type=meta_dict["model_type"],
                description=meta_dict.get("description", ""),
                is_loaded=False,  # Always start unloaded
                model_path=meta_dict.get("model_path"),
                params=meta_dict.get("params", {}),
            )

        logger.info(f"Loaded model registry from {filepath}")
        return True

    except Exception as e:
        logger.error(f"Failed to load registry: {str(e)}")
        return False
