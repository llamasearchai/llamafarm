#!/usr/bin/env python3
"""
LlamaFarmAI Command Line Interface

This CLI tool provides access to LlamaFarmAI functionality directly from the terminal,
enabling users to analyze images, query the AI assistant, and access other features
without using the web interface.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llamafarmai-cli")

# Default API endpoint
DEFAULT_API_URL = "http://localhost:8000/api/v1"


class LlamaFarmAICLI:
    """Command-line interface for LlamaFarmAI."""
    
    def __init__(self, api_url: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize the CLI tool.
        
        Args:
            api_url: URL of the LlamaFarmAI API (defaults to environment variable or localhost)
            token: API token (defaults to environment variable)
        """
        self.api_url = api_url or os.environ.get("LLAMAFARMAI_API_URL", DEFAULT_API_URL)
        self.token = token or os.environ.get("LLAMAFARMAI_API_TOKEN")
        
        # Initialize session for API requests
        self.session = requests.Session()
        
        if self.token:
            self.session.headers.update({"Author: Nik Jois
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an API request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Form data (for POST requests)
            files: Files to upload
            json_data: JSON data (for POST requests)
        
        Returns:
            Dict containing the API response
        """
        url = f"{self.api_url}/{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, params=data)
            elif method == "POST":
                response = self.session.post(url, data=data, files=files, json=json_data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            return response.json()
            
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    error_message = error_data.get('detail', str(e))
                except ValueError:
                    error_message = str(e)
            else:
                error_message = str(e)
                
            logger.error(f"API request failed: {error_message}")
            return {"error": error_message, "status_code": getattr(e.response, 'status_code', None)}
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Log in to the API and get an access token.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Dict containing the access token
        """
        data = {
            "username": username,
            "password": password
        }
        
        result = self._make_request("POST", "token", data=data)
        
        if "access_token" in result:
            self.token = result["access_token"]
            self.session.headers.update({"Author: Nik Jois
            
            # Save token to environment variable
            os.environ["LLAMAFARMAI_API_TOKEN"] = self.token
            
            logger.info("Successfully logged in")
        else:
            logger.error("Login failed")
        
        return result
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check API health.
        
        Returns:
            Dict containing health status
        """
        return self._make_request("GET", "health")
    
    def classify_crop(
        self,
        image_path: str,
        backend: str = "mlx",
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        Classify crop type in an image.
        
        Args:
            image_path: Path to image file
            backend: ML backend to use
            device: Device to run on
            
        Returns:
            Dict containing classification results
        """
        # Verify image file exists
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image file not found: {image_path}"}
        
        # Prepare files and data
        files = {
            "image": (image_path.name, open(image_path, "rb"), f"image/{image_path.suffix[1:]}")
        }
        
        data = {
            "backend": backend,
            "device": device
        }
        
        return self._make_request("POST", "crop/classify", data=data, files=files)
    
    def detect_disease(
        self,
        image_path: str,
        backend: str = "yolo",
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        Detect diseases in a plant image.
        
        Args:
            image_path: Path to image file
            backend: ML backend to use
            device: Device to run on
            
        Returns:
            Dict containing detection results
        """
        # Verify image file exists
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image file not found: {image_path}"}
        
        # Prepare files and data
        files = {
            "image": (image_path.name, open(image_path, "rb"), f"image/{image_path.suffix[1:]}")
        }
        
        data = {
            "backend": backend,
            "device": device
        }
        
        return self._make_request("POST", "disease/detect", data=data, files=files)
    
    def detect_weeds(
        self,
        image_path: str,
        backend: str = "yolo",
        device: str = "auto"
    ) -> Dict[str, Any]:
        """
        Detect weeds in a field image.
        
        Args:
            image_path: Path to image file
            backend: ML backend to use
            device: Device to run on
            
        Returns:
            Dict containing detection results
        """
        # Verify image file exists
        image_path = Path(image_path)
        if not image_path.exists():
            return {"error": f"Image file not found: {image_path}"}
        
        # Prepare files and data
        files = {
            "image": (image_path.name, open(image_path, "rb"), f"image/{image_path.suffix[1:]}")
        }
        
        data = {
            "backend": backend,
            "device": device
        }
        
        return self._make_request("POST", "weed/detect", data=data, files=files)
    
    def query_llm(
        self,
        query: str,
        prompt_template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        model_type: str = "openai",
        model_name: str = "gpt-4"
    ) -> Dict[str, Any]:
        """
        Query the agricultural language model.
        
        Args:
            query: Question to ask
            prompt_template: Name of the prompt template to use
            context: Variables to format the prompt template with
            model_type: LLM backend to use
            model_name: Name of the model to use
            
        Returns:
            Dict containing the response
        """
        json_data = {
            "query": query,
            "model_type": model_type,
            "model_name": model_name
        }
        
        if prompt_template:
            json_data["prompt_template"] = prompt_template
            
        if context:
            json_data["context"] = context
        
        return self._make_request("POST", "llm/query", json_data=json_data)
    
    def optimize_irrigation(
        self,
        field_data: Dict[str, Any],
        weather_forecast: List[Dict[str, Any]],
        constraints: Dict[str, Any],
        planning_horizon: int = 7
    ) -> Dict[str, Any]:
        """
        Optimize irrigation scheduling.
        
        Args:
            field_data: Field characteristics
            weather_forecast: Weather forecast data
            constraints: Resource constraints
            planning_horizon: Number of days to plan for
            
        Returns:
            Dict containing the irrigation schedule
        """
        json_data = {
            "field_data": field_data,
            "weather_forecast": weather_forecast,
            "constraints": constraints,
            "planning_horizon": planning_horizon
        }
        
        return self._make_request("POST", "irrigation/optimize", json_data=json_data)


def format_json(data: Dict[str, Any]) -> str:
    """Format JSON data for display."""
    return json.dumps(data, indent=2)


def main():
    """Entry point for the CLI tool."""
    # Create main parser
    parser = argparse.ArgumentParser(description="LlamaFarmAI Command Line Interface")
    parser.add_argument("--api-url", help="LlamaFarmAI API URL", default=os.environ.get("LLAMAFARMAI_API_URL", DEFAULT_API_URL))
    parser.add_argument("--token", help="API token", default=os.environ.get("LLAMAFARMAI_API_TOKEN"))
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Login command
    login_parser = subparsers.add_parser("login", help="Log in to the API")
    login_parser.add_argument("username", help="Username")
    login_parser.add_argument("password", help="Password")
    
    # Health check command
    subparsers.add_parser("health", help="Check API health")
    
    # Crop classification command
    classify_parser = subparsers.add_parser("classify-crop", help="Classify crop in image")
    classify_parser.add_argument("image", help="Path to image file")
    classify_parser.add_argument("--backend", choices=["mlx", "torch"], default="mlx", help="ML backend to use")
    classify_parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device to run on")
    
    # Disease detection command
    disease_parser = subparsers.add_parser("detect-disease", help="Detect diseases in plant image")
    disease_parser.add_argument("image", help="Path to image file")
    disease_parser.add_argument("--backend", choices=["yolo", "mlx", "torch"], default="yolo", help="ML backend to use")
    disease_parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device to run on")
    
    # Weed detection command
    weed_parser = subparsers.add_parser("detect-weeds", help="Detect weeds in field image")
    weed_parser.add_argument("image", help="Path to image file")
    weed_parser.add_argument("--backend", choices=["sam", "yolo", "mlx"], default="yolo", help="ML backend to use")
    weed_parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto", help="Device to run on")
    
    # LLM query command
    llm_parser = subparsers.add_parser("ask", help="Query the agricultural language model")
    llm_parser.add_argument("query", help="Question to ask")
    llm_parser.add_argument("--template", help="Prompt template to use")
    llm_parser.add_argument("--context", help="Context variables in JSON format")
    llm_parser.add_argument("--model-type", choices=["openai", "mlx", "hybrid"], default="openai", help="LLM backend to use")
    llm_parser.add_argument("--model-name", default="gpt-4", help="Model name to use")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create CLI instance
    cli = LlamaFarmAICLI(api_url=args.api_url, token=args.token)
    
    # Execute command
    start_time = time.time()
    
    if args.command == "login":
        result = cli.login(args.username, args.password)
    
    elif args.command == "health":
        result = cli.health_check()
    
    elif args.command == "classify-crop":
        result = cli.classify_crop(args.image, backend=args.backend, device=args.device)
    
    elif args.command == "detect-disease":
        result = cli.detect_disease(args.image, backend=args.backend, device=args.device)
    
    elif args.command == "detect-weeds":
        result = cli.detect_weeds(args.image, backend=args.backend, device=args.device)
    
    elif args.command == "ask":
        # Parse context JSON if provided
        context = None
        if args.context:
            try:
                context = json.loads(args.context)
            except json.JSONDecodeError:
                logger.error("Invalid JSON format for context")
                sys.exit(1)
        
        result = cli.query_llm(
            args.query,
            prompt_template=args.template,
            context=context,
            model_type=args.model_type,
            model_name=args.model_name
        )
    
    else:
        # Default to help if no command specified
        parser.print_help()
        sys.exit(0)
    
    elapsed_time = time.time() - start_time
    
    # Print result
    if "error" in result:
        logger.error(f"Command failed: {result['error']}")
        sys.exit(1)
    
    # Special handling for LLM query
    if args.command == "ask" and "text" in result:
        print(f"\n{result['text']}\n")
        print(f"Model: {result['model']} ({result['type']})")
        print(f"Time: {elapsed_time:.2f}s")
    else:
        # Print formatted JSON result
        print(format_json(result))


if __name__ == "__main__":
    main() 