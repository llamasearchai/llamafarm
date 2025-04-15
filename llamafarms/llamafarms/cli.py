#!/usr/bin/env python3
"""
LlamaFarms Command Line Interface

This CLI tool provides access to LlamaFarms functionality directly from the terminal,
enabling users to analyze images, query the AI assistant, and access other features.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
from . import logger


class LlamaFarmsCLI:
    """Command-line interface for LlamaFarms."""

    def __init__(self):
        """Initialize the CLI tool."""
        self.parser = self._create_parser()

    def _create_parser(self):
        """Create the argument parser."""
        # Create main parser
        parser = argparse.ArgumentParser(
            description="LlamaFarms - Advanced Precision Agriculture Platform",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        parser.add_argument(
            "--version", action="store_true", help="Show version information"
        )
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")

        # Create subparsers for commands
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Vision commands
        vision_parser = subparsers.add_parser("vision", help="Computer vision tools")
        vision_subparsers = vision_parser.add_subparsers(
            dest="vision_command", help="Vision command"
        )

        # Crop classification
        classify_parser = vision_subparsers.add_parser(
            "classify", help="Classify crop in image"
        )
        classify_parser.add_argument("image", type=str, help="Path to image file")
        classify_parser.add_argument(
            "--backend",
            choices=["mlx", "torch"],
            default="mlx",
            help="ML backend to use",
        )

        # Disease detection
        disease_parser = vision_subparsers.add_parser(
            "disease", help="Detect diseases in plant image"
        )
        disease_parser.add_argument("image", type=str, help="Path to image file")
        disease_parser.add_argument(
            "--backend",
            choices=["yolo", "mlx"],
            default="yolo",
            help="ML backend to use",
        )

        # Weed detection
        weed_parser = vision_subparsers.add_parser(
            "weed", help="Detect weeds in field image"
        )
        weed_parser.add_argument("image", type=str, help="Path to image file")
        weed_parser.add_argument(
            "--backend",
            choices=["sam", "yolo", "mlx"],
            default="yolo",
            help="ML backend to use",
        )

        # LLM commands
        llm_parser = subparsers.add_parser("llm", help="Language model tools")
        llm_subparsers = llm_parser.add_subparsers(
            dest="llm_command", help="LLM command"
        )

        # Ask question
        ask_parser = llm_subparsers.add_parser(
            "ask", help="Ask a question to the agricultural assistant"
        )
        ask_parser.add_argument("question", type=str, help="Question to ask")
        ask_parser.add_argument(
            "--model",
            choices=["gpt-4", "mlx-local"],
            default="gpt-4",
            help="Model to use",
        )

        # Irrigation commands
        irrigation_parser = subparsers.add_parser(
            "irrigation", help="Irrigation optimization tools"
        )
        irrigation_subparsers = irrigation_parser.add_subparsers(
            dest="irrigation_command", help="Irrigation command"
        )

        # Optimize irrigation
        optimize_parser = irrigation_subparsers.add_parser(
            "optimize", help="Optimize irrigation schedule"
        )
        optimize_parser.add_argument(
            "config", type=str, help="Path to irrigation config file (JSON)"
        )

        # Models commands
        models_parser = subparsers.add_parser("models", help="Model management tools")
        models_subparsers = models_parser.add_subparsers(
            dest="models_command", help="Models command"
        )

        # List models
        list_parser = models_subparsers.add_parser("list", help="List available models")
        list_parser.add_argument("--type", type=str, help="Filter by model type")

        # API commands
        api_parser = subparsers.add_parser("api", help="API server management")
        api_subparsers = api_parser.add_subparsers(
            dest="api_command", help="API command"
        )

        # Start API server
        start_parser = api_subparsers.add_parser("start", help="Start the API server")
        start_parser.add_argument(
            "--host", type=str, default="0.0.0.0", help="Host to bind to"
        )
        start_parser.add_argument(
            "--port", type=int, default=8000, help="Port to listen on"
        )

        return parser

    def run(self, args=None):
        """Run the CLI with the given arguments."""
        args = self.parser.parse_args(args)

        # Set up debug logging if requested
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Show version and exit
        if args.version:
            from . import __version__

            print(f"LlamaFarms version {__version__}")
            return 0

        # Handle commands
        if args.command == "vision":
            return self._handle_vision_command(args)
        elif args.command == "llm":
            return self._handle_llm_command(args)
        elif args.command == "irrigation":
            return self._handle_irrigation_command(args)
        elif args.command == "models":
            return self._handle_models_command(args)
        elif args.command == "api":
            return self._handle_api_command(args)
        else:
            # No command specified, show help
            self.parser.print_help()
            return 0

    def _handle_vision_command(self, args):
        """Handle vision commands."""
        if args.vision_command == "classify":
            return self._classify_crop(args.image, args.backend)
        elif args.vision_command == "disease":
            return self._detect_disease(args.image, args.backend)
        elif args.vision_command == "weed":
            return self._detect_weeds(args.image, args.backend)
        else:
            print("Please specify a vision command")
            return 1

    def _handle_llm_command(self, args):
        """Handle LLM commands."""
        if args.llm_command == "ask":
            return self._ask_question(args.question, args.model)
        else:
            print("Please specify an LLM command")
            return 1

    def _handle_irrigation_command(self, args):
        """Handle irrigation commands."""
        if args.irrigation_command == "optimize":
            return self._optimize_irrigation(args.config)
        else:
            print("Please specify an irrigation command")
            return 1

    def _handle_models_command(self, args):
        """Handle models commands."""
        if args.models_command == "list":
            return self._list_models(args.type)
        else:
            print("Please specify a models command")
            return 1

    def _handle_api_command(self, args):
        """Handle API commands."""
        if args.api_command == "start":
            return self._start_api_server(args.host, args.port)
        else:
            print("Please specify an API command")
            return 1

    def _classify_crop(self, image_path, backend):
        """Classify crop in an image."""
        logger.info(f"Classifying crop in {image_path} using {backend} backend")

        try:
            from .core.ai.vision import CropClassifier

            # Check if image exists
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return 1

            # Create classifier
            classifier = CropClassifier(backend=backend)

            # Load image
            import numpy as np
            from PIL import Image

            img = Image.open(image_path)
            img_array = np.array(img)

            # Run classification
            start_time = time.time()
            result = classifier.predict(img_array)
            elapsed = time.time() - start_time

            # Print result
            print(f"\nCrop Classification Result:")
            print(f"- Class: {result['class_name']}")
            print(f"- Confidence: {result['confidence']:.2f}")
            print(f"- Processing Time: {elapsed:.2f} seconds\n")

            return 0

        except ImportError as e:
            logger.error(f"Required module not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error classifying crop: {e}")
            return 1

    def _detect_disease(self, image_path, backend):
        """Detect diseases in a plant image."""
        logger.info(f"Detecting diseases in {image_path} using {backend} backend")

        try:
            from .core.ai.vision import PlantDiseaseDetector

            # Check if image exists
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return 1

            # Create detector
            detector = PlantDiseaseDetector(backend=backend)

            # Load image
            import numpy as np
            from PIL import Image

            img = Image.open(image_path)
            img_array = np.array(img)

            # Run detection
            start_time = time.time()
            result = detector.detect(img_array)
            elapsed = time.time() - start_time

            # Print result
            print(f"\nDisease Detection Result:")
            print(f"- Number of detections: {result['count']}")

            for i, detection in enumerate(result["detections"]):
                print(f"\nDetection {i+1}:")
                print(f"- Disease: {detection['class_name']}")
                print(f"- Confidence: {detection['confidence']:.2f}")
                print(f"- Location: {detection['bbox']}")

            print(f"\n- Processing Time: {elapsed:.2f} seconds\n")

            return 0

        except ImportError as e:
            logger.error(f"Required module not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error detecting diseases: {e}")
            return 1

    def _detect_weeds(self, image_path, backend):
        """Detect weeds in a field image."""
        logger.info(f"Detecting weeds in {image_path} using {backend} backend")

        # Placeholder for weed detection implementation
        print("Weed detection not yet implemented")
        return 0

    def _ask_question(self, question, model):
        """Ask a question to the agricultural assistant."""
        logger.info(f"Asking question using {model} model")

        try:
            from .core.ai.llm import AgricultureLLM

            # Create LLM
            if model == "gpt-4":
                llm = AgricultureLLM(model_type="openai", model_name="gpt-4")
            else:
                llm = AgricultureLLM(model_type="mlx", model_name="mlx-local")

            # Ask question
            start_time = time.time()
            response = llm.ask(question)
            elapsed = time.time() - start_time

            # Print response
            if isinstance(response, dict):
                text = response.get("text", "No response")
            else:
                text = response

            print(f"\nResponse ({elapsed:.2f}s):\n")
            print(text)
            print()

            return 0

        except ImportError as e:
            logger.error(f"Required module not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return 1

    def _optimize_irrigation(self, config_path):
        """Optimize irrigation schedule."""
        logger.info(f"Optimizing irrigation using config {config_path}")

        # Placeholder for irrigation optimization implementation
        print("Irrigation optimization not yet implemented")
        return 0

    def _list_models(self, model_type):
        """List available models."""
        logger.info(f"Listing models{' of type ' + model_type if model_type else ''}")

        try:
            from .core.ai.model_registry import list_models

            models = list_models(model_type)

            print(f"\nAvailable Models ({len(models)}):\n")

            for model in models:
                print(f"- {model['name']} v{model['version']} ({model['model_type']})")
                print(f"  {model['description']}")
                print(f"  Loaded: {'Yes' if model['is_loaded'] else 'No'}")
                print()

            return 0

        except ImportError as e:
            logger.error(f"Required module not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return 1

    def _start_api_server(self, host, port):
        """Start the API server."""
        logger.info(f"Starting API server on {host}:{port}")

        try:
            from .api.server import start_server

            print(f"Starting LlamaFarms API server on {host}:{port}...")
            print("Press Ctrl+C to stop")

            start_server(host=host, port=port)

            return 0

        except ImportError as e:
            logger.error(f"Required module not found: {e}")
            return 1
        except Exception as e:
            logger.error(f"Error starting API server: {e}")
            return 1


def main():
    """Entry point for the CLI."""
    cli = LlamaFarmsCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
