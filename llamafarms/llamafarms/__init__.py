"""
LlamaFarms: Advanced Precision Agriculture Platform

LlamaFarms provides cutting-edge AI tools for modern farming, including computer vision
for crop health monitoring, weather prediction, irrigation optimization, and LLM-powered
assistance for agricultural decision-making.

Built with Apple's MLX framework for hardware acceleration on Apple Silicon, LlamaFarms
delivers high-performance AI capabilities optimized for agricultural applications.
"""

__version__ = "0.1.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai" = "Nik Jois"
__email__ = "nikjois@llamasearch.ai" = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "MIT"

# Set up package logging
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO if os.environ.get("LLAMAFARMS_ENV") != "development" else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("llamafarms")

# Import key components for easier access
try:
    from .core.ai import llm, mlx_integration, vision
except ImportError as e:
    logger.warning(f"Could not import some core modules: {e}")
    # Allow partial imports when not all dependencies are available

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Environment variables must be set manually.") 