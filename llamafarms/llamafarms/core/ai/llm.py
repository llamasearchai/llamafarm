"""
LLM Integration Module

This module provides utilities for working with Large Language Models (LLMs)
for agricultural applications, including both API-based models like OpenAI's GPT
and local models running with MLX.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure logging
logger = logging.getLogger("llamafarms.core.ai.llm")

# Try to import optional dependencies
try:
    import openai

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI SDK not available. OpenAI models will not be accessible.")

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logger.warning("MLX not available. Local LLM inference will be limited.")

try:
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI as LangchainOpenAI
    from langchain.prompts import PromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("Langchain not available. Advanced LLM chains will be limited.")


class AgricultureLLM:
    """LLM interface specialized for agricultural applications."""

    def __init__(
        self,
        model_type: str = "openai",  # 'openai', 'mlx', 'hybrid'
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """
        Initialize an agriculture-specialized LLM.

        Args:
            model_type: Type of model to use
            model_name: Name of the model
            api_key: API key for hosted models
            model_path: Path to local model weights
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Set up API key
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")

        # Initialize model based on type
        if model_type == "openai":
            if not OPENAI_AVAILABLE:
                raise ImportError(
                    "OpenAI SDK not available. Please install openai package."
                )
            self._init_openai_model()
        elif model_type == "mlx":
            if not MLX_AVAILABLE:
                raise ImportError("MLX not available. Please install mlx package.")
            self._init_mlx_model(model_path)
        elif model_type == "hybrid":
            self._init_hybrid_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load agriculture-specific prompts
        self.prompts = self._load_agriculture_prompts()

        logger.info(f"Initialized AgricultureLLM with {model_type} model: {model_name}")

    def _init_openai_model(self):
        """Initialize OpenAI model."""
        if not self.api_key:
            raise ValueError("OpenAI API key is required for OpenAI models")

        # Configure OpenAI client
        openai.api_key = self.api_key

        # Validate model name
        valid_models = ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
        if self.model_name not in valid_models:
            logger.warning(f"Model {self.model_name} may not be supported by OpenAI")

        logger.info(f"Initialized OpenAI model: {self.model_name}")

    def _init_mlx_model(self, model_path: Optional[str] = None):
        """Initialize MLX-based local LLM."""
        # This is a placeholder for MLX LLM initialization
        # In a real implementation, this would load a model from the specified path
        logger.warning("MLX LLM support is limited and experimental")

        if not model_path:
            # Use default model path
            model_path = os.path.join(
                os.path.dirname(__file__), "../../../models/llm/mlx_llm.bin"
            )
            if not os.path.exists(model_path):
                logger.warning(f"Default model not found at {model_path}")

        # Placeholder for model initialization
        self.mlx_model = None
        logger.info(f"Initialized MLX model placeholder")

    def _init_hybrid_model(self):
        """Initialize hybrid model approach (local + API)."""
        # Initialize both OpenAI and MLX models if available
        try:
            self._init_openai_model()
            logger.info("Initialized OpenAI component of hybrid model")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI component: {e}")

        try:
            self._init_mlx_model()
            logger.info("Initialized MLX component of hybrid model")
        except Exception as e:
            logger.error(f"Failed to initialize MLX component: {e}")

    def _load_agriculture_prompts(self) -> Dict[str, str]:
        """Load agriculture-specific prompts."""
        # Default prompts
        prompts = {
            "crop_advice": (
                "You are an expert agricultural advisor. "
                "Provide detailed advice for growing {crop} in {climate} conditions "
                "with {soil_type} soil. Consider water requirements, common pests, "
                "and optimal fertilization."
            ),
            "disease_identification": (
                "You are an expert in plant pathology. "
                "Based on the following symptoms observed in {crop}: {symptoms}, "
                "identify the most likely diseases, their causes, and recommended treatments."
            ),
            "irrigation_planning": (
                "You are an irrigation specialist. "
                "Create an optimal irrigation plan for {crop} in {region} during {season}. "
                "Consider the following factors: {factors}."
            ),
            "market_analysis": (
                "You are an agricultural economist. "
                "Analyze the current and projected market conditions for {crop} "
                "in {region} for the next {timeframe}. Consider supply, demand, "
                "price trends, and external factors."
            ),
            "sustainable_practices": (
                "You are a sustainable agriculture expert. "
                "Recommend sustainable farming practices for {crop} production "
                "that can reduce environmental impact while maintaining yield. "
                "Consider {specific_concerns} as particular areas of focus."
            ),
        }

        # Try to load custom prompts if available
        try:
            prompts_path = os.path.join(
                os.path.dirname(__file__),
                "../../../data/prompts/agriculture_prompts.json",
            )
            if os.path.exists(prompts_path):
                with open(prompts_path, "r") as f:
                    custom_prompts = json.load(f)
                prompts.update(custom_prompts)
                logger.info(f"Loaded custom agriculture prompts from {prompts_path}")
        except Exception as e:
            logger.warning(f"Failed to load custom prompts: {e}")

        return prompts

    def ask(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Union[str, Dict[str, Any]]:
        """
        Ask a question to the agricultural LLM.

        Args:
            query: The question or prompt
            context: Additional context for the query

        Returns:
            Response from the LLM
        """
        context = context or {}

        # Log the query
        logger.info(f"LLM query: {query[:50]}...")

        # Process based on model type
        if self.model_type == "openai":
            return self._ask_openai(query, context)
        elif self.model_type == "mlx":
            return self._ask_mlx(query, context)
        elif self.model_type == "hybrid":
            return self._ask_hybrid(query, context)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def _ask_openai(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask a question using OpenAI API."""
        try:
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert agricultural assistant specializing in precision farming, crop management, and sustainable agriculture.",
                },
                {"role": "user", "content": query},
            ]

            # Add context if provided
            if context:
                context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
                messages[0]["content"] += f"\n\nAdditional context:\n{context_str}"

            # Call API
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Process response
            result = {
                "text": response.choices[0].message.content,
                "model": self.model_name,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens,
                },
                "processing_time": time.time() - start_time,
            }

            logger.info(f"OpenAI response received in {result['processing_time']:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Error querying OpenAI: {e}")
            return {
                "text": f"Error: {str(e)}",
                "error": str(e),
                "model": self.model_name,
            }

    def _ask_mlx(self, query: str, context: Dict[str, Any]) -> str:
        """Ask a question using local MLX model."""
        # This is a placeholder for MLX LLM inference
        logger.warning("MLX LLM inference is not fully implemented")

        # Return a placeholder response
        return f"[MLX Model Response] Query: {query}\nThis is a placeholder response from the MLX model."

    def _ask_hybrid(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask a question using hybrid approach."""
        # Determine which model to use based on query complexity, context, etc.
        # This is a simplified implementation

        # Use local model for simple queries, OpenAI for complex ones
        query_length = len(query.split())
        has_complex_terms = any(
            term in query.lower()
            for term in [
                "analyze",
                "compare",
                "evaluate",
                "synthesize",
                "recommend",
                "forecast",
            ]
        )

        if query_length < 20 and not has_complex_terms:
            # Use local model for simple queries
            try:
                response = self._ask_mlx(query, context)
                return {
                    "text": response,
                    "model": f"mlx-{self.model_name}",
                    "source": "local",
                }
            except Exception as e:
                logger.warning(f"Local model failed, falling back to OpenAI: {e}")
                # Fall back to OpenAI
                return self._ask_openai(query, context)
        else:
            # Use OpenAI for complex queries
            return self._ask_openai(query, context)

    def use_template(self, template_name: str, **kwargs) -> Union[str, Dict[str, Any]]:
        """
        Use a predefined prompt template.

        Args:
            template_name: Name of the template to use
            **kwargs: Variables to fill in the template

        Returns:
            Response from the LLM
        """
        if template_name not in self.prompts:
            raise ValueError(f"Template '{template_name}' not found")

        # Fill template
        template = self.prompts[template_name]
        try:
            prompt = template.format(**kwargs)
        except KeyError as e:
            raise ValueError(
                f"Missing required parameter for template '{template_name}': {e}"
            )

        # Send to LLM
        return self.ask(prompt)

    def analyze_crop_health(
        self, crop: str, symptoms: List[str], region: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze crop health based on symptoms.

        Args:
            crop: Type of crop
            symptoms: List of observed symptoms
            region: Optional growing region for context

        Returns:
            Analysis results
        """
        # Format symptoms
        symptoms_text = ", ".join(symptoms)

        # Create context
        context = {"crop": crop}
        if region:
            context["region"] = region

        # Use disease identification template
        return self.use_template(
            "disease_identification", crop=crop, symptoms=symptoms_text
        )

    def get_growing_advice(
        self, crop: str, climate: str, soil_type: str
    ) -> Dict[str, Any]:
        """
        Get advice for growing a specific crop.

        Args:
            crop: Type of crop
            climate: Climate conditions
            soil_type: Type of soil

        Returns:
            Growing advice
        """
        return self.use_template(
            "crop_advice", crop=crop, climate=climate, soil_type=soil_type
        )

    def create_irrigation_plan(
        self, crop: str, region: str, season: str, factors: List[str]
    ) -> Dict[str, Any]:
        """
        Create an irrigation plan.

        Args:
            crop: Type of crop
            region: Growing region
            season: Growing season
            factors: List of factors to consider

        Returns:
            Irrigation plan
        """
        factors_text = ", ".join(factors)

        return self.use_template(
            "irrigation_planning",
            crop=crop,
            region=region,
            season=season,
            factors=factors_text,
        )


# Advanced LLM chains for agriculture
if LANGCHAIN_AVAILABLE:

    class AgricultureLLMChain:
        """Advanced LLM chains for agricultural applications using Langchain."""

        def __init__(
            self,
            llm_type: str = "openai",
            model_name: str = "gpt-3.5-turbo",
            api_key: Optional[str] = None,
        ):
            """
            Initialize an agriculture LLM chain.

            Args:
                llm_type: Type of LLM to use
                model_name: Name of the model
                api_key: API key for the model
            """
            self.llm_type = llm_type
            self.model_name = model_name

            # Set up API key
            if api_key:
                self.api_key = api_key
            else:
                self.api_key = os.environ.get("OPENAI_API_KEY")

            # Initialize LLM
            if llm_type == "openai":
                self.llm = LangchainOpenAI(
                    model_name=model_name, openai_api_key=self.api_key, temperature=0.7
                )
            else:
                raise ValueError(f"Unsupported LLM type: {llm_type}")

            logger.info(
                f"Initialized AgricultureLLMChain with {llm_type} model: {model_name}"
            )

        def create_crop_rotation_chain(self):
            """Create a chain for crop rotation planning."""
            template = """
            You are an expert in crop rotation planning.
            
            Current crops: {current_crops}
            Soil type: {soil_type}
            Region: {region}
            Goals: {goals}
            
            Create a 3-year crop rotation plan that will improve soil health, manage pests naturally,
            and optimize yields. Explain the benefits of each rotation.
            """

            prompt = PromptTemplate(
                input_variables=["current_crops", "soil_type", "region", "goals"],
                template=template,
            )

            return LLMChain(llm=self.llm, prompt=prompt)

        def create_pest_management_chain(self):
            """Create a chain for integrated pest management."""
            template = """
            You are an expert in integrated pest management for agriculture.
            
            Crop: {crop}
            Pests observed: {pests}
            Current growth stage: {growth_stage}
            Farming approach: {approach}
            
            Provide a detailed integrated pest management plan that prioritizes biological and cultural
            controls before chemical interventions. For each pest, recommend specific actions,
            timing, and expected outcomes.
            """

            prompt = PromptTemplate(
                input_variables=["crop", "pests", "growth_stage", "approach"],
                template=template,
            )

            return LLMChain(llm=self.llm, prompt=prompt)

else:
    # Placeholder if Langchain is not available
    class AgricultureLLMChain:
        """Placeholder for AgricultureLLMChain when Langchain is not available."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Langchain is required for AgricultureLLMChain. Please install langchain package."
            )
