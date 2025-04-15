"""
Large Language Model Integration for Agricultural Applications

This module provides LLM capabilities specialized for agriculture, supporting
both cloud-based APIs (OpenAI) and on-device inference with MLX.
"""

import json
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Union

# Conditionally import dependencies
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI SDK not available. Cloud LLM features will be disabled.")

try:
    import mlx.core as mx
    import mlx.nn as nn

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    logging.warning("MLX not available. On-device LLM features will be limited.")

try:
    import langchain
    from langchain.chains import LLMChain
    from langchain.llms import OpenAI as LangchainOpenAI
    from langchain.prompts import PromptTemplate

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logging.warning(
        "Langchain not available. Advanced chain and agent features will be limited."
    )


# Configure logging
logger = logging.getLogger("llamafarm.ai.llm")


class AgricultureLLM:
    """
    LLM interface specialized for agricultural applications.
    Supports both cloud API calls and local inference with MLX.
    """

    def __init__(
        self,
        model_type: str = "openai",  # 'openai', 'mlx', 'hybrid'
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        local_model_path: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Initialize an agriculture-specialized LLM.

        Args:
            model_type: Type of model to use ('openai', 'mlx', 'hybrid')
            model_name: Name of the model to use
            api_key: API key for cloud services
            local_model_path: Path to local model weights
            temperature: Temperature for text generation
            max_tokens: Maximum tokens in generated responses
        """
        self.model_type = model_type
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.local_model_path = local_model_path

        # Initialize client based on model_type
        if model_type == "openai":
            self._init_openai()
        elif model_type == "mlx":
            self._init_mlx(local_model_path)
        elif model_type == "hybrid":
            self._init_openai()
            self._init_mlx(local_model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load specialized agriculture prompts
        self.prompts = self._load_agriculture_prompts()

    def _init_openai(self):
        """Initialize OpenAI client."""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI SDK required for OpenAI models. Install with 'pip install openai'"
            )

        if not self.api_key:
            raise ValueError("API key required for OpenAI models")

        logger.info(f"Initializing OpenAI client with model {self.model_name}")
        self.openai_client = OpenAI(api_key=self.api_key)

    def _init_mlx(self, model_path: Optional[str] = None):
        """Initialize MLX model for local inference."""
        if not MLX_AVAILABLE:
            raise ImportError(
                "MLX required for local models. Install with 'pip install mlx'"
            )

        logger.info("Initializing MLX model for local inference")

        # Placeholder for MLX LLM initialization
        # In a real implementation, this would load a language model optimized for MLX
        self.mlx_model = None
        self.mlx_tokenizer = None

        logger.info("MLX model initialized for local inference")

    def _load_agriculture_prompts(self) -> Dict[str, str]:
        """Load specialized agriculture prompts."""
        # These would typically be loaded from a file
        return {
            "crop_recommendation": """
            As an agricultural AI assistant, please analyze the following data about a farm:
            - Location: {location}
            - Soil type: {soil_type}
            - Soil pH: {soil_ph}
            - Local climate: {climate}
            - Available irrigation: {irrigation}
            
            Recommend the most suitable crops to plant this season, considering:
            1. Best fit for the soil and climate conditions
            2. Potential yield and market value
            3. Water requirements matching available irrigation
            4. Sustainability and environmental impact
            
            For each recommended crop, provide:
            - Optimal planting time
            - Expected growth duration
            - Key care requirements
            - Potential challenges and solutions
            """,
            "disease_diagnosis": """
            As an agricultural AI assistant, please analyze these symptoms observed in the {crop_type} crop:
            
            {symptoms}
            
            Additional context:
            - Growing region: {region}
            - Current growth stage: {growth_stage}
            - Recent weather conditions: {weather}
            - Any treatments already applied: {treatments}
            
            Provide a diagnosis of potential diseases or issues, including:
            1. Most likely causes ranked by probability
            2. Recommended treatments or interventions
            3. Preventive measures for the future
            4. Whether this requires urgent attention
            """,
            "irrigation_planning": """
            As an agricultural AI assistant, please help create an optimal irrigation schedule for:
            - Crop type: {crop_type}
            - Growth stage: {growth_stage}
            - Field size: {field_size} hectares
            - Soil type: {soil_type}
            - Current soil moisture: {soil_moisture}%
            - Weather forecast: {weather_forecast}
            - Available water: {available_water} cubic meters
            - Irrigation system: {irrigation_system}
            
            Create a 7-day irrigation plan that:
            1. Maintains optimal soil moisture for the crop
            2. Uses water efficiently
            3. Accounts for weather conditions
            4. Works with the existing irrigation system
            """,
            "market_analysis": """
            As an agricultural AI assistant, please analyze the market outlook for {crop_type} given the following:
            - Current season: {season}
            - Production region: {region}
            - Estimated yield: {yield_estimate} tons
            - Current market price: {current_price} per ton
            - Historical price trends: {price_trends}
            - Supply projections: {supply_projections}
            - Demand projections: {demand_projections}
            
            Provide insights on:
            1. Price forecasts for the next 3-6 months
            2. Optimal timing for selling the harvest
            3. Potential storage vs. immediate sale tradeoffs
            4. Alternative markets or value-added options to consider
            5. Risk factors that could impact price and how to mitigate them
            """,
        }

    def ask(
        self,
        query: str,
        prompt_template: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        stream: bool = False,
        use_local: Optional[bool] = None,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate a response to an agricultural query.

        Args:
            query: User's query
            prompt_template: Optional template name from self.prompts
            context: Variables to format the prompt template with
            stream: Whether to stream the response
            use_local: Override to use local model regardless of self.model_type

        Returns:
            Generated response as string or dict with metadata
        """
        # Determine whether to use local or cloud model
        use_local = (
            use_local
            if use_local is not None
            else (self.model_type in ["mlx", "hybrid"])
        )

        # Prepare the prompt
        if prompt_template and prompt_template in self.prompts:
            # Format specialized prompt with context
            context = context or {}
            prompt = self.prompts[prompt_template].format(**context)
            prompt = f"{prompt}\n\nQuery: {query}"
        else:
            # Use the query directly with a generic agriculture prefix
            prompt = (
                f"As an agricultural AI assistant, please answer the following: {query}"
            )

        # Generate response
        if use_local and MLX_AVAILABLE and self.mlx_model:
            return self._generate_local(prompt, stream)
        else:
            return self._generate_openai(prompt, stream)

    def _generate_openai(
        self, prompt: str, stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Generate response using OpenAI API."""
        if not OPENAI_AVAILABLE or not self.openai_client:
            raise RuntimeError("OpenAI client not available")

        logger.info(f"Generating response with OpenAI model: {self.model_name}")

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream,
            )

            if stream:
                # Return a generator for streaming
                def response_generator():
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return response_generator()
            else:
                # Return the complete response
                result = response.choices[0].message.content
                return {"text": result, "model": self.model_name, "type": "openai"}

        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}")
            raise

    def _generate_local(
        self, prompt: str, stream: bool = False
    ) -> Union[str, Dict[str, Any]]:
        """Generate response using local MLX model."""
        if not MLX_AVAILABLE or not self.mlx_model:
            raise RuntimeError("MLX model not available")

        logger.info("Generating response with local MLX model")

        # Placeholder for MLX-based text generation
        # In a real implementation, this would use MLX for inference

        # Simulate response for demonstration
        result = "This is a simulated response from a local MLX model."

        return {"text": result, "model": "mlx-local", "type": "mlx"}


class FarmAssistant:
    """
    AI assistant for farm management, powered by LLMs,
    with specialized tools for agricultural tasks.
    """

    def __init__(
        self,
        llm: Optional[AgricultureLLM] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        memory: bool = True,
    ):
        """
        Initialize a farm management assistant.

        Args:
            llm: AgricultureLLM instance
            tools: List of tool definitions
            memory: Whether to maintain conversation memory
        """
        self.llm = llm or AgricultureLLM()
        self.tools = tools or self._default_tools()
        self.memory = memory
        self.conversation_history = [] if memory else None

        # Initialize LangChain integration if available
        if LANGCHAIN_AVAILABLE:
            self._init_langchain()

    def _default_tools(self) -> List[Dict[str, Any]]:
        """Define default agricultural tools."""
        return [
            {
                "name": "weather_forecast",
                "description": "Get weather forecast for a specific location",
                "function": self._get_weather_forecast,
            },
            {
                "name": "crop_calendar",
                "description": "Get planting and harvesting dates for specific crops",
                "function": self._get_crop_calendar,
            },
            {
                "name": "market_prices",
                "description": "Get current market prices for agricultural products",
                "function": self._get_market_prices,
            },
            {
                "name": "disease_lookup",
                "description": "Look up information about crop diseases",
                "function": self._lookup_disease,
            },
        ]

    def _init_langchain(self):
        """Initialize LangChain components for advanced workflows."""
        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain not available, skipping initialization")
            return

        # Initialize LangChain components
        logger.info("Initializing LangChain components")

        # Example of setting up a chain for crop recommendations
        crop_rec_template = PromptTemplate(
            input_variables=["climate", "soil", "water"],
            template=self.llm.prompts["crop_recommendation"],
        )

        self.crop_chain = LLMChain(
            llm=LangchainOpenAI(
                temperature=self.llm.temperature,
                model=self.llm.model_name,
                openai_api_key=self.llm.api_key,
            ),
            prompt=crop_rec_template,
        )

    def chat(self, message: str) -> str:
        """
        Chat with the farm assistant.

        Args:
            message: User's message

        Returns:
            Assistant's response
        """
        # Add message to conversation history if memory is enabled
        if self.memory:
            self.conversation_history.append({"role": "user", "content": message})

        # Process the message to detect tool calls
        response = self._process_message(message)

        # Add response to conversation history if memory is enabled
        if self.memory:
            self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def _process_message(self, message: str) -> str:
        """Process user message and determine if tools need to be called."""
        # For a real implementation, this would use more sophisticated tool routing
        # based on the message content, possibly using an LLM itself to decide

        # For now, use a simple keyword approach
        for tool in self.tools:
            if tool["name"].replace("_", " ") in message.lower():
                logger.info(f"Tool detected: {tool['name']}")
                # Extract params (simplified)
                result = tool["function"](message)
                response = self.llm.ask(
                    f"Tool {tool['name']} returned: {result}. Respond to: {message}"
                )
                return response["text"] if isinstance(response, dict) else response

        # No tool needed, just use the LLM
        response = self.llm.ask(message)
        return response["text"] if isinstance(response, dict) else response

    # Tool implementation examples
    def _get_weather_forecast(self, query: str) -> str:
        """Get weather forecast for a location."""
        # In a real application, this would call a weather API
        return "Weather forecast: Sunny, 25°C, 5% chance of rain"

    def _get_crop_calendar(self, query: str) -> str:
        """Get planting and harvesting dates for crops."""
        # In a real application, this would look up a database
        return "Corn: Plant in April-May, harvest in September-October"

    def _get_market_prices(self, query: str) -> str:
        """Get current market prices for agricultural products."""
        # In a real application, this would query market data
        return "Corn: $175/ton, Wheat: $220/ton, Soybeans: $380/ton"

    def _lookup_disease(self, query: str) -> str:
        """Look up information about crop diseases."""
        # In a real application, this would query a disease database
        return "Corn leaf blight: Fungal disease, treat with fungicide, rotate crops"


# Utility decorator for MLX model caching
def mlx_model_cache(func: Callable) -> Callable:
    """Decorator to cache MLX model outputs for repeated inputs."""
    cache = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from args and kwargs
        key = str(args) + str(sorted(kwargs.items()))

        if key in cache:
            logger.debug("Using cached result for MLX model")
            return cache[key]

        result = func(*args, **kwargs)
        cache[key] = result
        return result

    return wrapper
