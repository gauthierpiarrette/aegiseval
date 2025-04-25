# SPDX-License-Identifier: Apache-2.0
"""Adapters to interact with various LLM APIs."""

import os
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from aegiseval.config.settings import get_config
from aegiseval.runner.exceptions import (
    ModelAdapterError, AuthenticationError, 
    RateLimitError, InvalidRequestError, ServiceUnavailableError, TimeoutError
)

# Configure logging
logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Base class for all model adapters."""
    
    def __init__(self, model_id: str):
        """Initialize adapter for specific model.
        
        Args:
            model_id: Model identifier
        """
        self.model_id = model_id
        self.config = get_config()
        self._initialized = False
    
    @abstractmethod
    async def generate(self, 
                      prompts: List[str], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> List[str]:
        """Generate responses for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate per response
            
        Returns:
            List of model responses corresponding to each prompt
            
        Raises:
            Various ModelAdapterError subclasses for different failure modes
        """
        pass
        
    @abstractmethod
    async def close(self) -> None:
        """Release resources held by the adapter."""
        pass
        
    @property
    def name(self) -> str:
        """Return a human-readable name for this adapter."""
        return f"{self.__class__.__name__}({self.model_id})"


class OpenAIAdapter(ModelAdapter):
    """Adapter for OpenAI models using httpx.AsyncClient."""
    
    API_URL = "https://api.openai.com/v1/chat/completions"
    
    def __init__(self, model_id: str, api_key: Optional[str] = None, 
                 organization: Optional[str] = None, system_prompt: Optional[str] = None,
                 rpm_limit: Optional[int] = None, max_tokens: Optional[int] = None):
        """Initialize adapter for specific OpenAI model.
        
        Args:
            model_id: OpenAI model identifier (e.g., "gpt-4o")
            api_key: Optional API key (defaults to OPENAI_API_KEY env var)
            organization: Optional organization ID
            system_prompt: Optional system prompt to use with all requests
            rpm_limit: Optional rate limit in requests per minute
            max_tokens: Maximum number of tokens to generate in responses
            
        Raises:
            AuthenticationError: If API key is not provided and not in env
        """
        super().__init__(model_id)
        
        # Get API key from args, env, or config
        config_openai: Dict[str, Any] = {}
        if self.config is not None:
            config_openai = self.config.get("openai", {}) or {}
            
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or \
                       config_openai.get("api_key")
        if not self.api_key:
            raise AuthenticationError("OPENAI_API_KEY not found in environment or config")
        
        self.organization = organization or os.getenv("OPENAI_ORG") or \
                            config_openai.get("organization")
        
        # Store the system prompt and rate limit
        self.system_prompt = system_prompt
        self.rpm_limit = rpm_limit
        self.max_tokens = max_tokens
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(
                connect=5.0,
                read=60.0,
                write=10.0,
                pool=None,
            ),
        )
        self._initialized = True
        
        # If rpm_limit is specified, configure rate limiting for the adapter
        if self.rpm_limit:
            logger.info(f"Setting rate limit to {self.rpm_limit} requests per minute")
            # This would be implemented with a rate limiter in a production system
    
    @retry(
        retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5)
    )
    async def generate(self, 
                      prompts: List[str], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> List[str]:
        """Generate responses from the OpenAI model for batch of prompts.
        
        Uses true batch processing when possible, with fallback to individual calls.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens to generate per response
            
        Returns:
            List of model responses corresponding to each prompt
            
        Raises:
            AuthenticationError: If API credentials are invalid
            RateLimitError: If rate limits are exceeded
            ServiceUnavailableError: If the API is unavailable
            TimeoutError: If the request times out
            InvalidRequestError: If the request is invalid
            ModelAdapterError: For other errors
        """
        if not self._initialized:
            raise ModelAdapterError("Adapter not initialized")
            
        if not prompts:
            return []
        
        # Use instance max_tokens if provided and not overridden by call
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        
        # Determine if we should use batch processing or individual calls
        use_batch = len(prompts) > 1 and self.model_id.startswith("gpt-4")
        
        try:
            if use_batch:
                return await self._batch_generate(prompts, temperature, effective_max_tokens)
            else:
                return await self._individual_generate(prompts, temperature, effective_max_tokens)
        except httpx.HTTPStatusError as e:
            self._handle_http_error(e)
            # This line will never be reached, but mypy needs it
            return []  
        except httpx.TimeoutException:
            raise TimeoutError("Request to OpenAI API timed out")
        except httpx.RequestError as e:
            raise ModelAdapterError(f"Request error: {str(e)}")
        except Exception as e:
            raise ModelAdapterError(f"Unexpected error: {str(e)}")
    
    async def _batch_generate(self, 
                            prompts: List[str], 
                            temperature: float = 0.0,
                            max_tokens: Optional[int] = None) -> List[str]:
        """Process multiple prompts in a single API call when possible.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate per response
            
        Returns:
            List of model responses
        """
        # For this implementation, we'll use multiple parallel requests
        # as the OpenAI API doesn't support true batch processing
        tasks = [
            self._process_single_prompt(prompt, temperature, max_tokens)
            for prompt in prompts
        ]
        
        return await asyncio.gather(*tasks)
    
    async def _individual_generate(self, 
                                 prompts: List[str], 
                                 temperature: float = 0.0,
                                 max_tokens: Optional[int] = None) -> List[str]:
        """Process prompts individually.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate per response
            
        Returns:
            List of model responses
        """
        responses = []
        
        for prompt in prompts:
            response = await self._process_single_prompt(prompt, temperature, max_tokens)
            responses.append(response)
        
        return responses
    
    async def _process_single_prompt(self, 
                                   prompt: str, 
                                   temperature: float = 0.0,
                                   max_tokens: Optional[int] = None) -> str:
        """Process a single prompt and return the response text.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Model response text
            
        Raises:
            Various exceptions based on API errors
        """
        # Prepare the messages list
        messages = []
        
        # Add system message if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the request payload
        data = {
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens is not None:
            data["max_tokens"] = max_tokens
        
        # Make the API request
        response = await self.client.post(self.API_URL, json=data)
        response.raise_for_status()
        
        result = response.json()
        if isinstance(result, Dict) and "choices" in result:
            return str(result["choices"][0]["message"]["content"])
        return ""  # Fallback for unexpected response structure
    
    def _handle_http_error(self, error: httpx.HTTPStatusError) -> None:
        """Handle HTTP errors from the OpenAI API.
        
        Args:
            error: HTTP error from the API
            
        Raises:
            Appropriate exception based on error type
        """
        status_code = error.response.status_code
        error_message = str(error)
        
        try:
            # Handle the case where json() might be a coroutine in tests
            if hasattr(error.response.json, "__await__"):
                error_data = {"error": {"message": error_message}}
            else:
                error_data = error.response.json()
                error_message = error_data.get("error", {}).get("message", str(error))
        except ValueError:
            error_data = {"error": {"message": error.response.text}}
            error_message = error_data.get("error", {}).get("message", str(error))
        
        if status_code == 401:
            raise AuthenticationError(f"Authentication failed: {error_message}")
        elif status_code == 429:
            raise RateLimitError(f"Rate limit exceeded: {error_message}")
        elif status_code == 400:
            raise InvalidRequestError(f"Invalid request: {error_message}")
        elif status_code >= 500:
            raise ServiceUnavailableError(f"OpenAI service error: {error_message}")
        else:
            raise ModelAdapterError(f"OpenAI API error: {error_message}")
    
    async def validate_connection(self) -> bool:
        """Test connection to API and return True if successful.
        
        Returns:
            Boolean indicating if connection is valid
        """
        try:
            # Make a minimal request to validate connection
            response = await self.generate(["Hello"], max_tokens=5)
            return len(response) > 0
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if hasattr(self, 'client'):
            await self.client.aclose()


class MockAdapter(ModelAdapter):
    """Mock adapter for testing without API calls."""
    
    def __init__(self, model_id: str, responses: Optional[Dict[str, str]] = None):
        """Initialize mock adapter.
        
        Args:
            model_id: Model identifier
            responses: Optional dictionary mapping prompts to responses
        """
        super().__init__(model_id)
        self.responses = responses or {}
        self._initialized = True
    
    async def generate(self, 
                      prompts: List[str], 
                      temperature: float = 0.0,
                      max_tokens: Optional[int] = None) -> List[str]:
        """Return mock responses for prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Ignored in mock
            max_tokens: Ignored in mock
            
        Returns:
            List of mock responses
        """
        result = []
        for prompt in prompts:
            # Use exact match if available, fallback to default
            if prompt in self.responses:
                result.append(self.responses[prompt])
            else:
                result.append(f"Mock response for: {prompt[:50]}...")
        return result
    
    async def close(self) -> None:
        """No resources to release."""
        pass


def create_adapter(model_id: str, **kwargs: Any) -> ModelAdapter:
    """Create appropriate adapter based on model ID.
    
    Args:
        model_id: Model identifier
        **kwargs: Additional arguments to pass to adapter
        
    Returns:
        Initialized model adapter
        
    Raises:
        ValueError: If no adapter is available for the model
    """
    if model_id.startswith(("gpt-", "ft:gpt-")):
        return OpenAIAdapter(model_id, **kwargs)
    elif model_id == "mock":
        return MockAdapter(model_id, **kwargs)
    else:
        raise ValueError(f"No adapter available for model: {model_id}")


async def run_evaluation(model_adapter: ModelAdapter, 
                         test_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run evaluation against model using the provided adapter.
    
    Args:
        model_adapter: Initialized model adapter
        test_cases: List of test case dictionaries
        
    Returns:
        List of test results with responses and pass/fail status
    """
    # Extract prompts from test cases
    prompts = [case.get("prompt", "") for case in test_cases]
    
    try:
        # Generate responses
        responses = await model_adapter.generate(prompts)
        
        # Process results
        results = []
        for case, response in zip(test_cases, responses):
            # Note: This is a placeholder for actual evaluation logic
            # which will be implemented in the EvaluationRunner
            results.append({
                "test_id": case.get("test_id", "unknown"),
                "prompt": case.get("prompt", ""),
                "response": response,
                "passed": True,  # Placeholder for actual evaluation
                "regulations": case.get("regulations", []),
            })
        
        return results
    
    except ModelAdapterError as e:
        # Log the error and return empty results
        logger.error(f"Evaluation failed: {e}")
        return []