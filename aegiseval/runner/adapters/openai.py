# SPDX-License-Identifier: Apache-2.0

"""Adapter for OpenAI models."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from aegiseval.runner.exceptions import (
    AuthenticationError, RateLimitError, ServiceUnavailableError, 
    InvalidRequestError, QuotaExceededError
)
from .base_http import AsyncHTTPAdapter


class OpenAIAdapter(AsyncHTTPAdapter):
    """Adapter for OpenAI models.

    This adapter connects to the OpenAI API to generate completions.

    Attributes:
        model: The OpenAI model identifier to use (e.g., "gpt-4o").
        _BASE_URL: The base URL for the OpenAI API.
        _HEADERS: The headers to use for API requests.
    """

    _BASE_URL = "https://api.openai.com/v1/chat/completions"
    _HEADERS = {}

    def __init__(self, model: str = "gpt-4o", system_prompt: Optional[str] = None, 
                 api_key: Optional[str] = None, organization: Optional[str] = None,
                 rpm_limit: Optional[int] = None, max_tokens: Optional[int] = None):
        """Initialize the OpenAI adapter.

        Args:
            model: The OpenAI model identifier to use.
            system_prompt: Optional system prompt to use with all requests.
            api_key: Optional API key (defaults to OPENAI_API_KEY env var).
            organization: Optional organization ID.
            rpm_limit: Optional rate limit in requests per minute.
            max_tokens: Maximum number of tokens to generate in responses.

        Raises:
            AuthenticationError: If the OPENAI_API_KEY environment variable is not set.
        """
        # Get API key from args or env
        key = api_key or os.getenv("OPENAI_API_KEY")
        if key is None:
            raise AuthenticationError("OPENAI_API_KEY not found in environment or config")
        
        # Initialize with base class constructor first
        super().__init__(model)
        
        # Store configuration
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.rpm_limit = rpm_limit
        
        # Set class headers to include API key
        headers = {"Authorization": f"Bearer {key}"}
        if organization:
            headers["OpenAI-Organization"] = organization
            
        type(self)._HEADERS = headers

    async def _post(self, url: str, json: Dict[str, Any]) -> Dict[str, Any]:
        """Make a POST request to the API and handle errors.

        Args:
            url: The URL to send the request to.
            json: The JSON payload to send.

        Returns:
            The JSON response from the API.

        Raises:
            AuthenticationError: If the API key is invalid.
            RateLimitError: If rate limits are hit.
            ServiceUnavailableError: If the service is unavailable.
            InvalidRequestError: If the request is invalid.
        """
        try:
            return await super()._post(url, json)
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            
            # Try to get error details from response
            error_message = str(e)
            try:
                error_data = e.response.json()
                if "error" in error_data and "message" in error_data["error"]:
                    error_message = error_data["error"]["message"]
            except (ValueError, KeyError):
                pass
                
            if status_code == 401:
                raise AuthenticationError(f"Authentication failed: {error_message}")
            elif status_code == 429:
                raise RateLimitError(f"Rate limit exceeded: {error_message}")
            elif status_code >= 500:
                raise ServiceUnavailableError(f"Service unavailable: {error_message}")
            elif status_code == 400:
                raise InvalidRequestError(f"Invalid request: {error_message}")
            elif status_code == 402 or status_code == 403:
                raise QuotaExceededError(f"Quota exceeded: {error_message}")
            else:
                raise  # Re-raise other errors

    @retry(
        retry=retry_if_exception_type((RateLimitError, ServiceUnavailableError)),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        stop=stop_after_attempt(5)
    )
    async def _generate_one(self, prompt: str) -> str:
        """Generate a completion for a single prompt.

        Args:
            prompt: The prompt string to generate a completion for.

        Returns:
            The generated completion string.
        """
        # Prepare the messages list
        messages = []
        
        # Add system message if provided
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Prepare the request payload
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2
        }
        
        # Add max_tokens if specified
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        else:
            payload["max_tokens"] = 1024  # Default
            
        data = await self._post(self._BASE_URL, payload)
        return str(data["choices"][0]["message"]["content"]) 

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose() 