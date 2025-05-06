# SPDX-License-Identifier: Apache-2.0

"""Adapter for Google Gemini models."""

from __future__ import annotations

import os
from typing import cast, Optional, Dict, Any

from .base_http import AsyncHTTPAdapter


class GeminiAdapter(AsyncHTTPAdapter):
    """Adapter for Google Gemini models.

    This adapter connects to the Google Generative Language API to generate completions.

    Attributes:
        model: The Gemini model identifier to use (e.g., "gemini-pro").
        _BASE_URL: The base URL for the Gemini API with a format placeholder for the model.
        _HEADERS: The headers to use for API requests.
        system_prompt: Optional system prompt to use with all requests.
    """

    _BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    _HEADERS = {}

    def __init__(self, model: str = "gemini-pro", system_prompt: Optional[str] = None,
                 rpm_limit: Optional[int] = None, max_tokens: Optional[int] = None):
        """Initialize the Gemini adapter.

        Args:
            model: The Gemini model identifier to use.
            system_prompt: Optional system prompt to use with all requests.
            rpm_limit: Optional rate limit in requests per minute.
            max_tokens: Maximum number of tokens to generate in responses.

        Raises:
            RuntimeError: If the GOOGLE_API_KEY environment variable is not set.
        """
        key = os.getenv("GOOGLE_API_KEY")
        if key is None:
            raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
        
        # Initialize with base class constructor first
        super().__init__(model)
        
        # Store configuration
        self.system_prompt = system_prompt
        self.rpm_limit = rpm_limit
        self.max_tokens = max_tokens
        
        # Set class headers to include API key
        type(self)._HEADERS = {"Content-Type": "application/json"}
        self._api_key = key

    async def _generate_one(self, prompt: str) -> str:
        """Generate a completion for a single prompt.

        Args:
            prompt: The prompt string to generate a completion for.

        Returns:
            The generated completion string.
        """
        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2}
        }
        
        # Add system instruction if provided
        if self.system_prompt:
            payload["system_instruction"] = {"parts": [{"text": self.system_prompt}]}
        
        # Add max_tokens if specified
        if self.max_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = self.max_tokens
        
        # Format URL with model and append API key
        url = f"{self._BASE_URL.format(model=self.model)}?key={self._api_key}"
        
        data = await self._post(url, payload)
        return cast(str, data["candidates"][0]["content"]["parts"][0]["text"]) 

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose() 