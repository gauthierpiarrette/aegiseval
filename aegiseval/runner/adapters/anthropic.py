# SPDX-License-Identifier: Apache-2.0

"""Adapter for Anthropic Claude models."""

from __future__ import annotations

import os
from typing import cast, Optional, Dict, Any

from .base_http import AsyncHTTPAdapter


class AnthropicAdapter(AsyncHTTPAdapter):
    """Adapter for Anthropic Claude models.

    This adapter connects to the Anthropic API to generate completions.

    Attributes:
        model: The Claude model identifier to use (e.g., "claude-3-opus-20240229").
        _BASE_URL: The base URL for the Anthropic API.
        _HEADERS: The headers to use for API requests.
        system_prompt: Optional system prompt to use with all requests.
    """

    _BASE_URL = "https://api.anthropic.com/v1/messages"
    _HEADERS = {}

    def __init__(self, model: str = "claude-3-sonnet-20240229", system_prompt: Optional[str] = None,
                 rpm_limit: Optional[int] = None, max_tokens: Optional[int] = None):
        """Initialize the Anthropic Claude adapter.

        Args:
            model: The Claude model identifier to use.
            system_prompt: Optional system prompt to use with all requests.
            rpm_limit: Optional rate limit in requests per minute.
            max_tokens: Maximum number of tokens to generate in responses.

        Raises:
            RuntimeError: If the ANTHROPIC_API_KEY environment variable is not set.
        """
        key = os.getenv("ANTHROPIC_API_KEY")
        if key is None:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
        
        # Initialize with base class constructor first
        super().__init__(model)
        
        # Store configuration
        self.system_prompt = system_prompt
        self.rpm_limit = rpm_limit
        self.max_tokens = max_tokens or 1024
        
        # Set class headers to include API key and version
        type(self)._HEADERS = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01"
        }

    async def _generate_one(self, prompt: str) -> str:
        """Generate a completion for a single prompt.

        Args:
            prompt: The prompt string to generate a completion for.

        Returns:
            The generated completion string.
        """
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        # Add system prompt if provided
        if self.system_prompt:
            payload["system"] = self.system_prompt
            
        data = await self._post(self._BASE_URL, payload)
        return cast(str, data["content"][0]["text"]) 

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose() 