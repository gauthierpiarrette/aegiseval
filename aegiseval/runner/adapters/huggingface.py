# SPDX-License-Identifier: Apache-2.0

"""Adapter for Hugging Face Inference API."""

from __future__ import annotations

import os
from typing import List

from .base_http import AsyncHTTPAdapter


class HuggingFaceAdapter(AsyncHTTPAdapter):
    """Adapter for Hugging Face Inference API.

    This adapter connects to the Hugging Face Inference API to generate completions.

    Attributes:
        model: The Hugging Face model identifier to use.
        _BASE_URL: The base URL for the Hugging Face Inference API.
        _HEADERS: The headers to use for API requests.
    """

    _BASE_URL = "https://api-inference.huggingface.co/models/{model}"
    _HEADERS = {}

    def __init__(self, model: str):
        """Initialize the Hugging Face adapter.

        Args:
            model: The Hugging Face model identifier to use.

        Raises:
            RuntimeError: If the HF_TOKEN environment variable is not set.
        """
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise RuntimeError("HF_TOKEN environment variable is not set")
        
        # Initialize with base class constructor first
        super().__init__(model)
        
        # Set class headers to include API token
        type(self)._HEADERS = {"Authorization": f"Bearer {token}"}

    async def _generate_one(self, prompt: str) -> str:
        """Generate a completion for a single prompt.

        Args:
            prompt: The prompt string to generate a completion for.

        Returns:
            The generated completion string.
        """
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": 0.2
            }
        }
        data = await self._post(self._BASE_URL.format(model=self.model), payload)
        # The Hugging Face Inference API returns a list where the first item is the response
        response_data = data if isinstance(data, List) else [data]
        return str(response_data[0]["generated_text"])

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()