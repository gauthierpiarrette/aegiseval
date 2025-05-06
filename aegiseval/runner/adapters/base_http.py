# SPDX-License-Identifier: Apache-2.0

"""Base HTTP adapter for LLM APIs."""

from __future__ import annotations

import asyncio
from typing import Protocol, Sequence, ClassVar, Any, cast

import httpx


class ModelAdapter(Protocol):
    """Protocol defining the interface for model adapters."""

    model: str

    async def generate(self, prompts: Sequence[str]) -> list[str]:
        """Generate completions for the given prompts.

        Args:
            prompts: A sequence of prompt strings to generate completions for.

        Returns:
            A list of completion strings, one for each prompt.
        """
        ...

    async def close(self) -> None:
        """Close any resources used by the adapter."""
        ...


class AsyncHTTPAdapter(ModelAdapter):
    """Base class for HTTP-based model adapters.

    This class provides common functionality for making HTTP requests to LLM APIs.
    Subclasses must implement the _generate_one method and set the _BASE_URL class
    attribute.

    Attributes:
        model: The model identifier to use for generation.
        client: The HTTP client for making requests.
    """

    _BASE_URL: ClassVar[str]  # To be set by subclass
    _HEADERS: ClassVar[dict[str, str]] = {}

    def __init__(self, model: str, timeout: float = 30):
        """Initialize the adapter.

        Args:
            model: The model identifier to use for generation.
            timeout: The timeout for HTTP requests in seconds.
        """
        self.model = model
        self.client = httpx.AsyncClient(timeout=timeout)

    async def _post(self, url: str, json: dict[str, Any]) -> dict[str, Any]:
        """Make a POST request to the API.

        Args:
            url: The URL to send the request to.
            json: The JSON payload to send.

        Returns:
            The JSON response from the API.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        r = await self.client.post(url, json=json, headers=self._HEADERS)
        r.raise_for_status()
        return cast(dict[str, Any], r.json())

    async def generate(self, prompts: Sequence[str]) -> list[str]:
        """Generate completions for the given prompts.

        Args:
            prompts: A sequence of prompt strings to generate completions for.

        Returns:
            A list of completion strings, one for each prompt.
        """
        tasks = [self._generate_one(p) for p in prompts]
        return await asyncio.gather(*tasks)

    async def _generate_one(self, prompt: str) -> str:
        """Generate a completion for a single prompt.

        Args:
            prompt: The prompt string to generate a completion for.

        Returns:
            The generated completion string.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _generate_one") 