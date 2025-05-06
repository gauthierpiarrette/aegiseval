# SPDX-License-Identifier: Apache-2.0

"""Adapters factory for different LLM providers."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, cast

from .base_http import ModelAdapter
from .openai import OpenAIAdapter
from .anthropic import AnthropicAdapter
from .gemini import GeminiAdapter
from .huggingface import HuggingFaceAdapter


class MockAdapter(ModelAdapter):
    """Mock adapter for testing without API calls."""
    
    def __init__(self, model: str, responses: Optional[Dict[str, str]] = None):
        """Initialize mock adapter.
        
        Args:
            model: Model identifier
            responses: Optional dictionary mapping prompts to responses
        """
        self.model = model
        self.responses = responses or {}
    
    async def generate(self, 
                      prompts: Sequence[str], 
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


# Mapping of scheme to adapter class
ADAPTERS: Dict[str, Any] = {
    "openai": OpenAIAdapter,
    "claude": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "hf": HuggingFaceAdapter,
}


def create_adapter(uri: str, **kwargs: Any) -> ModelAdapter:
    """Create appropriate adapter based on model URI.
    
    The URI format is: scheme:model
    If no scheme is provided, the URI is assumed to be an OpenAI model.
    
    Examples:
        "gpt-4o" -> OpenAIAdapter("gpt-4o")
        "gemini:gemini-1.5-pro" -> GeminiAdapter("gemini-1.5-pro")
        "claude:claude-3-opus-20240229" -> AnthropicAdapter("claude-3-opus-20240229")
        "hf:mistralai/Mistral-7B-Instruct-v0.1" -> HuggingFaceAdapter("mistralai/Mistral-7B-Instruct-v0.1")
        
    Args:
        uri: The URI identifying the model.
        **kwargs: Additional arguments to pass to the adapter constructor.
        
    Returns:
        Initialized model adapter.
        
    Raises:
        ValueError: If the scheme is not supported.
    """
    # Special case for backward compatibility with existing code
    if uri == "mock":
        return MockAdapter(uri, **kwargs)
    
    # Handle GPT models directly for backward compatibility
    if uri.startswith(("gpt-", "ft:gpt-")):
        return OpenAIAdapter(uri, **kwargs)
    
    # Parse URI in the format "scheme:model"
    scheme, _, model = uri.partition(":")
    if not _:
        scheme, model = "openai", scheme
    
    if scheme not in ADAPTERS:
        raise ValueError(f"Unsupported scheme: {scheme}")
    
    # Handle unsupported OpenAI models with a specific error
    if scheme == "openai" and not model.startswith(("gpt-", "ft:gpt-")):
        raise ValueError(f"No adapter available for model: {model}")
    
    adapter_class = ADAPTERS[scheme]
    return cast(ModelAdapter, adapter_class(model, **kwargs))


# For backward compatibility
get_adapter = create_adapter 