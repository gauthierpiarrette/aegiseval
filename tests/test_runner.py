# SPDX-License-Identifier: Apache-2.0
"""Tests for the AegisEval runner module."""

import json
import pytest
import asyncio
from typing import Generator, Any

from aegiseval.runner.adapters import create_adapter, OpenAIAdapter
from aegiseval.runner.exceptions import AuthenticationError


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Mock environment variables required for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    yield
    # No teardown needed as monkeypatch will restore the environment


@pytest.fixture
def openai_adapter() -> Generator[OpenAIAdapter, None, None]:
    """Create an OpenAIAdapter instance for testing."""
    adapter = OpenAIAdapter("gpt-4")
    yield adapter
    # Ensure we close the client after the test
    asyncio.run(adapter.close())


@pytest.mark.skip(reason="Mocking issue with httpx - needs investigation")
@pytest.mark.asyncio
async def test_openai_adapter_generate(mock_env: None, httpx_mock: Any) -> None:
    """Test that OpenAIAdapter.generate processes prompts correctly."""
    # Set up test data
    test_prompts = [
        "What is the capital of France?",
        "How do you make pancakes?",
        "Explain quantum physics briefly."
    ]
    
    # Mock responses
    for i in range(len(test_prompts)):
        httpx_mock.add_response(
            method="POST",
            url="https://api.openai.com/v1/chat/completions",
            json={
                "id": f"chatcmpl-mock{i}",
                "object": "chat.completion",
                "created": 1630000000,
                "model": "gpt-4",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Mock response {i+1}"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30
                }
            },
            match_content=lambda content: json.loads(content)["messages"][0]["content"] == test_prompts[i]
        )
    
    # Initialize adapter and run test
    adapter = OpenAIAdapter("gpt-4")
    
    try:
        responses = await adapter.generate(test_prompts)
        
        # Assert the number of responses matches the number of prompts
        assert len(responses) == len(test_prompts), "Number of responses should match number of prompts"
        
        # Assert the responses contain the expected mock content
        for i, response in enumerate(responses):
            assert response == f"Mock response {i+1}"
            
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_create_adapter_returns_correct_adapter(mock_env: None) -> None:
    """Test that create_adapter returns the correct adapter type."""
    # Test with OpenAI model ID
    adapter = create_adapter("gpt-4")
    assert isinstance(adapter, OpenAIAdapter)
    await adapter.close()
    
    # Test with fine-tuned OpenAI model ID
    adapter = create_adapter("ft:gpt-3.5-turbo:org:model:id")
    assert isinstance(adapter, OpenAIAdapter)
    await adapter.close()
    
    # Test with unsupported model ID
    with pytest.raises(ValueError, match=r"No adapter available for model: .*"):
        create_adapter("unsupported-model")


@pytest.mark.asyncio
async def test_adapter_error_handling(mock_env: None, httpx_mock: Any) -> None:
    """Test OpenAIAdapter handling of API errors."""
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
    )
    
    adapter = OpenAIAdapter("gpt-4")
    
    try:
        with pytest.raises(AuthenticationError):
            await adapter.generate(["Test prompt"])
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_missing_api_key() -> None:
    """Test error when API key is missing."""
    with pytest.raises(AuthenticationError, match="OPENAI_API_KEY not found in environment or config"):
        OpenAIAdapter("gpt-4")  # No API key in environment 