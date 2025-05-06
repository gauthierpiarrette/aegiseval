# SPDX-License-Identifier: Apache-2.0
"""Tests for the Anthropic adapter."""

import pytest
import pytest_httpx
from typing import Any
from aegiseval.runner.adapters import get_adapter


@pytest.mark.asyncio
async def test_anthropic_adapter(httpx_mock: pytest_httpx.HTTPXMock, monkeypatch: Any) -> None:
    """Test the Anthropic adapter with a mocked response."""
    # Set environment variables
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    
    # Mock HTTP response
    httpx_mock.add_response(
        json={
            "content": [
                {
                    "text": "world"
                }
            ]
        }
    )
    
    # Initialize adapter
    adapter = get_adapter("claude:demo-model")
    
    # Test generate method
    result = await adapter.generate(["hi"])
    
    # Assert expected response
    assert result == ["world"]
    
    # Close adapter to release resources
    await adapter.close() 