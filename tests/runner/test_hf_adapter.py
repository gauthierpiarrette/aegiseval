# SPDX-License-Identifier: Apache-2.0
"""Tests for the Hugging Face adapter."""

import pytest
import pytest_httpx
from typing import Any
from aegiseval.runner.adapters import get_adapter


@pytest.mark.asyncio
async def test_hf_adapter(httpx_mock: pytest_httpx.HTTPXMock, monkeypatch: Any) -> None:
    """Test the Hugging Face adapter with a mocked response."""
    # Set environment variables
    monkeypatch.setenv("HF_TOKEN", "test-token")
    
    # Mock HTTP response
    httpx_mock.add_response(
        json=[
            {
                "generated_text": "world"
            }
        ]
    )
    
    # Initialize adapter
    adapter = get_adapter("hf:demo-model")
    
    # Test generate method
    result = await adapter.generate(["hi"])
    
    # Assert expected response
    assert result == ["world"]
    
    # Close adapter to release resources
    await adapter.close() 