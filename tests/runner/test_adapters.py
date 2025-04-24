# SPDX-License-Identifier: Apache-2.0
"""Tests for the model adapters."""

import pytest
import httpx
import tenacity
from unittest.mock import patch, AsyncMock, MagicMock

from aegiseval.runner.adapters import OpenAIAdapter, MockAdapter, create_adapter
from aegiseval.runner.exceptions import (
    AuthenticationError, RateLimitError, InvalidRequestError,
    ServiceUnavailableError, TimeoutError, ModelAdapterError
)


class TestOpenAIAdapter:
    """Tests for the OpenAIAdapter class."""
    
    @pytest.mark.asyncio
    async def test_init_missing_api_key(self):
        """Test initialization with missing API key."""
        with patch.dict('os.environ', clear=True):
            with pytest.raises(AuthenticationError):
                OpenAIAdapter("gpt-4")
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_generate_success(self, mock_post):
        """Test successful response generation."""
        # Configure mock
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        # Test adapter
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            adapter = OpenAIAdapter("gpt-4")
            responses = await adapter.generate(["Test prompt"])
            
        assert len(responses) == 1
        assert responses[0] == "Test response"
        await adapter.close()
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient.post')
    async def test_generate_auth_error(self, mock_post):
        """Test authentication error handling."""
        # Create a proper HTTP error with a non-coroutine response
        mock_error_response = MagicMock()
        mock_error_response.status_code = 401
        mock_error_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        
        mock_error = httpx.HTTPStatusError(
            "401 Unauthorized", 
            request=MagicMock(), 
            response=mock_error_response
        )
        
        # Configure the mock to raise the error
        mock_post.side_effect = mock_error
        
        # Test adapter
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            adapter = OpenAIAdapter("gpt-4")
            with pytest.raises(AuthenticationError):
                await adapter.generate(["Test prompt"])
        
        await adapter.close()
    
    @pytest.mark.asyncio
    @patch('tenacity.wait.wait_exponential')
    @patch('tenacity.stop.stop_after_attempt')
    @patch('httpx.AsyncClient.post')
    async def test_generate_rate_limit_error(self, mock_post, mock_stop, mock_wait):
        """Test rate limit error handling.
        
        Note: Override the tenacity retry behavior to avoid actual waiting
        """
        # Configure tenacity to fail faster
        mock_stop.return_value = lambda retry_state: retry_state.attempt_number >= 1
        mock_wait.return_value = lambda retry_state: 0
        
        # Create a proper HTTP error with a non-coroutine response
        mock_error_response = MagicMock()
        mock_error_response.status_code = 429
        mock_error_response.json.return_value = {
            "error": {"message": "Rate limit exceeded"}
        }
        
        mock_error = httpx.HTTPStatusError(
            "429 Too Many Requests", 
            request=MagicMock(), 
            response=mock_error_response
        )
        
        # Configure the mock to raise the error
        mock_post.side_effect = mock_error
        
        # Test adapter
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            adapter = OpenAIAdapter("gpt-4")
            # Expect tenacity RetryError wrapping RateLimitError
            with pytest.raises(tenacity.RetryError) as excinfo:
                await adapter.generate(["Test prompt"])
            
            # Verify the inner exception is RateLimitError
            assert isinstance(excinfo.value.last_attempt.exception(), RateLimitError)
        
        await adapter.close()


class TestMockAdapter:
    """Tests for the MockAdapter class."""
    
    @pytest.mark.asyncio
    async def test_mock_responses(self):
        """Test that mock adapter returns expected responses."""
        responses = {
            "specific prompt": "specific response",
        }
        
        adapter = MockAdapter("mock", responses=responses)
        
        # Test specific response
        result = await adapter.generate(["specific prompt"])
        assert result[0] == "specific response"
        
        # Test default response
        result = await adapter.generate(["unknown prompt"])
        assert "Mock response for:" in result[0]
        
        await adapter.close()


class TestAdapterFactory:
    """Tests for the adapter factory function."""
    
    def test_create_openai_adapter(self):
        """Test creating OpenAI adapter."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test_key'}):
            adapter = create_adapter("gpt-4")
            assert isinstance(adapter, OpenAIAdapter)
    
    def test_create_mock_adapter(self):
        """Test creating mock adapter."""
        adapter = create_adapter("mock")
        assert isinstance(adapter, MockAdapter)
    
    def test_unsupported_model(self):
        """Test error for unsupported model."""
        with pytest.raises(ValueError):
            create_adapter("unsupported-model") 