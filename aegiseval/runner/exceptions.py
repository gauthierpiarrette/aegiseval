# SPDX-License-Identifier: Apache-2.0
"""Exceptions for the model runners."""


class ModelAdapterError(Exception):
    """Base class for all model adapter errors."""
    pass


class AuthenticationError(ModelAdapterError):
    """Raised when authentication to the model API fails."""
    pass


class QuotaExceededError(ModelAdapterError):
    """Raised when the API quota is exceeded."""
    pass


class RateLimitError(ModelAdapterError):
    """Raised when rate limits are hit."""
    pass


class InvalidRequestError(ModelAdapterError):
    """Raised when the request to the API is invalid."""
    pass


class ServiceUnavailableError(ModelAdapterError):
    """Raised when the API service is unavailable."""
    pass


class TimeoutError(ModelAdapterError):
    """Raised when the API request times out."""
    pass 