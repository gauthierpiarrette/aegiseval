# SPDX-License-Identifier: Apache-2.0
"""Configuration package for AegisEval Lite.

This package provides configuration management for AegisEval, including
schema validation, default configuration, and configuration loading.
"""

from aegiseval.config.settings import (
    get_config, load_config, get_validated_config
)
from aegiseval.config.schema import (
    AegisEvalConfig, create_default_config, validate_config
)

__all__ = [
    "get_config",
    "load_config",
    "get_validated_config",
    "AegisEvalConfig",
    "create_default_config",
    "validate_config",
] 