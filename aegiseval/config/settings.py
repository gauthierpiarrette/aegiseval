# SPDX-License-Identifier: Apache-2.0
"""Configuration management for the AegisEval framework."""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from aegiseval.config.schema import AegisEvalConfig, create_default_config, validate_config

# Configure logging
logger = logging.getLogger(__name__)

# Singleton config instance
_config: Dict[str, Any] = {}
_validated_config: Optional[AegisEvalConfig] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a YAML file and merge with defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration dictionary
    """
    global _config, _validated_config
    
    # Start with default config
    default_config = create_default_config()
    config = default_config.dict()
    
    # Try loading from environment variable first
    env_config = os.environ.get("AEGISEVAL_CONFIG")
    if env_config and Path(env_config).exists():
        config_path = env_config
    
    # Then use provided path if available
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r") as f:
                user_config = yaml.safe_load(f)
                if user_config:
                    # Validate the user config
                    try:
                        # Merge with defaults
                        merged_config = _deep_update(config.copy(), user_config)
                        
                        # Validate the merged config
                        _validated_config = validate_config(merged_config)
                        
                        # Update the config dictionary
                        config = merged_config
                    except Exception as e:
                        logger.error(f"Invalid configuration: {e}")
                        # Fall back to defaults
                        _validated_config = default_config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
    else:
        # No user config, use defaults
        _validated_config = default_config
    
    _config = config
    return config


def get_config() -> Dict[str, Any]:
    """Get the current configuration, loading defaults if not initialized.
    
    Returns:
        Current configuration dictionary
    """
    global _config
    if not _config:
        _config = load_config()
    # Ensure we never return None
    if _config is None:
        logger.warning("Configuration is None, falling back to empty dict")
        _config = {}
    return _config


def get_validated_config() -> AegisEvalConfig:
    """Get the validated configuration object.
    
    Returns:
        Validated configuration object
    """
    global _validated_config
    if not _validated_config:
        load_config()
    return _validated_config or create_default_config()


def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary.
    
    Args:
        d: Base dictionary to update
        u: Dictionary with updates
        
    Returns:
        Updated dictionary
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = _deep_update(d[k].copy(), v)
        else:
            d[k] = v
    return d 