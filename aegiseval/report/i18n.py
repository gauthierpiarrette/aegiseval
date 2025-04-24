# SPDX-License-Identifier: Apache-2.0
"""Internationalization helper for AegisEval reports."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from aegiseval.config.settings import get_config

# Configure logging
logger = logging.getLogger(__name__)


class I18n:
    """Helper class for loading and accessing localized strings.
    
    This class loads translation files from the locales directory and
    provides methods to access translated strings.
    """
    
    def __init__(self, language: str = "en") -> None:
        """Initialize the I18n helper.
        
        Args:
            language: The language code to use for translations. Defaults to "en".
        """
        self.language = language
        self.strings: Dict[str, Any] = {}
        self._load_strings()
    
    def _load_strings(self) -> None:
        """Load the strings for the current language."""
        # Determine the package directory
        package_dir = Path(__file__).resolve().parent.parent.parent
        locale_file = package_dir / "locales" / self.language / "strings.yaml"
        
        # Fall back to English if the requested language file doesn't exist
        if not locale_file.exists():
            logger.warning(f"Language file not found for {self.language}, falling back to English")
            self.language = "en"
            locale_file = package_dir / "locales" / "en" / "strings.yaml"
        
        # Load the strings file
        if locale_file.exists():
            try:
                with open(locale_file, "r", encoding="utf-8") as f:
                    self.strings = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Failed to load language file {locale_file}: {str(e)}")
                self.strings = {}
        else:
            # For test scenarios, keep the existing strings if the file doesn't exist
            # This allows test cases to set the strings dictionary directly
            if not self.strings:
                logger.warning(f"No language file found at {locale_file}")
                self.strings = {}
    
    def get(self, key: str, default: Optional[str] = None) -> str:
        """Get a translated string by key.
        
        Args:
            key: The string key to look up.
            default: The default value to return if the key is not found.
                  If None, the key itself is returned.
        
        Returns:
            The translated string, or the default value if not found.
        """
        # Split nested keys by dot notation
        parts = key.split('.')
        value = self.strings
        
        # Navigate through nested dictionary
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default if default is not None else key
        
        # Return the value if it's a string, otherwise the key or default
        return value if isinstance(value, str) else (default if default is not None else key)


# Create and cache translators for different languages
_translators: Dict[str, I18n] = {}


def get_translator(language: str = "en") -> I18n:
    """Get a translator for the specified language.
    
    Args:
        language: The language code
        
    Returns:
        An I18n instance for the language
    """
    global _translators
    
    # Check if we already have a translator for this language
    if language in _translators:
        return _translators[language]
    
    # Create a new translator
    translator = I18n(language)
    _translators[language] = translator
    
    return translator


def set_language(language: str) -> None:
    """Set the current language for translations.
    
    Args:
        language: The language code to use for translations.
    """
    # Get configuration and check supported languages
    config = get_config()
    supported = config.get("reporting", {}).get("supported_languages", ["en"])
    
    if language not in supported:
        logger.warning(f"Language {language} is not supported. Using English.")
        language = "en"
    
    # Clear the translator cache to force reloading
    global _translators
    _translators = {}
    
    # Load the translator for this language
    get_translator(language)

# Default i18n instance
i18n = get_translator()

def get(key: str, default: Optional[str] = None) -> str:
    """Get a translated string using the default translator.
    
    Args:
        key: The string key to look up.
        default: The default value to return if the key is not found.
              If None, the key itself is returned.
    
    Returns:
        The translated string, or the default value if not found.
    """
    return i18n.get(key, default) 