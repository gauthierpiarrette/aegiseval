# SPDX-License-Identifier: Apache-2.0
"""Tests for internationalization (i18n) functionality."""

import os
import tempfile
from pathlib import Path
import yaml


from aegiseval.report.i18n import I18n, set_language, get


def test_i18n_fallback_to_default() -> None:
    """Test that I18n falls back to English if requested language is not available."""
    i18n = I18n("nonexistent_language")
    assert i18n.language == "en"


def test_i18n_get_with_default() -> None:
    """Test that I18n.get returns the default value if key is not found."""
    i18n = I18n("en")
    assert i18n.get("nonexistent.key", "default_value") == "default_value"


def test_i18n_get_without_default() -> None:
    """Test that I18n.get returns the key itself if key is not found and no default provided."""
    i18n = I18n("en")
    assert i18n.get("nonexistent.key") == "nonexistent.key"


def test_i18n_nested_keys() -> None:
    """Test that I18n.get properly handles nested keys."""
    # Create a temporary test strings file
    with tempfile.TemporaryDirectory() as temp_dir:
        locale_dir = Path(temp_dir) / "en"
        os.makedirs(locale_dir, exist_ok=True)
        
        strings = {
            "parent": {
                "child": {
                    "grandchild": "nested value"
                }
            }
        }
        
        with open(locale_dir / "strings.yaml", "w", encoding="utf-8") as f:
            yaml.dump(strings, f)
        
        # Test with a custom file path (monkeypatch the _load_strings method)
        i18n = I18n("en")
        
        # Mock the strings directly
        i18n.strings = strings
        
        assert i18n.get("parent.child.grandchild") == "nested value"
        assert i18n.get("parent.child.missing", "default") == "default"
        assert i18n.get("parent.missing.key", "default") == "default"


def test_global_i18n_functions() -> None:
    """Test the global i18n helper functions."""
    # Create a temporary test strings file for English
    with tempfile.TemporaryDirectory() as temp_dir:
        en_dir = Path(temp_dir) / "en"
        fr_dir = Path(temp_dir) / "fr"
        os.makedirs(en_dir, exist_ok=True)
        os.makedirs(fr_dir, exist_ok=True)
        
        en_strings = {"greeting": "Hello"}
        fr_strings = {"greeting": "Bonjour"}
        
        with open(en_dir / "strings.yaml", "w", encoding="utf-8") as f:
            yaml.dump(en_strings, f)
        
        with open(fr_dir / "strings.yaml", "w", encoding="utf-8") as f:
            yaml.dump(fr_strings, f)
        
        # Mock the global i18n to use our temp files
        from aegiseval.report.i18n import i18n
        i18n.strings = en_strings
        
        # Test default language
        assert get("greeting") == "Hello"
        
        # Test switching language
        set_language("fr")
        i18n.strings = fr_strings  # Mock the loaded strings
        assert get("greeting") == "Bonjour" 