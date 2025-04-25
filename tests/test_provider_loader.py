# SPDX-License-Identifier: Apache-2.0

"""Tests for the provider profile loader module."""

import pytest

from aegiseval.config.provider_loader import load_profile, ProviderProfile


def test_load_openai_safe_profile() -> None:
    """Test loading the OpenAI safe profile."""
    profile = load_profile("openai_safe")
    
    assert isinstance(profile, ProviderProfile)
    assert "csam" in profile.excluded_tags
    assert "extremist" in profile.excluded_tags
    assert "copyright_long" in profile.excluded_tags
    assert "privacy" in profile.excluded_tags
    assert "accuracy" in profile.excluded_tags
    assert len(profile.excluded_tags) == 5
    assert "You are being red-teamed for safety evaluation" in profile.system_header
    assert "Follow OpenAI policy" in profile.system_header
    assert profile.rpm_limit == 60


def test_load_local_default_profile() -> None:
    """Test loading the local default profile."""
    profile = load_profile("local_default")
    
    assert isinstance(profile, ProviderProfile)
    assert profile.excluded_tags == []
    assert profile.system_header == ""
    assert profile.rpm_limit == 200


def test_unknown_profile_raises_key_error() -> None:
    """Test that loading an unknown profile raises a KeyError."""
    with pytest.raises(KeyError) as excinfo:
        load_profile("nonexistent_profile")
    
    assert "not found" in str(excinfo.value) 