# SPDX-License-Identifier: Apache-2.0
"""Tests for the test suite loader's tag validation."""

import os
import pytest
import tempfile
import yaml
from typing import List

from aegiseval.data.test_suite import load_test_suite, TestCase


def test_tags_validation_accepts_valid_tags() -> None:
    """Test that valid tags are accepted."""
    # Valid tags
    valid_tags: List[str] = [
        "csam", "extremist", "copyright_long", "jailbreak",
        "self_harm", "bias", "misinfo", "privacy", "hallucination"
    ]
    
    # Create a test suite with valid tags
    for tag in valid_tags:
        # Test each tag individually
        case = TestCase(
            id="test_1",
            prompt="Valid prompt",
            category="harmful_instructions",
            regulations=["EU-AI-Act-Article-5"],
            expected_behavior="Should refuse",
            tags=[tag]
        )
        assert case.tags is not None and tag in case.tags


def test_tags_validation_rejects_invalid_tags() -> None:
    """Test that invalid tags are rejected."""
    # Invalid tag
    with pytest.raises(ValueError) as excinfo:
        TestCase(
            id="test_1",
            prompt="Valid prompt",
            category="harmful_instructions",
            regulations=["EU-AI-Act-Article-5"],
            expected_behavior="Should refuse",
            tags=["invalid_tag"]
        )
    
    assert "Tag 'invalid_tag' is not allowed" in str(excinfo.value)


def test_load_test_suite_with_tags() -> None:
    """Test loading a test suite with tags from a file."""
    # Create a temporary test suite file with tags
    test_data = {
        "id": "test_suite",
        "name": "Test Suite",
        "description": "Test description",
        "version": "1.0",
        "tests": [
            {
                "id": "test_1",
                "prompt": "Prompt 1",
                "category": "harmful_instructions",
                "regulations": ["EU-AI-Act-Article-5"],
                "expected_behavior": "Behavior 1",
                "tags": ["jailbreak", "extremist"]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(test_data, temp)
    
    try:
        # Load the test suite
        test_suite = load_test_suite(temp.name)
        
        # Verify tags were loaded correctly
        assert len(test_suite.tests) == 1
        assert test_suite.tests[0].tags == ["jailbreak", "extremist"]
    finally:
        os.unlink(temp.name)


def test_load_test_suite_with_invalid_tags() -> None:
    """Test loading a test suite with invalid tags from a file."""
    # Create a temporary test suite file with an invalid tag
    test_data = {
        "id": "test_suite",
        "name": "Test Suite",
        "description": "Test description",
        "version": "1.0",
        "tests": [
            {
                "id": "test_1",
                "prompt": "Prompt 1",
                "category": "harmful_instructions",
                "regulations": ["EU-AI-Act-Article-5"],
                "expected_behavior": "Behavior 1",
                "tags": ["invalid_tag"]
            }
        ]
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".yaml", delete=False) as temp:
        yaml.dump(test_data, temp)
    
    try:
        # Try to load the test suite, should raise ValueError
        with pytest.raises(ValueError) as excinfo:
            load_test_suite(temp.name)
        
        assert "Tag 'invalid_tag' is not allowed" in str(excinfo.value)
    finally:
        os.unlink(temp.name)


def test_empty_tags_list() -> None:
    """Test that an empty tags list is valid."""
    case = TestCase(
        id="test_1",
        prompt="Valid prompt",
        category="harmful_instructions",
        regulations=["EU-AI-Act-Article-5"],
        expected_behavior="Should refuse",
        tags=[]
    )
    assert case.tags == []


def test_none_tags_becomes_empty_list() -> None:
    """Test that None tags becomes an empty list."""
    case = TestCase(
        id="test_1",
        prompt="Valid prompt",
        category="harmful_instructions",
        regulations=["EU-AI-Act-Article-5"],
        expected_behavior="Should refuse",
        tags=None
    )
    assert case.tags == [] 