# SPDX-License-Identifier: Apache-2.0
"""Tests for the configuration loading module."""

import os
import yaml
import pytest
from pathlib import Path

import jsonschema

from aegiseval.config.loader import (
    load_test_suites,
    load_eu_map,
    TestSuiteCfg,
)


@pytest.fixture
def valid_test_suites_yaml(tmp_path: Path) -> Path:
    """Create a valid test suites YAML file for testing."""
    # Create a prompts file
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    prompts_file = prompts_dir / "default_prompts.txt"
    prompts_file.write_text("Test prompt")
    
    # Create test suites YAML file
    config_path = tmp_path / "test_suites.yaml"
    config_content = {
        "default": {
            "weight": 1,
            "prompts": str(prompts_file.relative_to(tmp_path)),
        },
        "behavioral": {
            "weight": 2,
            "prompts": str(prompts_file.relative_to(tmp_path)),
        },
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    return config_path


@pytest.fixture
def valid_eu_map_yaml(tmp_path: Path) -> Path:
    """Create a valid EU AI Act mapping YAML file for testing."""
    config_path = tmp_path / "eu_map.yaml"
    config_content = {
        "jailbreak_test": ["EU-AI-Act-Article-5"],
        "bias_test": ["EU-AI-Act-Article-5", "EU-AI-Act-Article-10"],
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    return config_path


@pytest.fixture
def invalid_test_suites_yaml(tmp_path: Path) -> Path:
    """Create an invalid test suites YAML file with a negative weight."""
    # Create a prompts file
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    prompts_file = prompts_dir / "default_prompts.txt"
    prompts_file.write_text("Test prompt")
    
    # Create test suites YAML file with invalid weight
    config_path = tmp_path / "invalid_test_suites.yaml"
    config_content = {
        "default": {
            "weight": -1,  # Invalid: weight cannot be negative
            "prompts": str(prompts_file.relative_to(tmp_path)),
        },
    }
    
    with open(config_path, "w") as f:
        yaml.dump(config_content, f)
    
    return config_path


def test_load_test_suites(valid_test_suites_yaml: Path) -> None:
    """Test loading valid test suites configuration."""
    result = load_test_suites(valid_test_suites_yaml)
    
    assert len(result) == 2
    assert "default" in result
    assert "behavioral" in result
    
    assert isinstance(result["default"], TestSuiteCfg)
    assert result["default"].weight == 1
    assert result["behavioral"].weight == 2
    
    # Check that the paths exist
    assert os.path.exists(result["default"].prompts)
    assert os.path.exists(result["behavioral"].prompts)


def test_load_eu_map(valid_eu_map_yaml: Path) -> None:
    """Test loading valid EU AI Act mapping."""
    result = load_eu_map(valid_eu_map_yaml)
    
    assert len(result) == 2
    assert "jailbreak_test" in result
    assert "bias_test" in result
    
    assert result["jailbreak_test"] == ["EU-AI-Act-Article-5"]
    assert result["bias_test"] == ["EU-AI-Act-Article-5", "EU-AI-Act-Article-10"]


def test_invalid_test_suites_weight(invalid_test_suites_yaml: Path) -> None:
    """Test that a validation error is raised for invalid weight."""
    with pytest.raises(jsonschema.exceptions.ValidationError):
        load_test_suites(invalid_test_suites_yaml)


def test_missing_file() -> None:
    """Test that appropriate error is raised for missing files."""
    with pytest.raises(FileNotFoundError):
        load_test_suites(Path("/path/to/nonexistent/file.yaml")) 