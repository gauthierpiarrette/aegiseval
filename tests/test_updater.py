# SPDX-License-Identifier: Apache-2.0
"""Tests for the updater job module."""

import os
import yaml
import pytest
from unittest.mock import patch, mock_open
import vcr
from typing import Dict, List

from aegiseval.updater.job import check_for_updates


# Setup the VCR cassette directory
VCR_CASSETTES_DIR = os.path.join(os.path.dirname(__file__), 'vcr_cassettes')
os.makedirs(VCR_CASSETTES_DIR, exist_ok=True)


@pytest.fixture
def mock_local_yaml() -> Dict[str, List[str]]:
    """Create a mock local YAML file."""
    return {
        "test1": ["EU-AI-Act-Article-5"],
        "test2": ["EU-AI-Act-Article-6", "EU-AI-Act-Article-7"]
    }


@pytest.fixture
def mock_same_remote_json() -> Dict[str, List[str]]:
    """Create a mock remote JSON that's the same as the local YAML."""
    return {
        "test1": ["EU-AI-Act-Article-5"],
        "test2": ["EU-AI-Act-Article-6", "EU-AI-Act-Article-7"]
    }


@pytest.fixture
def mock_different_remote_json() -> Dict[str, List[str]]:
    """Create a mock remote JSON that differs from the local YAML."""
    return {
        "test1": ["EU-AI-Act-Article-5", "EU-AI-Act-Article-8"],  # Added an article
        "test2": ["EU-AI-Act-Article-6", "EU-AI-Act-Article-7"],
        "test3": ["EU-AI-Act-Article-9"]  # New test mapping
    }


@vcr.use_cassette(os.path.join(VCR_CASSETTES_DIR, 'no_updates.yaml'))
def test_check_for_updates_no_diff(mock_local_yaml: Dict[str, List[str]], mock_same_remote_json: Dict[str, List[str]]) -> None:
    """Test checking for updates when no differences exist."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_local_yaml))):
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_same_remote_json
            mock_get.return_value.raise_for_status = lambda: None
            
            result = check_for_updates()
            
            assert result is False


@vcr.use_cassette(os.path.join(VCR_CASSETTES_DIR, 'has_updates.yaml'))
def test_check_for_updates_with_diff(mock_local_yaml: Dict[str, List[str]], mock_different_remote_json: Dict[str, List[str]]) -> None:
    """Test checking for updates when differences exist."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_local_yaml))):
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = mock_different_remote_json
            mock_get.return_value.raise_for_status = lambda: None
            
            result = check_for_updates()
            
            assert result is True


@vcr.use_cassette(os.path.join(VCR_CASSETTES_DIR, 'request_error.yaml'))
def test_check_for_updates_with_request_error() -> None:
    """Test handling of request errors during update check."""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = Exception("Connection error")
        
        result = check_for_updates()
        
        assert result is False


def test_check_for_updates_with_custom_branch() -> None:
    """Test checking for updates from a custom branch."""
    with patch('requests.get') as mock_get, \
         patch('builtins.open', mock_open(read_data=yaml.dump({}))):
        mock_get.return_value.json.return_value = {}
        mock_get.return_value.raise_for_status = lambda: None
        
        check_for_updates(branch="develop")
        
        # Verify that the URL contains the correct branch name
        mock_get.assert_called_once()
        called_url = mock_get.call_args[0][0]
        assert "develop" in called_url 