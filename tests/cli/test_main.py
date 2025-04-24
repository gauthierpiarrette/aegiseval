# SPDX-License-Identifier: Apache-2.0
"""Tests for the AegisEval CLI main module."""

import pytest
import os
import tempfile
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from aegiseval.cli.main import cli, scan


class TestCLI:
    """Tests for the CLI interface."""
    
    def test_help(self):
        """Test help output."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "AegisEval Lite" in result.output
    
    def test_scan_help(self):
        """Test scan command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["scan", "--help"])
        assert result.exit_code == 0
        assert "Scan an LLM for safety issues" in result.output
    
    def test_offline_mode(self):
        """Test scan in offline mode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CliRunner()
            result = runner.invoke(cli, [
                "scan",
                "--model", "gpt-4",
                "--outdir", temp_dir,
                "--offline"
            ])
            
            assert result.exit_code == 0
            assert "Offline scan completed" in result.output
            
            # Check if mock files were created
            assert os.path.exists(os.path.join(temp_dir, "gpt-4_mock_report.json"))
            assert os.path.exists(os.path.join(temp_dir, "gpt-4_mock_badge.md"))
    
    @patch("aegiseval.cli.main.asyncio.run")
    def test_api_key_error(self, mock_run):
        """Test handling of API key errors."""
        # Make asyncio.run raise an AuthenticationError
        from aegiseval.runner.exceptions import AuthenticationError
        mock_run.side_effect = AuthenticationError("Invalid API key")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CliRunner()
            # Set OPENAI_API_KEY to a dummy value to prevent offline mode
            with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}):
                result = runner.invoke(cli, [
                    "scan",
                    "--model", "gpt-4",
                    "--outdir", temp_dir
                ])
            
            assert result.exit_code == 1
            assert "Authentication error" in result.output
    
    @patch("aegiseval.cli.main._run_evaluation")
    @patch("aegiseval.cli.main.asyncio.run")
    def test_successful_evaluation(self, mock_asyncio_run, mock_run_evaluation):
        """Test successful evaluation flow."""
        # Set up mocks
        mock_report_paths = {
            "json": "/tmp/model_report.json",
            "pdf": "/tmp/model_report.pdf",
            "badge_md": "/tmp/model_badge.md"
        }
        mock_result = {
            "report_paths": mock_report_paths,
            "overall_score": 0.85
        }
        mock_run_evaluation.return_value = mock_result
        
        # Make asyncio.run return the expected result
        mock_asyncio_run.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CliRunner()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
                result = runner.invoke(cli, [
                    "scan",
                    "--model", "gpt-4",
                    "--outdir", temp_dir,
                    "--api-key", "test-key"
                ])
            
            assert result.exit_code == 0
            assert "Safety Score: 0.85" in result.output
            
            # Verify mocks were called
            mock_asyncio_run.assert_called_once()
    
    def test_system_command(self):
        """Test system command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["system", "--list-models"])
        
        assert result.exit_code == 0
        assert "Available models" in result.output
        assert "gpt-4" in result.output
    
    @patch("aegiseval.cli.main._run_evaluation")
    @patch("aegiseval.cli.main.asyncio.run")
    def test_evaluation_with_redteam(self, mock_asyncio_run, mock_run_evaluation):
        """Test evaluation with red team option."""
        # Set up mocks
        mock_report_paths = {
            "json": "/tmp/model_report.json",
            "pdf": "/tmp/model_report.pdf",
            "badge_md": "/tmp/model_badge.md"
        }
        mock_result = {
            "report_paths": mock_report_paths,
            "overall_score": 0.75
        }
        mock_run_evaluation.return_value = mock_result
        
        # Make asyncio.run return the expected result
        mock_asyncio_run.return_value = mock_result
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CliRunner()
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"}):
                result = runner.invoke(cli, [
                    "scan",
                    "--model", "gpt-4",
                    "--outdir", temp_dir,
                    "--redteam", "2",
                    "--api-key", "test-key"
                ])
            
            assert result.exit_code == 0
            assert "Safety Score: 0.75" in result.output
    
    @patch("aegiseval.cli.main.asyncio.run")
    def test_model_adapter_error(self, mock_run):
        """Test handling of model adapter errors."""
        # Make asyncio.run raise a ModelAdapterError
        from aegiseval.runner.exceptions import ModelAdapterError
        mock_run.side_effect = ModelAdapterError("Model not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = CliRunner()
            # Set OPENAI_API_KEY to a dummy value to prevent offline mode
            with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key"}):
                result = runner.invoke(cli, [
                    "scan",
                    "--model", "nonexistent-model",
                    "--outdir", temp_dir
                ])
            
            assert result.exit_code == 1
            assert "Error accessing the model" in result.output 