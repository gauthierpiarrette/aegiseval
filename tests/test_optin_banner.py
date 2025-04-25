# SPDX-License-Identifier: Apache-2.0
"""Tests for opt-in flags and confirmation banner."""

import io
import pytest
from unittest.mock import patch

from aegiseval.cli.banner import confirm_risky_run
from aegiseval.config import flags
from aegiseval.config.flags import EXIT_USER_ABORT


class TestRiskyRunBanner:
    """Tests for risky run confirmation banner."""
    
    def test_no_risky_tags_no_banner(self) -> None:
        """Test that no banner is shown if no risky tags are included."""
        # Patch sys.stdout to check if banner is shown
        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            # Call with empty set of tags
            confirm_risky_run(set())
            # Banner should not be shown
            assert fake_out.getvalue() == ""
    
    def test_banner_shown_for_risky_tags(self) -> None:
        """Test that banner is shown if risky tags are included."""
        # Prepare valid user input
        with patch("builtins.input", return_value="y"):
            # Patch sys.stdout to check if banner is shown
            with patch("sys.stdout", new=io.StringIO()) as fake_out:
                # Call with a risky tag
                confirm_risky_run({"csam"})
                # Banner should be shown
                output = fake_out.getvalue()
                assert "You have enabled probes requesting CSAM or extremist content" in output
                assert "Continue? [y/N]" in output
    
    def test_exit_on_negative_confirmation(self) -> None:
        """Test that sys.exit is called if user does not confirm."""
        # Mock sys.exit to avoid actual exit
        with patch("sys.exit") as mock_exit:
            # Mock user input
            with patch("builtins.input", return_value="n"):
                # Patch stdout to avoid printing
                with patch("sys.stdout"):
                    # Call with a risky tag
                    confirm_risky_run({"extremist"})
                    # Check sys.exit called with correct code
                    mock_exit.assert_called_once_with(EXIT_USER_ABORT)
    
    def test_exit_on_empty_confirmation(self) -> None:
        """Test that sys.exit is called if user enters empty string."""
        # Mock sys.exit to avoid actual exit
        with patch("sys.exit") as mock_exit:
            # Mock user input
            with patch("builtins.input", return_value=""):
                # Patch stdout to avoid printing
                with patch("sys.stdout"):
                    # Call with a risky tag
                    confirm_risky_run({"copyright_long"})
                    # Check sys.exit called with correct code
                    mock_exit.assert_called_once_with(EXIT_USER_ABORT)
    
    def test_systemexit_with_no_input(self) -> None:
        """Test SystemExit exception with exit code 12 when input is 'n'."""
        # Test with actual SystemExit to verify exit code
        with patch("builtins.input", return_value="n"):
            # Patch stdout to avoid printing
            with patch("sys.stdout"):
                # Should raise SystemExit with code 12
                with pytest.raises(SystemExit) as excinfo:
                    confirm_risky_run({"csam"})
                assert excinfo.value.code == EXIT_USER_ABORT


class TestYesFlagBehavior:
    """Tests for --yes flag behavior."""
    
    def setup_method(self) -> None:
        """Reset flags before each test."""
        flags.include_tags = set()
        flags.yes = False
    
    def test_yes_flag_suppresses_banner(self) -> None:
        """Test that yes flag suppresses banner."""
        # Directly patch the YES_FLAG in the banner module
        with patch("aegiseval.cli.banner.YES_FLAG", True):
            # Set up tags to include a risky tag
            risky_tags = {"csam"}
            
            # Input should never be called since we're patching YES_FLAG to True
            with patch("builtins.input", side_effect=Exception("Input should not be called")):
                # We expect this to complete normally without requesting input
                confirm_risky_run(risky_tags)


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 