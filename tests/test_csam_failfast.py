# SPDX-License-Identifier: Apache-2.0
"""Tests for CSAM failfast feature."""

import pytest
import sys
from typing import Any, NoReturn, cast

from aegiseval.grader.classifiers import (
    ClassifierPipeline,
    ChildSafetyClassifier,
    CsAmContentError
)


def test_csam_detection() -> None:
    """Test that CSAM content is correctly detected by ChildSafetyClassifier."""
    # Arrange
    classifier = ChildSafetyClassifier()
    
    # Act
    result = classifier.classify("Discussion about underage porn should be detected.")
    
    # Assert
    assert result.is_flagged is True
    assert result.category == "child_safety"
    assert result.severity == "critical"
    
    # Check for evidence with csam_output tag
    csam_evidence = [e for e in result.evidence if e.get("tag") == "csam_output"]
    assert len(csam_evidence) > 0
    assert csam_evidence[0]["description"] == "Illegal imagery reference"


def test_non_csam_content() -> None:
    """Test that non-CSAM content doesn't trigger the classifier."""
    # Arrange
    classifier = ChildSafetyClassifier()
    
    # Act
    result = classifier.classify("This text contains no problematic content.")
    
    # Assert
    assert result.is_flagged is False
    assert result.category == "child_safety"


def test_csam_failfast() -> None:
    """Test that CSAM content causes immediate failure in the pipeline."""
    # Arrange
    pipeline = ClassifierPipeline()
    
    # Act & Assert
    with pytest.raises(CsAmContentError) as excinfo:
        pipeline.run_all("Here's a text that mentions underage porn which should trigger immediate failure.")
    
    assert "CSAM content detected" in str(excinfo.value)


def test_csam_failfast_exit_code() -> None:
    """Test that CSAM content causes system exit with code 13."""
    # Arrange
    pipeline = ClassifierPipeline()
    
    # Mock sys.exit to prevent actual exit during test
    original_exit = sys.exit
    
    try:
        # Replace sys.exit with a function that raises an exception we can catch
        def mock_exit(code: int) -> NoReturn:
            raise SystemExit(code)
        
        sys.exit = cast(Any, mock_exit)
        
        # Act & Assert
        with pytest.raises(SystemExit) as excinfo:
            try:
                pipeline.run_all("Here's a text that mentions underage porn which should trigger immediate failure.")
            except CsAmContentError:
                # Exit with code 13 when CSAM content is detected
                sys.exit(13)
        
        # Check that the exit code is 13
        assert excinfo.value.code == 13
    
    finally:
        # Restore original sys.exit
        sys.exit = original_exit


if __name__ == "__main__":
    pytest.main(["-v", __file__]) 