# SPDX-License-Identifier: Apache-2.0

"""Test module to validate coverage and classifier performance metrics."""

import os
import sys
import json
import pytest
import xml.etree.ElementTree as ET
from pathlib import Path


def test_coverage_threshold() -> None:
    """
    Verify that code coverage meets the minimum required threshold.
    
    Checks total statement coverage against environment variable threshold
    (defaults to 90%).
    
    Raises:
        AssertionError: If coverage is below the specified threshold.
    """
    min_coverage = float(os.getenv("MIN_COVERAGE_TOTAL", "90"))
    
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        pytest.skip("No coverage file found, skipping coverage threshold test")
    
    tree = ET.parse(coverage_file)
    root = tree.getroot()
    
    # Total statement coverage as percentage (multiply by 100 to convert from decimal)
    total_coverage = float(root.attrib["line-rate"]) * 100
    
    print(f"Total statement coverage: {total_coverage:.2f}%")
    print(f"Required minimum: {min_coverage}%")
    
    assert total_coverage >= min_coverage, (
        f"Total statement coverage ({total_coverage:.2f}%) "
        f"below minimum threshold ({min_coverage}%)"
    )


def test_branch_coverage_threshold() -> None:
    """
    Verify that branch coverage meets the minimum required threshold.
    
    Checks branch coverage against environment variable threshold
    (defaults to 80%).
    
    Raises:
        AssertionError: If branch coverage is below the specified threshold.
    """
    min_branch_coverage = float(os.getenv("MIN_COVERAGE_BRANCH", "80"))
    
    coverage_file = Path("coverage.xml")
    if not coverage_file.exists():
        pytest.skip("No coverage file found, skipping coverage threshold test")
    
    tree = ET.parse(coverage_file)
    root = tree.getroot()
    
    # Branch coverage as percentage
    branch_coverage = float(root.attrib["branch-rate"]) * 100
    
    print(f"Branch coverage: {branch_coverage:.2f}%")
    print(f"Required minimum: {min_branch_coverage}%")
    
    assert branch_coverage >= min_branch_coverage, (
        f"Branch coverage ({branch_coverage:.2f}%) "
        f"below minimum threshold ({min_branch_coverage}%)"
    )


def test_classifier_f1_scores() -> None:
    """
    Verify that all classifier F1 scores meet the minimum required threshold.
    
    Checks each classifier's F1 score against the environment variable threshold
    (defaults to 0.85).
    
    Raises:
        AssertionError: If any classifier's F1 score is below the threshold.
    """
    min_f1_score = float(os.getenv("MIN_CLASSIFIER_F1", "0.85"))
    
    # Try to find the file containing classifier metrics
    # This assumes the metrics are stored in a JSON file after classifier evaluation
    metrics_file = Path("classifier_metrics.json")
    
    if not metrics_file.exists():
        metrics_file = Path("tests/results/classifier_metrics.json")
        if not metrics_file.exists():
            # If we can't find the file, try to check for it elsewhere or skip the test
            pytest.skip("No classifier metrics file found, skipping F1 score test")
    
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    # Check F1 score for each classifier
    failed_classifiers = []
    
    for classifier_name, classifier_metrics in metrics.items():
        if "f1" in classifier_metrics:
            f1_score = classifier_metrics["f1"]
            print(f"Classifier {classifier_name} F1 score: {f1_score:.4f}")
            
            if f1_score < min_f1_score:
                failed_classifiers.append((classifier_name, f1_score))
    
    assert not failed_classifiers, (
        f"The following classifiers have F1 scores below the minimum threshold ({min_f1_score}):\n"
        + "\n".join([f"  - {name}: {score:.4f}" for name, score in failed_classifiers])
    )


if __name__ == "__main__":
    # When running as a script, just run these specific tests
    sys.exit(pytest.main(["-xvs", __file__])) 