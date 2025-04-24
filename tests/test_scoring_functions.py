# SPDX-License-Identifier: Apache-2.0
"""Tests for the core scoring module."""

from dataclasses import dataclass
from aegiseval.core.score import (
    aegis_score, 
    calculate_safety_score, 
    map_failures_to_regulations
)


@dataclass
class SuiteScore:
    """Simple data class for test scoring."""
    weight: int
    pass_rate: float


def test_calculate_safety_score_empty() -> None:
    """Test safety score calculation with empty results."""
    assert calculate_safety_score([]) == 0.0


def test_calculate_safety_score_all_pass() -> None:
    """Test safety score calculation with all passing tests."""
    test_results = [
        {"test_id": "test1", "passed": True},
        {"test_id": "test2", "passed": True},
    ]
    assert calculate_safety_score(test_results) == 1.0


def test_calculate_safety_score_mixed() -> None:
    """Test safety score calculation with mixed results."""
    test_results = [
        {"test_id": "test1", "passed": True},
        {"test_id": "test2", "passed": False},
        {"test_id": "test3", "passed": True},
        {"test_id": "test4", "passed": False},
    ]
    assert calculate_safety_score(test_results) == 0.5


def test_map_failures_to_regulations() -> None:
    """Test mapping failures to regulations."""
    test_results = [
        {"test_id": "test1", "passed": False, "regulations": ["EU-AI-Act-Article-5"]},
        {"test_id": "test2", "passed": True, "regulations": ["EU-AI-Act-Article-5"]},
        {"test_id": "test3", "passed": False, "regulations": ["EU-AI-Act-Article-6"]},
    ]
    
    regulation_map = map_failures_to_regulations(test_results)
    
    assert "EU-AI-Act-Article-5" in regulation_map
    assert len(regulation_map["EU-AI-Act-Article-5"]) == 1
    assert regulation_map["EU-AI-Act-Article-5"][0]["test_id"] == "test1"
    
    assert "EU-AI-Act-Article-6" in regulation_map
    assert len(regulation_map["EU-AI-Act-Article-6"]) == 1
    assert regulation_map["EU-AI-Act-Article-6"][0]["test_id"] == "test3"


def test_aegis_score_basic() -> None:
    """Test basic aegis_score functionality."""
    results = [
        SuiteScore(weight=1, pass_rate=0.5),
        SuiteScore(weight=1, pass_rate=0.0),
        SuiteScore(weight=1, pass_rate=1.0),
    ]
    assert aegis_score(results) == 50


def test_aegis_score_weighted() -> None:
    """Test weighted aegis_score calculation."""
    results = [
        SuiteScore(weight=1, pass_rate=0.0),
        SuiteScore(weight=3, pass_rate=1.0),
    ]
    assert aegis_score(results) == 75


def test_aegis_score_empty() -> None:
    """Test aegis_score with empty results."""
    assert aegis_score([]) == 0


def test_aegis_score_zero_weights() -> None:
    """Test aegis_score with zero weights."""
    results = [
        SuiteScore(weight=0, pass_rate=0.5),
        SuiteScore(weight=0, pass_rate=1.0),
    ]
    assert aegis_score(results) == 0


def test_aegis_score_range() -> None:
    """Test aegis_score always returns a value between 0 and 100."""
    test_cases = [
        # Empty case
        [],
        # All passing
        [SuiteScore(weight=1, pass_rate=1.0)],
        # All failing
        [SuiteScore(weight=1, pass_rate=0.0)],
        # Mixed results
        [SuiteScore(weight=1, pass_rate=0.5), SuiteScore(weight=1, pass_rate=0.7)],
        # Different weights
        [SuiteScore(weight=1, pass_rate=0.3), SuiteScore(weight=3, pass_rate=0.9)],
    ]
    
    for test_case in test_cases:
        score = aegis_score(test_case)
        assert 0 <= score <= 100, f"Score {score} for {test_case} not in range [0, 100]"


def test_aegis_score_all_pass() -> None:
    """Test if all pass_rates are 1.0, score should be 100."""
    test_cases = [
        [SuiteScore(weight=1, pass_rate=1.0)],
        [SuiteScore(weight=1, pass_rate=1.0), SuiteScore(weight=2, pass_rate=1.0)],
        [SuiteScore(weight=10, pass_rate=1.0), SuiteScore(weight=5, pass_rate=1.0)],
    ]
    
    for test_case in test_cases:
        score = aegis_score(test_case)
        assert score == 100


def test_aegis_score_weight_scaling() -> None:
    """Test score should not change when all weights are scaled by the same factor."""
    test_cases = [
        ([SuiteScore(weight=1, pass_rate=0.5)], 2),
        ([SuiteScore(weight=2, pass_rate=0.7), SuiteScore(weight=3, pass_rate=0.3)], 5),
    ]
    
    for base_results, scaling_factor in test_cases:
        # Create a copy with scaled weights
        scaled_results = [
            SuiteScore(weight=suite.weight * scaling_factor, pass_rate=suite.pass_rate) 
            for suite in base_results if suite.weight > 0
        ]
        
        # Skip test if we have no valid results after filtering
        if not scaled_results or not base_results:
            continue
            
        # The scores should be very close (may differ slightly due to rounding)
        base_score = aegis_score(base_results)
        scaled_score = aegis_score(scaled_results)
        
        # Allow for small differences due to rounding
        assert abs(base_score - scaled_score) <= 1 