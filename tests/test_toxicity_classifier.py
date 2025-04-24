# SPDX-License-Identifier: Apache-2.0
"""Tests for the toxicity classifier."""

from aegiseval.grader.classifiers import ToxicityClassifier, Metric


def test_toxicity_classifier_init() -> None:
    """Test that the ToxicityClassifier can be initialized."""
    classifier = ToxicityClassifier()
    assert classifier is not None
    # In test environments, the model might be None if transformers is not installed
    # Just verify the classifier works without crashing


def test_toxicity_classifier_predict() -> None:
    """Test that the predict method returns the correct number of results."""
    classifier = ToxicityClassifier()
    
    # Test with empty list
    results = classifier.predict([])
    assert len(results) == 0
    
    # Test with single item
    results = classifier.predict(["This is a test."])
    assert len(results) == 1
    assert isinstance(results[0], bool)
    
    # Test with multiple items
    texts = [
        "This is a pleasant message with no harmful content.",
        "I hate all people from that country, they are all terrible people.",
        "What a beautiful day! I hope everyone is enjoying the sunshine."
    ]
    results = classifier.predict(texts)
    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, bool)


def test_toxicity_classifier_benchmark() -> None:
    """Test that the benchmark method returns a Metric with F1 score >= 0.85."""
    classifier = ToxicityClassifier()
    
    # Run benchmark on sample dataset
    metrics = classifier.benchmark()
    
    # Check that we have a valid Metric object
    assert isinstance(metrics, Metric)
    assert metrics.tp + metrics.fp + metrics.fn + metrics.tn > 0
    
    # Check F1 score requirement
    assert metrics.f1 >= 0.85, f"F1 score is {metrics.f1}, expected >= 0.85" 