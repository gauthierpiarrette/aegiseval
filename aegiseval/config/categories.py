# SPDX-License-Identifier: Apache-2.0
"""Test categories for the AegisEval framework, mapped to EU AI Act articles."""

from typing import Dict, Any


# Categories mapped to EU AI Act articles
CATEGORIES = {
    "harmful_instructions": {
        "description": "Tests for harmful content generation",
        "regulations": ["EU-AI-Act-Article-5"],
        "severity": "high"
    },
    "misinformation": {
        "description": "Tests for misinformation generation",
        "regulations": ["EU-AI-Act-Article-5", "EU-AI-Act-Article-13"],
        "severity": "high"
    },
    "bias": {
        "description": "Tests for biased or discriminatory content",
        "regulations": ["EU-AI-Act-Article-5", "EU-AI-Act-Article-10"],
        "severity": "medium"
    },
    "privacy": {
        "description": "Tests for privacy-infringing content",
        "regulations": ["EU-AI-Act-Article-10"],
        "severity": "high"
    }
}


def get_categories() -> Dict[str, Dict[str, Any]]:
    """Get the defined test categories.
    
    Returns:
        Dictionary of categories with their descriptions, regulations, and default severity.
    """
    return CATEGORIES 