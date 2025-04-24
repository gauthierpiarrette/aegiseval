# SPDX-License-Identifier: Apache-2.0
"""Core functionality for AegisEval."""

from aegiseval.core.score import (
    calculate_safety_score, get_improvement_areas,
    SafetyScore, SafetyLevel, CategoryScore, RegulationScore, ImprovementArea
)
from aegiseval.core.regulations import (
    get_regulation_definitions, get_regulation_mapping, map_test_to_regulations
)

__all__ = [
    'calculate_safety_score',
    'get_improvement_areas',
    'SafetyScore',
    'SafetyLevel',
    'CategoryScore',
    'RegulationScore',
    'ImprovementArea',
    'get_regulation_definitions',
    'get_regulation_mapping',
    'map_test_to_regulations'
] 