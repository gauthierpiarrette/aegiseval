# SPDX-License-Identifier: Apache-2.0
"""Model runners and adapters for AegisEval."""

from aegiseval.runner.adapters import (
    ModelAdapter, create_adapter, OpenAIAdapter, MockAdapter
)
from aegiseval.runner.evaluator import (
    EvaluationRunner, evaluate_model, evaluate_model_sync,
    TestCaseResult, EvaluationResult
)

# Explicitly define what symbols are exported from this module
__all__ = [
    # Adapters
    'ModelAdapter', 'create_adapter', 'OpenAIAdapter', 'MockAdapter',
    
    # Evaluator
    'EvaluationRunner', 'evaluate_model', 'evaluate_model_sync',
    'TestCaseResult', 'EvaluationResult',
] 