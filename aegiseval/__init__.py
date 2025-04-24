# SPDX-License-Identifier: Apache-2.0
"""AegisEval - A tool for safety testing LLMs."""

__version__ = "0.1.0"

from aegiseval.data.test_suite import TestCase, TestSuite, load_test_suite
from aegiseval.config.settings import get_config, load_config, get_validated_config
from aegiseval.config.schema import AegisEvalConfig
from aegiseval.grader.classifiers import (
    ClassifierPipeline, ClassificationResult, run_classifiers,
    HarmfulContentClassifier, MisinformationClassifier, BiasClassifier
)
from aegiseval.redteam.attacker import Attacker, Attack, AttackType, AttackLevel
from aegiseval.runner.adapters import ModelAdapter, create_adapter
from aegiseval.runner.evaluator import (
    EvaluationRunner, evaluate_model, evaluate_model_sync,
    TestCaseResult, EvaluationResult
)
from aegiseval.core.score import (
    calculate_safety_score, SafetyScore, SafetyLevel
)
from aegiseval.report.builder import generate_reports
from aegiseval.cli.main import main
from aegiseval.integration import (
    EvaluationPipeline, run_evaluation, check_system_compatibility
)

# Explicitly define what symbols are exported from this package
__all__ = [
    # Data/test suite
    'TestCase', 'TestSuite', 'load_test_suite',
    
    # Configuration 
    'get_config', 'load_config', 'get_validated_config', 'AegisEvalConfig',
    
    # Graders/classifiers
    'ClassifierPipeline', 'ClassificationResult', 'run_classifiers',
    'HarmfulContentClassifier', 'MisinformationClassifier', 'BiasClassifier',
    
    # Red teaming
    'Attacker', 'Attack', 'AttackType', 'AttackLevel',
    
    # Model adapters
    'ModelAdapter', 'create_adapter',
    
    # Evaluation
    'EvaluationRunner', 'evaluate_model', 'evaluate_model_sync',
    'TestCaseResult', 'EvaluationResult',
    
    # Scoring
    'calculate_safety_score', 'SafetyScore', 'SafetyLevel',
    
    # Reporting
    'generate_reports',
    
    # CLI
    'main',
    
    # Integration
    'EvaluationPipeline', 'run_evaluation', 'check_system_compatibility',
] 