# SPDX-License-Identifier: Apache-2.0
"""Grader package for AegisEval Lite."""

from typing import Dict, Type

from aegiseval.grader.classifiers import ClassifierBase
from aegiseval.grader.classifiers_numeric import NumericFactClassifier
from aegiseval.grader.classifiers_pii import PIILeakClassifier
from aegiseval.grader.classifiers_selfid import SelfIdentificationClassifier

# Registry of all available classifiers
CLASSIFIER_REGISTRY: Dict[str, Type[ClassifierBase]] = {
    "numeric_fact": NumericFactClassifier,
    "pii_leak": PIILeakClassifier,
    "self_identification": SelfIdentificationClassifier,
}

__all__ = [
    "CLASSIFIER_REGISTRY",
    "NumericFactClassifier",
    "PIILeakClassifier",
    "SelfIdentificationClassifier",
] 