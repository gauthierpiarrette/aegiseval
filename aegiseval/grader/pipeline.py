# SPDX-License-Identifier: Apache-2.0
"""Pipeline for LLM response classification and grading.

This module provides a classification pipeline that applies the classifiers from
the classifiers module and implements fail-fast behavior for illegal content.
"""

import logging
import sys
from typing import Dict, List, Optional

from aegiseval.grader.classifiers import (
    ClassificationResult,
    ClassifierPipeline,
    CsAmContentError
)
from aegiseval.grader.classifiers_numeric import NumericFactClassifier
from aegiseval.grader.classifiers_pii import PIILeakClassifier
from aegiseval.grader.classifiers_selfid import SelfIdentificationClassifier
from aegiseval.config.flags import EXIT_ILLEGAL_CONTENT

# Configure logging
logger = logging.getLogger(__name__)


class IllegalContentError(Exception):
    """Raised when illegal content is detected during classification.
    
    This error indicates that the model response contains content that
    violates legal standards or platform policies and should trigger
    immediate termination of the evaluation.
    """
    pass


class GraderPipeline:
    """Pipeline for classifying and grading LLM responses with fail-fast behavior."""
    
    def __init__(self, classifier_pipeline: Optional[ClassifierPipeline] = None):
        """Initialize the grader pipeline.
        
        Args:
            classifier_pipeline: Optional custom classifier pipeline
                                 (default: creates standard pipeline)
        """
        if classifier_pipeline is None:
            # Create a standard classifier pipeline with all available classifiers
            # Include the new classifiers in the initial list
            classifiers = [
                NumericFactClassifier(),
                PIILeakClassifier(),
                SelfIdentificationClassifier()
            ]
            self.classifier_pipeline = ClassifierPipeline(classifiers)
        else:
            self.classifier_pipeline = classifier_pipeline
    
    def process(self, text: str) -> Dict[str, ClassificationResult]:
        """Process a single text through the classifier pipeline.
        
        Args:
            text: The text to classify and grade
            
        Returns:
            Dictionary mapping category names to classification results
            
        Raises:
            IllegalContentError: If illegal content is detected
        """
        try:
            return self.classifier_pipeline.run_all(text)
        except CsAmContentError as e:
            logger.critical("💥 Illegal CSAM output detected – scan aborted")
            # Re-raise as IllegalContentError for higher-level handling
            raise IllegalContentError("Illegal CSAM output detected") from e
    
    def process_batch(self, texts: List[str]) -> List[Dict[str, ClassificationResult]]:
        """Process multiple texts through the classifier pipeline.
        
        Args:
            texts: List of texts to classify and grade
            
        Returns:
            List of dictionaries mapping category names to classification results
            
        Raises:
            IllegalContentError: If illegal content is detected in any text
        """
        results = []
        for text in texts:
            # Process each text individually to maintain fail-fast behavior
            results.append(self.process(text))
        return results


def handle_illegal_content() -> None:
    """Handle illegal content detection with appropriate error message and exit code.
    
    This function should be called by higher-level components when catching
    IllegalContentError to ensure consistent error handling and exit behavior.
    """
    print("\033[31m💥 Illegal CSAM output detected – scan aborted\033[0m")
    sys.exit(EXIT_ILLEGAL_CONTENT) 