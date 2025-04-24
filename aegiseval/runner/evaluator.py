# SPDX-License-Identifier: Apache-2.0
"""Evaluation runner for coordinating model testing."""

import asyncio
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable

from pydantic import BaseModel, Field

from aegiseval.data.test_suite import TestCase, TestSuite
from aegiseval.runner.adapters import ModelAdapter
from aegiseval.grader.classifiers import (
    ClassifierPipeline, ClassificationResult
)
from aegiseval.config.settings import get_config

# Configure logging
logger = logging.getLogger(__name__)

class TestCaseResult(BaseModel):
    """Result of evaluating a single test case."""
    test_case_id: str = Field(..., description="ID of the test case")
    prompt: str = Field(..., description="The prompt that was tested")
    response: str = Field(..., description="The model's response")
    passed: bool = Field(..., description="Whether the test was passed")
    classification_results: Dict[str, ClassificationResult] = Field(
        ..., description="Results from each classifier"
    )
    latency_ms: int = Field(..., description="Response time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if test failed due to error")
    regulations: List[str] = Field(default_factory=list, description="Relevant regulations")
    category: str = Field(..., description="Test case category")
    severity: str = Field(..., description="Test case severity")
    timestamp: datetime = Field(default_factory=datetime.now, description="When test was run")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class EvaluationResult(BaseModel):
    """Comprehensive results of an evaluation run."""
    model_id: str = Field(..., description="ID of the model tested")
    test_suite_id: str = Field(..., description="ID of the test suite")
    results: List[TestCaseResult] = Field(..., description="Individual test results")
    start_time: datetime = Field(..., description="When evaluation started")
    end_time: datetime = Field(..., description="When evaluation finished")
    total_tests: int = Field(..., description="Total number of tests")
    passed_tests: int = Field(..., description="Number of tests that passed")
    failed_tests: int = Field(..., description="Number of tests that failed")
    error_tests: int = Field(0, description="Number of tests that errored")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class EvaluationRunner:
    """Runs evaluations against LLMs with parallel processing."""
    
    def __init__(
        self,
        model_adapter: ModelAdapter,
        classifier_pipeline: Optional[ClassifierPipeline] = None,
        max_parallel: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None
    ):
        """Initialize the evaluation runner.
        
        Args:
            model_adapter: Adapter for the model to evaluate
            classifier_pipeline: Pipeline for classifying responses
            max_parallel: Maximum number of parallel requests
            timeout: Timeout for each request in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.model_adapter = model_adapter
        self.classifier_pipeline = classifier_pipeline or ClassifierPipeline()
        
        # Load config with defaults
        config = get_config()
        eval_config = config.get("evaluation", {})
        
        self.max_parallel = max_parallel or eval_config.get("parallel", 4)
        self.timeout = timeout or eval_config.get("timeout", 30)
        self.max_retries = max_retries or eval_config.get("max_retries", 3)
        
        # Semaphore for controlling parallel execution
        self._semaphore = asyncio.Semaphore(self.max_parallel)
    
    async def evaluate_suite(
        self, 
        test_suite: TestSuite,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> EvaluationResult:
        """Evaluate an entire test suite against the model.
        
        Args:
            test_suite: Test suite to evaluate
            progress_callback: Optional callback for progress updates
            
        Returns:
            Evaluation results
            
        Raises:
            ValueError: If test_suite is empty
        """
        if not test_suite.tests:
            raise ValueError("Test suite is empty")
        
        logger.info(f"Starting evaluation of test suite: {test_suite.id}")
        logger.info(f"Model: {self.model_adapter.model_id}")
        logger.info(f"Total tests: {len(test_suite.tests)}")
        
        start_time = datetime.now()
        
        # Create tasks for all test cases
        tasks = []
        for i, test_case in enumerate(test_suite.tests):
            task = self._evaluate_test_case_with_retry(test_case)
            tasks.append(task)
        
        # Process in batches to control concurrency and report progress
        results = []
        total = len(tasks)
        completed = 0
        
        # Create batches based on max_parallel
        for i in range(0, total, self.max_parallel):
            batch = tasks[i:i + self.max_parallel]
            batch_results = await asyncio.gather(*batch)
            results.extend(batch_results)
            
            # Update progress
            completed += len(batch)
            if progress_callback:
                progress_callback(completed, total)
            
            logger.debug(f"Completed {completed}/{total} tests")
        
        end_time = datetime.now()
        
        # Count passed and failed tests
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = sum(1 for r in results if not r.passed and not r.error)
        error_tests = sum(1 for r in results if r.error)
        
        logger.info(f"Evaluation completed. Passed: {passed_tests}, Failed: {failed_tests}, Errors: {error_tests}")
        
        return EvaluationResult(
            model_id=self.model_adapter.model_id,
            test_suite_id=test_suite.id,
            results=results,
            start_time=start_time,
            end_time=end_time,
            total_tests=total,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            error_tests=error_tests,
            metadata={
                "duration_seconds": (end_time - start_time).total_seconds(),
                "test_suite_name": test_suite.name,
                "test_suite_version": test_suite.version,
            }
        )
    
    async def _evaluate_test_case_with_retry(self, test_case: TestCase) -> TestCaseResult:
        """Evaluate a test case with retries.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            Test case result
        """
        retries = 0
        while retries <= self.max_retries:
            try:
                return await self._evaluate_test_case(test_case)
            except Exception as e:
                retries += 1
                if retries <= self.max_retries:
                    # Wait before retrying with exponential backoff
                    delay = 2 ** retries
                    logger.warning(f"Retrying test case {test_case.id} after error: {str(e)}. "
                                  f"Retry {retries}/{self.max_retries} in {delay}s")
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded, return error result
                    logger.error(f"Test case {test_case.id} failed after {retries} retries: {str(e)}")
                    return TestCaseResult(
                        test_case_id=test_case.id,
                        prompt=test_case.prompt,
                        response="",
                        passed=False,
                        classification_results={},
                        latency_ms=0,
                        error=str(e),
                        regulations=test_case.regulations,
                        category=test_case.category,
                        severity=test_case.severity,
                        metadata={"retries": retries}
                    )
        
        # This should never be reached due to the return statements above,
        # but mypy needs it for type safety
        return TestCaseResult(
            test_case_id=test_case.id,
            prompt=test_case.prompt,
            response="",
            passed=False,
            classification_results={},
            latency_ms=0,
            error="Maximum retries exceeded",
            regulations=test_case.regulations,
            category=test_case.category,
            severity=test_case.severity,
            metadata={"retries": retries}
        )
    
    async def _evaluate_test_case(self, test_case: TestCase) -> TestCaseResult:
        """Evaluate a single test case.
        
        Args:
            test_case: Test case to evaluate
            
        Returns:
            Test case result
        """
        async with self._semaphore:
            start_time = time.time()
            
            try:
                # Get response from model
                responses = await self.model_adapter.generate([test_case.prompt])
                response = responses[0] if responses else ""
                
                # Measure latency
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Classify the response
                classification_results = self.classifier_pipeline.run_all(response)
                
                # Determine if the test passed (no unsafe content detected)
                passed = all(not result.is_flagged for result in classification_results.values())
                
                return TestCaseResult(
                    test_case_id=test_case.id,
                    prompt=test_case.prompt,
                    response=response,
                    passed=passed,
                    classification_results=classification_results,
                    latency_ms=latency_ms,
                    error="",  # No error
                    regulations=test_case.regulations,
                    category=test_case.category,
                    severity=test_case.severity
                )
            except Exception as e:
                # Re-raise to be handled by the retry mechanism
                logger.debug(f"Error evaluating test case {test_case.id}: {str(e)}")
                raise
    
    def evaluate_suite_sync(self, test_suite: TestSuite) -> EvaluationResult:
        """Synchronous wrapper for evaluate_suite.
        
        Args:
            test_suite: Test suite to evaluate
            
        Returns:
            Evaluation results
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.evaluate_suite(test_suite))
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.model_adapter.close()


async def evaluate_model(
    model_adapter: ModelAdapter,
    test_suite: TestSuite,
    max_parallel: int = 4,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> EvaluationResult:
    """Evaluate a model using the provided test suite.
    
    Args:
        model_adapter: Adapter for the model to evaluate
        test_suite: Test suite to evaluate
        max_parallel: Maximum number of parallel requests
        progress_callback: Optional callback for progress updates
        
    Returns:
        Evaluation results
    """
    runner = EvaluationRunner(
        model_adapter=model_adapter,
        max_parallel=max_parallel
    )
    
    try:
        return await runner.evaluate_suite(test_suite, progress_callback)
    finally:
        await runner.close()


def evaluate_model_sync(
    model_adapter: ModelAdapter,
    test_suite: TestSuite,
    max_parallel: int = 4
) -> EvaluationResult:
    """Synchronous wrapper for evaluate_model.
    
    Args:
        model_adapter: Adapter for the model to evaluate
        test_suite: Test suite to evaluate
        max_parallel: Maximum number of parallel requests
        
    Returns:
        Evaluation results
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Create new event loop if none exists
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        evaluate_model(
            model_adapter=model_adapter,
            test_suite=test_suite,
            max_parallel=max_parallel
        )
    )

def _create_test_result(test: TestCase, response: str, passed: bool, reason: Optional[str] = None) -> TestCaseResult:
    """Create a test case result from a test and response.
    
    Args:
        test: The test case that was evaluated
        response: The model response to the test
        passed: Whether the test passed
        reason: Optional explanation for the result
        
    Returns:
        A TestCaseResult object
    """
    # Use empty dict for classification results since we don't have actual results
    classification_results: Dict[str, ClassificationResult] = {}
    
    return TestCaseResult(
        test_case_id=test.id,
        category=test.category,
        regulations=test.regulations,
        prompt=test.prompt,
        response=response,
        passed=passed,
        severity=test.severity,
        classification_results=classification_results,
        latency_ms=0,  # Not measured in this simple implementation
        error="",  # No error
        metadata={"reason": reason or ""}
    ) 