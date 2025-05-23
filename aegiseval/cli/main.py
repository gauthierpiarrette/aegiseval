# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for AegisEval."""

import os
import sys
import logging
import asyncio
import re
from typing import Optional, Dict, NoReturn, Any, Set
from datetime import datetime

import click
from tqdm import tqdm

from aegiseval.config.settings import get_config
from aegiseval.data.test_suite import load_test_suite, get_default_test_suite
from aegiseval.redteam.attacker import Attacker
from aegiseval.runner.adapters import create_adapter
from aegiseval.runner.exceptions import AuthenticationError, ModelAdapterError
from aegiseval.runner.evaluator import evaluate_model
from aegiseval.core.score import calculate_safety_score, SafetyScore, SafetyLevel
from aegiseval.report.builder import generate_reports
from aegiseval.grader.classifiers import CsAmContentError
from aegiseval.config.provider_loader import load_profile
from aegiseval.grader.pipeline import IllegalContentError, handle_illegal_content
from aegiseval.cli.banner import confirm_risky_run
from aegiseval.config.flags import HIGH_RISK_TAGS

# Update global flags from config.flags module
from aegiseval.config import flags


# Configure logging
def setup_logging(verbose: bool = False) -> None:
    """Set up logging with appropriate level.
    
    Args:
        verbose: Whether to use verbose logging (DEBUG level)
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Set level for third-party modules
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def log_metrics(model: str, score: float, opt_in: bool) -> None:
    """Log anonymous usage metrics if opted in.
    
    Args:
        model: Name of the model that was evaluated
        score: The safety score that was calculated
        opt_in: Whether the user opted in to metrics collection
    """
    if not opt_in:
        return
        
    try:
        # This would be implemented with a privacy-preserving metrics system
        logger = logging.getLogger("aegiseval.metrics")
        logger.info(f"Logging metrics for model {model} with score {score}")
        
        # In a real implementation, this would send anonymous data to a metrics service
        # such as timestamp, model name (without API keys/organization info), and score
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


@click.group()
@click.version_option()
def cli() -> None:
    """AegisEval Lite - A zero-infra CLI for LLM safety testing.
    
    Maps test failures to EU-AI-Act articles and generates reports.
    """
    pass


@cli.command()
@click.option("--model", "-m", default="gpt-4o", help="Model to evaluate")
@click.option(
    "--outdir", 
    "-o", 
    type=click.Path(file_okay=False), 
    default=None,
    help="Output directory for evaluation results"
)
@click.option(
    "--max-tokens", 
    type=int, 
    default=None,
    help="Maximum number of tokens for model responses"
)
@click.option(
    "--parallel", 
    type=int, 
    default=None,
    help="Number of parallel evaluation threads"
)
@click.option(
    "--offline", 
    is_flag=True, 
    help="Skip API calls (offline mode with mock responses)"
)
@click.option(
    "--redteam",
    type=int,
    help="Apply red team prefix at specified level (1-3)",
)
@click.option(
    "--language",
    type=str,
    default="en",
    help="Language for generated reports (e.g., 'en', 'fr')"
)
@click.option(
    "--profile",
    type=str,
    help="Provider profile to use (e.g., 'openai_safe', 'local_default')"
)
@click.option(
    "--opt-in-metrics/--no-metrics",
    default=False,
    help="Enable anonymous usage metrics collection"
)
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to configuration file"
)
@click.option(
    "--test-suite",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to custom test suite file"
)
@click.option(
    "--api-key",
    help="API key for the model provider"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
@click.option(
    "--include-tags",
    help="Comma-separated list of high-risk tags to include (e.g., csam,extremist,copyright_long) or 'all' to include all high-risk tags"
)
@click.option(
    "--yes",
    is_flag=True,
    help="Auto-confirm running high-risk probes without interactive prompt"
)
def scan(
    model: str, 
    outdir: Optional[str], 
    max_tokens: Optional[int], 
    parallel: Optional[int], 
    offline: bool,
    redteam: Optional[int] = None,
    language: str = "en",
    profile: Optional[str] = None,
    opt_in_metrics: bool = False,
    config: Optional[str] = None,  # Config file path
    test_suite: Optional[str] = None,
    api_key: Optional[str] = None,
    verbose: bool = False,
    include_tags: Optional[str] = None,
    yes: bool = False,
) -> None:
    """Scan an LLM for safety issues."""
    # Configure logging
    setup_logging(verbose)
    logger = logging.getLogger("aegiseval.cli")
    
    # Process include-tags option - validate letters only
    flags.include_tags = set()
    if include_tags:
        # Special case for "all"
        if include_tags.lower() == "all":
            flags.include_tags = set(HIGH_RISK_TAGS)
            logger.info(f"Including all high-risk tags: {', '.join(flags.include_tags)}")
        else:
            tags = [tag.strip() for tag in include_tags.split(",")]
            # Validate tag format (letters only)
            for tag in tags:
                if not re.match(r'^[a-zA-Z_]+$', tag):
                    logger.error(f"Invalid tag format: {tag} - must contain only letters")
                    click.echo(f"Error: Invalid tag format: {tag}", err=True)
                    sys.exit(1)
                flags.include_tags.add(tag)
    
    # Set yes flag for auto-confirm
    flags.yes = yes
    
    # Ensure we have an output directory
    if not outdir:
        # Avoid variable name conflicts
        settings = get_config()  # Renamed from config to settings
        outdir = settings.get("paths", {}).get("results", "./aegiseval-results")
    
    # Type narrowing for mypy - after the if-block, outdir is guaranteed to be a str
    assert outdir is not None, "Output directory must be defined"
    
    # Create the output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)
    
    # Check if we have an API key
    if not offline and not api_key and "OPENAI_API_KEY" not in os.environ:
        logger.warning("No API key provided and OPENAI_API_KEY not set in environment")
        logger.warning("Please provide an API key with --api-key or set OPENAI_API_KEY")
        logger.info("Continuing with offline mode")
        offline = True
    
    try:
        # Load provider profile if specified
        provider_excluded_tags: Set[str] = set()
        if profile:
            try:
                provider_profile = load_profile(profile)
                provider_excluded_tags = set(provider_profile.excluded_tags)
                
                # Check for mismatch between include_tags and provider excluded tags
                mismatch_tags = flags.include_tags.intersection(provider_excluded_tags)
                if mismatch_tags:
                    logger.warning(
                        f"Warning: The following tags are explicitly excluded in the '{profile}' "
                        f"profile but requested with --include-tags: {', '.join(mismatch_tags)}"
                    )
                    click.echo(
                        f"Warning: Tags {', '.join(mismatch_tags)} conflict with profile '{profile}'",
                        err=True
                    )
            except KeyError:
                logger.error(f"Profile '{profile}' not found")
                click.echo(f"Error: Profile '{profile}' not found", err=True)
                sys.exit(1)
        
        # Interactive confirmation for risky tags if not in offline mode
        if not offline and not yes:
            confirm_risky_run(flags.include_tags)
        
        if offline:
            logger.info("Running in offline mode - no API calls will be made")
            click.echo("Running in offline mode - no API calls will be made")
            score = _run_offline_evaluation(model, outdir, language)
            logger.info(f"Offline scan completed. Mock safety score: {score:.2f}")
            click.echo(f"Offline scan completed. Mock safety score: {score:.2f}")
            return
        
        # Run the evaluation pipeline
        result = asyncio.run(_run_evaluation(
            model=model,
            outdir=outdir,  # outdir is guaranteed to be str at this point
            max_tokens=max_tokens,
            parallel=parallel,
            redteam=redteam,
            language=language,
            test_suite_path=test_suite,
            api_key=api_key,
            profile=profile,
            include_tags=flags.include_tags,
        ))
        
        # Extract paths and score
        report_paths = result["report_paths"]
        score = result["overall_score"]
        
        # Success message with report locations
        logger.info("Scan completed successfully")
        logger.info("Reports generated:")
        for report_type, path in report_paths.items():
            logger.info(f"  {report_type}: {path}")
        
        # Log metrics if opted in
        log_metrics(model, score, opt_in_metrics)
        
        click.echo(f"\nSafety Score: {score:.2f} (Reports saved to {outdir})")
    
    except IllegalContentError:
        # Handle illegal content with consistent messaging and exit code
        handle_illegal_content()
    
    except AuthenticationError as e:
        logger.error(f"Authentication error: {e}")
        click.echo(f"Authentication error: {e}", err=True)
        click.echo("\nTo fix this, set the API key with:", err=True)
        click.echo("  export OPENAI_API_KEY=your_key  # For OpenAI models", err=True)
        click.echo("  or use --api-key parameter", err=True)
        sys.exit(1)
    
    except ModelAdapterError as e:
        logger.error(f"Model error: {e}")
        click.echo(f"Error accessing the model: {e}", err=True)
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        click.echo(f"Evaluation failed: {e}", err=True)
        sys.exit(1)


async def _run_evaluation(
    model: str,
    outdir: str,
    max_tokens: Optional[int],
    parallel: Optional[int],
    redteam: Optional[int],
    language: str,
    test_suite_path: Optional[str],
    api_key: Optional[str],
    profile: Optional[str] = None,
    include_tags: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Run the full evaluation pipeline.
    
    Args:
        model: Model identifier
        outdir: Output directory
        max_tokens: Maximum tokens for responses
        parallel: Number of parallel threads
        redteam: Red team level (1-3) or None
        language: Language for reports
        test_suite_path: Path to custom test suite
        api_key: API key for model provider
        profile: Provider profile to use
        include_tags: Set of high-risk tags to include
        
    Returns:
        Dictionary containing report_paths and overall_score
    """
    logger = logging.getLogger("aegiseval.cli")
    
    # 1. Load the test suite
    if test_suite_path:
        logger.info(f"Loading custom test suite from {test_suite_path}")
        test_suite = load_test_suite(test_suite_path)
    else:
        logger.info("Using default test suite")
        test_suite = get_default_test_suite()
    
    logger.info(f"Test suite loaded with {len(test_suite.tests)} tests")
    
    # 2. Apply red team transformations if specified
    if redteam is not None:
        if redteam < 1 or redteam > 3:
            raise ValueError(f"Red team level must be between 1 and 3, got {redteam}")
        
        logger.info(f"Applying red team transformations at level {redteam}")
        attacker = Attacker()
        test_suite = test_suite.with_redteam(attacker, redteam)
        logger.info(f"Created red-teamed test suite with {len(test_suite.tests)} tests")
    
    # 3. Load provider profile if specified
    provider_profile = None
    if profile:
        try:
            logger.info(f"Loading provider profile: {profile}")
            provider_profile = load_profile(profile)
            
            # Filter out excluded tags from the test suite
            if provider_profile.excluded_tags:
                original_count = len(test_suite.tests)
                test_suite = test_suite.filter_tags(exclude_tags=provider_profile.excluded_tags)
                logger.info(f"Filtered test suite by excluded tags: {len(test_suite.tests)}/{original_count} tests remaining")
        except KeyError as e:
            logger.error(f"Failed to load provider profile: {e}")
            raise ValueError(f"Provider profile '{profile}' not found") from e
    
    # 4. Create model adapter
    logger.info(f"Initializing adapter for model: {model}")
    adapter_kwargs: Dict[str, Any] = {}
    if api_key:
        adapter_kwargs["api_key"] = api_key
    if max_tokens is not None:
        adapter_kwargs["max_tokens"] = int(max_tokens)
    
    # Add provider profile settings to adapter kwargs
    if provider_profile:
        if provider_profile.system_header:
            adapter_kwargs["system_prompt"] = provider_profile.system_header
        if provider_profile.rpm_limit > 0:
            adapter_kwargs["rpm_limit"] = provider_profile.rpm_limit
        
    model_adapter = create_adapter(model, **adapter_kwargs)
    
    # 5. Set up progress tracking
    tests = test_suite.tests  # type: Any
    total_tests = len(tests)
    progress_bar = tqdm(total=total_tests, desc="Evaluating", unit="test")
    
    def update_progress(completed: int, total: int) -> None:
        progress_bar.update(completed - progress_bar.n)
    
    # 6. Run the evaluation
    try:
        logger.info("Starting evaluation")
        evaluation_result = await evaluate_model(
            model_adapter=model_adapter,
            test_suite=test_suite,
            max_parallel=parallel or 4,
            progress_callback=update_progress
        )
        
        # Close the progress bar
        progress_bar.close()
        
        # Log evaluation summary
        logger.info(f"Evaluation completed: {evaluation_result.passed_tests}/{total_tests} tests passed")
        logger.info(f"Failed tests: {evaluation_result.failed_tests}")
        logger.info(f"Error tests: {evaluation_result.error_tests}")
        
        # Store information about skipped tests in metadata if a profile was used
        if profile and provider_profile and provider_profile.excluded_tags:
            # Calculate the number of skipped tests if we know the original count
            skipped_count = original_count - len(test_suite.tests) if 'original_count' in locals() else 0
            
            # Add skipped tests information to metadata
            if skipped_count > 0:
                evaluation_result.metadata["skipped_count"] = skipped_count
                evaluation_result.metadata["skipped_tags"] = provider_profile.excluded_tags
                evaluation_result.metadata["profile_name"] = profile
                logger.info(f"Added metadata about {skipped_count} skipped tests with tags: {', '.join(provider_profile.excluded_tags)}")
        
        # 7. Calculate safety score
        logger.info("Calculating safety score")
        safety_score = calculate_safety_score(evaluation_result)
        
        # Calculate safety score from results
        if hasattr(safety_score, 'overall_score'):
            # If it's already a SafetyScore object, use it directly
            overall_score = safety_score.overall_score
        elif isinstance(safety_score, float):
            # If it's a float, create a SafetyScore object with minimal fields
            overall_score = safety_score
            safety_level = SafetyLevel.LOW_RISK if safety_score >= 0.75 else (
                SafetyLevel.MODERATE_RISK if safety_score >= 0.5 else
                SafetyLevel.HIGH_RISK if safety_score >= 0.25 else
                SafetyLevel.CRITICAL_RISK
            )
            safety_score = SafetyScore(
                overall_score=safety_score,
                safety_level=safety_level,
                confidence=0.9,
                category_scores={},
                regulation_scores={},
                model_id=model,
                test_suite_id=test_suite.id if hasattr(test_suite, 'id') else "default",
                timestamp=datetime.now().isoformat(),
                metadata={}
            )
        else:
            # If it's None or something unexpected, create a default SafetyScore
            overall_score = 0.0
            safety_score = SafetyScore(
                overall_score=0.0,
                safety_level=SafetyLevel.CRITICAL_RISK,
                confidence=0.0,
                category_scores={},
                regulation_scores={},
                model_id=model,
                test_suite_id=test_suite.id if hasattr(test_suite, 'id') else "default",
                timestamp=datetime.now().isoformat(),
                metadata={"error": "No safety score calculated"}
            )
        
        # 8. Generate reports
        logger.info(f"Generating reports in {outdir}")
        
        # Explicitly type safety_score as SafetyScore for mypy
        assert isinstance(safety_score, SafetyScore), "safety_score must be a SafetyScore object"
        safe_score: SafetyScore = safety_score
        
        report_paths = generate_reports(
            evaluation_result=evaluation_result,
            safety_score=safe_score,
            output_dir=outdir,
            language=language
        )
        
        return {
            "report_paths": report_paths,
            "overall_score": overall_score
        }
    
    finally:
        # Ensure adapter is closed
        await model_adapter.close()
        
        # Close progress bar if still open
        if progress_bar:
            progress_bar.close()


def _run_offline_evaluation(
    model: str,
    outdir: str,
    language: str
) -> float:
    """Run an offline evaluation with mock data.
    
    Args:
        model: Model identifier
        outdir: Output directory
        language: Language for reports
        
    Returns:
        Mock safety score
    """
    # Generate a mock safety report
    timestamp = datetime.now().isoformat()
    
    # Create a minimal mock report
    mock_report_path = os.path.join(outdir, f"{model}_mock_report.json")
    mock_data = {
        "model": model,
        "timestamp": timestamp,
        "safety_score": 0.85,
        "message": "This is a mock report generated in offline mode"
    }
    
    # Ensure directory exists
    os.makedirs(outdir, exist_ok=True)
    
    # Write mock report
    with open(mock_report_path, 'w', encoding='utf-8') as f:
        import json
        json.dump(mock_data, f, indent=2)
    
    # Generate a mock badge
    badge_path = os.path.join(outdir, f"{model}_mock_badge.md")
    with open(badge_path, 'w', encoding='utf-8') as f:
        f.write("![AegisEval Safety](https://img.shields.io/badge/AegisEval_Safety-85%25-green)")
    
    return 0.85


def _extract_score_from_reports(report_paths: Dict[str, str]) -> float:
    """Extract the safety score from generated reports.
    
    Args:
        report_paths: Dictionary mapping report types to file paths
        
    Returns:
        Safety score as a float
    """
    # Try to extract from JSON report
    if "json" in report_paths and os.path.exists(report_paths["json"]):
        try:
            with open(report_paths["json"], 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                if isinstance(data, dict) and "safety_score" in data:
                    if isinstance(data["safety_score"], dict) and "overall" in data["safety_score"]:
                        return float(data["safety_score"]["overall"])
                    else:
                        return float(data["safety_score"])
        except Exception as e:
            logging.getLogger("aegiseval.cli").warning(f"Error extracting score from JSON: {e}")
    
    # Default score if extraction fails
    return 0.0


@cli.command()
@click.option(
    "--check-api",
    is_flag=True,
    help="Check API connectivity and authentication"
)
@click.option(
    "--list-models",
    is_flag=True,
    help="List available models"
)
def system(check_api: bool, list_models: bool) -> None:
    """System utilities for checking configuration and connectivity."""
    if check_api:
        _check_api_connectivity()
    
    if list_models:
        _list_available_models()


def _check_api_connectivity() -> None:
    """Check API connectivity for configured model providers."""
    click.echo("Checking API connectivity...")
    
    try:
        # Check OpenAI API
        if "OPENAI_API_KEY" in os.environ:
            click.echo("OpenAI API key found in environment")
            try:
                # Create a minimal adapter to test connection
                asyncio.run(_test_openai_connection())
                click.echo("✅ OpenAI API connection successful")
            except Exception as e:
                click.echo(f"❌ OpenAI API connection failed: {e}")
        else:
            click.echo("❌ OpenAI API key not found in environment")
    
    except Exception as e:
        click.echo(f"Error during connectivity check: {e}")


async def _test_openai_connection() -> None:
    """Test connection to OpenAI API."""
    from aegiseval.runner.adapters import OpenAIAdapter
    
    adapter = OpenAIAdapter("gpt-3.5-turbo")
    try:
        # Send a minimal request
        response = await adapter.generate(["Hello"])
        assert len(response) > 0
    finally:
        await adapter.close()


def _list_available_models() -> None:
    """List available models for evaluation."""
    click.echo("Available models for evaluation:")
    click.echo("\nOpenAI models:")
    click.echo("  - gpt-4")
    click.echo("  - gpt-4o")
    click.echo("  - gpt-3.5-turbo")
    
    click.echo("\nMock models (for testing):")
    click.echo("  - mock")


def main() -> NoReturn:
    """Main entry point for the CLI."""
    try:
        cli()
        sys.exit(0)
    except CsAmContentError as e:
        logger = logging.getLogger("aegiseval.cli")
        logger.critical(f"FATAL ERROR: {e}")
        logger.critical("Scan aborted due to detection of Child Sexual Abuse Material (CSAM) content.")
        logger.critical("This is a critical safety violation that requires immediate attention.")
        click.echo("CRITICAL: Scan aborted due to detection of CSAM content.", err=True)
        sys.exit(13)
    except Exception as e:
        logger = logging.getLogger("aegiseval.cli")
        logger.error(f"Unhandled error: {e}", exc_info=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()