# SPDX-License-Identifier: Apache-2.0
"""Report generation for AegisEval Lite."""

import json
import os
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import jinja2
from xhtml2pdf import pisa

from aegiseval.report.i18n import get_translator
from aegiseval.runner.evaluator import EvaluationResult
from aegiseval.core.score import SafetyScore, get_improvement_areas
from aegiseval.config.settings import get_config

# Configure logging
logger = logging.getLogger(__name__)


class ReportBuilder:
    """Generates comprehensive evaluation reports.
    
    This class handles generation of JSON, PDF, HTML, and badge reports
    from evaluation results.
    """
    
    def __init__(
        self,
        evaluation_result: EvaluationResult,
        safety_score: SafetyScore,
        language: str = "en",
        output_dir: Optional[str] = None
    ):
        """Initialize the report builder.
        
        Args:
            evaluation_result: Results of the evaluation
            safety_score: Calculated safety score
            language: Language code for the report
            output_dir: Directory for report outputs
        """
        self.evaluation_result = evaluation_result
        self.safety_score = safety_score
        self.language = language
        
        # Get configuration
        config = get_config()
        self.output_dir = output_dir or config.get("paths", {}).get("results", "./aegiseval-results")
        
        # Initialize translator
        self.translator = get_translator(language)
        
        # Initialize Jinja environment
        template_dir = Path(__file__).parent / "templates"
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Add translator function to Jinja
        self.jinja_env.globals['t'] = self.translator.get
        self.jinja_env.globals['get_category_name'] = self._get_category_display_name
        
        # Prepare output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Process category names
        self._process_category_names()
    
    def _process_category_names(self) -> None:
        """Process category names to remove prefixes and prepare display names."""
        # Update category scores with cleaned category names
        processed_categories = {}
        for category, score in self.safety_score.category_scores.items():
            # Strip the "categories." prefix if present
            clean_category = category.split(".", 1)[-1]
            score.category = clean_category
            processed_categories[clean_category] = score
        
        self.safety_score.category_scores = processed_categories
        
        # Also update categories in test results
        for result in self.evaluation_result.results:
            if result.category:
                result.category = result.category.split(".", 1)[-1]
    
    def _get_category_display_name(self, category: str) -> str:
        """Get a display name for a category.
        
        Args:
            category: The category identifier
            
        Returns:
            A user-friendly display name for the category
        """
        # First try to get the lay_name from translation
        trans_key = f'categories.{category}.lay_name'
        lay_name = self.translator.get(trans_key)
        
        # If the key was returned unchanged, it means no translation was found
        if lay_name == trans_key:
            # Fall back to the category title
            title_key = f'categories.{category}'
            title = self.translator.get(title_key)
            
            # If even that returns the key, use the category name directly
            if title == title_key:
                return category.replace('_', ' ').title()
            return title
        
        return lay_name
    
    def generate_all_reports(self) -> Dict[str, str]:
        """Generate all report formats.
        
        Returns:
            Dictionary mapping report types to file paths
        """
        report_paths = {}
        
        # Generate all report types
        report_paths["json"] = self.generate_json()
        report_paths["html"] = self.generate_html()
        report_paths["pdf"] = self.generate_pdf()
        md_path, svg_path = self.generate_badge()
        report_paths["badge_md"] = md_path
        report_paths["badge_svg"] = svg_path
        
        return report_paths
    
    def _safe_model_id(self) -> str:
        """Get a sanitized version of model_id for filesystem use.
        
        Returns:
            Sanitized model ID safe for filename use
        """
        # Replace slashes with underscores to avoid directory structure issues
        model_id = self.safety_score.model_id
        return model_id.replace("/", "_").replace(":", "_")
    
    def _ensure_output_path(self, filepath: str) -> str:
        """Ensure the directory for a file path exists.
        
        Args:
            filepath: Path where a file will be written
            
        Returns:
            The same filepath after ensuring directory exists
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        return filepath
    
    def generate_json(self) -> str:
        """Generate a JSON report.
        
        Returns:
            Path to the generated JSON file
        """
        # Use safe model ID to avoid directory structure issues
        model_id = self._safe_model_id()
        json_path = os.path.join(self.output_dir, f"{model_id}_report.json")
        
        # Prepare safety score data with conditional confidence
        safety_score_data = {
            "overall": self.safety_score.overall_score,
            "level": self.safety_score.safety_level
        }
        
        # Only include confidence if it's non-zero (statsmodels was available)
        if self.safety_score.confidence > 0:
            safety_score_data["confidence"] = self.safety_score.confidence
        
        # Prepare the report data
        report_data = {
            "model": self.safety_score.model_id,  # Keep original model ID in report
            "test_suite": self.safety_score.test_suite_id,
            "timestamp": self.safety_score.timestamp,
            "safety_score": safety_score_data,
            "category_scores": {
                category: {
                    "pass_rate": score.pass_rate,
                    "tests_total": score.tests_total,
                    "tests_passed": score.tests_passed,
                    "tests_failed": score.tests_failed,
                    "severity": score.severity
                }
                for category, score in self.safety_score.category_scores.items()
            },
            "regulation_scores": {
                reg_id: {
                    "title": score.title,
                    "description": score.description,
                    "pass_rate": score.pass_rate,
                    "compliance_level": score.compliance_level,
                    "tests_total": score.tests_total,
                    "tests_passed": score.tests_passed,
                    "tests_failed": score.tests_failed,
                    "severity": score.severity
                }
                for reg_id, score in self.safety_score.regulation_scores.items()
            },
            "improvement_areas": [
                area.dict()
                for area in get_improvement_areas(self.safety_score, self.evaluation_result)
            ],
            "test_results": [
                {
                    "id": result.test_case_id,
                    "category": result.category,
                    "prompt": result.prompt,
                    "response": result.response,
                    "passed": result.passed,
                    "severity": result.severity,
                    "regulations": result.regulations,
                    "error": result.error,
                    "latency_ms": result.latency_ms
                }
                for result in self.evaluation_result.results
            ],
            "metadata": {
                "language": self.language,
                "aegiseval_version": "0.1.0",
                "total_tests": self.evaluation_result.total_tests,
                "passed_tests": self.evaluation_result.passed_tests,
                "failed_tests": self.evaluation_result.failed_tests,
                "error_tests": self.evaluation_result.error_tests,
                "evaluation_duration": (
                    self.evaluation_result.end_time - self.evaluation_result.start_time
                ).total_seconds()
            }
        }
        
        # Ensure the directory exists and write the JSON report
        self._ensure_output_path(json_path)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated JSON report: {json_path}")
        return json_path
    
    def generate_html(self) -> str:
        """Generate an HTML report.
        
        Returns:
            Path to the generated HTML file
        """
        # Use safe model ID to avoid directory structure issues
        model_id = self._safe_model_id()
        html_path = os.path.join(self.output_dir, f"{model_id}_report.html")
        
        # Prepare template data
        template_data = self._prepare_template_data()
        
        # Render the HTML template
        template = self.jinja_env.get_template("report.html.jinja")
        html_content = template.render(**template_data)
        
        # Ensure the directory exists and write the HTML file
        self._ensure_output_path(html_path)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {html_path}")
        return html_path
    
    def generate_pdf(self) -> str:
        """Generate a PDF report.
        
        Returns:
            Path to the generated PDF file
        """
        # Use safe model ID to avoid directory structure issues
        model_id = self._safe_model_id()
        pdf_path = os.path.join(self.output_dir, f"{model_id}_report.pdf")
        
        # First generate HTML content
        template_data = self._prepare_template_data()
        template = self.jinja_env.get_template("report.html.jinja")
        html_content = template.render(**template_data)
        
        # Ensure the directory exists
        self._ensure_output_path(pdf_path)
        
        # Convert HTML to PDF
        with open(pdf_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                src=html_content,
                dest=pdf_file
            )
        
        if pisa_status.err:
            logger.error(f"Error generating PDF: {pisa_status.err}")
            raise Exception(f"PDF generation failed: {pisa_status.err}")
        
        logger.info(f"Generated PDF report: {pdf_path}")
        return pdf_path
    
    def generate_badge(self) -> Tuple[str, str]:
        """Generate Markdown and SVG badges.
        
        Returns:
            Tuple of (markdown_path, svg_path)
        """
        # Use safe model ID to avoid directory structure issues
        model_id = self._safe_model_id()
        md_path = os.path.join(self.output_dir, f"{model_id}_badge.md")
        svg_path = os.path.join(self.output_dir, f"{model_id}_badge.svg")
        
        # Generate Markdown badge
        score_percent = int(self.safety_score.overall_score * 100)
        
        # Map score to color
        if score_percent >= 90:
            color = "brightgreen"
        elif score_percent >= 70:
            color = "green"
        elif score_percent >= 50:
            color = "yellow"
        else:
            color = "red"
        
        # Create Markdown content
        md_content = f"![AegisEval Safety](https://img.shields.io/badge/AegisEval_Safety-{score_percent}%25-{color})"
        
        # Ensure the directories exist
        self._ensure_output_path(md_path)
        self._ensure_output_path(svg_path)
        
        # Write Markdown file
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        # Generate SVG badge
        svg_content = self._generate_svg_badge(score_percent, color)
        
        # Write SVG file
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)
        
        logger.info(f"Generated badge files: {md_path}, {svg_path}")
        return md_path, svg_path
    
    def _prepare_template_data(self) -> Dict[str, Any]:
        """Prepare data for templates.
        
        Returns:
            Dictionary of template variables
        """
        # Get improvement areas
        improvement_areas = get_improvement_areas(self.safety_score, self.evaluation_result)
        
        # Find weakest category (lowest pass rate, ties broken by severity)
        weakest_category = None
        lowest_pass_rate = 1.0
        highest_severity = -1
        severity_map = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        
        # Create a list of category results
        category_results = list(self.safety_score.category_scores.values())
        
        # Sort category results by severity (descending) and pass rate (ascending)
        category_results = sorted(
            category_results,
            key=lambda r: (-severity_map.get(r.severity.lower(), 0), r.pass_rate)
        )
        
        # Create a sorted dictionary of category scores
        sorted_category_scores = {r.category: r for r in category_results}
        
        # Update the category scores in the safety score
        self.safety_score.category_scores = sorted_category_scores
        
        for category, score in self.safety_score.category_scores.items():
            if score.pass_rate < lowest_pass_rate or (
                score.pass_rate == lowest_pass_rate and 
                severity_map.get(score.severity, 0) > highest_severity
            ):
                lowest_pass_rate = score.pass_rate
                highest_severity = severity_map.get(score.severity, 0)
                weakest_category = category
        
        # Prepare example test results
        failed_examples_objects = []
        passed_examples_objects = []
        
        # Find examples for the weakest category
        if weakest_category:
            # Get failed examples for weakest category
            category_fails = [
                r for r in self.evaluation_result.results
                if r.category == weakest_category and not getattr(r, "passed", True)
            ]
            
            # Add one failed example if available
            if category_fails:
                failed_examples_objects.append(random.choice(category_fails))
            
            # Get passed examples for weakest category
            category_passes = [
                r for r in self.evaluation_result.results
                if r.category == weakest_category and getattr(r, "passed", False)
            ]
            
            # For misinformation, find a good example with corrective text
            if weakest_category == "misinformation" and category_passes:
                # Try to find a response that contains debunking language
                debunk_passes = [
                    r for r in category_passes 
                    if "debunk" in r.response.lower() or "no evidence" in r.response.lower()
                ]
                
                if debunk_passes:
                    passed_examples_objects.append(random.choice(debunk_passes))
                else:
                    # Fallback to first pass
                    passed_examples_objects.append(category_passes[0])
            elif category_passes:
                # For other categories, just pick a random passed example
                passed_examples_objects.append(random.choice(category_passes))
        
        # Convert TestCaseResult objects to normalized dictionaries for the template
        failed_examples: List[Dict[str, Any]] = [normalize_example(r) for r in failed_examples_objects]
        passed_examples: List[Dict[str, Any]] = [normalize_example(r) for r in passed_examples_objects]
        
        # If we don't have examples from the weakest category, get some from all results
        if not failed_examples:
            failed_examples = [normalize_example(r) for r in self.evaluation_result.results 
                               if not getattr(r, "passed", True)][:5]
        if not passed_examples:
            passed_examples = [normalize_example(r) for r in self.evaluation_result.results 
                               if getattr(r, "passed", False)][:5]
        
        # Generate ASCII progress bars for each category
        category_bars = {}
        for category, score in self.safety_score.category_scores.items():
            bar_length = 10
            filled = round(score.pass_rate * bar_length)
            bar = f"[{'█' * filled}{' ' * (bar_length - filled)}] {int(score.pass_rate * 100)} %"
            category_bars[category] = bar
        
        # Get skipped test information if available
        skipped_tests_info = {}
        metadata = getattr(self.evaluation_result, 'metadata', {})
        
        # Try multiple sources for skipped test information
        skipped_count = 0
        skipped_tags: List[str] = []
        
        # Check direct attributes first (some implementations might use these)
        if hasattr(self.evaluation_result, 'skipped_count'):
            skipped_count = getattr(self.evaluation_result, 'skipped_count')
            if hasattr(self.evaluation_result, 'skipped_tags'):
                skipped_tags = getattr(self.evaluation_result, 'skipped_tags', [])
        
        # If not found, check metadata dictionary
        elif metadata:
            # Check for standard keys
            if 'skipped_count' in metadata:
                skipped_count = metadata.get('skipped_count', 0)
                skipped_tags = metadata.get('skipped_tags', [])
            # Check for alternate keys that might be used
            elif 'profile' in metadata and metadata['profile']:
                profile_data = metadata['profile']
                if isinstance(profile_data, dict):
                    if 'skipped_tests' in profile_data:
                        skipped_count = profile_data.get('skipped_tests', 0)
                    elif 'excluded_tags' in profile_data:
                        skipped_tags = profile_data.get('excluded_tags', [])
                        # If we have tags but no count, estimate skipped count from tags
                        if skipped_tags and not skipped_count:
                            skipped_count = len(skipped_tags)
            # Check for profile name directly
            elif 'profile_name' in metadata:
                profile_name = metadata.get('profile_name', '')
                # If we're using the OpenAI safe profile, we know it skips some tests
                if profile_name and ('openai' in profile_name.lower() and 'safe' in profile_name.lower()):
                    # Hardcoded knowledge that OpenAI safe profile skips adult, csam, and violence tags
                    skipped_tags = ['adult', 'csam', 'violence']
                    skipped_count = len(skipped_tags)  # Estimate based on tags
        
        # Only add to template data if we actually have skipped tests
        if skipped_count > 0:
            skipped_tests_info = {
                'count': skipped_count,
                'tags': ', '.join(skipped_tags) if skipped_tags else 'N/A'
            }
        
        return {
            "model_id": self.safety_score.model_id,
            "test_suite_id": self.safety_score.test_suite_id,
            "timestamp": self.safety_score.timestamp,
            "language": self.language,
            "safety_score": self.safety_score,
            "evaluation_result": self.evaluation_result,
            "improvement_areas": improvement_areas,
            "passed_examples": passed_examples,
            "failed_examples": failed_examples,
            "category_bars": category_bars,
            "skipped_tests_info": skipped_tests_info,
            "get_category_name": self._get_category_display_name,
            "t": self.translator.get  # Translator function
        }
    
    def _generate_svg_badge(self, score: int, color: str) -> str:
        """Generate an SVG badge.
        
        Args:
            score: Score as percentage (0-100)
            color: Badge color
            
        Returns:
            SVG content as string
        """
        # Map color name to hex code
        color_map = {
            "brightgreen": "#4c1",
            "green": "#97CA00",
            "yellow": "#dfb317",
            "red": "#e05d44"
        }
        hex_color = color_map.get(color, "#4c1")
        
        # Get badge template
        template = self.jinja_env.get_template("badge.svg.jinja")
        
        # Render badge template
        return template.render(
            score=score,
            color=hex_color
        )


def generate_reports(
    evaluation_result: EvaluationResult,
    safety_score: SafetyScore,
    output_dir: Optional[str] = None,
    language: str = "en"
) -> Dict[str, str]:
    """Generate all reports for an evaluation.
    
    Args:
        evaluation_result: Results of the evaluation
        safety_score: Calculated safety score
        output_dir: Output directory for reports
        language: Language code for the reports
        
    Returns:
        Dictionary mapping report types to file paths
    """
    builder = ReportBuilder(
        evaluation_result=evaluation_result,
        safety_score=safety_score,
        language=language,
        output_dir=output_dir
    )
    
    return builder.generate_all_reports()

# Add these functions required by tests
def generate_json_report(model_name: str, results: List[Dict[str, Any]], safety_score: float) -> Dict[str, Any]:
    """Generate a JSON report from test results.
    
    Args:
        model_name: Name of the model being evaluated
        results: List of test results
        safety_score: Overall safety score (0.0-1.0)
        
    Returns:
        Dictionary containing the JSON report data
    """
    # Create a simple JSON structure
    report = {
        "model": model_name,
        "safety_score": safety_score,
        "timestamp": None,  # Would normally have a timestamp
        "results": results,
        "metadata": {
            "aegiseval_version": "0.1.0",
            "total_tests": len(results),
            "passed_tests": sum(1 for r in results if r.get("passed", False)),
            "failed_tests": sum(1 for r in results if not r.get("passed", False))
        }
    }
    
    return report

def generate_html_report(json_report: Dict[str, Any]) -> str:
    """Generate an HTML report from a JSON report.
    
    Args:
        json_report: Dictionary containing the evaluation report data
        
    Returns:
        HTML content as a string
    """
    # Get the Jinja environment
    template_dir = Path(__file__).parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    
    # Get translator
    language = json_report.get("metadata", {}).get("language", "en")
    translator = get_translator(language)
    
    # Add translator to the environment
    env.globals['t'] = translator.get
    
    # Add category name helper function
    def get_category_name(category: str) -> str:
        """Get a display name for a category."""
        return category.replace('_', ' ').title()
    
    env.globals['get_category_name'] = get_category_name
    
    # Extract values with defaults for missing fields
    model_id = json_report.get("model", "Unknown Model")
    timestamp = json_report.get("timestamp", datetime.now().isoformat())
    test_suite_id = json_report.get("test_suite", "default")
    
    # Extract safety score with defaults
    safety_score_data = json_report.get("safety_score", {})
    
    # Create a dictionary with score data
    safety_score_dict: Dict[str, Any] = {}
    
    if isinstance(safety_score_data, float):
        # Handle case where safety_score is a float (in test data)
        safety_score_dict = {
            "overall_score": safety_score_data,
            "safety_level": "low_risk" if safety_score_data >= 0.7 else "moderate_risk",
            "confidence": 0.9,
            "category_scores": {},
            "regulation_scores": {}
        }
    else:
        # Handle case where safety_score is a dictionary
        safety_score_dict = {
            "overall_score": safety_score_data.get("overall", 0.0),
            "safety_level": safety_score_data.get("level", "unknown"),
            "confidence": safety_score_data.get("confidence", 0.0),
            "category_scores": json_report.get("category_scores", {}),
            "regulation_scores": json_report.get("regulation_scores", {})
        }
    
    # Prepare test results data
    test_results = json_report.get("test_results", [])
    if not test_results and "results" in json_report:
        # Handle old format where results is a list of dictionaries
        test_results = json_report["results"]
    
    # Create failed and passed examples
    failed_examples = [normalize_example(r) for r in test_results if not r.get("passed", False)][:5]
    passed_examples = [normalize_example(r) for r in test_results if r.get("passed", True)][:5]
    
    # Prepare template data
    template_data = {
        "model_id": model_id,
        "timestamp": timestamp,
        "test_suite_id": test_suite_id,
        "safety_score": safety_score_dict,
        "improvement_areas": json_report.get("improvement_areas", []),
        "evaluation_result": json_report.get("metadata", {}),
        "failed_examples": failed_examples,
        "passed_examples": passed_examples,
        "language": language
    }
    
    # Render the template
    template = env.get_template("report.html.jinja")
    html_content = template.render(**template_data)
    
    return html_content

def generate_pdf_report(html_content: str, output_path: str) -> bool:
    """Generate a PDF report from HTML content.
    
    Args:
        html_content: HTML report content
        output_path: Path to save the PDF
        
    Returns:
        True if successful, False otherwise
    """
    # Create directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure PDF extension
    if not output_path.lower().endswith('.pdf'):
        output_path += '.pdf'
    
    # Generate PDF from HTML
    try:
        # Add required CSS for PDF generation if needed
        # For demonstration, we're using a very simple conversion
        with open(output_path, "wb") as pdf_file:
            pisa_status = pisa.CreatePDF(
                src=html_content,  # HTML content
                dest=pdf_file      # File handle to receive PDF
            )
            
        # Return True if successful, False otherwise
        if pisa_status.err == 0:
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        return False

def safe_model_id(model_id: str) -> str:
    """Convert a model ID to a format safe for filenames.
    
    Args:
        model_id: The original model ID
        
    Returns:
        A sanitized version safe for filesystem use
    """
    return model_id.replace("/", "_").replace(":", "_")

def generate_report(
    model_name: str,
    results: List[Dict[str, Any]],
    safety_score: float,
    output_dir: Optional[str] = None
) -> Dict[str, str]:
    """Generate all report formats for a model evaluation.
    
    Args:
        model_name: Name of the evaluated model
        results: List of test results
        safety_score: Overall safety score (0-1)
        output_dir: Directory to save reports (default: ./aegiseval-results)
        
    Returns:
        Dictionary with paths to the generated report files
    """
    # Ensure output directory exists
    if output_dir is None:
        output_dir = "./aegiseval-results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize output paths
    output_paths = {}
    
    # Sanitize model name for filesystem
    safe_name = safe_model_id(model_name)
    
    # Generate JSON report
    json_report = generate_json_report(model_name, results, safety_score)
    json_path = os.path.join(output_dir, f"{safe_name}_report.json")
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    output_paths["json"] = json_path
    
    # Add language info to the JSON report
    json_report["metadata"] = json_report.get("metadata", {})
    json_report["metadata"]["language"] = "en"
    
    # Generate HTML report
    html_content = generate_html_report(json_report)
    html_path = os.path.join(output_dir, f"{safe_name}_report.html")
    os.makedirs(os.path.dirname(os.path.abspath(html_path)), exist_ok=True)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    output_paths["html"] = html_path
    
    # Generate PDF report
    pdf_path = os.path.join(output_dir, f"{safe_name}_report.pdf")
    generate_pdf_report(html_content, pdf_path)
    output_paths["pdf"] = pdf_path
    
    # Generate badge
    badge_md_path = write_badge(safety_score, os.path.join(output_dir, f"{safe_name}_badge.md"))
    output_paths["badge_md"] = badge_md_path
    
    return output_paths

# Add missing functions needed by tests
def write_json(data: Dict[str, Any], output_path: str) -> str:
    """Write data to a JSON file with pretty printing.
    
    Args:
        data: The data to write
        output_path: Path to write the file to
        
    Returns:
        The output path
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write with proper indentation and UTF-8 encoding
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_path

def write_badge(score: float, output_path: str) -> str:
    """Write a safety score badge to a file.
    
    Args:
        score: Safety score (0.0-1.0)
        output_path: Path to write badge SVG
        
    Returns:
        Path to the written badge file
    """
    # Calculate score as percentage
    score_int = int(score * 100)
    
    # Determine color based on score
    if score_int >= 90:
        color = "#4c1"  # green
    elif score_int >= 70:
        color = "#97CA00"  # yellowgreen
    elif score_int >= 50:
        color = "#dfb317"  # yellow
    elif score_int >= 30:
        color = "#fe7d37"  # orange
    else:
        color = "#e05d44"  # red
    
    # Generate SVG content
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="124" height="20" role="img">
  <title>AegisEval Score: {score_int}/100</title>
  <linearGradient id="s" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <clipPath id="r">
    <rect width="124" height="20" rx="3" fill="#fff"/>
  </clipPath>
  <g clip-path="url(#r)">
    <rect width="71" height="20" fill="#555"/>
    <rect x="71" width="53" height="20" fill="{color}"/>
    <rect width="124" height="20" fill="url(#s)"/>
  </g>
  <g fill="#fff" text-anchor="middle" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" text-rendering="geometricPrecision" font-size="11">
    <text aria-hidden="true" x="36.5" y="15" fill="#fff" fill-opacity=".3">AegisEval</text>
    <text x="36.5" y="14">AegisEval</text>
    <text aria-hidden="true" x="97.5" y="15" fill="#fff" fill-opacity=".3">{score_int}/100</text>
    <text x="97.5" y="14">{score_int}/100</text>
  </g>
</svg>"""
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Write SVG to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg)
    
    return output_path

def normalize_example(example: Any) -> Dict[str, Any]:
    """Normalize test example format for the template.
    
    Args:
        example: Original test example data (either dict or TestCaseResult)
        
    Returns:
        Normalized example data as dictionary
    """
    # Map common keys to expected format
    normalized = {}
    
    # Helper function to get attribute from either dict or object
    def get_value(obj: Any, key: str, default: Any = None) -> Any:
        if hasattr(obj, key):
            return getattr(obj, key, default)
        elif isinstance(obj, dict) and key in obj:
            return obj.get(key, default)
        return default
    
    # Map ID field from test_id or id to test_case_id
    normalized["test_case_id"] = get_value(example, "test_case_id", 
                                 get_value(example, "test_id", 
                                 get_value(example, "id", "unknown")))
    
    # Map category field
    normalized["category"] = get_value(example, "category", "general")
    
    # Map name field
    normalized["name"] = get_value(example, "name", "Unnamed Test")
    
    # Map prompt
    normalized["prompt"] = get_value(example, "prompt", 
                           get_value(example, "details", "No prompt available"))
    
    # Map response
    normalized["response"] = get_value(example, "response", "No response recorded")
    
    return normalized 