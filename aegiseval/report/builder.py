# SPDX-License-Identifier: Apache-2.0
"""Report generation for AegisEval Lite."""

import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import jinja2
from xhtml2pdf import pisa

from aegiseval.report.i18n import get_translator
from aegiseval.runner.evaluator import EvaluationResult, TestCaseResult
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
        
        # Prepare output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
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
    
    def generate_json(self) -> str:
        """Generate a JSON report.
        
        Returns:
            Path to the generated JSON file
        """
        model_id = self.safety_score.model_id
        json_path = os.path.join(self.output_dir, f"{model_id}_report.json")
        
        # Prepare the report data
        report_data = {
            "model": model_id,
            "test_suite": self.safety_score.test_suite_id,
            "timestamp": self.safety_score.timestamp,
            "safety_score": {
                "overall": self.safety_score.overall_score,
                "level": self.safety_score.safety_level,
                "confidence": self.safety_score.confidence
            },
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
        
        # Write the JSON report
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated JSON report: {json_path}")
        return json_path
    
    def generate_html(self) -> str:
        """Generate an HTML report.
        
        Returns:
            Path to the generated HTML file
        """
        model_id = self.safety_score.model_id
        html_path = os.path.join(self.output_dir, f"{model_id}_report.html")
        
        # Prepare template data
        template_data = self._prepare_template_data()
        
        # Render the HTML template
        template = self.jinja_env.get_template("report.html.jinja")
        html_content = template.render(**template_data)
        
        # Write the HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {html_path}")
        return html_path
    
    def generate_pdf(self) -> str:
        """Generate a PDF report.
        
        Returns:
            Path to the generated PDF file
        """
        model_id = self.safety_score.model_id
        pdf_path = os.path.join(self.output_dir, f"{model_id}_report.pdf")
        
        # First generate HTML content
        template_data = self._prepare_template_data()
        template = self.jinja_env.get_template("report.html.jinja")
        html_content = template.render(**template_data)
        
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
        model_id = self.safety_score.model_id
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
        
        # Filter test results for examples
        passed_examples: list[TestCaseResult] = []
        failed_examples: list[TestCaseResult] = []
        
        # Get a few examples of passed and failed tests
        for result in self.evaluation_result.results:
            if len(passed_examples) < 3 and result.passed:
                passed_examples.append(result)
            elif len(failed_examples) < 3 and not result.passed and not result.error:
                failed_examples.append(result)
            
            if len(passed_examples) >= 3 and len(failed_examples) >= 3:
                break
        
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
    # Fix example format if needed (adapt test format to template expectations)
    def normalize_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize test example format for the template.
        
        Args:
            example: Original test example data
            
        Returns:
            Normalized example data
        """
        # Map common keys to expected format
        normalized = {}
        
        # Map ID field from test_id or id to test_case_id
        normalized["test_case_id"] = example.get("test_id", example.get("id", "unknown"))
        
        # Map category field
        normalized["category"] = example.get("category", "general")
        
        # Map prompt
        normalized["prompt"] = example.get("prompt", example.get("details", "No prompt available"))
        
        # Map response
        normalized["response"] = example.get("response", "No response recorded")
        
        return normalized
        
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
        html_content: HTML string to convert
        output_path: Path to save the PDF file
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Convert HTML to PDF
    with open(output_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(
            src=html_content,
            dest=pdf_file
        )
    
    # Return True if successful
    return not pisa_status.err

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
    
    # Generate JSON report
    json_report = generate_json_report(model_name, results, safety_score)
    json_path = os.path.join(output_dir, f"{model_name}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2, ensure_ascii=False)
    output_paths["json"] = json_path
    
    # Add language info to the JSON report
    json_report["metadata"] = json_report.get("metadata", {})
    json_report["metadata"]["language"] = "en"
    
    # Generate HTML report
    html_content = generate_html_report(json_report)
    html_path = os.path.join(output_dir, f"{model_name}_report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    output_paths["html"] = html_path
    
    # Generate PDF report
    pdf_path = os.path.join(output_dir, f"{model_name}_report.pdf")
    generate_pdf_report(html_content, pdf_path)
    output_paths["pdf"] = pdf_path
    
    # Generate badge
    badge_md_path = write_badge(safety_score, os.path.join(output_dir, f"{model_name}_badge.md"))
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
    """Generate a Markdown badge showing the model's safety score.
    
    Args:
        score: The safety score (0-1)
        output_path: Path to write the badge Markdown
        
    Returns:
        Path to the generated badge file
    """
    # Convert score to percentage
    score_percent = int(score * 100)
    
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
    
    # Write Markdown file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    logger.info(f"Generated badge markdown: {output_path}")
    return output_path 