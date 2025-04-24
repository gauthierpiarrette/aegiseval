# SPDX-License-Identifier: Apache-2.0
"""Tests for PDF generation in AegisEval."""

import os
import tempfile
import unittest

from pdfminer.high_level import extract_text

from aegiseval.report.builder import (
    generate_json_report,
    generate_html_report,
    generate_pdf_report,
    generate_report,
)


class TestPDFGeneration(unittest.TestCase):
    """Tests for PDF report generation functionality."""

    def setUp(self) -> None:
        """Set up test data for PDF generation tests."""
        self.model_name = "test-model"
        self.safety_score = 0.85
        self.results = [
            {
                "test_id": "test-1",
                "name": "Harmful Content Test",
                "passed": True,
                "score": 0.9,
                "details": "Model refused to generate harmful content",
                "regulations": ["EU-AI-Act-Article-5"]
            },
            {
                "test_id": "test-2",
                "name": "Bias Test",
                "passed": False,
                "score": 0.6,
                "details": "Model exhibited some bias in responses",
                "regulations": ["EU-AI-Act-Article-10"]
            }
        ]
        
        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self) -> None:
        """Clean up temporary files after tests."""
        # Remove temp files if they exist
        for filename in os.listdir(self.temp_dir):
            file_path = os.path.join(self.temp_dir, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        
        # Remove the temp directory
        os.rmdir(self.temp_dir)
    
    def test_html_report_generation(self) -> None:
        """Test that HTML report is generated correctly."""
        # Generate a JSON report
        json_report = generate_json_report(
            self.model_name, self.results, self.safety_score
        )
        
        # Generate HTML from the JSON report
        html_content = generate_html_report(json_report)
        
        # Verify the HTML contains expected content
        self.assertIn("AegisEval Safety Report", html_content)
        self.assertIn(self.model_name, html_content)
        
        # Verify the HTML contains the score
        score_percent = str(int(self.safety_score * 100))
        self.assertIn(f"{score_percent}/100", html_content)
        
        # Verify the test results are included
        for test in self.results:
            self.assertIn(test["name"], html_content)
    
    def test_pdf_generation(self) -> None:
        """Test that PDF report is generated successfully with the correct content."""
        # Generate a full report
        output_paths = generate_report(
            self.model_name,
            self.results,
            self.safety_score,
            output_dir=self.temp_dir
        )
        
        # Verify that the PDF file was created
        pdf_path = output_paths["pdf"]
        self.assertTrue(os.path.exists(pdf_path))
        
        # Extract text from the PDF
        pdf_text = extract_text(pdf_path)
        
        # Verify the PDF contains the required text
        self.assertIn("AegisEval Safety Report", pdf_text)
        
        # Verify other important content
        self.assertIn(self.model_name, pdf_text)
        score_percent = str(int(self.safety_score * 100))
        self.assertIn(score_percent, pdf_text)
        
        # Verify test results are included
        for test in self.results:
            self.assertIn(test["name"], pdf_text)
    
    def test_html_to_pdf_conversion(self) -> None:
        """Test direct HTML to PDF conversion function."""
        # Create some simple HTML
        html_content = """
        <!DOCTYPE html>
        <html>
        <head><title>Test PDF</title></head>
        <body>
            <h1>Aegis Score: 85%</h1>
            <p>This is a test document.</p>
        </body>
        </html>
        """
        
        # Set output path
        pdf_path = os.path.join(self.temp_dir, "test_conversion.pdf")
        
        # Convert HTML to PDF
        success = generate_pdf_report(html_content, pdf_path)
        
        # Verify conversion was successful
        self.assertTrue(success)
        self.assertTrue(os.path.exists(pdf_path))
        
        # Extract text from the PDF
        pdf_text = extract_text(pdf_path)
        
        # Verify the PDF contains the expected content
        self.assertIn("Aegis Score", pdf_text)
        self.assertIn("85%", pdf_text)
        self.assertIn("This is a test document", pdf_text)


if __name__ == "__main__":
    unittest.main() 