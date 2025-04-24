# SPDX-License-Identifier: Apache-2.0
"""Tests for report generation functionality in AegisEval."""

import json
import os
import re
import tempfile
import unittest

from aegiseval.report.builder import write_json, write_badge


class TestReportFunctions(unittest.TestCase):
    """Tests for individual report generation functions."""

    def test_write_json_roundtrip(self) -> None:
        """Test that JSON data round-trips correctly through write_json."""
        test_data = {
            "model": "test-model",
            "safety_score": 0.75,
            "timestamp": "2023-01-01T00:00:00Z",
            "results": [
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
        }

        # Use a temp file for testing
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            temp_path = tmp.name

        try:
            # Write the test data to the temp file
            write_json(test_data, temp_path)

            # Verify the file exists
            self.assertTrue(os.path.exists(temp_path))

            # Read back and compare
            with open(temp_path, 'r', encoding='utf-8') as f:
                read_data = json.load(f)

            # Check that the data round-trips correctly
            self.assertEqual(test_data, read_data)

            # Check the file is UTF-8 encoded and pretty-printed
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Check for indentation (pretty-printing)
                self.assertIn('  "model":', content)
                # Ensure content can be read as UTF-8
                self.assertTrue(content)
        finally:
            # Clean up the temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_write_badge(self) -> None:
        """Test that badge SVG is created correctly and contains the score."""
        # Test with different scores
        scores = [0.42, 0.75, 0.95]

        for score in scores:
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp:
                temp_path = tmp.name

            try:
                # Generate badge
                write_badge(score, temp_path)

                # Verify the file exists
                self.assertTrue(os.path.exists(temp_path))

                # Read the SVG content
                with open(temp_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Check that it's a valid SVG
                self.assertTrue(content.startswith('<svg'))
                self.assertTrue(content.endswith('</svg>'))

                # Check that it contains the score with expected format
                score_int = int(score * 100)
                score_str = f"{score_int}/100"
                self.assertIn(score_str, content)

                # Verify score matches the regex pattern
                score_pattern = r"\d{1,3}/100"
                matches = re.findall(score_pattern, content)
                self.assertTrue(matches)
                self.assertEqual(matches[0], score_str)
            finally:
                # Clean up the temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main() 