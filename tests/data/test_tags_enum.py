# SPDX-License-Identifier: Apache-2.0
"""Tests to verify that tags enum matches allowed tags."""

def test_default_suite_tags_allowed():
    """Test that the default suite loads with tags validation working properly."""
    from aegiseval.data.test_suite import load_test_suite
    load_test_suite("tests/data/test_suites/default.yaml") 