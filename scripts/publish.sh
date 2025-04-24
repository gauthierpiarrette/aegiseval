#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "Publishing AegisEval to PyPI using Trusted Publishing..."

# Ensure we have the artifacts
if [ ! -d "artifacts/dist" ]; then
  echo "Error: artifacts/dist directory not found. Download artifacts first."
  exit 1
fi

# Install required packages for publishing
pip install --upgrade pip twine

# Use twine to verify the distribution files
echo "Verifying distribution files..."
twine check artifacts/dist/*

# Use twine to upload to PyPI using PyPI Trusted Publishing (OIDC)
# No API tokens needed as OIDC handles auth automatically
echo "Uploading to PyPI via Trusted Publishing..."
twine upload --verbose artifacts/dist/*

echo "Package published successfully to PyPI." 