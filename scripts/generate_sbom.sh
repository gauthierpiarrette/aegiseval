#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Generate SBOM in CycloneDX format
echo "Generating SBOM in CycloneDX format..."

# Ensure we have the dist directory
if [ ! -d "dist" ]; then
  echo "Error: dist directory not found. Build the package first."
  exit 1
fi

# Use syft to generate the SBOM
echo "Running syft to generate SBOM..."
syft packages dir:. -o cyclonedx-json=sbom.json

# Validate the SBOM was generated
if [ -f "sbom.json" ]; then
  echo "SBOM successfully generated: sbom.json"
else
  echo "Error: Failed to generate SBOM."
  exit 1
fi

echo "SBOM generation complete." 