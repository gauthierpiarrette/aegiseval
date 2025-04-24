# SPDX-License-Identifier: Apache-2.0

# Release Checklist for AegisEval Lite

This document outlines the steps to follow when preparing a release and what to do in case a release needs to be revoked.

## Pre-Release Checklist

Before creating a new release tag, ensure:

1. All tests pass locally and in CI: `pytest tests/`
2. Documentation is up-to-date
3. Version number is updated in `pyproject.toml`
4. Changelog is updated
5. All dependencies are properly pinned
6. Security scan has been completed

## Release Process

1. Create and push a version tag:
   ```bash
   git tag -a v0.x.y -m "Release v0.x.y"
   git push origin v0.x.y
   ```

2. Monitor the GitHub Actions workflow at:
   `.github/workflows/release.yml`

3. Verify the release artifacts on GitHub:
   - Wheels (.whl files)
   - Source distribution (.tar.gz)
   - SBOM (sbom.json)
   - Sigstore attestations

4. Confirm the package is published on PyPI:
   https://pypi.org/project/aegiseval/

## Revocation Steps

If a release needs to be revoked, follow these steps:

### 1. PyPI Package Revocation

PyPI doesn't allow true deletion of packages to prevent supply chain attacks. Instead:

1. Mark the release as yanked on PyPI:
   ```bash
   pip install twine
   twine yank aegiseval==0.x.y
   ```

2. If severe (e.g., security vulnerability), contact PyPI admins:
   https://pypi.org/help/#admin-request

### 2. GitHub Release Revocation

1. Delete the GitHub release:
   - Go to Releases page
   - Click the "Delete" button for the affected release

2. Delete the tag:
   ```bash
   git tag -d v0.x.y
   git push --delete origin v0.x.y
   ```

### 3. Sigstore Attestation Revocation

Sigstore attestations can be revoked using:

1. Install the Sigstore CLI:
   ```bash
   go install github.com/sigstore/cosign/cmd/cosign@latest
   ```

2. Revoke the attestation:
   ```bash
   cosign attest-revoke --signature [SIG_URL]
   ```

### 4. Notification Process

1. Create a security advisory on GitHub
2. Send notification to users via:
   - GitHub Discussions
   - Project mailing list
   - Twitter/social media accounts
   - Update documentation

### 5. Recovery Process

1. Fix the issue in the codebase
2. Create a new patched version
3. Follow the standard release process with the fixed version
4. Document the issue and solution in the changelog

## Post-Revocation Verification

After revocation, verify:

1. The package is no longer installable with `pip install aegiseval==0.x.y`
2. The GitHub release is no longer accessible
3. All attestations are properly revoked
4. Users have been properly notified 