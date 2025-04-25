# SPDX-License-Identifier: Apache-2.0
"""Central definitions for CLI flags and configuration options."""

from typing import Set

# Default empty set of tags to include
include_tags: Set[str] = set()

# Auto-confirm flag for risky probes
yes: bool = False

# Set of tags requiring explicit opt-in and confirmation
HIGH_RISK_TAGS = {"csam", "extremist", "copyright_long"}

# Exit codes
EXIT_USER_ABORT = 12  # User aborted risky run
EXIT_ILLEGAL_CONTENT = 13  # Illegal content detected 