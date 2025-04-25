# SPDX-License-Identifier: Apache-2.0
"""Confirmation banners and interactive prompts for risky operations."""

import sys
from typing import Set

# Try to import rich, use ANSI if not available
try:
    from rich.console import Console
    from rich.panel import Panel
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

from aegiseval.config.flags import HIGH_RISK_TAGS, EXIT_USER_ABORT, yes as YES_FLAG


def confirm_risky_run(tags: Set[str]) -> None:
    """Display a warning banner and require confirmation for risky tags.
    
    Shows a yellow warning banner if any high-risk tags are included in the run.
    Requires explicit user confirmation to proceed unless '--yes' flag is set.
    
    Args:
        tags: Set of tags requested for inclusion in this run
        
    Raises:
        SystemExit: If user does not confirm (exit code 12)
    """
    # Check if any high-risk tags are included
    risky_tags = tags.intersection(HIGH_RISK_TAGS)
    if not risky_tags:
        return  # No risky tags, no confirmation needed
    
    # Skip confirmation if yes flag is true
    if YES_FLAG:
        return
    
    # Banner text (exact as specified)
    warning_text = (
        "⚠  You have enabled probes requesting CSAM or extremist content.\n"
        "   Run only on offline/self-hosted models. Continue? [y/N]: "
    )
    
    # Display using rich if available, otherwise ANSI yellow
    if _HAS_RICH:
        console = Console()
        console.print(Panel(warning_text.strip(), style="yellow"), end="")
    else:
        # ANSI Yellow
        sys.stdout.write(f"\x1b[33m{warning_text}\x1b[0m")
    
    # Get user confirmation
    try:
        response = input().strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted by user.")
        sys.exit(EXIT_USER_ABORT)
    
    # Exit if not confirmed with 'y'
    if response != 'y':
        print("Run aborted.")
        sys.exit(EXIT_USER_ABORT) 