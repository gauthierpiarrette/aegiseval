# SPDX-License-Identifier: Apache-2.0

"""Provider profile configuration loader for AegisEval.

This module provides functionality to load provider profiles from YAML
configuration files.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class ProviderProfile:
    """Provider profile configuration.

    This dataclass encapsulates the configuration for a provider profile,
    including safety settings and rate limits.

    Attributes:
        excluded_tags: List of evaluation tags to exclude for this provider.
        system_header: System prompt header to use with this provider.
        rpm_limit: Rate limit in requests per minute for this provider.
    """

    excluded_tags: List[str]
    system_header: str
    rpm_limit: int


def load_profile(name: str) -> ProviderProfile:
    """Load a provider profile by name.

    Args:
        name: The name of the provider profile to load.

    Returns:
        A ProviderProfile instance containing the profile configuration.

    Raises:
        KeyError: If the specified profile does not exist.
    """
    # Define the base path for provider profiles
    base_path = Path(__file__).parent / "provider_profiles"
    profile_path = base_path / f"{name}.yaml"

    # Check if the profile exists
    if not profile_path.exists():
        raise KeyError(f"Provider profile '{name}' not found")

    # Load the profile from the YAML file
    with open(profile_path, "r", encoding="utf-8") as file:
        profile_data = yaml.safe_load(file)

    # Create and return a ProviderProfile instance
    return ProviderProfile(
        excluded_tags=profile_data.get("excluded_tags", []),
        system_header=profile_data.get("system_header", ""),
        rpm_limit=profile_data.get("rpm_limit", 0),
    ) 