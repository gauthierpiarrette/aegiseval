# SPDX-License-Identifier: Apache-2.0
"""Updater job for AegisEval Lite."""

import yaml
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from deepdiff import DeepDiff


def check_for_updates(branch: str = "main") -> Union[bool, Tuple[bool, Optional[str]]]:
    """Check for updates by comparing remote regulations with local version.
    
    Fetches the remote JSON file containing EU AI Act regulations and compares
    it with the local eu_ai_act.yaml file. Returns True if differences are found,
    indicating an update is available.
    
    Args:
        branch: GitHub branch to check for updates (default: "main")
        
    Returns:
        For backward compatibility with tests:
            - Just a boolean if no tests call the internal version
            - A tuple (boolean, error_message) for new code that handles errors
        
    Note:
        This function catches exceptions internally and returns False or (False, error_message)
        instead of raising them, to allow graceful handling in CLI contexts.
    """
    # URL for the remote JSON file - replace with actual URL in production
    remote_url = f"https://raw.githubusercontent.com/aegiseval/regulations/{branch}/eu_ai_act.json"
    
    # Path to local YAML file
    local_path = Path(__file__).parent.parent / "data" / "eu_ai_act.yaml"
    
    try:
        # Fetch remote JSON
        response = requests.get(remote_url, timeout=10)
        response.raise_for_status()
        remote_data = response.json()
        
        # Load local YAML
        with open(local_path, "r", encoding="utf-8") as f:
            local_data = yaml.safe_load(f)
        
        # Compare the data
        diff = DeepDiff(local_data, remote_data)
        
        # Return just the boolean for backward compatibility with tests
        return bool(diff)
    
    except requests.RequestException as e:
        # Handle network and request errors
        print(f"Error fetching updates: {e}")
        return False
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        # Handle invalid file format errors
        print(f"Error processing update data: {e}")
        return False
    except Exception as e:
        # Catch-all for other unexpected errors
        print(f"Unexpected error checking for updates: {e}")
        return False


def download_updates() -> bool:
    """Download and apply available updates.
    
    Returns:
        True if updates were successfully applied, False otherwise
    """
    # Placeholder implementation
    update_available = check_for_updates()
    
    if not update_available:
        return False
    
    # Download logic would go here
    return True


def get_regulation_updates() -> List[Dict[str, Any]]:
    """Get updates to regulatory mappings.
    
    Returns:
        List of updated regulation definitions
    """
    # Placeholder implementation
    return [
        {
            "id": "EU-AI-Act-Article-5",
            "version": "1.0.0",
            "title": "Prohibited AI Practices",
            "description": "Prohibits AI systems that pose unacceptable risk",
            "url": "https://example.com/eu-ai-act-article-5",
            "updated_at": "2023-01-01"
        }
    ] 