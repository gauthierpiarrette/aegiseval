# SPDX-License-Identifier: Apache-2.0
"""Configuration loading and validation for AegisEval Lite."""

import os
import json
import yaml
import jsonschema
from pathlib import Path
from typing import Dict, Any, List, Optional, Mapping

from pydantic import (
    BaseModel, 
    Field, 
    PositiveInt, 
    FilePath,
    field_validator,
)


class ModelConfig(BaseModel):
    """Configuration for a model to evaluate."""
    
    name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Provider of the model (e.g., 'openai')")
    api_config: Dict[str, Any] = Field(default_factory=dict, 
                                       description="API-specific configuration")


class EvaluationConfig(BaseModel):
    """Configuration for an evaluation run."""
    
    model: ModelConfig = Field(..., description="Model to evaluate")
    test_suite: str = Field("default", description="Test suite to use")
    output_dir: str = Field("./results", description="Directory for outputs")
    regulations: List[str] = Field(default_factory=list, 
                                   description="Regulations to check against")


class TestSuiteCfg(BaseModel):
    """Configuration for a test suite.
    
    Attributes:
        weight: Weight of this test suite in the overall evaluation
        prompts: Path to the prompts file for this test suite
    """
    weight: PositiveInt
    prompts: FilePath
    
    @field_validator("prompts")
    @classmethod
    def validate_file_exists(cls, v: FilePath) -> FilePath:
        """Validate that the prompts file exists."""
        if not v.exists():
            raise ValueError(f"Prompts file does not exist: {v}")
        return v


def load_config(config_path: Optional[str] = None) -> EvaluationConfig:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Optional path to config file. If None, will try to locate a default.
        
    Returns:
        Configuration object
    """
    # Define a reasonable default config
    default_config = {
        "model": {
            "name": "gpt-4",
            "provider": "openai",
            "api_config": {"api_key": ""}
        },
        "test_suite": "default",
        "output_dir": "./results",
        "regulations": ["EU-AI-Act-Article-5"]
    }
    
    if config_path and os.path.exists(config_path):
        # Load from file - would implement JSON loading here
        loaded_config = default_config  # Placeholder
    else:
        loaded_config = default_config
    
    # Create a ModelConfig from the nested dictionary first
    model_dict = loaded_config.get("model", {})
    if not isinstance(model_dict, Mapping):
        model_dict = {}
    model_config = ModelConfig(**model_dict)
    
    # Then create the EvaluationConfig with the proper model config
    test_suite = loaded_config.get("test_suite", "default")
    output_dir = loaded_config.get("output_dir", "./results")
    regulations_list = loaded_config.get("regulations", [])
    
    if not isinstance(test_suite, str):
        test_suite = "default"
    if not isinstance(output_dir, str):
        output_dir = "./results"
    if not isinstance(regulations_list, list):
        regulations_list = []
    
    return EvaluationConfig(
        model=model_config,
        test_suite=test_suite,
        output_dir=output_dir,
        regulations=regulations_list
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file and return as dictionary.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary containing the YAML content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        if not isinstance(data, dict):
            return {}
        return data


def validate_against_schema(data: dict, schema_path: Path) -> None:
    """Validate data against a JSON schema.
    
    Args:
        data: Data to validate
        schema_path: Path to JSON schema file
        
    Raises:
        jsonschema.exceptions.ValidationError: If validation fails
        FileNotFoundError: If schema file doesn't exist
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    
    jsonschema.validate(data, schema)


def load_test_suites(path: Path) -> Dict[str, TestSuiteCfg]:
    """Load and validate test suites configuration.
    
    Args:
        path: Path to test suites YAML file
        
    Returns:
        Dictionary mapping test suite names to their configurations
        
    Raises:
        FileNotFoundError: If file doesn't exist
        jsonschema.exceptions.ValidationError: If validation fails
        ValueError: If Pydantic validation fails
    """
    # Load YAML
    data = load_yaml(path)
    
    # Validate against JSON schema
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "test_suites.schema.json"
    validate_against_schema(data, schema_path)
    
    # Convert to Pydantic models
    result: Dict[str, TestSuiteCfg] = {}
    for suite_name, suite_config in data.items():
        if not isinstance(suite_config, dict):
            continue
            
        # Convert relative file paths to absolute based on the config file location
        if "prompts" in suite_config and not os.path.isabs(suite_config["prompts"]):
            suite_config["prompts"] = str(path.parent / suite_config["prompts"])
        
        result[suite_name] = TestSuiteCfg(**suite_config)
    
    return result


def load_eu_map(path: Path) -> Dict[str, List[str]]:
    """Load and validate EU AI Act mapping.
    
    Args:
        path: Path to EU AI Act mapping YAML file
        
    Returns:
        Dictionary mapping test identifiers to lists of EU AI Act articles
        
    Raises:
        FileNotFoundError: If file doesn't exist
        jsonschema.exceptions.ValidationError: If validation fails
    """
    # Load YAML
    data = load_yaml(path)
    
    # Validate against JSON schema
    schema_path = Path(__file__).parent.parent.parent / "schemas" / "eu_ai_act.schema.json"
    validate_against_schema(data, schema_path)
    
    # Ensure data is a dictionary with string keys and list values
    result: Dict[str, List[str]] = {}
    for key, value in data.items():
        if isinstance(value, list):
            str_values: List[str] = []
            for item in value:
                if isinstance(item, str):
                    str_values.append(item)
            result[key] = str_values
    
    return result


def get_default_config_path() -> str:
    """Get the default configuration file path.
    
    Returns:
        Path to default config file
    """
    # Check for config in user's home directory
    home_config = os.path.expanduser("~/.aegiseval/config.json")
    if os.path.exists(home_config):
        return home_config
    
    # Fall back to current directory
    return "./aegiseval_config.json" 