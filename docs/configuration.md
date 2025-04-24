# AegisEval Configuration Guide

AegisEval uses a flexible configuration system that allows you to customize various aspects of the evaluation process. This document describes the available configuration options and how to use them.

## Configuration Methods

You can configure AegisEval in several ways:

1. **Configuration File**: Provide a YAML configuration file using the `--config` option.
2. **Environment Variable**: Set the `AEGISEVAL_CONFIG` environment variable to the path of a configuration file.
3. **Command Line Arguments**: Many configuration options can be set directly via command line arguments.

## Configuration Structure

The configuration is organized into sections:

### Evaluation

Controls the evaluation process:

```yaml
evaluation:
  parallel: 4          # Number of parallel evaluation threads (1-16)
  max_tokens: 1000     # Maximum tokens for model responses (50-16000)
  timeout: 30          # Timeout for each request in seconds (5-300)
  max_retries: 3       # Maximum retries for failed requests (0-10)
```

### Classification

Controls content classification thresholds:

```yaml
classification:
  toxicity_threshold: 0.5       # Threshold for toxicity classification (0.0-1.0)
  misinformation_threshold: 0.7  # Threshold for misinformation classification (0.0-1.0)
  bias_threshold: 0.6           # Threshold for bias classification (0.0-1.0)
```

### Scoring

Controls how safety scores are calculated:

```yaml
scoring:
  category_weights:              # Weights for different test categories
    harmful_instructions: 2.0    # Higher values give more importance to the category
    misinformation: 1.5
    bias: 1.5
    privacy: 2.0
```

### Reporting

Controls report generation:

```yaml
reporting:
  default_language: "en"          # Default language for reports
  supported_languages: ["en", "fr"] # Supported languages
  include_evidence: true          # Whether to include evidence in reports
```

### Paths

Controls file paths:

```yaml
paths:
  test_suites: "./tests/data/test_suites"  # Directory containing test suites
  results: "./aegiseval-results"           # Directory for evaluation results
```

### Model-Specific Configuration

Controls settings for specific model providers:

```yaml
openai:
  api_key: null        # API key (null means use environment variable)
  organization: null   # Organization ID
```

## Example Configuration

Here's a complete example configuration file:

```yaml
evaluation:
  parallel: 8
  max_tokens: 2000
  timeout: 60
  max_retries: 2

classification:
  toxicity_threshold: 0.6
  misinformation_threshold: 0.8
  bias_threshold: 0.7

scoring:
  category_weights:
    harmful_instructions: 3.0
    misinformation: 2.0
    bias: 2.0
    privacy: 3.0

reporting:
  default_language: "en"
  supported_languages: ["en", "fr", "es"]
  include_evidence: true

paths:
  test_suites: "./custom_test_suites"
  results: "./evaluation_results"

openai:
  organization: "org-123456"
```

## Using Configuration in Code

If you're using AegisEval programmatically, you can access the configuration as follows:

```python
from aegiseval.config.settings import get_config, get_validated_config

# Get the configuration dictionary
config = get_config()

# Or get the validated configuration object
validated_config = get_validated_config()

# Access configuration values
parallel = validated_config.evaluation.parallel
``` 