# AegisEval

A tool for safety testing LLMs, mapping failures to EU-AI-Act articles, and generating JSON, PDF, and README badges.

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![REUSE status](https://api.reuse.software/badge/github.com/aegiseval/aegiseval)](https://api.reuse.software/info/github.com/aegiseval/aegiseval)

## Setup

### Environment Setup

1. Set up your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

2. Optionally, set up your OpenAI organization ID:

```bash
export OPENAI_ORG=your_organization_id_here
```

### Configuration

You can use the provided configuration file `aegiseval-config.yaml` or create your own with custom settings.

## Usage

Run a safety evaluation on a model:

```bash
aegiseval scan --model gpt-4o --outdir ./aegiseval-results
```

### Options

- `--model`: The OpenAI model to evaluate (e.g., gpt-4o, gpt-4, etc.)
- `--outdir`: Directory for evaluation outputs
- `--config`: Path to a custom configuration file
- `--redteam-level`: Red team level (1-3) for more aggressive testing
- `--language`: Language for reports (default: en)
- `--profile`: Provider-specific profile (e.g., `openai_safe`, `anthropic_default`, `gemini_low_temp`)
- `--include-tags`: Comma-separated list of high-risk tags to include in testing (e.g., `csam,extremist,copyright_long`). Use `--include-tags all` to include all high-risk tags.
- `--yes`: Auto-confirm running high-risk probes without interactive prompt

### High-Risk Tags

By default, AegisEval excludes certain high-risk tags from testing to avoid generating harmful content. You can selectively include these tags for more comprehensive testing:

```bash
# Include specific high-risk tags
aegiseval scan --model gpt-4o --include-tags csam,extremist

# Include all high-risk tags
aegiseval scan --model gpt-4o --include-tags all
```

When including high-risk tags, you'll be prompted to confirm before testing begins, unless the `--yes` flag is provided.

> ⚠️ **Warning**: Including high-risk tags may result in your model being prompted to generate potentially harmful content as part of the evaluation. Use with caution.

## Troubleshooting

If you encounter an error message like:

```
AttributeError: 'NoneType' object has no attribute 'get'
```

It's likely that your OpenAI API key is not properly set up. Make sure to:

1. Set your OpenAI API key as an environment variable
2. Or add it to the configuration file
3. Or provide it directly with the `--api-key` option

## Features

### Internationalization

Reports can be generated in different languages:

```bash
aegiseval scan --model gpt-4o --language fr
```

Currently supported languages:
- English (en) - default
- French (fr)

### Usage Metrics

AegisEval Lite includes optional anonymous metrics collection (disabled by default):

```bash
# Enable metrics collection
aegiseval scan --model gpt-4o --opt-in-metrics

# Explicitly disable metrics collection (default)
aegiseval scan --model gpt-4o --no-metrics
```

When enabled, this collects:
- Timestamp of evaluation
- Model name
- Safety score

See [privacy policy](docs/privacy.md) for more details.

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting and type checking
ruff check aegiseval
mypy aegiseval
```

## License

Apache-2.0 