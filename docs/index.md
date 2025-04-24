# AegisEval Lite

## Zero-Infrastructure LLM Safety Testing

AegisEval Lite is a command-line tool designed to safety-test Large Language Models (LLMs) with zero infrastructure requirements. It provides comprehensive evaluation against safety criteria, regulatory mapping to EU AI Act articles, and generates detailed reports in multiple formats.

## Features

- **Comprehensive Safety Testing**: Tests LLMs against a wide range of safety scenarios
- **EU AI Act Compliance**: Maps failures to specific EU AI Act regulatory articles
- **Multi-Format Reporting**: Generates JSON, PDF reports and README badges
- **Zero-Infrastructure**: Runs entirely on your local machine with minimal setup
- **Offline Capability**: Can run evaluations without internet connectivity
- **Extensible**: Support for multiple model providers and custom test suites

## Quick Start

Test your LLM model for safety compliance with a single command:

```bash
aegiseval scan --model gpt-4o --outdir ./aegiseval-results
```

## Evaluation Process

AegisEval Lite follows a systematic approach to evaluate LLM safety:

1. **Test Suite Selection**: Loads predefined test scenarios across safety categories
2. **Model Response Generation**: Runs prompts against the target model
3. **Response Grading**: Analyzes responses for safety compliance
4. **Regulatory Mapping**: Maps failures to relevant EU AI Act articles
5. **Safety Scoring**: Calculates an overall safety score (0-100)
6. **Report Generation**: Creates detailed JSON and PDF reports
7. **Badge Creation**: Generates visual badge for README display

## Example Safety Score Badge

![AegisEval Safety](https://img.shields.io/badge/AegisEval_Safety-85%25-green)

## License

AegisEval Lite is released under the Apache 2.0 license. 