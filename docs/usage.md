# Usage

AegisEval Lite provides a simple command-line interface for evaluating LLM safety.

## Basic Usage

The simplest way to run an evaluation is with the `scan` command:

```bash
aegiseval scan --model gpt-4o
```

This will:
1. Run the default test suite against the specified model
2. Generate a safety report in the default output directory (`./aegiseval-out`)
3. Create a README badge showing the safety score

## Command-Line Options

The `scan` command accepts several options to customize the evaluation:

```bash
aegiseval scan --model MODEL_NAME [OPTIONS]
```

### Required Parameters

- `--model`, `-m`: The model to evaluate (e.g., `gpt-4o`, `claude-3-opus-20240229`)

### Optional Parameters

- `--outdir`, `-o`: Output directory for evaluation results (default: `./aegiseval-out`)
- `--max-tokens`: Maximum tokens for model responses (default: 6000)
- `--parallel`: Number of parallel evaluation threads (default: 4)
- `--offline`: Run in offline mode, skipping API calls

## Examples

### Evaluating a Specific Model

```bash
aegiseval scan --model gpt-4o
```

### Custom Output Directory

```bash
aegiseval scan --model claude-3-opus-20240229 --outdir ./my-safety-reports
```

### Limited Response Length

```bash
aegiseval scan --model mistral-medium --max-tokens 1024
```

### Running in Offline Mode (for testing)

```bash
aegiseval scan --model gpt-4o --offline
```

## Understanding the Results

After running a scan, AegisEval Lite generates several output files in the specified directory:

- **JSON Report**: Contains detailed evaluation results with test-by-test breakdowns
- **PDF Report**: A human-readable summary of the evaluation results
- **Markdown Badge**: A badge showing the safety score for inclusion in README files
- **SVG Badge**: A visual representation of the safety score

The safety score is calculated on a scale of 0-100, where:

- **90-100**: Excellent - Very low risk
- **70-89**: Good - Low risk
- **50-69**: Fair - Moderate risk
- **0-49**: Poor - High risk

## Environment Variables

AegisEval Lite uses environment variables for API authentication:

- `OPENAI_API_KEY`: Required for evaluating OpenAI models
- `ANTHROPIC_API_KEY`: Required for evaluating Anthropic models

You can set these in your environment or use a `.env` file in the current directory. 