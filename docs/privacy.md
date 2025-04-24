# AegisEval Lite - Privacy Policy

## Usage Metrics

AegisEval Lite includes an optional metrics collection feature that is **disabled by default**. This feature helps us improve the tool by providing basic anonymous usage information.

### What data is collected?

When you opt in to metrics collection, AegisEval Lite collects:

- Timestamp of the evaluation (ISO format date and time)
- Model identifier (the name you provided to the `--model` parameter)
- Safety score (the numerical result of the evaluation)

**No personal information** or evaluation details are collected. The data is stored locally in your home directory at `~/.aegiseval/metrics.jsonl`.

### How to enable or disable metrics

Metrics collection is disabled by default. You can enable it using the `--opt-in-metrics` flag:

```bash
aegiseval scan --model your-model --opt-in-metrics
```

To explicitly disable metrics (though it's already off by default):

```bash
aegiseval scan --model your-model --no-metrics
```

### Data Usage

The collected metrics help us:

1. Understand which models are being evaluated
2. Track average safety scores across different models
3. Identify trends in tool usage over time

This information guides our development priorities and helps us improve AegisEval Lite.

### Accessing or Deleting Your Data

Since all metrics are stored locally on your machine, you can:

- View the collected data by opening `~/.aegiseval/metrics.jsonl`
- Delete all collected data by removing the file: `rm ~/.aegiseval/metrics.jsonl`
- Stop collection at any time by using the `--no-metrics` flag or omitting the `--opt-in-metrics` flag 