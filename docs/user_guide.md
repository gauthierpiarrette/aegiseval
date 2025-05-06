# User Guide

### Estimated token cost per scan
| Model | Price /1 K in-tokens | Price /1 K out-tokens | Avg cost per scan |
|-------|---------------------|-----------------------|-------------------|
| GPT-4o | $0.01 | $0.03 | ≈ $0.07 |
| Gemini-1.5-Pro | $0.007 | $0.014 | ≈ $0.05 |
| Llama-3-70B (HF Inference) | $0.012 | $0.020 | ≈ $0.08 |

*(assumes 3 K in + 1 K out tokens per scan)* 

### Supported Models

AegisEval Lite supports the following model providers:

| Scheme | Example | Env var |  |
|--------|---------|---------|--|
| openai | gpt-4o | OPENAI_API_KEY |  |
| gemini | gemini:gemini-1.5-pro | GOOGLE_API_KEY |  |
| hf | hf:meta-llama/Llama-3-70B | HF_TOKEN |  |
| claude | claude:claude-3-opus-202402 | ANTHROPIC_API_KEY |  |

### Using with OpenAI keys
```bash
aegiseval scan --model gpt-4o --profile openai_safe
```

When using the `openai_safe` profile, AegisEval Lite will use appropriate parameters for safety evaluations with OpenAI models. The CLI will automatically use your `OPENAI_API_KEY` environment variable. 

You can also explicitly specify your API key:
```bash
export OPENAI_API_KEY=your_key_here
aegiseval scan --model gpt-4o --profile openai_safe
```

The `openai_safe` profile sets conservative parameter values such as:
- temperature: 0.7
- top_p: 0.95
- max_tokens: 2048 

### Provider Safety

#### High-Risk Probes

AegisEval Lite supports testing models against high-risk scenarios, but requires explicit opt-in:

```bash
aegiseval scan --model your_model --include-tags csam,extremist
```

When including high-risk tags, you'll be prompted to confirm before proceeding:

```
⚠  You have enabled probes requesting CSAM or extremist content.
   Run only on offline/self-hosted models. Continue? [y/N]:
```

You can bypass this confirmation with the `--yes` flag:

```bash
aegiseval scan --model your_model --include-tags csam,extremist --yes
```

#### Use a separate evaluation API key

We recommend creating a **second** OpenAI/Anthropic key strictly for
`aegiseval scan` runs.  A policy strike on this key will not disrupt
production traffic. 

### Categories and EU AI Act Regulation Mapping

The following table shows how AegisEval Lite categories map to EU AI Act articles, and which categories might be skipped by certain provider profiles:

| Category | Description | EU AI Act Articles | Provider Skip Note |
|----------|-------------|-------------------|-------------------|
| harmful_instructions | Tests for harmful content generation | EU-AI-Act-Article-5 | None |
| misinformation | Tests for misinformation generation | EU-AI-Act-Article-5, EU-AI-Act-Article-13 | None |
| bias | Tests for biased or discriminatory content | EU-AI-Act-Article-5, EU-AI-Act-Article-10 | None |
| privacy | Tests for privacy-infringing content | EU-AI-Act-Article-10 | Skipped by openai_safe profile |
| extremist_content | Tests for extremist content generation | EU-AI-Act-Article-5 | Skipped by openai_safe profile |
| child_safety | Tests for child safety concerns | EU-AI-Act-Article-5 | Skipped by openai_safe profile (csam tag) |
| jailbreak_override | Tests for safety bypass attempts | EU-AI-Act-Article-9, EU-AI-Act-Article-15 | None |
| copyright_leak | Tests for copyright material leakage | EU-AI-Act-Recital-60 | Skipped by openai_safe profile (copyright_long tag) |
| hallucination_factuality | Tests for factual accuracy | EU-AI-Act-Article-13 | Skipped by openai_safe profile (accuracy tag) |
| self_harm | Tests for self-harm content | EU-AI-Act-Article-5(1)(d) | None |
