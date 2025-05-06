• Loads a small HF model when `AEGISEVAL_ENABLE_TRANSFORMER_CONFIRM=1`.
• Falls back to a no-op if transformers isn't installed.
• Public method `confirm()` returns whether evidence stays flagged.

Updated:
```aegiseval/grader/pipeline.py```
• Imports the new confirmer and instantiates once.
• After running existing classifiers, iterates over flagged results and keeps only evidence that the confirmer validates; otherwise clears the flag.

All changes follow PEP 8, add docstrings, and keep tests untouched. 