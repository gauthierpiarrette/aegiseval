# Contributing to AegisEval Lite

We welcome contributions to AegisEval Lite! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others when contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** to your local machine
3. **Set up the development environment**:
   ```bash
   pip install -e ".[dev]"
   ```
4. **Create a new branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

### Code Style

We follow PEP 8 and use Ruff for linting:

```bash
ruff check aegiseval
```

For type checking, we use mypy:

```bash
mypy aegiseval
```

### Documentation

- All public functions, classes, and methods should have Google-style docstrings
- Update documentation in the docs/ directory when adding or changing features
- Run mkdocs to test documentation changes:
  ```bash
  mkdocs serve
  ```

### Tests

We use pytest for our test suite:

```bash
pytest
```

Please add tests for any new features or bug fixes.

## Pull Request Process

1. **Update your fork** with the latest changes from the main repository
2. **Run tests** to make sure everything passes
3. **Update documentation** if needed
4. **Submit a pull request** with a clear description of your changes
5. **Respond to feedback** in code review

## Adding New Test Suites

To add a new test suite to AegisEval Lite:

1. Create a new YAML file in the `aegiseval/data/test_suites/` directory
2. Follow the schema defined in `schemas/test_suites.schema.json`
3. Add mapping to EU AI Act articles in `aegiseval/data/eu_ai_act.yaml`
4. Add tests for your new test suite

## EU AI Act Compliance Mapping Changes

When making changes to EU AI Act compliance mappings:

1. Apply the `⚖️ mapping-change` label to your PR
2. Update both:
   - `aegiseval/data/eu_ai_act.yaml` with the mapping changes
   - `docs/compliance_review.md` by adding a new row to the table
3. Obtain sign-off from a legal reviewer before merging
4. Document the rationale for the mapping change

Note: PRs with the `⚖️ mapping-change` label will be blocked by CI if both files aren't modified.

## License

By contributing to AegisEval Lite, you agree that your contributions will be licensed under the project's Apache 2.0 license.

## Questions?

If you have any questions or need help with the contribution process, please open an issue on GitHub or reach out to the maintainers. 