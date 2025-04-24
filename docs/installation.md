# Installation

AegisEval Lite is designed to be easy to install with minimal dependencies.

## Prerequisites

- Python 3.11 or later
- pip (Python package installer)

## Installing from PyPI

The recommended way to install AegisEval Lite is from the Python Package Index (PyPI):

```bash
pip install aegiseval
```

This will install the latest stable version of AegisEval Lite along with all required dependencies.

## Installing from Source

For development or to get the latest features before they're released to PyPI, you can install from source:

```bash
# Clone the repository
git clone https://github.com/aegiseval/aegiseval.git
cd aegiseval

# Install in development mode
pip install -e ".[dev]"
```

Installing in development mode allows you to make changes to the code and have them immediately reflected in the installed package.

## Verifying Installation

To verify that AegisEval Lite was installed correctly, run:

```bash
aegiseval --version
```

This should display the current version of AegisEval Lite.

## Dependencies

AegisEval Lite depends on the following key packages:

- **click**: Command-line interface creation
- **pydantic**: Data validation and settings management
- **httpx**: Asynchronous HTTP client
- **jinja2**: Template engine for report generation
- **xhtml2pdf**: PDF report generation

For development, additional dependencies include:

- **pytest**: Testing framework
- **ruff**: Python linter
- **mypy**: Static type checker

These dependencies are automatically installed when using pip. 