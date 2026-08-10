# Contributing to UniTSFM

Thank you for your interest in contributing! We welcome contributions of all forms: bug reports, feature requests, documentation improvements, and code changes.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/royxforge/uniftsm.git
cd uniftsm

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

## Code Style

This project uses:

- **Black** for code formatting
- **Ruff** for linting
- **mypy** for type checking

Run all checks before committing:

```bash
black uniftsm/
ruff check uniftsm/
mypy uniftsm/
```

## Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=uniftsm

# Specific test file
pytest tests/test_core/test_base.py
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If adding a new model wrapper, use the `@registry.register()` decorator.
3. Add tests for any new functionality.
4. Ensure all tests pass and code style checks are clean.
5. Update documentation if needed.
6. Submit a pull request with a clear description of changes.

## Adding a New Model

New TSFMs can be added via the plugin system without modifying core code:

```python
from uniftsm.core.base import BaseForecaster
from uniftsm.core.registry import registry

@registry.register(
    "my_model",
    metadata={
        "family": "my_model",
        "supports_multivariate": False,
        "supports_probabilistic": True,
        "min_length": 64,
        "max_context_length": 2048,
    },
)
class MyModelForecaster(BaseForecaster):
    def _load_model(self):
        ...
    def fit(self, y, X=None, freq=None, **kwargs):
        ...
    def predict(self, horizon, return_quantiles=False, quantiles=None, **kwargs):
        ...
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
