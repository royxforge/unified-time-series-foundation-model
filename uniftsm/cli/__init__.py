"""Command-line interface for UniTSFM."""

from uniftsm.cli.benchmark import benchmark
from uniftsm.cli.compare import compare
from uniftsm.cli.forecast import forecast
from uniftsm.cli.main import main
from uniftsm.cli.serve import serve

__all__ = ["forecast", "benchmark", "serve", "compare", "main"]
