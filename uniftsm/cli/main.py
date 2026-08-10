"""Main CLI entry point — ``uniftsm`` command.

Usage:

    uniftsm forecast data.csv --horizon 24
    uniftsm benchmark --datasets m4_hourly
    uniftsm serve --port 8080
    uniftsm compare chronos timesfm --dataset electricity
"""

from __future__ import annotations

import click

from uniftsm.cli.benchmark import benchmark
from uniftsm.cli.compare import compare
from uniftsm.cli.forecast import forecast
from uniftsm.cli.serve import serve


@click.group()
@click.version_option(version="0.1.0", prog_name="uniftsm")
@click.option("--verbose", "-v", is_flag=True, help="Increase verbosity.")
def cli(verbose: bool) -> None:
    """UniTSFM — Unified Time Series Foundation Model Toolkit.

    A unified toolkit wrapping Chronos, TimesFM, MOIRAI, Lag-Llama,
    and TinyTimeMixers behind a single API.
    """
    if verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)


cli.add_command(forecast)
cli.add_command(benchmark)
cli.add_command(serve)
cli.add_command(compare)


def main() -> None:
    """Entry point for package console_scripts."""
    cli()


if __name__ == "__main__":
    main()
