"""``uniftsm forecast`` — forecast from a CSV file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd


def _json_scalar(value: Any) -> Any:
    """Convert a single pandas/numpy value into a JSON-serialisable scalar.

    Timestamps become ISO-8601 strings, numpy scalars become Python scalars,
    and missing values become ``null`` (``json.dumps`` would otherwise emit the
    invalid ``NaN`` token).
    """
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def _forecast_payload(
    output_df: pd.DataFrame,
    *,
    model: str,
    horizon: int,
    freq: str | None,
    context_length: int,
) -> dict[str, Any]:
    """Build a machine-readable forecast payload.

    Quantile columns keep their names (e.g. ``q_0.5``) and each series is
    emitted alongside the shared forecast index, so consumers can align the
    values with their timestamps without parsing a rendered table.
    """
    return {
        "model": model,
        "horizon": horizon,
        "freq": freq,
        "context_length": context_length,
        "index": [_json_scalar(value) for value in output_df.index],
        "forecasts": {
            str(column): [_json_scalar(value) for value in output_df[column].tolist()]
            for column in output_df.columns
        },
    }


@click.command(name="forecast")
@click.argument("data_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--horizon", "-h", type=int, default=24, help="Forecast horizon.")
@click.option(
    "--model",
    "-m",
    default="chronos",
    help="Model name. Use 'auto' for automatic selection.",
)
@click.option(
    "--ensemble",
    is_flag=True,
    default=False,
    help="Use uncertainty-weighted ensemble of all models.",
)
@click.option(
    "--target-col",
    default=None,
    help="Name of the target column (for multivariate data).",
)
@click.option(
    "--freq",
    default=None,
    help="Pandas frequency alias (auto-detected if not provided).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save the forecast to a file (format follows --format).",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "csv", "json"]),
    default="table",
    show_default=True,
    help="Render the forecast as a table, CSV, or JSON. 'json' prints only JSON so it can be piped.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def forecast(
    data_path: str,
    horizon: int,
    model: str,
    ensemble: bool,
    target_col: str | None,
    freq: str | None,
    output: str | None,
    output_format: str,
    verbose: bool,
) -> None:
    """Generate forecasts for time series data from DATA_PATH (CSV).

    Examples:

        uniftsm forecast data.csv --horizon 90 --model chronos

        uniftsm forecast data.csv --horizon 168 --model auto --ensemble

        uniftsm forecast data.csv --format json | jq '.forecasts'
    """
    import logging

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("uniftsm.cli.forecast")

    # Load data
    logger.info("Loading data from %s...", data_path)
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    if target_col:
        y = df[target_col]
    elif isinstance(df, pd.DataFrame):
        y = df.iloc[:, 0]
    else:
        y = df

    logger.info("Loaded series with %d observations.", len(y))

    # Select / initialise model
    selected_model_name = model
    if ensemble:
        logger.info("Using uncertainty-weighted ensemble of all available models.")
        from uniftsm.core.base import BaseForecaster
        from uniftsm.core.registry import registry
        from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble

        models = []
        for model_name in registry.list_models():
            try:
                model_cls = registry.get(model_name)
                m = model_cls(device="auto")
                m.fit(y, freq=freq)
                models.append(m)
                logger.info("  Fitted model: %s", model_name)
            except Exception as exc:
                logger.warning("  Failed to load model '%s': %s", model_name, exc)

        if not models:
            # A non-zero exit is required here: callers pipeline this command,
            # and silently succeeding with no output looks like a valid result.
            raise click.ClickException("No models could be loaded for the ensemble.")

        selected_model_name = "ensemble"
        forecaster: BaseForecaster | UncertaintyWeightedEnsemble = UncertaintyWeightedEnsemble(
            models
        )
    elif model == "auto":
        logger.info("Using auto model selection...")
        from uniftsm.core.registry import registry
        from uniftsm.selection.meta_features import MetaFeatureExtractor

        features = MetaFeatureExtractor.extract(y)
        suggestions = registry.suggest(
            series_length=int(features["length"]),
            frequency=str(features.get("frequency", 0)),
            is_multivariate=False,
            needs_uncertainty=True,
            top_k=1,
        )
        if not suggestions:
            raise click.ClickException("No suitable model found for automatic selection.")
        best_model_name = suggestions[0][0]
        logger.info("Auto-selected model: %s (score=%.3f)", best_model_name, suggestions[0][1])
        selected_model_name = best_model_name
        model_cls = registry.get(best_model_name)
        forecaster = model_cls(device="auto")
        forecaster.fit(y, freq=freq)
    else:
        from uniftsm.core.registry import registry

        model_cls = registry.get(model)
        forecaster = model_cls(device="auto")
        forecaster.fit(y, freq=freq)

    # Generate forecast
    logger.info("Generating %d-step forecast...", horizon)
    result = forecaster.predict(horizon, return_quantiles=True)
    forecast_df = result  # dict of quantile DataFrames

    # Combine into a single DataFrame
    output_df = pd.DataFrame(
        {q: df.values.ravel() for q, df in forecast_df.items()},
        index=next(iter(forecast_df.values())).index,
    )

    payload = None
    if output_format == "json":
        payload = _forecast_payload(
            output_df,
            model=selected_model_name,
            horizon=horizon,
            freq=freq,
            context_length=len(y),
        )

    # Print — machine-readable modes emit only the data so stdout stays pipeable.
    machine_readable = output_format in {"json", "csv"}
    if output_format == "json":
        click.echo(json.dumps(payload, indent=2))
    elif output_format == "csv":
        click.echo(output_df.to_csv(), nl=False)
    else:
        click.echo(output_df.to_string())

    # Save
    if output:
        output_path = Path(output)
        if output_format == "json":
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            output_df.to_csv(output_path)
        # Log lines go to stderr in machine-readable modes so they cannot be
        # mistaken for forecast values by a downstream parser.
        click.echo(f"Forecast saved to {output_path.resolve()}", err=machine_readable)

    logger.info("Forecast complete.")
