"""``uniftsm forecast`` — forecast from a CSV file."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd


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
    help="Save forecast to CSV.",
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
    verbose: bool,
) -> None:
    """Generate forecasts for time series data from DATA_PATH (CSV).

    Examples:

        uniftsm forecast data.csv --horizon 90 --model chronos

        uniftsm forecast data.csv --horizon 168 --model auto --ensemble
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
            logger.error("No models could be loaded. Exiting.")
            return

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
            logger.error("No suitable model found.")
            return
        best_model_name = suggestions[0][0]
        logger.info("Auto-selected model: %s (score=%.3f)", best_model_name, suggestions[0][1])
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

    # Print
    click.echo(output_df.to_string())

    # Save
    if output:
        output_path = Path(output)
        output_df.to_csv(output_path)
        click.echo(f"\nForecast saved to {output_path.resolve()}")

    logger.info("Forecast complete.")
