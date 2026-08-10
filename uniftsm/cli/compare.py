"""``uniftsm compare`` — compare two or more models on a dataset."""

from __future__ import annotations

import click
import numpy as np
import pandas as pd


@click.command(name="compare")
@click.argument("models", nargs=-1, required=True)
@click.option("--dataset", "-d", default="m4_hourly", help="Dataset name.")
@click.option("--horizon", "-h", type=int, default=24, help="Forecast horizon.")
@click.option(
    "--max-series",
    type=int,
    default=5,
    help="Number of series to evaluate.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save comparison table to CSV.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def compare(
    models: tuple[str, ...],
    dataset: str,
    horizon: int,
    max_series: int,
    output: str | None,
    verbose: bool,
) -> None:
    """Compare two or more models on a given dataset.

    Examples:

        uniftsm compare chronos timesfm --dataset electricity

        uniftsm compare chronos timesfm moirai --horizon 48
    """
    import logging

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("uniftsm.cli.compare")

    from uniftsm.core.registry import registry
    from uniftsm.datasets.monash import MonashDatasetLoader
    from uniftsm.evaluation.metrics import compute_metrics
    from uniftsm.evaluation.significance import (
        diebold_mariano_test,
        wilcoxon_signed_rank_test,
    )

    # Load dataset
    loader = MonashDatasetLoader()
    logger.info("Loading dataset '%s'...", dataset)
    try:
        series_dict = loader.load(dataset, max_series=max_series)
    except Exception as exc:
        click.echo(f"Error loading dataset: {exc}")
        return

    if not series_dict:
        click.echo(f"No series found for dataset '{dataset}'.")
        return

    # Evaluate each model
    model_results: dict[str, list[dict]] = {m: [] for m in models}

    for series_id, y in series_dict.items():
        split = int(len(y) * 0.8)
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        actual_horizon = min(horizon, len(y_test))

        for model_name in models:
            try:
                model_cls = registry.get(model_name)
                forecaster = model_cls(device="cpu", seed=42)
                forecaster.fit(y_train)
                pred = forecaster.predict(actual_horizon)

                metrics = compute_metrics(
                    y_true=y_test.values[:actual_horizon],
                    y_pred=pred["mean"].values,
                    y_insample=y_train.values,
                )
                model_results[model_name].append(metrics)
            except Exception as exc:
                logger.warning("Model '%s' failed on series '%s': %s", model_name, series_id, exc)

    # Aggregate results
    click.echo("\n=== Model Comparison ===")
    summary_rows = []
    for model_name in models:
        results = model_results.get(model_name, [])
        if not results:
            continue
        avg = pd.DataFrame(results).mean().to_dict()
        summary_rows.append({"model": model_name, **avg})
        click.echo(f"\n{model_name}:")
        for metric, value in avg.items():
            click.echo(f"  {metric}: {value:.4f}")

    # Statistical significance tests
    if len(models) >= 2:
        click.echo("\n=== Statistical Significance ===")
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                r1 = model_results.get(m1, [])
                r2 = model_results.get(m2, [])
                if not r1 or not r2:
                    continue
                errors1 = np.array([r.get("MAE", 0) for r in r1])
                errors2 = np.array([r.get("MAE", 0) for r in r2])

                dm = diebold_mariano_test(errors1, errors2)
                wx = wilcoxon_signed_rank_test(errors1, errors2)

                click.echo(f"\n{m1} vs {m2}:")
                click.echo(
                    f"  DM test: stat={dm['dm_stat']:.3f}, p={dm['p_value']:.4f}, "
                    f"significant={dm['significant_at_5pct']}"
                )
                click.echo(
                    f"  Wilcoxon: stat={wx['statistic']:.1f}, p={wx['p_value']:.4f}, "
                    f"significant={wx['significant_at_5pct']}"
                )

    summary_df = pd.DataFrame(summary_rows)
    if output:
        summary_df.to_csv(output, index=False)
        click.echo(f"\nComparison table saved to {output}")

    logger.info("Comparison complete.")
