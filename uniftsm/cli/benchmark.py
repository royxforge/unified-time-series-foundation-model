"""``uniftsm benchmark`` — run evaluation benchmarks."""

from __future__ import annotations

import click
import numpy as np
import pandas as pd


@click.command(name="benchmark")
@click.option(
    "--datasets",
    "-d",
    default="m4_hourly",
    help="Comma-separated list of datasets from the Monash archive.",
)
@click.option(
    "--models",
    "-m",
    default="chronos,timesfm,moirai,lag_llama,ttm",
    help="Comma-separated list of model names.",
)
@click.option(
    "--horizon",
    "-h",
    type=int,
    default=24,
    help="Forecast horizon.",
)
@click.option(
    "--max-series",
    type=int,
    default=10,
    help="Limit number of series per dataset for faster runs.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    type=click.Path(dir_okay=False),
    help="Save results to CSV.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose output.")
def benchmark(
    datasets: str,
    models: str,
    horizon: int,
    max_series: int,
    output: str | None,
    verbose: bool,
) -> None:
    """Run evaluation benchmarks across datasets and models.

    Examples:

        uniftsm benchmark --datasets m4_hourly,electricity --models chronos,timesfm

        uniftsm benchmark --datasets weather --max-series 5
    """
    import logging

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger("uniftsm.cli.benchmark")

    dataset_list = [d.strip() for d in datasets.split(",")]
    model_list = [m.strip() for m in models.split(",")]

    from uniftsm.core.registry import registry
    from uniftsm.datasets.monash import MonashDatasetLoader
    from uniftsm.evaluation.metrics import compute_metrics

    loader = MonashDatasetLoader()

    all_results = []
    for dataset_name in dataset_list:
        logger.info("Loading dataset: %s", dataset_name)
        try:
            series_dict = loader.load(dataset_name, max_series=max_series)
        except Exception as exc:
            logger.error("Failed to load dataset '%s': %s", dataset_name, exc)
            continue

        for series_id, y in series_dict.items():
            # Use 80% for training, 20% for testing
            split = int(len(y) * 0.8)
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            actual_horizon = min(horizon, len(y_test))

            for model_name in model_list:
                try:
                    model_cls = registry.get(model_name)
                    forecaster = model_cls(device="cpu", seed=42)
                    forecaster.fit(y_train)
                    pred = forecaster.predict(
                        actual_horizon,
                        return_quantiles=True,
                    )
                    quantiles = [0.1, 0.5, 0.9]
                    quantile_preds = {
                        f"q_{q}": pred[f"q_{q}"]["forecast"].values for q in quantiles
                    }

                    metrics = compute_metrics(
                        y_true=y_test.values[:actual_horizon],
                        y_pred=pred["q_0.5"]["forecast"].values,
                        y_insample=y_train.values,
                        mean=pred["q_0.5"]["forecast"].values,
                        std=np.std([quantile_preds[f"q_{q}"] for q in quantiles], axis=0),
                    )

                    all_results.append(
                        {
                            "model": model_name,
                            "dataset": dataset_name,
                            "series_id": series_id,
                            **metrics,
                        }
                    )
                    logger.info(
                        "  %s on %s/%s: MASE=%.3f, sMAPE=%.3f",
                        model_name,
                        dataset_name,
                        series_id,
                        metrics.get("MASE", -1),
                        metrics.get("sMAPE", -1),
                    )
                except Exception as exc:
                    logger.warning(
                        "  %s on %s/%s failed: %s",
                        model_name,
                        dataset_name,
                        series_id,
                        exc,
                    )

    results_df = pd.DataFrame(all_results)
    click.echo("\n=== Summary ===")
    if not results_df.empty:
        summary = results_df.groupby("model")[["MASE", "sMAPE", "CRPS", "RMSE"]].mean()
        click.echo(summary.to_string())

    if output:
        results_df.to_csv(output, index=False)
        click.echo(f"\nResults saved to {output}")

    logger.info("Benchmark complete. %d results collected.", len(all_results))
