#!/usr/bin/env python3
"""Preliminary *real* benchmark: statistical baselines + a real TSFM.

This script produces genuine numbers (no mocks): it runs the statistical
baselines shipped with UniTSFM and the smallest verified TSFM wrapper
(Chronos ``tiny``) against synthetic series with known structure
(trend, seasonality, random walk, and a complex multi-seasonal series).

It is intentionally lightweight — it runs on CPU in a few minutes and
requires no HuggingFace authentication.  The full paper benchmark over
the 7 Monash datasets with all 5 TSFMs is reproduced with
``python benchmarks/run_all.py`` (see ``benchmarks/README.md``).

Outputs (written to ``benchmarks/results/``):
    - benchmark_results.csv  — per model × dataset metrics
    - summary.csv            — mean metrics per model × dataset
    - benchmark_table.tex    — LaTeX-ready comparison table
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmarks.mini")

RESULTS_DIR = Path(__file__).parent / "results"

# Model configurations used here.  ``chronos`` is the smallest real TSFM
# wrapper; add any other installed backend to extend the comparison.
MODELS: dict[str, dict] = {
    "naive": {},
    "seasonal_naive": {"seasonal_period": 12},
    "theta": {},
    "chronos_tiny": {"model_size": "tiny", "num_samples": 20},
}


def generate_datasets() -> dict[str, pd.Series]:
    """Synthetic series with known ground-truth structure."""
    from uniftsm.datasets.synthetic import SyntheticDatasetGenerator

    return {
        "linear_trend": SyntheticDatasetGenerator.linear_trend(n=240, slope=0.1),
        "seasonal": SyntheticDatasetGenerator.seasonal(n=240, period=12, amplitude=2.0),
        "random_walk": SyntheticDatasetGenerator.random_walk(n=240, std=1.0),
        "complex": SyntheticDatasetGenerator.complex(n=240, freq="D"),
    }


def run() -> list[dict]:
    """Run the mini benchmark and return per-run result rows."""
    from uniftsm.evaluation.metrics import compute_metrics
    from uniftsm.evaluation.statistical_baselines import (
        NaiveForecaster,
        SeasonalNaiveForecaster,
        ThetaForecaster,
    )

    baselines = {
        "naive": NaiveForecaster,
        "seasonal_naive": SeasonalNaiveForecaster,
        "theta": ThetaForecaster,
    }
    from uniftsm.models.chronos import ChronosForecaster

    horizon = 24
    results: list[dict] = []

    for dataset_name, y in generate_datasets().items():
        split = int(len(y) * 0.8)
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        actual_horizon = min(horizon, len(y_test))

        for model_name, kwargs in MODELS.items():
            t0 = time.time()
            try:
                if model_name in baselines:
                    forecaster = baselines[model_name](**kwargs)
                else:
                    forecaster = ChronosForecaster(device="cpu", seed=42, **kwargs)
                forecaster.fit(y_train)
                pred = forecaster.predict(actual_horizon, return_quantiles=False)

                metrics = compute_metrics(
                    y_true=y_test.values[:actual_horizon],
                    y_pred=pred["mean"].values,
                    y_insample=y_train.values,
                    mean=pred["mean"].values,
                    std=pred["std"].values,
                )
                elapsed = time.time() - t0
                results.append(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "series_id": "synthetic_1",
                        "horizon": actual_horizon,
                        "time_seconds": round(elapsed, 3),
                        **metrics,
                    }
                )
                logger.info(
                    "%s on %s: MASE=%.3f (%.1fs)",
                    model_name,
                    dataset_name,
                    metrics.get("MASE", float("nan")),
                    elapsed,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("%s on %s failed: %s", model_name, dataset_name, exc)

    return results


def save(results: list[dict]) -> None:
    """Persist CSV / LaTeX artifacts and print a summary table."""
    from uniftsm.evaluation.report import latex_table

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)

    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)

    metric_names = ["MASE", "sMAPE", "CRPS", "RMSE", "MAE"]
    latex = latex_table(
        results,
        metric_names=metric_names,
        caption="UniTSFM preliminary benchmark: statistical baselines and Chronos-tiny "
        "on synthetic series (lower is better).",
        label="tab:preliminary",
    )
    (RESULTS_DIR / "benchmark_table.tex").write_text(latex)

    print("\n=== Preliminary benchmark summary (mean per model) ===")
    if not df.empty:
        summary = df.groupby("model")[metric_names].mean().round(4).sort_values("MASE")
        print(summary.to_string())
    print(f"\nResults written to {RESULTS_DIR}")


def main() -> None:
    print("=" * 62)
    print("UniTSFM preliminary real benchmark (baselines + Chronos-tiny)")
    print("=" * 62)
    results = run()
    save(results)
    print(f"Complete. {len(results)} results.")


if __name__ == "__main__":
    main()
