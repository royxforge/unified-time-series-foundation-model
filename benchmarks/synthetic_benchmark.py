#!/usr/bin/env python3
"""Synthetic benchmark that runs without external TSFM dependencies.

Uses statistical baselines to validate the benchmarking pipeline.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent / "results"


def generate_benchmark_data() -> dict[str, pd.Series]:
    """Generate synthetic time series for benchmarking.

    Uses static methods directly (no class instantiation needed).
    """
    from uniftsm.datasets.synthetic import SyntheticDatasetGenerator

    rng = np.random.default_rng(42)

    datasets: dict[str, pd.Series] = {
        "linear_trend": SyntheticDatasetGenerator.linear_trend(n=200, slope=0.1, freq="D"),
        "seasonal": SyntheticDatasetGenerator.seasonal(n=200, period=12, freq="D"),
        "random_walk": SyntheticDatasetGenerator.random_walk(n=200, freq="D"),
        "complex": SyntheticDatasetGenerator.complex(n=200, freq="D"),
        "manual_rw": pd.Series(
            np.cumsum(rng.normal(0, 1, 200)),
            index=pd.date_range("2020-01-01", periods=200, freq="D"),
            name="random_walk",
        ),
    }
    return datasets


def run_benchmark(quick: bool = False, dataset: str | None = None) -> list[dict]:
    """Run synthetic benchmark using statistical baselines."""
    from uniftsm.evaluation.metrics import compute_metrics
    from uniftsm.evaluation.statistical_baselines import (
        NaiveForecaster,
        SeasonalNaiveForecaster,
        ThetaForecaster,
    )

    datasets = generate_benchmark_data()
    if dataset:
        datasets = {k: v for k, v in datasets.items() if k == dataset}
    if quick:
        datasets = dict(list(datasets.items())[:2])

    horizon = 24
    all_results = []

    for ds_name, y in datasets.items():
        split = int(len(y) * 0.8)
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        actual_horizon = min(horizon, len(y_test))

        baselines = {
            "naive": NaiveForecaster(),
            "seasonal_naive": SeasonalNaiveForecaster(seasonal_period=12),
            "theta": ThetaForecaster(),
        }

        for model_name, model in baselines.items():
            t0 = time.time()
            try:
                model.fit(y_train)
                pred = model.predict(actual_horizon)
                metrics = compute_metrics(
                    y_true=y_test.values[:actual_horizon],
                    y_pred=pred["mean"].values,
                    y_insample=y_train.values,
                    mean=pred["mean"].values,
                    std=pred["std"].values,
                )
                elapsed = time.time() - t0
                all_results.append(
                    {
                        "dataset": ds_name,
                        "model": model_name,
                        "horizon": actual_horizon,
                        "time_seconds": round(elapsed, 4),
                        **metrics,
                    }
                )
                print(
                    f"  {ds_name}/{model_name}: MASE={metrics.get('MASE', -1):.4f} ({elapsed:.2f}s)"
                )
            except Exception as e:
                print(f"  {ds_name}/{model_name} failed: {e}")

    return all_results


def save_results(results: list[dict]) -> None:
    """Save benchmark results to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "synthetic_benchmark.csv", index=False)
    with open(RESULTS_DIR / "synthetic_benchmark.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    if not results:
        print("No results to summarize.")
        return
    summary = df.groupby("model")[["MASE", "sMAPE", "CRPS", "time_seconds"]].mean().round(4)
    summary.to_csv(RESULTS_DIR / "synthetic_summary.csv")
    print("\nSynthetic benchmark summary saved.")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UniTSFM synthetic benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick run (2 datasets)")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset name")
    args = parser.parse_args()
    print("=" * 60)
    print("UniTSFM Synthetic Benchmark")
    print("=" * 60)
    results = run_benchmark(quick=args.quick, dataset=args.dataset)
    save_results(results)
    print(f"\nComplete. {len(results)} results.")


if __name__ == "__main__":
    main()
