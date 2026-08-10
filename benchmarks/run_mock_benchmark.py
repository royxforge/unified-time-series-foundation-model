"""Mock benchmark for CI — runs without any external TSFM packages.

Uses MockForecaster to validate the benchmarking infrastructure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path so conftest is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from conftest import MockForecaster

RESULTS_DIR = Path(__file__).parent / "results"


def run_mock_benchmark() -> list[dict]:
    """Run benchmark using MockForecaster (no real TSFM dependencies)."""
    from uniftsm.evaluation.metrics import compute_metrics

    horizon = 24
    rng = np.random.default_rng(42)

    # Generate mock data
    idx = pd.date_range("2020-01-01", periods=200, freq="D")
    y_train = pd.Series(np.cumsum(rng.normal(0, 1, 176)), index=idx[:176], name="value")
    y_test = pd.Series(np.cumsum(rng.normal(0, 1, 24)), index=idx[176:], name="value")
    actual_horizon = min(horizon, len(y_test))

    datasets = ["electricity", "weather", "traffic", "solar"]
    models = ["chronos", "timesfm", "moirai", "lag_llama", "ttm"]

    all_results = []

    for ds_name in datasets:
        for model_name in models:
            t0 = time.time()
            try:
                forecaster = MockForecaster(
                    model_name=model_name,
                    mean_values=np.linspace(0, 5, actual_horizon),
                    std_values=np.linspace(0.5, 1.0, actual_horizon),
                )
                forecaster.fit(y_train)
                pred = forecaster.predict(actual_horizon)

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
                print(f"  {ds_name}/{model_name}: MASE={metrics.get('MASE', -1):.4f}")

            except Exception as e:
                print(f"  {ds_name}/{model_name} failed: {e}")

    return all_results


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UniTSFM Mock Benchmark (CI-safe, no TSFM deps)")
    print("=" * 60)

    results = run_mock_benchmark()

    # Save results
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "mock_benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    with open(RESULTS_DIR / "mock_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    # Summary
    if not results:
        print("No results generated.")
        return

    summary = df.groupby("model")[["MASE", "sMAPE", "CRPS", "time_seconds"]].mean().round(4)
    summary.to_csv(RESULTS_DIR / "mock_summary.csv")
    print("\nSummary (averaged across datasets):")
    print(summary.to_string())
    print(f"\nMock benchmark complete. {len(results)} results saved to {csv_path}")


if __name__ == "__main__":
    main()
