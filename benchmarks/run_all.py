#!/usr/bin/env python3
"""Script to reproduce all benchmark results from the UniTSFM paper.

Usage:
    python benchmarks/run_all.py              # Run full benchmark
    python benchmarks/run_all.py --quick      # Quick run (2 series per dataset)
    python benchmarks/run_all.py --dataset electricity  # Single dataset

Output is written to benchmarks/results/ as CSV and LaTeX files.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmarks.run_all")

# ── Configuration ──────────────────────────────────────────────────────────

DATASETS = [
    "m4_hourly",
    "electricity",
    "traffic",
    "weather",
    "solar",
    "exchange_rate",
    "influenza",
]

MODELS = [
    "chronos",
    "timesfm",
    "moirai",
    "lag_llama",
    "ttm",
]

METRICS = ["MASE", "sMAPE", "CRPS", "RMSE", "MAE"]

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UniTSFM benchmarks")
    parser.add_argument("--quick", action="store_true", help="Quick run (2 series)")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset (default: all)",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=10,
        help="Max series per dataset",
    )
    parser.add_argument("--horizon", type=int, default=24, help="Forecast horizon")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else DATASETS
    max_series = 2 if args.quick else args.max_series

    logger.info(
        "Starting benchmark: %d datasets, %d models, %d series max",
        len(datasets),
        len(MODELS),
        max_series,
    )

    from uniftsm.core.registry import registry
    from uniftsm.datasets.monash import MonashDatasetLoader
    from uniftsm.evaluation.metrics import compute_metrics

    loader = MonashDatasetLoader()
    all_results: list[dict] = []

    for dataset_name in datasets:
        logger.info("=" * 60)
        logger.info("Dataset: %s", dataset_name)

        try:
            series_dict = loader.load(dataset_name, max_series=max_series)
        except Exception as exc:
            logger.error("  Failed to load: %s", exc)
            continue

        for series_id, y in series_dict.items():
            split = int(len(y) * 0.8)
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            actual_horizon = min(args.horizon, len(y_test))

            for model_name in MODELS:
                t0 = time.time()
                try:
                    model_cls = registry.get(model_name)
                    forecaster = model_cls(device="cpu", seed=42)
                    forecaster.fit(y_train)
                    pred = forecaster.predict(
                        actual_horizon,
                        return_quantiles=False,
                    )

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
                            "model": model_name,
                            "dataset": dataset_name,
                            "series_id": series_id,
                            "time_seconds": round(elapsed, 3),
                            **metrics,
                        }
                    )
                    logger.info(
                        "  %s/%s: MASE=%.4f (%.1fs)",
                        model_name,
                        series_id,
                        metrics.get("MASE", -1),
                        elapsed,
                    )
                except Exception as exc:
                    logger.warning("  %s/%s failed: %s", model_name, series_id, exc)

    # ── Save results ────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # ── Summary ─────────────────────────────────────────────────────────
    if not results_df.empty:
        summary = results_df.groupby(["model", "dataset"])[METRICS].mean().round(4)
        summary_path = RESULTS_DIR / "summary.csv"
        summary.to_csv(summary_path)
        logger.info("Summary saved to %s", summary_path)

        # Print best model per dataset
        best_models = results_df.groupby("dataset").apply(lambda g: g.loc[g["MASE"].idxmin()])[
            ["model", "MASE"]
        ]
        print("\nBest model per dataset (by MASE):")
        print(best_models.to_string())

    # ── Generate LaTeX table ────────────────────────────────────────────
    from uniftsm.evaluation.report import latex_table

    latex = latex_table(
        all_results,
        metric_names=METRICS,
        caption="UniTSFM benchmark results across 7 Monash datasets and 5 TSFMs.",
        label="tab:benchmark",
    )
    latex_path = RESULTS_DIR / "benchmark_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    logger.info("LaTeX table saved to %s", latex_path)

    logger.info("Benchmark complete. %d results total.", len(all_results))


if __name__ == "__main__":
    main()
