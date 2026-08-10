"""Precomputed performance profiles for model selection.

Stores performance benchmarks used by the meta-classifier and the
rule-based ``ModelRegistry.suggest()`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class ModelBenchmarkProfile:
    """Performance profile of a model on a benchmark dataset."""

    model_name: str
    dataset: str
    frequency: str
    metric: str
    value: float
    std: float = 0.0


_BENCHMARK_DATA: list[dict[str, Any]] = [
    # M4-Hourly
    {
        "model": "chronos",
        "dataset": "m4_hourly",
        "freq": "h",
        "MASE": 0.85,
        "sMAPE": 9.2,
        "CRPS": 0.12,
    },
    {
        "model": "timesfm",
        "dataset": "m4_hourly",
        "freq": "h",
        "MASE": 0.82,
        "sMAPE": 8.9,
        "CRPS": 0.14,
    },
    {
        "model": "moirai",
        "dataset": "m4_hourly",
        "freq": "h",
        "MASE": 0.88,
        "sMAPE": 9.8,
        "CRPS": 0.13,
    },
    {
        "model": "lag_llama",
        "dataset": "m4_hourly",
        "freq": "h",
        "MASE": 0.91,
        "sMAPE": 10.1,
        "CRPS": 0.15,
    },
    {
        "model": "ttm",
        "dataset": "m4_hourly",
        "freq": "h",
        "MASE": 0.95,
        "sMAPE": 11.0,
        "CRPS": 0.18,
    },
    # Electricity
    {
        "model": "chronos",
        "dataset": "electricity",
        "freq": "h",
        "MASE": 0.72,
        "sMAPE": 7.1,
        "CRPS": 0.09,
    },
    {
        "model": "timesfm",
        "dataset": "electricity",
        "freq": "h",
        "MASE": 0.70,
        "sMAPE": 6.8,
        "CRPS": 0.10,
    },
    {
        "model": "moirai",
        "dataset": "electricity",
        "freq": "h",
        "MASE": 0.68,
        "sMAPE": 6.5,
        "CRPS": 0.08,
    },
    {
        "model": "lag_llama",
        "dataset": "electricity",
        "freq": "h",
        "MASE": 0.76,
        "sMAPE": 7.8,
        "CRPS": 0.11,
    },
    {
        "model": "ttm",
        "dataset": "electricity",
        "freq": "h",
        "MASE": 0.81,
        "sMAPE": 8.5,
        "CRPS": 0.14,
    },
    # Traffic
    {
        "model": "chronos",
        "dataset": "traffic",
        "freq": "h",
        "MASE": 0.79,
        "sMAPE": 8.5,
        "CRPS": 0.11,
    },
    {
        "model": "timesfm",
        "dataset": "traffic",
        "freq": "h",
        "MASE": 0.77,
        "sMAPE": 8.2,
        "CRPS": 0.12,
    },
    {
        "model": "moirai",
        "dataset": "traffic",
        "freq": "h",
        "MASE": 0.75,
        "sMAPE": 8.0,
        "CRPS": 0.10,
    },
    {
        "model": "lag_llama",
        "dataset": "traffic",
        "freq": "h",
        "MASE": 0.82,
        "sMAPE": 9.0,
        "CRPS": 0.13,
    },
    {"model": "ttm", "dataset": "traffic", "freq": "h", "MASE": 0.88, "sMAPE": 9.8, "CRPS": 0.16},
    # Weather
    {
        "model": "chronos",
        "dataset": "weather",
        "freq": "D",
        "MASE": 0.65,
        "sMAPE": 5.8,
        "CRPS": 0.07,
    },
    {
        "model": "timesfm",
        "dataset": "weather",
        "freq": "D",
        "MASE": 0.63,
        "sMAPE": 5.5,
        "CRPS": 0.08,
    },
    {
        "model": "moirai",
        "dataset": "weather",
        "freq": "D",
        "MASE": 0.67,
        "sMAPE": 6.0,
        "CRPS": 0.07,
    },
    {
        "model": "lag_llama",
        "dataset": "weather",
        "freq": "D",
        "MASE": 0.71,
        "sMAPE": 6.5,
        "CRPS": 0.09,
    },
    {"model": "ttm", "dataset": "weather", "freq": "D", "MASE": 0.69, "sMAPE": 6.2, "CRPS": 0.10},
    # Solar
    {
        "model": "chronos",
        "dataset": "solar",
        "freq": "min",
        "MASE": 0.90,
        "sMAPE": 12.0,
        "CRPS": 0.16,
    },
    {
        "model": "timesfm",
        "dataset": "solar",
        "freq": "min",
        "MASE": 0.88,
        "sMAPE": 11.5,
        "CRPS": 0.17,
    },
    {
        "model": "moirai",
        "dataset": "solar",
        "freq": "min",
        "MASE": 0.85,
        "sMAPE": 11.0,
        "CRPS": 0.15,
    },
    {
        "model": "lag_llama",
        "dataset": "solar",
        "freq": "min",
        "MASE": 0.93,
        "sMAPE": 12.8,
        "CRPS": 0.19,
    },
    {"model": "ttm", "dataset": "solar", "freq": "min", "MASE": 0.96, "sMAPE": 13.5, "CRPS": 0.22},
]


class BenchmarkDB:
    """In-memory store of precomputed benchmark results.

    Used by the meta-learner and the rule-based model suggestor.
    """

    def __init__(self) -> None:
        self._data = pd.DataFrame(_BENCHMARK_DATA)

    def get_best_model(
        self,
        dataset: str,
        metric: str = "MASE",
    ) -> str | None:
        """Return the best model for a given dataset and metric.

        Args:
            dataset: Dataset name.
            metric: Metric to optimise (lower is better).

        Returns:
            Best model name or ``None`` if not found.
        """
        subset = self._data[(self._data["dataset"] == dataset)]
        if subset.empty:
            return None
        row = subset.loc[subset[metric].idxmin()]
        return str(row["model"])

    def get_performance(
        self,
        model_name: str,
        dataset: str,
        metric: str = "MASE",
    ) -> float | None:
        """Return a specific performance value."""
        row = self._data[(self._data["model"] == model_name) & (self._data["dataset"] == dataset)]
        if row.empty:
            return None
        return float(row[metric].iloc[0])

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full benchmarks as a DataFrame."""
        return self._data.copy()


# Singleton
benchmark_db = BenchmarkDB()
