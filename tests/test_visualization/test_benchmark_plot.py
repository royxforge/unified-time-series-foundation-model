"""Tests for the benchmark results plotting script.

Uses matplotlib's Agg backend (no display required).
"""

import matplotlib

matplotlib.use("Agg")

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"
PLOT_SCRIPT = BENCHMARKS_DIR / "plot_results.py"


@pytest.fixture
def sample_results(tmp_path: Path) -> Path:
    """A small, deterministic benchmark_results.csv with the real schema."""
    rows = []
    for dataset in ["electricity", "solar", "exchange_rate"]:
        for model in ["chronos", "timesfm", "ttm"]:
            rows.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "series_id": "s1",
                    "time_seconds": 1.0,
                    "MASE": {"electricity": 0.5, "solar": 0.1, "exchange_rate": 2.0}[dataset]
                    * (1 if model == "timesfm" else 2),
                    "sMAPE": 10.0,
                    "CRPS": 5.0,
                    "RMSE": 8.0,
                    "MAE": 6.0,
                }
            )
    csv_path = tmp_path / "benchmark_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def test_plot_results_generates_png(sample_results: Path, tmp_path: Path) -> None:
    out = tmp_path / "chart.png"
    result = subprocess.run(
        [sys.executable, str(PLOT_SCRIPT), "--csv", str(sample_results), "--output", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_results_missing_csv(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--csv",
            str(tmp_path / "nope.csv"),
            "--output",
            str(tmp_path / "chart.png"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not found" in result.stderr


def test_plot_results_unknown_metric(sample_results: Path, tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PLOT_SCRIPT),
            "--csv",
            str(sample_results),
            "--metric",
            "NOPE",
            "--output",
            str(tmp_path / "chart.png"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "not in benchmark_results.csv" in result.stderr
