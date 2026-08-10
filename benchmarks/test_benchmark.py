"""Tests for the benchmark scripts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestSyntheticBenchmark:
    """Tests for the synthetic benchmark script."""

    def test_synthetic_benchmark_generates_results(self):
        from benchmarks.synthetic_benchmark import RESULTS_DIR, run_benchmark, save_results

        original_dir = RESULTS_DIR
        try:
            tmpdir = tempfile.mkdtemp()
            import benchmarks.synthetic_benchmark as sb_mod

            sb_mod.RESULTS_DIR = Path(tmpdir)

            results = run_benchmark(quick=True)
            assert len(results) > 0

            for r in results:
                assert "dataset" in r
                assert "model" in r
                assert "MASE" in r
                assert "sMAPE" in r

            save_results(results)

            csv_path = sb_mod.RESULTS_DIR / "synthetic_benchmark.csv"
            json_path = sb_mod.RESULTS_DIR / "synthetic_benchmark.json"
            assert csv_path.exists()
            assert json_path.exists()

            df = pd.read_csv(csv_path)
            assert len(df) == len(results)

            with open(json_path) as f:
                loaded = json.load(f)
            assert len(loaded) == len(results)
        finally:
            import benchmarks.synthetic_benchmark as sb_mod

            sb_mod.RESULTS_DIR = original_dir

    def test_benchmark_data_generation(self):
        from benchmarks.synthetic_benchmark import generate_benchmark_data

        datasets = generate_benchmark_data()
        assert len(datasets) > 0
        for series in datasets.values():
            assert len(series) == 200
            assert isinstance(series.index, pd.DatetimeIndex)


class TestMockBenchmark:
    """Tests for the mock (CI-safe) benchmark."""

    def test_mock_benchmark_runs(self):
        from benchmarks.run_mock_benchmark import run_mock_benchmark

        results = run_mock_benchmark()
        assert len(results) > 0

        for r in results:
            assert "dataset" in r
            assert "model" in r
            assert "MASE" in r
            assert np.isfinite(r["MASE"])

    def test_mock_benchmark_output_format(self):
        from benchmarks.run_mock_benchmark import run_mock_benchmark

        results = run_mock_benchmark()
        required_keys = {
            "dataset",
            "model",
            "horizon",
            "time_seconds",
            "MASE",
            "sMAPE",
            "MAE",
            "RMSE",
        }
        for r in results:
            missing = required_keys - set(r.keys())
            assert not missing, f"Missing keys: {missing}"


class TestBenchmarkRunnerConfig:
    """Tests for the main benchmark runner configuration."""

    def test_benchmark_runner_config(self):
        from benchmarks.run_all import DATASETS, METRICS, MODELS

        assert len(DATASETS) > 0
        assert len(MODELS) >= 5
        assert "MASE" in METRICS
        assert "sMAPE" in METRICS
        assert "CRPS" in METRICS


class TestBenchmarkDataValidation:
    """Validate benchmark data integrity."""

    def test_results_schema_consistency(self):
        from benchmarks.run_mock_benchmark import run_mock_benchmark

        results = run_mock_benchmark()
        if not results:
            pytest.skip("No results")

        keys = set(results[0].keys())
        for r in results[1:]:
            assert set(r.keys()) == keys, "Inconsistent keys"

    def test_metrics_are_finite(self):
        from benchmarks.run_mock_benchmark import run_mock_benchmark

        results = run_mock_benchmark()
        for r in results:
            for key in ["MASE", "sMAPE", "MAE", "RMSE"]:
                val = r.get(key, np.nan)
                assert np.isfinite(
                    val
                ), f"{key}={val} is not finite for {r['model']}/{r['dataset']}"


class TestResultsReproducibility:
    """Benchmark results should be reproducible with fixed seed."""

    def test_synthetic_benchmark_reproducible(self):
        from benchmarks.synthetic_benchmark import generate_benchmark_data

        d1 = generate_benchmark_data()
        d2 = generate_benchmark_data()

        for key in d1:
            np.testing.assert_allclose(
                d1[key].values, d2[key].values, err_msg=f"{key} not reproducible"
            )
