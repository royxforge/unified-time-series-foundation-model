"""Tests for statistical baselines, significance, and report generation."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.evaluation.report import ComparisonReport, latex_table
from uniftsm.evaluation.significance import (
    diebold_mariano_test,
    wilcoxon_signed_rank_test,
)
from uniftsm.evaluation.statistical_baselines import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def _make_series(values):
    """Create a Series with DatetimeIndex for baselines that need it."""
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


class TestNaiveForecaster:
    def test_basic_forecast(self):
        y = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        model = NaiveForecaster()
        model.fit(y)
        pred = model.predict(horizon=3)
        assert "mean" in pred.columns
        assert pred["mean"].iloc[0] == 5.0

    def test_predict_quantiles(self):
        y = _make_series([1.0, 2.0, 3.0, 4.0, 5.0])
        model = NaiveForecaster()
        model.fit(y)
        result = model.predict(horizon=3, return_quantiles=True)
        assert isinstance(result, dict)
        assert "q_0.5" in result

    def test_get_params(self):
        model = NaiveForecaster(seed=123)
        params = model.get_params()
        assert params["model_name"] == "naive"
        assert params["seed"] == 123

    def test_unfitted_raises(self):
        model = NaiveForecaster()
        with pytest.raises(RuntimeError):
            model.predict(horizon=3)


class TestSeasonalNaiveForecaster:
    def test_basic_forecast(self):
        y = _make_series(np.arange(20.0))
        model = SeasonalNaiveForecaster(season_length=4)
        model.fit(y)
        pred = model.predict(horizon=4)
        assert "mean" in pred.columns
        assert pred["mean"].iloc[0] == 16.0

    def test_insufficient_data_raises(self):
        y = _make_series([1.0, 2.0])
        model = SeasonalNaiveForecaster(season_length=4)
        with pytest.raises(ValueError):
            model.fit(y)

    def test_get_params(self):
        model = SeasonalNaiveForecaster(season_length=12)
        params = model.get_params()
        assert params["season_length"] == 12


class TestDieboldMariano:
    def test_identical_forecasts(self):
        np.random.seed(42)
        e1 = np.random.randn(50) * 0.1
        e2 = e1.copy()
        result = diebold_mariano_test(e1, e2, h=1)
        assert abs(result["dm_stat"]) < 1e-10

    def test_different_forecasts(self):
        np.random.seed(42)
        e1 = np.random.randn(50) * 0.1
        e2 = np.random.randn(50) * 0.5
        result = diebold_mariano_test(e1, e2, h=1)
        assert isinstance(result["dm_stat"], float)
        assert 0 <= result["p_value"] <= 1

    def test_unequal_length_raises(self):
        e1 = np.array([1.0, 2.0])
        e2 = np.array([1.0])
        with pytest.raises(ValueError, match="same length"):
            diebold_mariano_test(e1, e2)


class TestWilcoxon:
    def test_identical_forecasts(self):
        np.random.seed(42)
        e1 = np.random.randn(30) * 0.1
        e2 = e1.copy()
        result = wilcoxon_signed_rank_test(e1, e2)
        assert 0 <= result["p_value"] <= 1

    def test_different_forecasts(self):
        np.random.seed(42)
        e1 = np.random.randn(30) * 0.1
        e2 = np.random.randn(30) * 0.5
        result = wilcoxon_signed_rank_test(e1, e2)
        assert isinstance(result["statistic"], float)
        assert 0 <= result["p_value"] <= 1


class TestComparisonReport:
    def test_summary(self):
        report = ComparisonReport(
            [
                {"model": "a", "dataset": "d1", "MASE": 0.5, "sMAPE": 5.0},
                {"model": "b", "dataset": "d1", "MASE": 0.6, "sMAPE": 6.0},
            ],
            metric_names=["MASE", "sMAPE"],
        )
        summary = report.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "MASE" in summary.columns

    def test_best_model_per_dataset(self):
        report = ComparisonReport(
            [
                {"model": "a", "dataset": "d1", "MASE": 0.5},
                {"model": "b", "dataset": "d1", "MASE": 0.6},
                {"model": "a", "dataset": "d2", "MASE": 0.4},
                {"model": "b", "dataset": "d2", "MASE": 0.3},
            ],
            metric_names=["MASE"],
        )
        best = report.best_model_per_dataset("MASE")
        assert isinstance(best, pd.Series)

    def test_rank_matrix(self):
        report = ComparisonReport(
            [
                {"model": "a", "dataset": "d1", "MASE": 0.5},
                {"model": "b", "dataset": "d1", "MASE": 0.6},
            ],
            metric_names=["MASE"],
        )
        ranks = report.rank_matrix("MASE")
        assert isinstance(ranks, pd.DataFrame)

    def test_win_count(self):
        report = ComparisonReport(
            [
                {"model": "a", "dataset": "d1", "MASE": 0.5},
                {"model": "b", "dataset": "d1", "MASE": 0.6},
            ],
            metric_names=["MASE"],
        )
        wins = report.win_count("MASE")
        assert isinstance(wins, pd.Series)


class TestLatexTable:
    def _sample_results(self):
        return [
            {"model": "chronos", "dataset": "d1", "MASE": 0.5, "CRPS": 1.0},
            {"model": "timesfm", "dataset": "d1", "MASE": 0.4, "CRPS": 0.8},
            {"model": "chronos", "dataset": "d2", "MASE": 0.6, "CRPS": 1.2},
            {"model": "timesfm", "dataset": "d2", "MASE": 0.7, "CRPS": 1.5},
        ]

    def test_structure_one_row_per_metric(self):
        latex = latex_table(
            self._sample_results(),
            metric_names=["MASE", "CRPS"],
        )
        lines = latex.splitlines()
        header = [ln for ln in lines if "Dataset" in ln and "Metric" in ln]
        assert header, "table must have a 'Dataset & Metric' header"
        assert "Dataset & Metric & chronos & timesfm" in header[0]
        assert r"\begin{tabular}{llcc}" in latex, "column spec must be ll + 2c"
        # One data row per dataset × metric = 2 × 2 = 4.
        data_rows = [
            ln for ln in lines if ln.strip().startswith("d1") or ln.strip().startswith("d2")
        ]
        assert len(data_rows) == 4

    def test_best_value_bold(self):
        latex = latex_table(
            self._sample_results(),
            metric_names=["MASE"],
        )
        assert r"\textbf{0.400}" in latex, "best MASE (timesfm/d1) must be bold"

    def test_missing_model_dash(self):
        # timesfm only ran on d1 — its d2 cell must render as '--'.
        results = [
            {"model": "chronos", "dataset": "d1", "MASE": 0.5},
            {"model": "timesfm", "dataset": "d1", "MASE": 0.4},
            {"model": "chronos", "dataset": "d2", "MASE": 0.6},
        ]
        latex = latex_table(results, metric_names=["MASE"])
        d2_row = [ln for ln in latex.splitlines() if ln.startswith("d2")][0]
        assert "--" in d2_row, "missing model on d2 must render as '--'"
