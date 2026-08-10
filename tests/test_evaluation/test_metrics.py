"""Tests for forecasting metrics: MASE, sMAPE, CRPS, NMAE, OWA, MRAE."""

import numpy as np
import pytest

from uniftsm.evaluation.metrics import (
    compute_metrics,
    coverage,
    crps,
    mase,
    mrae,
    nmae,
    owa,
    smape,
)


class TestMASE:
    def test_perfect_forecast(self):
        val = mase(np.ones(10), np.ones(10), seasonality=1)
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_bad_forecast(self):
        val = mase(
            np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            np.array([10.0, 10.0, 10.0, 10.0, 10.0]),
            seasonality=1,
        )
        assert val > 1.0

    def test_with_insample(self):
        val = mase(
            np.array([5.0, 6.0]),
            np.array([4.5, 6.5]),
            y_insample=np.array([1.0, 2.0, 3.0, 4.0]),
            seasonality=1,
        )
        assert val >= 0
        assert isinstance(val, float)


class TestSMAPE:
    def test_perfect_forecast(self):
        val = smape(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert val == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        val = smape(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
        assert val >= 0


class TestCRPS:
    def test_via_mean_std(self):
        val = crps(
            np.array([1.0, 2.0, 3.0]),
            mean=np.array([1.0, 2.0, 3.0]),
            std=np.array([0.1, 0.1, 0.1]),
        )
        assert val >= 0

    def test_larger_uncertainty_increases_crps(self):
        low = crps(np.array([1.0]), mean=np.array([1.0]), std=np.array([0.1]))
        high = crps(np.array([1.0]), mean=np.array([1.0]), std=np.array([2.0]))
        assert high > low

    def test_raises_without_args(self):
        with pytest.raises(ValueError, match="Either"):
            crps(np.array([1.0]))


class TestNMAE:
    def test_perfect(self):
        val = nmae(np.array([1.0, 2.0]), np.array([1.0, 2.0]), y_insample=np.array([1.0, 2.0]))
        assert val == pytest.approx(0.0, abs=1e-6)


class TestOWA:
    def test_default(self):
        val = owa(0.8, 9.0, smape_naive=100.0)
        assert val == pytest.approx((0.8 / 1.0 + 9.0 / 100.0) / 2.0)


class TestMRAE:
    def test_forecast_better_than_baseline(self):
        val = mrae(np.array([1.0, 2.0]), np.array([1.1, 1.9]), np.array([5.0, 5.0]))
        assert val < 1.0

    def test_forecast_worse_than_baseline(self):
        val = mrae(np.array([1.0, 2.0]), np.array([5.0, 5.0]), np.array([1.1, 1.9]))
        assert val > 1.0


class TestCoverage:
    def test_perfect_coverage(self):
        assert coverage(np.array([0.0, 0.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0])) == 1.0

    def test_no_coverage(self):
        assert coverage(np.array([10.0]), np.array([-1.0]), np.array([1.0])) == 0.0


class TestComputeMetrics:
    def test_all_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 4.2, 4.8])
        y_insample = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        results = compute_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_insample=y_insample,
            mean=y_pred,
            std=np.full(5, 0.1),
            lower=y_pred - 0.2,
            upper=y_pred + 0.2,
        )
        for key in ["MASE", "sMAPE", "CRPS", "NMAE", "OWA", "Coverage", "MAE", "RMSE"]:
            assert key in results, f"Missing metric: {key}"
