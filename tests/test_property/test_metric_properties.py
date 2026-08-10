"""Property-based tests for evaluation metrics using Hypothesis.

Tests statistical invariants that must hold for any valid input.
"""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given
from hypothesis import strategies as st

# ── Float array strategy for metric inputs ─────────────────────────────────

finite_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)


@given(
    y_true=st.lists(finite_floats, min_size=2, max_size=100),
    y_pred=st.lists(finite_floats, min_size=2, max_size=100),
)
def test_mase_non_negative(y_true: list[float], y_pred: list[float]):
    """MASE is always non-negative for any valid input."""
    from uniftsm.evaluation.metrics import mase

    yt = np.array(y_true)
    yp = np.array(y_pred)
    assume(len(yt) == len(yp))
    assume(not np.allclose(yt, yp))  # avoid degenerate case

    result = mase(yt, yp)
    assert result >= 0, f"MASE was negative: {result}"
    assert np.isfinite(result), f"MASE was non-finite: {result}"


@given(
    y_true=st.lists(finite_floats, min_size=1, max_size=100),
    y_pred=st.lists(finite_floats, min_size=1, max_size=100),
)
def test_smape_bounded(y_true: list[float], y_pred: list[float]):
    """sMAPE is in [0, 200] for any valid input."""
    from uniftsm.evaluation.metrics import smape

    yt = np.array(y_true)
    yp = np.array(y_pred)
    assume(len(yt) == len(yp))

    result = smape(yt, yp)
    assert 0 <= result <= 200, f"sMAPE out of bounds: {result}"


@given(
    y_true=st.lists(finite_floats, min_size=1, max_size=50),
    mean=st.lists(finite_floats, min_size=1, max_size=50),
    std=st.lists(st.floats(min_value=0.01, max_value=10.0), min_size=1, max_size=50),
)
def test_crps_parametric_non_negative(y_true: list[float], mean: list[float], std: list[float]):
    """CRPS (parametric) is always non-negative."""
    from uniftsm.evaluation.metrics import crps

    yt = np.array(y_true)
    mn = np.array(mean)
    sd = np.array(std)
    assume(len(yt) == len(mn) == len(sd))

    result = crps(yt, mean=mn, std=sd)
    assert result >= 0, f"CRPS was negative: {result}"
    assert np.isfinite(result)


@given(
    y_true=st.lists(finite_floats, min_size=1, max_size=50),
    y_pred=st.lists(finite_floats, min_size=1, max_size=50),
    y_insample=st.lists(finite_floats, min_size=2, max_size=100),
)
def test_nmae_non_negative(y_true: list[float], y_pred: list[float], y_insample: list[float]):
    """NMAE is always non-negative."""
    from uniftsm.evaluation.metrics import nmae

    yt = np.array(y_true)
    yp = np.array(y_pred)
    yi = np.array(y_insample)
    assume(len(yt) == len(yp))

    result = nmae(yt, yp, yi)
    assert result >= 0, f"NMAE was negative: {result}"
    assert np.isfinite(result)


@given(
    mase_val=st.floats(min_value=0.0, max_value=100.0),
    smape_val=st.floats(min_value=0.0, max_value=200.0),
)
def test_owa_positive(mase_val: float, smape_val: float):
    """OWA is always positive for valid MASE and sMAPE."""
    from uniftsm.evaluation.metrics import owa

    result = owa(mase_val, smape_val)
    assert result >= 0, f"OWA was negative: {result}"
    assert not np.isnan(result)


@given(
    y_true=st.lists(finite_floats, min_size=1, max_size=50),
    lower=st.lists(finite_floats, min_size=1, max_size=50),
    upper=st.lists(finite_floats, min_size=1, max_size=50),
)
def test_coverage_between_zero_and_one(
    y_true: list[float], lower: list[float], upper: list[float]
):
    """Coverage is always in [0, 1]."""
    from uniftsm.evaluation.metrics import coverage

    yt = np.array(y_true)
    lo = np.array(lower)
    up = np.array(upper)
    assume(len(yt) == len(lo) == len(up))
    assume(np.all(lo <= up))

    result = coverage(yt, lo, up)
    assert 0 <= result <= 1, f"Coverage out of bounds: {result}"


class TestMetricInvariants:
    """Deterministic metric invariants."""

    def test_perfect_forecast_mase_zero(self):
        """MASE = 0 for a perfect forecast."""
        from uniftsm.evaluation.metrics import mase

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        result = mase(y_true, y_pred, y_insample=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert result == 0.0, f"Perfect MASE should be 0, got {result}"

    def test_perfect_forecast_smape_zero(self):
        """sMAPE = 0 for a perfect forecast."""
        from uniftsm.evaluation.metrics import smape

        y_true = np.array([1.0, 2.0, 3.0])
        result = smape(y_true, y_true)
        assert result == 0.0, f"Perfect sMAPE should be 0, got {result}"

    def test_mase_higher_for_worse_forecast(self):
        """MASE should increase as forecast quality degrades."""
        from uniftsm.evaluation.metrics import mase

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_insample = np.array([1.0, 2.0, 3.0, 4.0])

        good = mase(y_true, y_true, y_insample)
        bad = mase(y_true, np.array([10, 20, 30, 40, 50]), y_insample)
        assert bad > good, "Worse forecast should have higher MASE"

    def test_coverage_perfect_intervals(self):
        """Coverage = 1.0 when all true values are within bounds."""
        from uniftsm.evaluation.metrics import coverage

        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        result = coverage(y_true, lower, upper)
        assert result == 1.0

    def test_coverage_no_coverage(self):
        """Coverage = 0.0 when no true values are within bounds."""
        from uniftsm.evaluation.metrics import coverage

        y_true = np.array([10.0, 20.0, 30.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        result = coverage(y_true, lower, upper)
        assert result == 0.0

    def test_owa_naive_equals_one(self):
        """OWA = 1.0 when forecast matches naive baseline."""
        from uniftsm.evaluation.metrics import owa

        result = owa(1.0, 100.0, mase_naive=1.0, smape_naive=100.0)
        assert result == 1.0
