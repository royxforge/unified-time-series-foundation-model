"""Property-based tests for SeriesScaler invariants."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

finite_floats = st.floats(min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False)

# Data may be as large as ~1e4 in magnitude, so the absolute roundtrip error of
# float64 arithmetic is bounded by ``eps * 1e4 ~ 2e-12``.  An ``atol`` of 1e-9
# is the tightest tolerance that is satisfiable for zero-valued observations;
# the meaningful precision guarantee is the relative tolerance ``rtol=1e-5``.
RTOL = 1e-5
ATOL = 1e-9


@given(values=st.lists(finite_floats, min_size=3, max_size=30))
def test_standard_scaler_roundtrip(values: list[float]):
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array(values, dtype=np.float64)
    scaler = SeriesScaler("standard")
    reconstructed = scaler.inverse_transform(scaler.fit_transform(arr))
    np.testing.assert_allclose(reconstructed, arr, rtol=RTOL, atol=ATOL)


@given(values=st.lists(finite_floats, min_size=3, max_size=30))
def test_minmax_scaler_roundtrip(values: list[float]):
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array(values, dtype=np.float64)
    # Skip constant series
    if np.allclose(arr, arr[0]):
        return
    scaler = SeriesScaler("minmax")
    reconstructed = scaler.inverse_transform(scaler.fit_transform(arr))
    np.testing.assert_allclose(reconstructed, arr, rtol=RTOL, atol=ATOL)


@given(values=st.lists(finite_floats, min_size=3, max_size=30))
def test_robust_scaler_roundtrip(values: list[float]):
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array(values, dtype=np.float64)
    if np.std(arr) < 1e-10:
        return  # Skip constant/near-constant
    scaler = SeriesScaler("robust")
    reconstructed = scaler.inverse_transform(scaler.fit_transform(arr))
    np.testing.assert_allclose(reconstructed, arr, rtol=RTOL, atol=ATOL)


@given(values=st.lists(finite_floats, min_size=3, max_size=30))
def test_identity_scaler_no_change(values: list[float]):
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array(values, dtype=np.float64)
    np.testing.assert_allclose(SeriesScaler("identity").fit_transform(arr), arr)


@given(values=st.lists(finite_floats, min_size=3, max_size=30))
def test_minmax_scaler_range(values: list[float]):
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array(values, dtype=np.float64)
    if np.allclose(arr, arr[0]):
        return
    scaled = SeriesScaler("minmax").fit_transform(arr)
    assert np.min(scaled) >= 0.0 - 1e-10
    assert np.max(scaled) <= 1.0 + 1e-10


def test_constant_series_standard():
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.ones(20) * 5.0
    np.testing.assert_allclose(SeriesScaler("standard").fit_transform(arr), np.zeros(20))


def test_constant_series_minmax():
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.ones(20) * 5.0
    np.testing.assert_allclose(SeriesScaler("minmax").fit_transform(arr), np.zeros(20))


def test_constant_series_robust():
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.ones(20) * 5.0
    np.testing.assert_allclose(SeriesScaler("robust").fit_transform(arr), np.zeros(20))


def test_scaler_not_fitted_raises():
    from uniftsm.pipeline.scaler import SeriesScaler

    with pytest.raises(RuntimeError, match="not been fitted"):
        SeriesScaler("standard").transform(np.array([1.0, 2.0, 3.0]))


def test_scaler_get_params():
    from uniftsm.pipeline.scaler import SeriesScaler

    scaler = SeriesScaler("standard")
    scaler.fit(np.array([1.0, 2.0, 3.0]))
    params = scaler.get_params()
    assert params["method"] == "standard"
    assert "mean" in params
    assert "std" in params


def test_standard_scaler_zero_mean():
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(np.mean(SeriesScaler("standard").fit_transform(arr))) < 1e-10


def test_robust_scaler_median_zero():
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert abs(np.median(SeriesScaler("robust").fit_transform(arr))) < 1e-10


@given(
    values=st.lists(finite_floats, min_size=3, max_size=30),
    shift=st.floats(min_value=-500, max_value=500),
)
def test_standard_scaler_shift_invariance(values: list[float], shift: float):
    from uniftsm.pipeline.scaler import SeriesScaler

    arr = np.array(values, dtype=np.float64)
    if np.std(arr) < 1e-10:
        return
    scaler = SeriesScaler("standard")
    scaled = scaler.fit_transform(arr)
    scaled_shifted = scaler.transform(arr + shift)
    np.testing.assert_allclose(np.std(scaled), np.std(scaled_shifted), rtol=1e-5)
