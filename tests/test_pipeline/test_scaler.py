"""Tests for SeriesScaler."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.pipeline.scaler import SeriesScaler


class TestSeriesScaler:
    def test_standard_roundtrip(self, simple_series):
        scaler = SeriesScaler("standard")
        scaled = scaler.fit_transform(simple_series)
        assert abs(scaled.mean()) < 1e-10
        assert abs(scaled.std(ddof=1) - 1.0) < 1e-6
        restored = scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(restored, simple_series.values, decimal=6)

    def test_minmax_roundtrip(self, simple_series):
        scaler = SeriesScaler("minmax")
        scaled = scaler.fit_transform(simple_series)
        assert scaled.min() >= 0.0
        assert scaled.max() <= 1.0
        restored = scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(restored, simple_series.values, decimal=6)

    def test_robust_roundtrip(self, simple_series):
        scaler = SeriesScaler("robust")
        scaled = scaler.fit_transform(simple_series)
        restored = scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(restored, simple_series.values, decimal=6)

    def test_identity_no_change(self, simple_series):
        scaler = SeriesScaler("identity")
        scaled = scaler.fit_transform(simple_series)
        np.testing.assert_array_almost_equal(scaled, simple_series.values)

    def test_not_fitted_raises(self):
        scaler = SeriesScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(np.array([1.0, 2.0]))

    def test_is_fitted(self, simple_series):
        scaler = SeriesScaler()
        assert not scaler.is_fitted
        scaler.fit(simple_series)
        assert scaler.is_fitted

    def test_get_params(self, simple_series):
        scaler = SeriesScaler("robust")
        scaler.fit(simple_series)
        params = scaler.get_params()
        assert params["method"] == "robust"
        assert "median" in params
        assert "iqr" in params

    def test_constant_series_standard(self):
        series = pd.Series(np.ones(50))
        scaler = SeriesScaler("standard")
        scaled = scaler.fit_transform(series)
        assert len(scaled) == 50

    def test_constant_series_minmax(self):
        series = pd.Series(np.ones(50))
        scaler = SeriesScaler("minmax")
        scaled = scaler.fit_transform(series)
        np.testing.assert_array_almost_equal(scaled, np.zeros(50))

    def test_invalid_method_raises_on_transform(self):
        scaler = SeriesScaler("invalid")
        series = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            scaler.fit_transform(series)

    def test_repr(self):
        scaler = SeriesScaler("standard")
        r = repr(scaler)
        assert "standard" in r
        assert "fitted=" in r

    def test_numpy_input(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler = SeriesScaler("minmax")
        scaled = scaler.fit_transform(arr)
        assert scaled[0] == 0.0
        assert scaled[-1] == 1.0
