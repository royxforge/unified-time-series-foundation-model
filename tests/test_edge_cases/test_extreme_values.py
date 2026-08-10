"""Tests for extreme values, NaN patterns, and edge-case series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uniftsm.core.exceptions import InsufficientDataError


class TestExtremeValues:
    """Tests that pipeline handles extreme numeric values gracefully."""

    def test_very_large_values(self):
        from uniftsm.pipeline.scaler import SeriesScaler

        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        large = pd.Series(np.ones(50) * 1e12, index=idx)
        scaled = SeriesScaler("standard").fit_transform(large)
        assert np.all(np.isfinite(scaled))
        assert np.allclose(scaled, 0.0)

    def test_very_small_values(self):
        from uniftsm.pipeline.scaler import SeriesScaler

        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        small = pd.Series(np.ones(50) * 1e-12, index=idx)
        scaled = SeriesScaler("standard").fit_transform(small)
        assert np.all(np.isfinite(scaled))

    def test_mixed_sign_values(self):
        from uniftsm.pipeline.scaler import SeriesScaler

        arr = np.random.randn(100) * 100
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        scaler = SeriesScaler("minmax")
        scaled = scaler.fit_transform(pd.Series(arr, index=idx))
        assert np.min(scaled) >= 0.0 - 1e-10
        assert np.max(scaled) <= 1.0 + 1e-10


class TestNaNPatterns:
    """Tests for various missing value configurations."""

    def test_forward_fill_finite_gaps(self):
        from uniftsm.pipeline.missing import MissingValueImputer

        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        values = np.arange(20.0)
        values[5:8] = np.nan
        series = pd.Series(values, index=idx)
        result = MissingValueImputer("forward_fill", max_gap=10, warn_on_fill=False).impute(series)
        assert not result.isna().any()
        assert result.iloc[5] == 4.0

    def test_leading_nans_backward_fill(self):
        from uniftsm.pipeline.missing import MissingValueImputer

        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        values = np.arange(20.0)
        values[:5] = np.nan
        result = MissingValueImputer("backward_fill", max_gap=10, warn_on_fill=False).impute(
            pd.Series(values, index=idx)
        )
        assert not result.isna().any()
        assert result.iloc[0] == 5.0

    def test_all_nan_median_fallback(self):
        from uniftsm.pipeline.missing import MissingValueImputer

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        all_nan = pd.Series(np.nan, index=idx)
        result = MissingValueImputer("median", warn_on_fill=False).impute(all_nan)
        assert not result.isna().any()
        assert result.iloc[0] == 0.0

    def test_trailing_nans(self):
        from uniftsm.pipeline.missing import MissingValueImputer

        idx = pd.date_range("2020-01-01", periods=20, freq="D")
        values = np.arange(20.0)
        values[-5:] = np.nan
        result = MissingValueImputer("forward_fill", max_gap=10, warn_on_fill=False).impute(
            pd.Series(values, index=idx)
        )
        assert not result.isna().any()
        assert result.iloc[-1] == 14.0

    def test_zero_strategy(self):
        from uniftsm.pipeline.missing import MissingValueImputer

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        values = np.arange(10.0)
        values[3:6] = np.nan
        result = MissingValueImputer("zero", warn_on_fill=False).impute(
            pd.Series(values, index=idx)
        )
        assert result.iloc[3] == 0.0


class TestSeriesLengthEdgeCases:
    """Tests for very short and very long series."""

    def test_short_series_raises(self):
        from uniftsm.core.base import BaseForecaster

        class MinimalForecaster(BaseForecaster):
            def _load_model(self):
                pass

            def fit(self, y, X=None, freq=None, **kwargs):
                self._validate_inputs(y)
                self._fitted = True
                return self

            def predict(self, horizon, **kwargs):
                return pd.DataFrame({"mean": np.zeros(horizon), "std": np.ones(horizon)})

        forecaster = MinimalForecaster(model_name="test")
        short = pd.Series(np.arange(5.0), index=pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises(InsufficientDataError):
            forecaster.fit(short)

    def test_long_series_scaler(self):
        from uniftsm.pipeline.scaler import SeriesScaler

        n = 10_000
        arr = np.random.randn(n)
        idx = pd.date_range("2020-01-01", periods=n, freq="min")
        scaled = SeriesScaler("standard").fit_transform(pd.Series(arr, index=idx))
        assert np.all(np.isfinite(scaled))

    def test_two_point_series_scaler(self):
        from uniftsm.pipeline.scaler import SeriesScaler

        series = pd.Series([1.0, 2.0], index=pd.date_range("2020-01-01", periods=2, freq="D"))
        scaler = SeriesScaler("standard")
        scaler.fit(series)
        assert np.all(np.isfinite(scaler.transform(series)))


class TestFrequencyEdgeCases:
    """Tests for unusual frequency patterns."""

    def test_irregular_time_index(self):
        from uniftsm.pipeline.frequency import FrequencyDetector

        idx = pd.DatetimeIndex(
            ["2020-01-01", "2020-01-03", "2020-01-10", "2020-02-01", "2020-03-01"]
        )
        freq = FrequencyDetector.detect(pd.Series(np.arange(5.0), index=idx))
        assert freq is None or isinstance(freq, str)

    def test_daily_frequency(self):
        from uniftsm.pipeline.frequency import FrequencyDetector

        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        freq = FrequencyDetector.detect(pd.Series(np.arange(10.0), index=idx))
        assert freq == "D"

    def test_hourly_frequency(self):
        from uniftsm.pipeline.frequency import FrequencyDetector

        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        freq = FrequencyDetector.detect(pd.Series(np.arange(10.0), index=idx))
        assert freq == "h"

    def test_non_datetime_index_raises(self):
        from uniftsm.pipeline.frequency import FrequencyResampler

        with pytest.raises(TypeError):
            FrequencyResampler().resample(pd.Series(np.arange(10.0)), "D")
