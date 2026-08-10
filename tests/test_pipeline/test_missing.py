"""Tests for MissingValueImputer."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.pipeline.missing import MissingValueImputer


class TestMissingValueImputer:
    def test_no_missing_unchanged(self, simple_series):
        imp = MissingValueImputer()
        result = imp.impute(simple_series)
        assert not result.isna().any()
        assert len(result) == len(simple_series)

    def test_forward_fill(self, series_with_nans):
        imp = MissingValueImputer("forward_fill", max_gap=5)
        result = imp.impute(series_with_nans)
        assert not result.isna().any()

    def test_backward_fill(self, series_with_nans):
        imp = MissingValueImputer("backward_fill", max_gap=5)
        result = imp.impute(series_with_nans)
        assert not result.isna().any()

    def test_linear_interpolation(self, series_with_nans):
        imp = MissingValueImputer("linear", max_gap=10)
        result = imp.impute(series_with_nans)
        assert not result.isna().any()

    def test_cubic_interpolation(self, series_with_nans):
        imp = MissingValueImputer("cubic", max_gap=10)
        result = imp.impute(series_with_nans)
        assert not result.isna().any()

    def test_mean_imputation(self, series_with_nans):
        imp = MissingValueImputer("mean")
        result = imp.impute(series_with_nans)
        assert not result.isna().any()

    def test_median_imputation(self, series_with_nans):
        imp = MissingValueImputer("median")
        result = imp.impute(series_with_nans)
        assert not result.isna().any()

    def test_zero_imputation(self, series_with_nans):
        imp = MissingValueImputer("zero")
        result = imp.impute(series_with_nans)
        assert not result.isna().any()
        # Check that NaN positions are now 0
        original_nans = series_with_nans.isna()
        assert np.allclose(result[original_nans].values, 0.0)

    def test_max_gap_respected(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        values = np.ones(50)
        values[10:25] = np.nan  # 15 consecutive NaNs
        s = pd.Series(values, index=idx)
        imp = MissingValueImputer("forward_fill", max_gap=5)
        result = imp.impute(s)
        assert not result.isna().any()  # Remaining should be filled with median

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            MissingValueImputer("invalid").impute(pd.Series([1.0, np.nan]))

    def test_repr(self):
        imp = MissingValueImputer("linear", max_gap=10)
        r = repr(imp)
        assert "linear" in r
        assert "10" in r
