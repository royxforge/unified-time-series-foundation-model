"""Tests for the BaseForecaster abstract class."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.core.base import BaseForecaster
from uniftsm.core.exceptions import InsufficientDataError


class SimpleForecaster(BaseForecaster):
    """Minimal concrete implementation for testing."""

    def __init__(self, **kwargs):
        super().__init__(model_name="test", **kwargs)
        self.loaded = False

    def _load_model(self):
        self.loaded = True

    def fit(self, y, X=None, freq=None, **kwargs):
        self._validate_inputs(y)
        self._context = y
        self._fitted = True
        return self

    def predict(self, horizon, return_quantiles=False, quantiles=None, **kwargs):
        quantiles = quantiles or [0.1, 0.5, 0.9]
        if return_quantiles:
            return {f"q_{q}": pd.DataFrame({"forecast": [0.0] * horizon}) for q in quantiles}
        return pd.DataFrame({"mean": [0.0] * horizon, "std": [1.0] * horizon})


class TestBaseForecaster:
    def test_init_defaults(self):
        f = SimpleForecaster()
        assert f.model_name == "test"
        assert not f.is_fitted
        assert not f.is_probabilistic

    def test_fit_and_predict(self):
        y = pd.Series(np.random.randn(100))
        f = SimpleForecaster()
        f.fit(y)
        assert f.is_fitted
        pred = f.predict(horizon=10)
        assert "mean" in pred.columns
        assert "std" in pred.columns
        assert len(pred) == 10

    def test_predict_interval(self):
        y = pd.Series(np.random.randn(100))
        f = SimpleForecaster()
        f.fit(y)
        lower, upper = f.predict_interval(horizon=10, alpha=0.05)
        assert len(lower) == 10
        assert len(upper) == 10

    def test_insufficient_data(self):
        y = pd.Series([1.0, 2.0, 3.0])
        f = SimpleForecaster()
        with pytest.raises(InsufficientDataError):
            f.fit(y)

    def test_get_params(self):
        f = SimpleForecaster(seed=42)
        params = f.get_params()
        assert params["seed"] == 42
        assert params["model_name"] == "test"

    def test_repr(self):
        f = SimpleForecaster()
        assert "SimpleForecaster" in repr(f)

    def test_make_index_with_multiplier_freq(self):
        # Regression: multiplier aliases like "10T" previously crashed
        # ``pd.Timedelta(1, unit="10T")`` inside ``_make_index``.
        idx = pd.date_range("2020-01-01", periods=50, freq="10min")
        y = pd.Series(np.arange(50.0), index=idx)
        f = SimpleForecaster()
        f._freq = "10T"
        out = f._make_index(y, horizon=5)
        assert isinstance(out, pd.DatetimeIndex)
        assert len(out) == 5
        diffs = out[1:] - out[:-1]
        assert (diffs == pd.Timedelta(minutes=10)).all()

    def test_make_index_with_anchored_freq(self):
        idx = pd.date_range("2020-01-05", periods=12, freq="W-SUN")
        y = pd.Series(np.arange(12.0), index=idx)
        f = SimpleForecaster()
        f._freq = "W-SUN"
        out = f._make_index(y, horizon=3)
        assert isinstance(out, pd.DatetimeIndex)
        assert len(out) == 3
        assert out[0] == y.index[-1] + pd.Timedelta(weeks=1)

    def test_make_index_no_freq_falls_back(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        y = pd.Series(np.arange(10.0), index=idx)
        f = SimpleForecaster()
        f._freq = None
        out = f._make_index(y, horizon=4)
        assert len(out) == 4
