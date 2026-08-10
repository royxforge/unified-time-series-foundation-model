"""Shared fixtures and helpers for the UniTSFM test suite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from uniftsm.core.base import BaseForecaster

# ── Shared test data ────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def simple_series(rng: np.random.Generator) -> pd.Series:
    """A simple 100-point random walk with datetime index."""
    steps = np.cumsum(rng.normal(0, 1, 100))
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(steps, index=idx, name="value")


@pytest.fixture
def constant_series() -> pd.Series:
    """A constant-valued series with datetime index."""
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    return pd.Series(np.ones(50), index=idx, name="value")


@pytest.fixture
def short_series() -> pd.Series:
    """Series with fewer than 10 observations (triggers InsufficientDataError)."""
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    return pd.Series(np.arange(5, dtype=float), index=idx, name="value")


@pytest.fixture
def multivariate_df(rng: np.random.Generator) -> pd.DataFrame:
    """A 50-point multivariate DataFrame with 3 columns."""
    steps = np.cumsum(rng.normal(0, 1, (50, 3)), axis=0)
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    return pd.DataFrame(steps, index=idx, columns=["a", "b", "c"])


@pytest.fixture
def hourly_series(rng: np.random.Generator) -> pd.Series:
    """Hourly series with 200 points."""
    steps = np.cumsum(rng.normal(0, 1, 200))
    idx = pd.date_range("2020-01-01", periods=200, freq="h")
    return pd.Series(steps, index=idx, name="value")


@pytest.fixture
def series_with_nans() -> pd.Series:
    """Series with ~10% missing values."""
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    values = np.sin(np.linspace(0, 4 * np.pi, 100))
    values[[10, 11, 12, 30, 31, 50, 51, 52, 53, 80]] = np.nan
    return pd.Series(values, index=idx, name="value")


# ── Mock forecaster for ensemble tests ─────────────────────────────────────


class MockForecaster(BaseForecaster):
    """Minimal forecaster with controllable output for testing."""

    def __init__(
        self,
        model_name: str = "mock",
        mean_values: np.ndarray | None = None,
        std_values: np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._mean_values = mean_values
        self._std_values = std_values
        self._loaded = False

    def _load_model(self) -> None:
        self._loaded = True

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> MockForecaster:
        self._validate_inputs(y)
        self._context = y
        self._fitted = True
        self._freq = freq
        self._original_fitted_length = len(y)
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        quantiles = quantiles or [0.1, 0.5, 0.9]

        if self._mean_values is not None and len(self._mean_values) >= horizon:
            mean = self._mean_values[:horizon]
        else:
            mean = np.zeros(horizon)

        if self._std_values is not None and len(self._std_values) >= horizon:
            std = self._std_values[:horizon]
        else:
            std = np.ones(horizon)

        forecast_index = pd.RangeIndex(horizon)

        if return_quantiles:
            from scipy.stats import norm

            result = {}
            for q in quantiles:
                z = norm.ppf(q)
                vals = mean + z * std
                result[f"q_{q}"] = pd.DataFrame(vals, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame(
            {"mean": mean, "std": std},
            index=forecast_index,
        )

    @property
    def is_probabilistic(self) -> bool:
        return True


@pytest.fixture
def mock_forecaster() -> MockForecaster:
    return MockForecaster(model_name="mock")


@pytest.fixture
def mock_forecasters() -> list[MockForecaster]:
    """Three mock forecasters with different mean/std profiles."""
    horizon = 20
    return [
        MockForecaster(
            model_name="model_a",
            mean_values=np.linspace(10, 12, horizon),
            std_values=np.linspace(0.5, 1.0, horizon),
        ),
        MockForecaster(
            model_name="model_b",
            mean_values=np.linspace(9, 13, horizon),
            std_values=np.linspace(0.8, 1.5, horizon),
        ),
        MockForecaster(
            model_name="model_c",
            mean_values=np.linspace(11, 11.5, horizon),
            std_values=np.linspace(0.3, 0.6, horizon),
        ),
    ]


# ── PyTorch fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def torch_device() -> torch.device:
    return torch.device("cpu")
