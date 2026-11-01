"""Shared fixtures and helpers for the UniTSFM test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from benchmarks.mock_forecaster import MockForecaster

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


# ── Mock forecaster fixtures (class: benchmarks/mock_forecaster.py) ─────────────────────────────────────


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
