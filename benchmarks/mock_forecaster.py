"""Mock forecaster shared by the test suite and the CI-safe mock benchmark.

This class deliberately lives outside ``tests/``: ``benchmarks/run_mock_benchmark.py``
is advertised in the README as running after a plain ``pip install -e .``, and the
test ``conftest`` imports pytest (a ``[dev]``-extra dependency). Importing the class
from ``tests/conftest.py`` made the advertised demo fail with ModuleNotFoundError on
any clean machine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from uniftsm.core.base import BaseForecaster


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
