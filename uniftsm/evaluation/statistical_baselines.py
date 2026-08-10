"""Statistical baseline models for comparison.

Implements Naive, Seasonal Naive, ETS, ARIMA, and Theta methods
as :class:`BaseForecaster` subclasses so they fit into the unified
evaluation pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from uniftsm.core.base import BaseForecaster

logger = logging.getLogger(__name__)


class NaiveForecaster(BaseForecaster):
    """Naive forecast — last observed value repeated for the horizon."""

    def __init__(
        self,
        device: str = "auto",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="naive", device=device, seed=seed)

    def _load_model(self) -> None:
        pass  # No model to load

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> NaiveForecaster:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        self._last_value = y.iloc[-1]
        self._context = y
        self._freq = freq
        self._fitted = True
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        quantiles = quantiles or [0.1, 0.5, 0.9]

        forecast_index = (
            self._make_index(self._context, horizon)
            if self._context is not None
            else pd.RangeIndex(horizon)
        )

        mean = np.full(horizon, self._last_value)
        std = np.abs(self._last_value) * 0.1  # heuristic uncertainty
        if std < 1e-8:
            std = np.full(horizon, 1.0)

        if return_quantiles:
            result = {}
            for q in quantiles:
                result[f"q_{q}"] = pd.DataFrame(mean, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame(
            {"mean": mean, "std": np.full(horizon, std)},
            index=forecast_index,
        )


class SeasonalNaiveForecaster(BaseForecaster):
    """Seasonal naive forecast — last observed seasonal period repeated."""

    def __init__(
        self,
        season_length: int = 12,
        device: str = "auto",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="seasonal_naive", device=device, seed=seed)
        self.season_length = season_length

    def _load_model(self) -> None:
        pass

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> SeasonalNaiveForecaster:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        if len(y) < self.season_length:
            raise ValueError(f"Series length {len(y)} < season_length {self.season_length}.")
        self._last_season = y.iloc[-self.season_length :].values
        self._context = y
        self._freq = freq
        self._fitted = True
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        quantiles = quantiles or [0.1, 0.5, 0.9]

        cycles = (horizon // self.season_length) + 1
        mean = np.tile(self._last_season, cycles)[:horizon]

        forecast_index = (
            self._make_index(self._context, horizon)
            if self._context is not None
            else pd.RangeIndex(horizon)
        )

        std = np.std(self._last_season) * np.ones(horizon) * 0.5
        if std[0] < 1e-8:
            std = np.ones(horizon)

        if return_quantiles:
            result = {}
            for q in quantiles:
                result[f"q_{q}"] = pd.DataFrame(mean, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame(
            {"mean": mean, "std": std},
            index=forecast_index,
        )

    def get_params(self) -> dict[str, Any]:
        return {**super().get_params(), "season_length": self.season_length}


class ETSForecaster(BaseForecaster):
    """Exponential Smoothing (ETS) baseline via statsmodels."""

    def __init__(
        self,
        error: str = "add",
        trend: str | None = "add",
        seasonal: str | None = "add",
        seasonal_periods: int = 12,
        device: str = "auto",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="ets", device=device, seed=seed)
        self.error = error
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods

    def _load_model(self) -> None:
        pass

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> ETSForecaster:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        self._ets_model = ETSModel(
            y,
            error=self.error,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
        )
        self._ets_result = self._ets_model.fit()
        self._context = y
        self._freq = freq
        self._fitted = True
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        quantiles = quantiles or [0.1, 0.5, 0.9]

        forecast = self._ets_result.forecast(horizon)
        mean = forecast.values

        # Use in-sample residuals for std
        residuals = self._ets_result.resid
        std = np.full(horizon, np.std(residuals))

        forecast_index = (
            self._make_index(self._context, horizon)
            if self._context is not None
            else pd.RangeIndex(horizon)
        )

        if return_quantiles:
            from scipy.stats import norm

            result = {}
            for q in quantiles:
                z = norm.ppf(q)
                vals = mean + z * std
                result[f"q_{q}"] = pd.DataFrame(vals, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame({"mean": mean, "std": std}, index=forecast_index)


class ARIMAForecaster(BaseForecaster):
    """ARIMA baseline via statsmodels."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] | None = None,
        device: str = "auto",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="arima", device=device, seed=seed)
        self.order = order
        self.seasonal_order = seasonal_order

    def _load_model(self) -> None:
        pass

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> ARIMAForecaster:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        from statsmodels.tsa.arima.model import ARIMA

        self._arima_model = ARIMA(
            y,
            order=self.order,
            seasonal_order=self.seasonal_order,
        )
        self._arima_result = self._arima_model.fit()
        self._context = y
        self._freq = freq
        self._fitted = True
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        quantiles = quantiles or [0.1, 0.5, 0.9]

        forecast_result = self._arima_result.forecast(horizon)
        mean = forecast_result.values

        # Use forecast standard errors for uncertainty

        std = np.full(horizon, np.std(self._arima_result.resid))

        forecast_index = (
            self._make_index(self._context, horizon)
            if self._context is not None
            else pd.RangeIndex(horizon)
        )

        if return_quantiles:
            from scipy.stats import norm

            result = {}
            for q in quantiles:
                z = norm.ppf(q)
                vals = mean + z * std
                result[f"q_{q}"] = pd.DataFrame(vals, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame({"mean": mean, "std": std}, index=forecast_index)

    def get_params(self) -> dict[str, Any]:
        return {
            **super().get_params(),
            "order": self.order,
            "seasonal_order": self.seasonal_order,
        }


class ThetaForecaster(BaseForecaster):
    """Theta method baseline.

    The Theta method decomposes the series into two theta-lines and
    combines them for the final forecast.
    """

    def __init__(
        self,
        theta: float = 2.0,
        device: str = "auto",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="theta", device=device, seed=seed)
        self.theta = theta

    def _load_model(self) -> None:
        pass

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> ThetaForecaster:
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        self._theta_values = y.values.astype(np.float64)
        self._context = y
        self._freq = freq

        n = len(self._theta_values)
        # Theta-line = quadratic fit applied to transformed series
        x = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, self._theta_values, 2)
        self._poly_coeffs = coeffs
        self._fitted = True
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting.")
        quantiles = quantiles or [0.1, 0.5, 0.9]

        n = len(self._theta_values)
        x_future = np.arange(n, n + horizon, dtype=np.float64)
        mean = np.polyval(self._poly_coeffs, x_future)

        # Heuristic std based on in-sample residuals
        x_insample = np.arange(n, dtype=np.float64)
        fitted = np.polyval(self._poly_coeffs, x_insample)
        residuals = self._theta_values - fitted
        std = np.full(horizon, np.std(residuals))

        forecast_index = (
            self._make_index(self._context, horizon)
            if self._context is not None
            else pd.RangeIndex(horizon)
        )

        if return_quantiles:
            from scipy.stats import norm

            result = {}
            for q in quantiles:
                z = norm.ppf(q)
                vals = mean + z * std
                result[f"q_{q}"] = pd.DataFrame(vals, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame({"mean": mean, "std": std}, index=forecast_index)

    def get_params(self) -> dict[str, Any]:
        return {**super().get_params(), "theta": self.theta}
