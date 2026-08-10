"""Cross-Model Uncertainty-Weighted Ensemble (CMUE).

This module implements **Novel Contribution #2**: a method that weights
models by their prediction uncertainty (inverse variance), calibrated
across qualitatively different TSFM architectures.

Intuition: If a model is uncertain about a forecast (high variance),
it should contribute less.  If it is confident (low variance), it
contributes more.  Because TSFMs produce uncertainty differently
(e.g., Chronos uses sampling, Lag-Llama uses distributional heads,
TimesFM produces parametric std), this method normalises those
heterogeneous signals into a common weighting scheme.

Members whose backend is not installed (e.g. in a partial install) are
skipped with a warning rather than failing the whole ensemble.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from uniftsm.core.exceptions import EnsembleError
from uniftsm.ensemble.base import BaseEnsemble

logger = logging.getLogger(__name__)


class UncertaintyWeightedEnsemble(BaseEnsemble):
    """Weight models by inverse prediction uncertainty.

    Uncertainty is measured as the inter-quartile range (IQR) from
    each model's quantile predictions.  IQR is a robust, scale-invariant
    measure that works across qualitatively different TSFM architectures.

    Parameters
    ----------
    models:
        List of fitted :class:`BaseForecaster` instances.
    temperature:
        Controls the sharpness of the weight distribution.
        ``temperature=0`` → uniform weights.
        ``temperature=1`` → proportional to inverse variance.
        ``temperature>1`` → winner-takes-all.
    name:
        Optional ensemble name.
    """

    def __init__(
        self,
        models: list,
        temperature: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(models, name=name)
        if temperature < 0:
            raise ValueError("Temperature must be non-negative.")
        self.temperature = temperature

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        quantiles = quantiles or [0.1, 0.5, 0.9]

        # Collect predictions and uncertainty estimates from members that
        # can actually run (skips unavailable backends with a warning).
        predictions = []
        variances = []
        for model in self.models:
            try:
                pred = model.predict(
                    horizon,
                    return_quantiles=True,
                    quantiles=[0.25, 0.5, 0.75],
                    **kwargs,
                )
                q25 = pred["q_0.25"].values.squeeze()
                q50 = pred["q_0.5"].values.squeeze()
                q75 = pred["q_0.75"].values.squeeze()

                predictions.append(q50)
                # IQR as robust uncertainty proxy
                iqr = q75 - q25
                variances.append(np.clip(iqr, 1e-8, None) ** 2)
            except Exception as exc:
                logger.warning(
                    "Ensemble member '%s' failed to predict and was skipped: %s",
                    model.model_name,
                    exc,
                )

        if not predictions:
            raise EnsembleError(
                self.name,
                "No ensemble member could produce a forecast.",
            )

        predictions_arr = np.asarray(predictions, dtype=np.float64)
        variances_arr = np.asarray(variances, dtype=np.float64)

        # Inverse variance weighting with temperature
        inv_var = 1.0 / variances_arr
        weights = inv_var**self.temperature
        weights = weights / (weights.sum(axis=0, keepdims=True) + 1e-8)

        # Weighted average
        ensemble_mean = (weights * predictions_arr).sum(axis=0)

        # Combined uncertainty (law of total variance)
        within_var = (weights * variances).sum(axis=0)
        between_var = (weights * (predictions - ensemble_mean[None, :]) ** 2).sum(axis=0)
        ensemble_std = np.sqrt(within_var + between_var)

        forecast_index = pd.RangeIndex(horizon)

        if return_quantiles:
            from scipy.stats import norm

            result = {}
            for q in quantiles:
                z = norm.ppf(q)
                vals = ensemble_mean + z * ensemble_std
                result[f"q_{q}"] = pd.DataFrame(vals, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame(
            {"mean": ensemble_mean, "std": ensemble_std},
            index=forecast_index,
        )

    def get_weights(self, horizon: int, **kwargs: Any) -> pd.DataFrame:
        """Return the weight assigned to each model at each step.

        Useful for interpretability: which models dominate which forecast
        horizons?

        Args:
            horizon: Forecast horizon.
            **kwargs: Forwarded to each model's ``predict`` method.

        Returns:
            DataFrame with shape ``(horizon, n_models)``.
        """
        variances = []
        active_models = []
        for model in self.models:
            try:
                pred = model.predict(
                    horizon,
                    return_quantiles=True,
                    quantiles=[0.25, 0.5, 0.75],
                    **kwargs,
                )
                q25 = pred["q_0.25"].values.squeeze()
                q75 = pred["q_0.75"].values.squeeze()
                iqr = q75 - q25
                variances.append(np.clip(iqr, 1e-8, None) ** 2)
                active_models.append(model.model_name)
            except Exception as exc:
                logger.warning(
                    "Ensemble member '%s' failed to predict and was skipped: %s",
                    model.model_name,
                    exc,
                )

        if not variances:
            raise EnsembleError(
                self.name,
                "No ensemble member could produce a forecast.",
            )

        variances_arr = np.asarray(variances, dtype=np.float64)
        inv_var = 1.0 / variances_arr
        weights = inv_var**self.temperature
        weights = weights / (weights.sum(axis=0, keepdims=True) + 1e-8)

        return pd.DataFrame(weights.T, columns=active_models)
