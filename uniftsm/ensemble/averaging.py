"""Simple and weighted average ensemble strategies.

Combines forecasts by taking a (possibly weighted) average of the
individual model predictions.  The simplest and most robust ensemble
method.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from uniftsm.core.exceptions import EnsembleError
from uniftsm.ensemble.base import BaseEnsemble


class AverageEnsemble(BaseEnsemble):
    """Simple uniform average across all models.

    Each model contributes equally to the final forecast.
    """

    def __init__(self, models: list, name: str | None = None) -> None:
        super().__init__(models, name=name)

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        quantiles = quantiles or [0.1, 0.5, 0.9]

        # Collect individual predictions
        all_means = []
        all_stds = []
        for model in self.models:
            pred = model.predict(horizon, return_quantiles=False, **kwargs)
            all_means.append(pred["mean"].values)
            all_stds.append(pred["std"].values)

        # Uniform average.  Between-model variance needs ddof=0 when there is
        # only a single member (otherwise numpy warns on ddof >= sample size).
        n_models = len(self.models)
        ddof = 1 if n_models > 1 else 0
        ensemble_mean = np.mean(all_means, axis=0)
        ensemble_std = np.sqrt(
            np.mean(np.square(all_stds), axis=0) + np.var(all_means, axis=0, ddof=ddof)
        )

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


class WeightedAverageEnsemble(BaseEnsemble):
    """Weighted average ensemble with user-specified weights.

    Parameters
    ----------
    models:
        List of fitted forecaster instances.
    weights:
        Optional list of weights (same length as ``models``).  If
        ``None``, uniform weights are used.  Weights are normalised
        to sum to 1.
    name:
        Optional ensemble name.
    """

    def __init__(
        self,
        models: list,
        weights: list[float] | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(models, name=name)
        if weights is not None:
            if len(weights) != len(models):
                raise EnsembleError(
                    self.name,
                    f"Number of weights ({len(weights)}) must match "
                    f"number of models ({len(models)}).",
                )
            w = np.array(weights, dtype=np.float64)
            self._weights = w / w.sum()
        else:
            self._weights = np.ones(len(models)) / len(models)

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        quantiles = quantiles or [0.1, 0.5, 0.9]

        all_means = []
        all_stds = []
        for model in self.models:
            pred = model.predict(horizon, return_quantiles=False, **kwargs)
            all_means.append(pred["mean"].values)
            all_stds.append(pred["std"].values)

        all_means_arr = np.array(all_means)
        all_stds_arr = np.array(all_stds)

        # Weighted average
        ensemble_mean = np.sum(self._weights[:, None] * all_means_arr, axis=0)
        within_var = np.sum(self._weights[:, None] ** 2 * all_stds_arr**2, axis=0)
        between_var = np.sum(
            self._weights[:, None] * (all_means_arr - ensemble_mean[None, :]) ** 2,
            axis=0,
        )
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
