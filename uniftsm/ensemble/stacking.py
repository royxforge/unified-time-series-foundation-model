"""Stacking ensemble — train a meta-model on top of base model predictions.

The meta-learner learns to combine the individual forecasts optimally,
potentially capturing non-linear interactions between model outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from uniftsm.core.exceptions import EnsembleError
from uniftsm.ensemble.base import BaseEnsemble


class StackingEnsemble(BaseEnsemble):
    """Stacking ensemble with a trainable meta-learner.

    The meta-learner is a regression model that takes the predictions
    of all base models as features and produces the final forecast.

    Parameters
    ----------
    models:
        List of fitted :class:`BaseForecaster` instances.
    meta_model:
        A scikit-learn regressor.  If ``None``, uses
        :class:`~sklearn.linear_model.LinearRegression`.
    name:
        Optional ensemble name.
    """

    def __init__(
        self,
        models: list,
        meta_model: Any = None,
        name: str | None = None,
    ) -> None:
        super().__init__(models, name=name)
        self._meta_model = meta_model or LinearRegression()
        self._scaler = StandardScaler()
        self._is_trained = False

    def train(
        self,
        y_train: pd.Series | pd.DataFrame,
        horizon: int,
        **kwargs: Any,
    ) -> StackingEnsemble:
        """Train the meta-learner on a validation set.

        For each model, generate forecasts on the training series, then
        use those forecasts as features to predict the true values.

        Args:
            y_train: The actual (held-out) target values for the training
                window.
            horizon: Forecast horizon used during training.
            **kwargs: Forwarded to each model's ``predict`` method.

        Returns:
            ``self`` for method chaining.
        """
        n_models = len(self.models)

        # Collect predictions from each model
        used_len = min(len(y_train), horizon)
        X_stack = np.zeros((used_len, n_models))
        for i, model in enumerate(self.models):
            pred = model.predict(horizon, return_quantiles=False, **kwargs)
            X_stack[:, i] = pred["mean"].values[:used_len]

        # Scale features
        X_scaled = self._scaler.fit_transform(X_stack)

        # Fit meta-model on matching number of targets
        if isinstance(y_train, pd.Series):
            y_vals = y_train.iloc[:used_len].values
        else:
            y_vals = y_train.values[:used_len].ravel()
        self._meta_model.fit(X_scaled, y_vals)
        self._is_trained = True
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._is_trained:
            raise EnsembleError(
                self.name,
                "Stacking ensemble has not been trained. "
                "Call .train() first with a validation set.",
            )

        quantiles = quantiles or [0.1, 0.5, 0.9]
        n_models = len(self.models)

        # Collect predictions
        X_stack = np.zeros((horizon, n_models))
        for i, model in enumerate(self.models):
            pred = model.predict(horizon, return_quantiles=False, **kwargs)
            X_stack[:, i] = pred["mean"].values

        X_scaled = self._scaler.transform(X_stack)
        ensemble_mean = self._meta_model.predict(X_scaled)

        # Estimate std from residuals of meta-model predictions
        residuals = []
        for model in self.models:
            pred = model.predict(horizon, return_quantiles=False, **kwargs)
            residuals.append(pred["mean"].values - ensemble_mean)
        ensemble_std = np.std(residuals, axis=0)

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
