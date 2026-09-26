"""Stacking ensemble — train a meta-model on top of base model predictions.

The meta-learner learns to combine the individual forecasts optimally,
potentially capturing non-linear interactions between model outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
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
        # Ridge (regularised) with a small default alpha: plain OLS overfits
        # the tiny backtest design matrix when models >> holdout points.
        self._meta_model = meta_model or Ridge(alpha=1.0)
        self._scaler = StandardScaler()
        self._is_trained = False

    def train(
        self,
        y_train: pd.Series | pd.DataFrame,
        horizon: int,
        **kwargs: Any,
    ) -> StackingEnsemble:
        """Train the meta-learner on a holdout window (no leakage).

        Splits ``y_train`` into fit ``[:-horizon]`` / holdout ``[-horizon:]``:
        each member forecasts the holdout from history only, then the
        meta-learner maps those out-of-sample forecasts to the realised
        holdout values.

        The previous implementation trained member→target on the *same*
        window (fitting member forecasts of ``y_train`` to ``y_train``
        itself), which leaks future information into the meta-model.

        Requires ``len(y_train) >= 2 * horizon`` so the member-fit window is
        at least as long as the horizon it must forecast.

        Args:
            y_train: Full history. The last ``horizon`` points form the
                holdout that the meta-learner is trained on.
            horizon: Forecast horizon used during training.
            **kwargs: Forwarded to each model's ``predict`` method.

        Returns:
            ``self`` for method chaining.
        """
        n_models = len(self.models)

        if isinstance(y_train, pd.Series):
            history_vals = y_train.iloc[:-horizon]
            holdout_vals = y_train.iloc[-horizon:]
        else:
            history_vals = y_train.iloc[:-horizon] if hasattr(y_train, "iloc") else y_train[:-horizon]
            holdout_vals = y_train.iloc[-horizon:] if hasattr(y_train, "iloc") else y_train[-horizon:]

        if len(history_vals) < horizon:
            raise EnsembleError(
                self.name,
                f"Need at least 2*horizon={2 * horizon} history points for "
                f"leakage-free stacking training, got {len(y_train)}.",
            )

        # Refit each member on history-only, then forecast the holdout window.
        # Members that fail are dropped for this training round.
        y_holdout = np.asarray(holdout_vals).ravel()
        used_len = len(y_holdout)
        X_stack = np.zeros((used_len, n_models))
        active: list[int] = []
        for i, model in enumerate(self.models):
            try:
                fitted = model.fit(history_vals)
                pred = fitted.predict(horizon, return_quantiles=False, **kwargs)
                X_stack[:, i] = pred["mean"].values[:used_len]
                active.append(i)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning(
                    "Stacking member '%s' failed during training and was skipped: %s",
                    getattr(model, "model_name", type(model).__name__),
                    exc,
                )

        if not active:
            raise EnsembleError(
                self.name,
                "No ensemble member could produce a training forecast.",
            )

        # Scale features
        X_scaled = self._scaler.fit_transform(X_stack)

        # Fit meta-model on the realised holdout values
        self._meta_model.fit(X_scaled, y_holdout)
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
