"""Conformal prediction sets for uncertainty quantification.

Conformal prediction provides distribution-free prediction intervals
with finite-sample coverage guarantees.  This module implements
split conformal prediction for TSFM ensemble forecasts.

Reference: Angelopoulos & Bates 2021, https://arxiv.org/abs/2107.07511
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from uniftsm.core.base import BaseForecaster
from uniftsm.core.exceptions import EnsembleError


class ConformalPrediction:
    """Split conformal prediction for a single :class:`BaseForecaster`.

    Parameters
    ----------
    model:
        A fitted :class:`BaseForecaster` instance.
    alpha:
        Significance level (0.05 → 95% prediction intervals).
    """

    def __init__(
        self,
        model: BaseForecaster,
        alpha: float = 0.05,
    ) -> None:
        if not model.is_fitted:
            raise EnsembleError(
                "ConformalPrediction",
                "Model must be fitted before building conformal sets.",
            )
        self.model = model
        self.alpha = alpha
        self._calibration_scores: np.ndarray | None = None
        self._calibrated = False

    def calibrate(
        self,
        y_holdout: pd.Series,
        horizon: int,
        **kwargs: Any,
    ) -> ConformalPrediction:
        """Calibrate prediction intervals using a holdout set.

        Args:
            y_holdout: The actual (held-out) target values.
            horizon: Forecast horizon used to generate predictions.
            **kwargs: Forwarded to the model's ``predict`` method.

        Returns:
            ``self`` for method chaining.
        """
        pred = self.model.predict(horizon, return_quantiles=False, **kwargs)
        y_true = y_holdout.values[:horizon]

        # Conformal scores = |y_true - y_pred| / std
        mean = pred["mean"].values[: len(y_true)]
        std = np.clip(pred["std"].values[: len(y_true)], 1e-8, None)
        scores = np.abs(y_true - mean) / std

        self._calibration_scores = scores
        self._calibrated = True
        return self

    def predict_interval(
        self,
        horizon: int,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Predict with conformalised intervals.

        Returns
        -------
        lower:
            DataFrame with conformal lower bound.
        upper:
            DataFrame with conformal upper bound.
        """
        if not self._calibrated or self._calibration_scores is None:
            raise EnsembleError(
                "ConformalPrediction",
                "Model must be calibrated before prediction. " "Call .calibrate() first.",
            )

        pred = self.model.predict(horizon, return_quantiles=False, **kwargs)
        assert isinstance(pred, pd.DataFrame)  # point forecast by construction
        mean = pred["mean"].values
        std = np.clip(pred["std"].values, 1e-8, None)

        # Conformal adjustment: quantile of calibration scores
        n = len(self._calibration_scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        conformal_quantile = np.quantile(self._calibration_scores, q)

        half_width = conformal_quantile * std

        index = pred.index
        lower = pd.DataFrame(mean - half_width, index=index, columns=["lower"])
        upper = pd.DataFrame(mean + half_width, index=index, columns=["upper"])
        return lower, upper
