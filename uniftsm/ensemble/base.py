"""Abstract base class for all ensemble strategies.

An ensemble takes a list of fitted :class:`BaseForecaster` models and
combines their predictions into a single (hopefully better) forecast.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from uniftsm.core.base import BaseForecaster
from uniftsm.core.exceptions import EnsembleError


class BaseEnsemble(ABC):
    """Abstract base for all ensemble strategies.

    Parameters
    ----------
    models:
        A list of fitted :class:`BaseForecaster` instances.  At least
        one model is required.
    name:
        Optional human-readable name for the ensemble.
    """

    def __init__(
        self,
        models: list[BaseForecaster],
        name: str | None = None,
    ) -> None:
        if not models:
            raise EnsembleError(
                "base",
                "At least one model is required to build an ensemble.",
            )
        self.models = models
        self.name = name or f"Ensemble({', '.join(m.model_name for m in models)})"

        # Ensure models are fitted
        for m in models:
            if not m.is_fitted:
                raise EnsembleError(
                    self.name,
                    f"Model '{m.model_name}' is not fitted. "
                    f"Call .fit() on all models before ensembling.",
                )

    @abstractmethod
    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Generate ensemble forecasts.

        Args:
            horizon: Forecast horizon.
            return_quantiles: If ``True``, return a dict of quantile
                DataFrames.
            quantiles: Sorted quantile levels.
            **kwargs: Forwarded to each model's ``predict`` method.

        Returns:
            Single DataFrame with ``"mean"`` and ``"std"`` columns, or
            a dict of quantile DataFrames.
        """

    def predict_interval(
        self,
        horizon: int,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prediction intervals via quantile ensemble.

        Args:
            horizon: Forecast horizon.
            alpha: Significance level (0.05 → 95% intervals).
            **kwargs: Forwarded to :meth:`predict`.

        Returns:
            A pair ``(lower, upper)`` of DataFrames.
        """
        result = self.predict(
            horizon=horizon,
            return_quantiles=True,
            quantiles=[alpha / 2, 1 - alpha / 2],
            **kwargs,
        )
        assert isinstance(result, dict)
        lower = result[f"q_{alpha / 2}"]
        upper = result[f"q_{1 - alpha / 2}"]
        return lower, upper

    @property
    def is_probabilistic(self) -> bool:
        """Whether all member models are probabilistic."""
        return all(m.is_probabilistic for m in self.models)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, n_models={len(self.models)})"
