"""Convenience class that provides the ``model.fit(X) → model.predict(h)`` API.

This is the primary user-facing entry point, wrapping auto-selection,
ensembling, and forecasting into a single interface.

Example::

    from uniftsm import UniTSFM
    import pandas as pd

    model = UniTSFM(auto_select=True, ensemble=True)
    model.fit(electricity["load"])
    forecast = model.predict(horizon=168, return_quantiles=True)
    lower, upper = model.predict_interval(horizon=168)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from uniftsm.core.base import BaseForecaster
from uniftsm.core.config import config
from uniftsm.core.registry import registry
from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble
from uniftsm.selection.meta_features import MetaFeatureExtractor

logger = logging.getLogger(__name__)


class UniTSFM:
    """Unified interface for time series forecasting with TSFMs.

    Wraps auto-selection, ensembling, and prediction into a single
    user-facing object.  This is the class advertised in the README
    as the primary way to use the library.

    Parameters
    ----------
    model_name:
        Specific model to use (e.g. ``"chronos"``).  Overridden by
        ``auto_select``.
    auto_select:
        If ``True``, use the meta-feature extractor to automatically
        choose the best model for the data.
    ensemble:
        If ``True``, use the uncertainty-weighted ensemble of all
        available models.
    ensemble_temperature:
        Temperature for the uncertainty-weighted ensemble
        (default from global config).
    **model_kwargs:
        Additional keyword arguments passed to the model constructor(s).
    """

    def __init__(
        self,
        model_name: str | None = None,
        auto_select: bool = False,
        ensemble: bool = False,
        ensemble_temperature: float | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.auto_select = auto_select
        self.ensemble = ensemble
        self.ensemble_temperature = ensemble_temperature or config.ensemble_temperature
        self._model_kwargs = model_kwargs
        self._forecaster: BaseForecaster | UncertaintyWeightedEnsemble | None = None
        self._fitted = False

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> UniTSFM:
        """Fit the model(s) to the provided time series.

        Args:
            y: Target time series.
            X: Optional exogenous features.
            freq: Pandas frequency alias.  Auto-detected if not provided.
            **kwargs: Additional fit arguments.

        Returns:
            ``self`` for method chaining.
        """
        if self.ensemble:
            self._forecaster = self._build_ensemble(y, freq, **kwargs)
        elif self.auto_select:
            self._forecaster = self._build_auto_selected(y, freq, **kwargs)
        elif self.model_name:
            model_cls = registry.get(self.model_name)
            forecaster = model_cls(**self._model_kwargs)
            forecaster.fit(y, X=X, freq=freq, **kwargs)
            self._forecaster = forecaster
        else:
            # Default: use chronos
            model_cls = registry.get("chronos")
            forecaster = model_cls(**self._model_kwargs)
            forecaster.fit(y, X=X, freq=freq, **kwargs)
            self._forecaster = forecaster

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
        """Generate forecasts.

        Args:
            horizon: Number of steps to forecast.
            return_quantiles: If ``True``, return a dict of quantile
                DataFrames.
            quantiles: Sorted quantile levels.
            X_future: Future exogenous features.
            **kwargs: Additional predict arguments.

        Returns:
            Forecast DataFrame or dict of quantile DataFrames.
        """
        if not self._fitted or self._forecaster is None:
            raise RuntimeError("UniTSFM must be fitted before predicting. Call .fit() first.")
        return self._forecaster.predict(
            horizon=horizon,
            return_quantiles=return_quantiles,
            quantiles=quantiles,
            X_future=X_future,
            **kwargs,
        )

    def predict_interval(
        self,
        horizon: int,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prediction intervals.

        Args:
            horizon: Forecast horizon.
            alpha: Significance level (0.05 → 95% intervals).
            **kwargs: Additional predict arguments.

        Returns:
            ``(lower, upper)`` DataFrames.
        """
        if self._forecaster is None:
            raise RuntimeError("UniTSFM must be fitted before predicting.")
        return self._forecaster.predict_interval(horizon, alpha=alpha, **kwargs)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_ensemble(
        self,
        y: pd.Series | pd.DataFrame,
        freq: str | None = None,
        **kwargs: Any,
    ) -> UncertaintyWeightedEnsemble:
        """Build an uncertainty-weighted ensemble of all available models."""
        models = []
        for model_name in registry.list_models():
            try:
                model_cls = registry.get(model_name)
                m = model_cls(**self._model_kwargs)
                m.fit(y, freq=freq, **kwargs)
                models.append(m)
                logger.debug("  Added model '%s' to ensemble.", model_name)
            except Exception as exc:
                logger.warning("  Could not add model '%s': %s", model_name, exc)

        if not models:
            raise RuntimeError(
                "No models could be loaded for ensembling. "
                "Check that at least one TSFM dependency is installed."
            )
        return UncertaintyWeightedEnsemble(
            models,
            temperature=self.ensemble_temperature,
        )

    def _build_auto_selected(
        self,
        y: pd.Series | pd.DataFrame,
        freq: str | None = None,
        **kwargs: Any,
    ) -> BaseForecaster:
        """Select the best model using meta-features and fit it."""
        y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else y

        features = MetaFeatureExtractor.extract(y_series)
        suggestions = registry.suggest(
            series_length=int(features["length"]),
            frequency=str(features.get("frequency", 0)),
            is_multivariate=isinstance(y, pd.DataFrame) and y.shape[1] > 1,
            needs_uncertainty=True,
            top_k=1,
        )
        if not suggestions:
            raise RuntimeError("No suitable model could be auto-selected.")

        best_name = suggestions[0][0]
        logger.info("Auto-selected model: '%s' (score=%.3f)", best_name, suggestions[0][1])
        model_cls = registry.get(best_name)
        forecaster = model_cls(**self._model_kwargs)
        forecaster.fit(y, freq=freq, **kwargs)
        return forecaster

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def is_probabilistic(self) -> bool:
        """Whether the underlying forecaster outputs distributions."""
        if self._forecaster is not None:
            return self._forecaster.is_probabilistic
        return False

    def get_params(self) -> dict[str, Any]:
        """Return configuration for reproducibility."""
        return {
            "model_name": self.model_name,
            "auto_select": self.auto_select,
            "ensemble": self.ensemble,
            "ensemble_temperature": self.ensemble_temperature,
            "model_kwargs": self._model_kwargs,
        }

    @property
    def forecaster(self) -> BaseForecaster | UncertaintyWeightedEnsemble | None:
        """The underlying fitted forecaster (for advanced use)."""
        return self._forecaster

    def __repr__(self) -> str:
        parts = [f"UniTSFM(fitted={self._fitted}"]
        if self.ensemble:
            parts.append("ensemble=True")
        if self.auto_select:
            parts.append("auto_select=True")
        if self.model_name:
            parts.append(f"model='{self.model_name}'")
        return ", ".join(parts) + ")"
