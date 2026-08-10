"""Abstract base class for all TSFM wrappers.

Every model registered in :class:`ModelRegistry` MUST implement this
interface.  It defines the contract that makes the unified API possible.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
import torch

from uniftsm.core.exceptions import ModelNotLoadedError

logger = logging.getLogger(__name__)


class BaseForecaster(ABC):
    """Abstract base for all TSFM wrappers.

    Subclasses **must** override :meth:`_load_model`, :meth:`fit`,
    and :meth:`predict`.  Optionally override :meth:`predict_interval`
    and :meth:`_validate_inputs`.

    Parameters
    ----------
    model_name:
        Human-readable identifier used in messages and the registry.
    device:
        Torch device string.  ``"auto"`` resolves to CUDA if available.
    seed:
        Random seed for reproducibility.
    **kwargs:
        Model-specific hyperparameters forwarded to the subclass.
    """

    def __init__(
        self,
        model_name: str = "",
        device: str = "auto",
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        # ``model_name`` defaults to "" because concrete subclasses always
        # pass it to ``super().__init__``; the default only exists so that
        # registry-based construction (``registry.get(name)(**kwargs)``)
        # type-checks against this abstract signature.
        self.model_name = model_name

        resolved_device = self._resolve_device(device)
        self.device = torch.device(resolved_device)

        self.seed = seed
        self._set_seed()

        # Storage for data prepared during fit()
        self._context: pd.Series | pd.DataFrame | None = None
        self._freq: str | None = None
        self._fitted: bool = False

        # Model weights — populated by _load_model() / fit()
        self._model: Any = None

    # ── Abstract interface ───────────────────────────────────────────────

    @abstractmethod
    def _load_model(self) -> None:
        """Download or initialise model weights on the configured device.

        Called lazily on the first call to :meth:`predict` (or explicitly
        by the user via :meth:`fit`).
        """

    @abstractmethod
    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> BaseForecaster:
        """Prepare the model for prediction on the provided series.

        For most zero-shot TSFMs this stores the scaled context and is
        effectively a no-op.  Fine-tunable models should trigger training
        here.

        Args:
            y: Target time series.  A ``Series`` for univariate or a
               ``DataFrame`` for multivariate data.
            X: Exogenous features (not yet widely supported by TSFMs).
            freq: Pandas frequency alias (e.g. ``"H"``, ``"D"``, ``"M"``).
                  Auto-detected from the index if not provided.
            **kwargs: Model-specific fit arguments.

        Returns:
            ``self`` for method chaining.
        """

    @abstractmethod
    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Generate forecasts for the given horizon.

        Args:
            horizon: Number of time steps to forecast ahead.
            return_quantiles: If ``True``, return a dict of DataFrames, one
                per requested quantile.  Otherwise return a single DataFrame
                with columns ``"mean"`` and ``"std"``.
            quantiles: Sorted list of quantile levels in ``[0, 1]``.
                Ignored when ``return_quantiles=False``.
            X_future: Future exogenous features.
            **kwargs: Model-specific prediction arguments.

        Returns:
            Single DataFrame with ``"mean"`` / ``"std"`` columns, **or**
            a dict mapping quantile strings (e.g. ``"q_0.1"``) to DataFrames.
        """

    # ── Concrete helpers ─────────────────────────────────────────────────

    def predict_interval(
        self,
        horizon: int,
        alpha: float = 0.05,
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prediction intervals via conformal or parametric methods.

        The default implementation uses quantile predictions at
        ``alpha/2`` and ``1 - alpha/2``.  Subclasses may override with
        a more efficient method.

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

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_probabilistic(self) -> bool:
        """Whether this model outputs distributions natively.

        Override in subclasses that produce probabilistic forecasts
        (e.g. Chronos via sampling, Lag-Llama via distribution heads).
        """
        return False

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called at least once."""
        return self._fitted

    # ── Serialisation / reproducibility ──────────────────────────────────

    def get_params(self) -> dict[str, Any]:
        """Return model configuration for reproducibility.

        Subclasses should extend this with their own hyperparameters.
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "seed": self.seed,
        }

    def set_seed(self, seed: int) -> None:
        """Override the random seed after construction."""
        self.seed = seed
        self._set_seed()

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve 'auto' to a concrete device string."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def _set_seed(self) -> None:
        """Set random seeds for reproducibility."""
        import random

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def _validate_inputs(
        self,
        y: pd.Series | pd.DataFrame,
        horizon: int | None = None,
    ) -> None:
        """Validate input types and dimensions.

        Raises
        ------
        InsufficientDataError
            If the series is too short.
        """
        from uniftsm.core.exceptions import InsufficientDataError

        length = len(y)
        if length < 10:
            raise InsufficientDataError(
                required=10,
                actual=length,
                model_name=self.model_name,
            )

    def _ensure_model_loaded(self) -> None:
        """Ensure model weights are loaded, raising if not."""
        if self._model is None:
            self._load_model()
        if self._model is None:
            raise ModelNotLoadedError(self.model_name)

    def _make_index(
        self,
        y: pd.Series,
        horizon: int,
    ) -> pd.DatetimeIndex:
        """Build a DatetimeIndex for the forecast horizon.

        Uses ``self._freq`` inferred from the training data.  Frequencies
        are handled via ``pd.tseries.frequencies.to_offset`` so that
        multiplier aliases (``"10T"``), anchored offsets (``"W-SUN"``)
        and legacy aliases (``"min"``, ``"H"``) all work.
        """
        freq = self._freq or pd.infer_freq(y.index)
        last_date = y.index[-1]
        if freq is None:
            return pd.date_range(start=last_date, periods=horizon, freq="D")
        offset = pd.tseries.frequencies.to_offset(freq)
        return pd.date_range(
            start=last_date + offset,
            periods=horizon,
            freq=offset,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device='{self.device}', "
            f"fitted={self._fitted})"
        )
