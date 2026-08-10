"""Google TimesFM wrapper — patched-decoder foundation model.

TimesFM uses a patched-decoder architecture that operates on non-overlapping
patches of the input series.  It is pre-trained on ~100B time steps.

Reference: Das et al. 2024, https://arxiv.org/abs/2310.10688
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from uniftsm.core.base import BaseForecaster
from uniftsm.core.exceptions import PredictionError
from uniftsm.core.registry import registry
from uniftsm.pipeline.scaler import SeriesScaler

logger = logging.getLogger(__name__)


@registry.register(
    "timesfm",
    metadata={
        "family": "timesfm",
        "supports_multivariate": False,
        "supports_probabilistic": False,
        "min_length": 32,
        "max_context_length": 512,
        "pretrained_sizes": ["small", "medium", "large"],
        "paper": "A decoder-only foundation model for time-series forecasting",
        "paper_url": "https://arxiv.org/abs/2310.10688",
    },
)
class TimesFMForecaster(BaseForecaster):
    """Google TimesFM wrapper — patched-decoder forecasting.

    Parameters
    ----------
    model_size:
        One of ``"small"``, ``"medium"``, ``"large"``.
    device:
        Torch device string.
    seed:
        Random seed.
    context_length:
        Maximum number of time steps to use as context.  TimesFM
        typically uses 512.
    **kwargs:
        Additional keyword arguments for the TimesFM model.
    """

    # TimesFM 2.x ships a single open 200M checkpoint (torch + flax
    # variants).  The legacy 1.x repos (``google/timesfm-small`` etc.) no
    # longer exist, so all size aliases resolve to the 2.5 200M PyTorch
    # checkpoint.  The ``model_size`` argument is retained for API
    # compatibility and future releases.
    MODEL_SIZE_MAP: dict[str, str] = {
        "small": "google/timesfm-2.5-200m-pytorch",
        "medium": "google/timesfm-2.5-200m-pytorch",
        "large": "google/timesfm-2.5-200m-pytorch",
    }

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        seed: int = 42,
        context_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="timesfm", device=device, seed=seed)
        if model_size not in self.MODEL_SIZE_MAP:
            raise ValueError(
                f"Unknown model_size '{model_size}'. " f"Choose from {list(self.MODEL_SIZE_MAP)}"
            )
        self.model_size = model_size
        self.context_length = context_length
        self._extra_kwargs = kwargs
        self._model = None
        self._scaled_context: np.ndarray | None = None
        self._scaler = SeriesScaler(
            method="standard"
        )  # Fixed quantile levels produced by the TimesFM 2.5 quantile head.

    TIMESFM_QUANTILES: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def _load_model(self) -> None:
        """Download and initialise the TimesFM 2.x model.

        Uses the ``TimesFM_2p5_200M_torch`` entry point shipped by the
        installed ``timesfm`` package.  The ``google/timesfm-2.5-200m-pytorch``
        checkpoint may require a HuggingFace token: run
        ``huggingface-cli login`` first if you see a 401 error.
        """
        if self._model is not None:
            return

        repo_id = self.MODEL_SIZE_MAP[self.model_size]
        logger.info("Loading TimesFM model '%s' from %s...", self.model_size, repo_id)
        t0 = time.time()

        try:
            from timesfm import TimesFM_2p5_200M_torch as TimesFmModel
        except ImportError as exc:
            raise ImportError(
                "TimesFM requires the 'timesfm' package. " "Install with: pip install timesfm"
            ) from exc

        # ``TimesFM_2p5_200M_torch`` is a hub-mixin wrapper, not an
        # ``nn.Module`` — it has no ``.eval()`` and no parameters to manage.
        self._model = TimesFmModel.from_pretrained(repo_id)
        logger.info("TimesFM model loaded in %.2fs", time.time() - t0)

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> TimesFMForecaster:
        """Prepare TimesFM for prediction (zero-shot)."""
        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                logger.warning(
                    "TimesFM does not support multivariate forecasting. "
                    "Using the first column only."
                )
            y = y.iloc[:, 0]

        self._validate_inputs(y)

        # Detect frequency
        if freq is None:
            from uniftsm.pipeline.frequency import FrequencyDetector

            detected = FrequencyDetector.detect(y)
            self._freq = detected
        else:
            self._freq = freq

        # Scale
        scaled = self._scaler.fit_transform(y)

        # Truncate context
        if len(scaled) > self.context_length:
            scaled = scaled[-self.context_length :]

        self._scaled_context = scaled
        self._context = y
        self._fitted = True
        logger.debug("TimesFM fit complete. Context length: %d", len(scaled))
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        if not self._fitted or self._scaled_context is None:
            raise RuntimeError("Model must be fitted before predicting.")

        self._ensure_model_loaded()
        quantiles = quantiles or [0.1, 0.5, 0.9]

        try:
            from timesfm import ForecastConfig

            # TimesFM requires the compiled decoder; ``max_context`` and
            # ``max_horizon`` are rounded up internally to patch multiples.
            config_obj = ForecastConfig(
                max_context=self.context_length,
                max_horizon=horizon,
                per_core_batch_size=1,
            )
            self._model.compile(config_obj)

            # ``forecast`` expects a list of 1-D series and returns
            # (point_forecast, quantile_forecast) at fixed levels.
            points, quantile_forecast = self._model.forecast(horizon, [self._scaled_context])
        except Exception as exc:
            raise PredictionError(self.model_name, str(exc)) from exc

        # Inverse scale the point forecast
        mean_orig = self._scaler.inverse_transform(np.asarray(points).ravel())

        # Derive std from the inter-quantile range in the original scale
        q_hi = np.asarray(quantile_forecast[0, :, -1])  # q_0.9
        q_lo = np.asarray(quantile_forecast[0, :, 0])  # q_0.1
        std_scaled = (q_hi - q_lo) / (2 * 1.2816)  # z_0.9
        std_orig = std_scaled * self._scaler._std

        # Build index
        if self._context is not None:
            forecast_index = self._make_index(
                (
                    self._context
                    if isinstance(self._context, pd.Series)
                    else pd.Series(index=self._context.index)
                ),
                horizon,
            )
        else:
            forecast_index = pd.RangeIndex(horizon)

        if return_quantiles:
            result = {}
            for q in quantiles:
                from scipy.stats import norm

                z = norm.ppf(q)
                vals = mean_orig + z * std_orig
                result[f"q_{q}"] = pd.DataFrame(vals, index=forecast_index, columns=["forecast"])
            return result

        return pd.DataFrame({"mean": mean_orig, "std": std_orig}, index=forecast_index)

    def get_params(self) -> dict[str, Any]:
        return {
            **super().get_params(),
            "model_size": self.model_size,
            "context_length": self.context_length,
        }
