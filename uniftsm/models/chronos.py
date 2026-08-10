"""Amazon Chronos wrapper — treats time series as language.

Architectural insight: Chronos scales and quantises continuous values
into discrete tokens, then uses a T5-based transformer decoder for
autoregressive generation of future tokens.

Reference: Ansari et al. 2024, https://arxiv.org/abs/2403.07815
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import torch

from uniftsm.core.base import BaseForecaster
from uniftsm.core.exceptions import PredictionError
from uniftsm.core.registry import registry
from uniftsm.pipeline.scaler import SeriesScaler

logger = logging.getLogger(__name__)


@registry.register(
    "chronos",
    metadata={
        "family": "chronos",
        "supports_multivariate": False,
        "supports_probabilistic": True,
        "min_length": 64,
        "max_context_length": 2048,
        "pretrained_sizes": ["tiny", "small", "base", "large"],
        "paper": "Chronos: Learning the Language of Time Series",
        "paper_url": "https://arxiv.org/abs/2403.07815",
    },
)
class ChronosForecaster(BaseForecaster):
    """Amazon Chronos wrapper — zero-shot probabilistic forecasting.

    Parameters
    ----------
    model_size:
        One of ``"tiny"``, ``"small"``, ``"base"``, ``"large"``.
    device:
        Torch device string.  ``"auto"`` resolves to CUDA if available.
    seed:
        Random seed.
    context_length:
        Maximum number of time steps to use as context.
    num_samples:
        Number of sample trajectories for probabilistic forecasts.
    **kwargs:
        Additional keyword arguments forwarded to the Chronos pipeline.
    """

    MODEL_SIZE_MAP: dict[str, str] = {
        "tiny": "amazon/chronos-t5-tiny",
        "small": "amazon/chronos-t5-small",
        "base": "amazon/chronos-t5-base",
        "large": "amazon/chronos-t5-large",
    }

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        seed: int = 42,
        context_length: int = 512,
        num_samples: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="chronos", device=device, seed=seed)
        if model_size not in self.MODEL_SIZE_MAP:
            raise ValueError(
                f"Unknown model_size '{model_size}'. " f"Choose from {list(self.MODEL_SIZE_MAP)}"
            )
        self.model_size = model_size
        self.context_length = context_length
        self.num_samples = num_samples
        self._extra_kwargs = kwargs
        self._model = None
        self._scaled_context: np.ndarray | None = None
        self._scaler = SeriesScaler(method="standard")
        self._freq = None

    def _load_model(self) -> None:
        """Download and initialise the Chronos pipeline."""
        if self._model is not None:
            return

        import chronos  # noqa: F401  # ensure package installed

        repo_id = self.MODEL_SIZE_MAP[self.model_size]
        logger.info("Loading Chronos model '%s' from %s...", self.model_size, repo_id)
        t0 = time.time()

        from chronos import ChronosPipeline

        self._model = ChronosPipeline.from_pretrained(
            repo_id,
            device_map=str(self.device),
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        logger.info("Chronos model loaded in %.2fs", time.time() - t0)

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> ChronosForecaster:
        """Prepare the Chronos model for prediction.

        Chronos is zero-shot, so ``fit`` stores the scaled context
        without updating model weights.
        """
        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                logger.warning(
                    "Chronos does not support multivariate forecasting. "
                    "Using the first column only."
                )
            y = y.iloc[:, 0]

        self._validate_inputs(y)

        # Detect frequency from index if not provided
        if freq is None:
            from uniftsm.pipeline.frequency import FrequencyDetector

            detected = FrequencyDetector.detect(y)
            if detected:
                self._freq = detected
            else:
                self._freq = pd.infer_freq(y.index)
        else:
            self._freq = freq

        # Scale the series
        scaled = self._scaler.fit_transform(y)

        # Truncate to max context length
        if len(scaled) > self.context_length:
            scaled = scaled[-self.context_length :]

        self._scaled_context = scaled
        self._context = y
        self._fitted = True

        logger.debug(
            "Chronos fit complete. Context length: %d, freq: %s",
            len(scaled),
            self._freq,
        )
        return self

    def predict(
        self,
        horizon: int,
        return_quantiles: bool = False,
        quantiles: list[float] | None = None,
        X_future: pd.DataFrame | None = None,
        **kwargs: Any,
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Generate probabilistic forecasts.

        Chronos returns raw sample trajectories.  These are summarised
        into mean/std or quantile outputs.

        Args:
            horizon: Forecast horizon.
            return_quantiles: If ``True``, return quantile DataFrames.
            quantiles: Sorted quantile levels.  Defaults to
                ``[0.1, 0.5, 0.9]``.
            X_future: Ignored by Chronos.
            **kwargs: Additional Chronos pipeline arguments (e.g.
                ``temperature``, ``top_k``).

        Returns:
            DataFrame with ``"mean"`` and ``"std"`` columns, **or** a
            dict of quantile DataFrames when ``return_quantiles=True``.
        """
        if not self._fitted or self._scaled_context is None:
            raise RuntimeError("Model must be fitted before predicting. Call .fit() first.")

        self._ensure_model_loaded()
        quantiles = quantiles or [0.1, 0.5, 0.9]

        # Chronos expects a 1D numpy array
        context_tensor = torch.tensor(self._scaled_context, dtype=torch.float32)

        # Generate samples
        try:
            with torch.no_grad():
                samples = self._model.predict(
                    inputs=context_tensor,
                    prediction_length=horizon,
                    num_samples=self.num_samples,
                    **{**self._extra_kwargs, **kwargs},
                )  # -> (num_samples, horizon)
        except Exception as exc:
            raise PredictionError(self.model_name, str(exc)) from exc

        # Convert to numpy and handle batch dimension
        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        elif isinstance(samples, (list, tuple)):
            samples = np.array(samples)

        # Chronos 2.x returns (batch, num_samples, horizon) - squeeze batch
        if samples.ndim == 3:
            samples = samples.squeeze(0)
        # Now samples is (num_samples, horizon)

        # Inverse-transform samples to original scale
        n_samples = samples.shape[0]
        samples_orig = np.zeros_like(samples)
        for i in range(n_samples):
            samples_orig[i] = self._scaler.inverse_transform(samples[i])

        # Build forecast index
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
            result: dict[str, pd.DataFrame] = {}
            for q in quantiles:
                vals = np.quantile(samples_orig, q, axis=0)
                result[f"q_{q}"] = pd.DataFrame(vals, index=forecast_index, columns=["forecast"])
            return result

        mean = np.mean(samples_orig, axis=0)
        std = np.std(samples_orig, axis=0)
        return pd.DataFrame({"mean": mean, "std": std}, index=forecast_index)

    @property
    def is_probabilistic(self) -> bool:
        return True

    def get_params(self) -> dict[str, Any]:
        return {
            **super().get_params(),
            "model_size": self.model_size,
            "context_length": self.context_length,
            "num_samples": self.num_samples,
        }
