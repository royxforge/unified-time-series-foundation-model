"""IBM TinyTimeMixers (TTM) wrapper — lightweight, fine-tunable TSFM.

TTM is a lightweight TSFM (~1M parameters) that uses a patched
encoder-decoder architecture.  It is designed for efficient zero-shot
forecasting and can be fine-tuned on downstream tasks.

Reference: Ekambaram et al. 2024, https://arxiv.org/abs/2401.03955
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
    "ttm",
    metadata={
        "family": "ttm",
        "supports_multivariate": True,
        "supports_probabilistic": False,
        "min_length": 32,
        "max_context_length": 1024,
        # ``model_size`` accepts small/medium/large as aliases; the public
        # r2 release is a single checkpoint.
        "pretrained_sizes": ["small", "medium", "large"],
        "paper": "Tiny Time Mixers (TTMs): Fast Pre-trained Models for Zero-Shot Forecasting",
        "paper_url": "https://arxiv.org/abs/2401.03955",
    },
)
class TTMForecaster(BaseForecaster):
    """IBM TinyTimeMixers wrapper — lightweight, fine-tunable forecasting.

    Parameters
    ----------
    model_size:
        One of ``"small"``, ``"medium"``, ``"large"``.
    device:
        Torch device string.
    seed:
        Random seed.
    context_length:
        Maximum number of time steps to use as context.
    **kwargs:
        Additional keyword arguments for the TTM model.
    """

    # The legacy ``ibm/ttm-*`` repositories are gated and no longer the
    # canonical source.  ``granite-tsfm`` (the package UniTSFM wraps)
    # ships the public ``ibm-granite/granite-timeseries-ttm-r2`` release,
    # so all size aliases resolve to it.
    MODEL_SIZE_MAP: dict[str, str] = {
        "small": "ibm-granite/granite-timeseries-ttm-r2",
        "medium": "ibm-granite/granite-timeseries-ttm-r2",
        "large": "ibm-granite/granite-timeseries-ttm-r2",
    }

    def __init__(
        self,
        model_size: str = "small",
        device: str = "auto",
        seed: int = 42,
        context_length: int = 512,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="ttm", device=device, seed=seed)
        if model_size not in self.MODEL_SIZE_MAP:
            raise ValueError(
                f"Unknown model_size '{model_size}'. " f"Choose from {list(self.MODEL_SIZE_MAP)}"
            )
        self.model_size = model_size
        self.context_length = context_length
        self._extra_kwargs = kwargs
        self._model = None
        self._scaled_context: np.ndarray | None = None
        self._scaler: SeriesScaler | dict[str, SeriesScaler] = SeriesScaler(method="standard")
        self._is_multivariate = False

    def _load_model(self) -> None:
        """Download and initialise the TTM model.

        TTM checkpoints are loaded through ``TinyTimeMixerForPrediction``
        from the ``tsfm_public`` package (shipped by ``granite-tsfm``),
        which exposes the forecasting head used at inference time.
        """
        if self._model is not None:
            return

        repo_id = self.MODEL_SIZE_MAP[self.model_size]
        logger.info("Loading TTM model '%s' from %s...", self.model_size, repo_id)
        t0 = time.time()

        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        self._model = TinyTimeMixerForPrediction.from_pretrained(
            repo_id,
            torch_dtype=torch.bfloat16 if self.device.type == "cuda" else torch.float32,
        )
        if self.device.type != "cuda":
            self._model = self._model.to(self.device)
        self._model.eval()
        logger.info("TTM model loaded in %.2fs", time.time() - t0)

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> TTMForecaster:
        """Prepare TTM for prediction."""
        self._is_multivariate = isinstance(y, pd.DataFrame) and y.shape[1] > 1

        if isinstance(y, pd.DataFrame) and self._is_multivariate:
            scaled = np.zeros_like(y.values, dtype=np.float64)
            scaler_map: dict[str, SeriesScaler] = {}
            for col_idx, col in enumerate(y.columns):
                s = SeriesScaler(method="standard")
                scaled[:, col_idx] = s.fit_transform(y[col])
                scaler_map[col] = s
            self._scaler = scaler_map
            self._scaled_context = scaled
        else:
            if isinstance(y, pd.DataFrame):
                y = y.iloc[:, 0]
            self._validate_inputs(y)
            scaler = self._scaler
            assert isinstance(scaler, SeriesScaler)  # univariate path uses a single scaler
            scaled = scaler.fit_transform(y)
            if len(scaled) > self.context_length:
                scaled = scaled[-self.context_length :]
            self._scaled_context = scaled

        if freq is None:
            from uniftsm.pipeline.frequency import FrequencyDetector

            detected = FrequencyDetector.detect(y)
            self._freq = detected
        else:
            self._freq = freq

        self._context = y
        self._fitted = True
        logger.debug(
            "TTM fit complete. Multivariate: %s, Context length: %d",
            self._is_multivariate,
            len(self._scaled_context) if self._scaled_context is not None else 0,
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
        if not self._fitted or self._scaled_context is None:
            raise RuntimeError("Model must be fitted before predicting.")

        self._ensure_model_loaded()
        quantiles = quantiles or [0.1, 0.5, 0.9]

        try:
            # TTM expects ``past_values`` of shape
            # (batch_size, sequence_length, num_input_channels) and a
            # fixed context length.  Short series are left-padded with
            # zeros (the series is standardised during fit, so zeros are
            # the neutral value); long series are truncated to the last
            # ``context_length`` steps in ``fit``.
            context = np.asarray(self._scaled_context, dtype=np.float32)
            if context.ndim == 1:
                context = context[:, np.newaxis]
            model_ctx = getattr(self._model.config, "context_length", 512)
            if context.shape[0] < model_ctx:
                context = np.pad(
                    context,
                    ((model_ctx - context.shape[0], 0), (0, 0)),
                    mode="constant",
                )
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                outputs = self._model(
                    past_values=context_tensor.to(self.device),
                    return_loss=False,
                    return_dict=True,
                )
            preds = outputs.prediction_outputs.cpu().numpy().squeeze(0)
            # TTM emits its configured prediction length (e.g. 96) even when
            # a shorter horizon is requested — keep only the first steps.
            preds = preds[:horizon]
            # (horizon, 1) → (horizon,) for the univariate case
            if preds.ndim == 2 and preds.shape[1] == 1:
                preds = preds[:, 0]
        except Exception as exc:
            raise PredictionError(self.model_name, str(exc)) from exc

        # Inverse scale
        if isinstance(self._scaler, dict):
            preds_orig = np.zeros_like(preds)
            for i, scaler in enumerate(self._scaler.values()):
                preds_orig[:, i] = scaler.inverse_transform(preds[:, i])
        else:
            preds_orig = self._scaler.inverse_transform(preds)

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
                result[f"q_{q}"] = pd.DataFrame(
                    preds_orig, index=forecast_index, columns=["forecast"]
                )
            return result

        # TTM is point-forecast; use a heuristic std based on residuals
        std = np.abs(preds_orig) * 0.05  # 5% heuristic
        return pd.DataFrame({"mean": preds_orig, "std": std}, index=forecast_index)

    def get_params(self) -> dict[str, Any]:
        return {
            **super().get_params(),
            "model_size": self.model_size,
            "context_length": self.context_length,
        }
