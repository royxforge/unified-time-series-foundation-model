"""Lag-Llama wrapper: LLM-based forecasting with lag features.

Lag-Llama uses a LLaMA-style architecture augmented with lag features
(lags of the time series at multiple scales) and a distribution head
for probabilistic forecasting.

Reference: Rasul et al. 2024, https://arxiv.org/abs/2310.08278

Note
----
This wrapper targets the ``lag-llama`` package (installed from GitHub) and
its ``LagLlamaEstimator`` API.  The installed fork predates the high-level
``get_predictor``/``predict`` helpers and relies on pandas aliases that
changed in pandas 2.x, so this wrapper reads the architecture (including
the trained lag sequence) from the checkpoint itself.  Verified end-to-end
on CPU; report any further API drift in the upstream package as an issue.
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


def _load_lagllama_module(estimator: Any) -> Any:
    """Build the Lag-Llama Lightning module from the estimator.

    The git-pinned ``lag-llama`` package relies on
    ``pytorch_lightning``'s ``load_from_checkpoint``, which calls
    ``torch.load`` without a ``weights_only`` argument.  PyTorch 2.6+
    defaults that argument to ``True``, which rejects the pickled
    checkpoint.  We temporarily default it to ``False`` (safe here
    because the checkpoint comes from the trusted official repository).
    """
    import torch

    original_load = torch.load

    def _permissive_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _permissive_load
    try:
        return estimator.create_lightning_module(use_kv_cache=False)
    finally:
        torch.load = original_load


@registry.register(
    "lag_llama",
    metadata={
        "family": "lag_llama",
        "supports_multivariate": False,
        "supports_probabilistic": True,
        "min_length": 64,
        "max_context_length": 2048,
        "pretrained_sizes": ["base"],
        "paper": "Lag-Llama: Towards Foundation Models for Time Series Forecasting",
        "paper_url": "https://arxiv.org/abs/2310.08278",
    },
)
class LagLlamaForecaster(BaseForecaster):
    """Lag-Llama wrapper: LLM-based probabilistic forecasting.

    Parameters
    ----------
    model_size:
        Lag-Llama currently only has a ``"base"`` size.
    device:
        Torch device string.
    seed:
        Random seed.
    context_length:
        Maximum number of time steps to use as context.
    num_samples:
        Number of sample trajectories for probabilistic forecasts.
    **kwargs:
        Additional keyword arguments for Lag-Llama inference.
    """

    # The canonical public repository (the ``-base`` suffix resolves to a
    # gated/404 repo and must not be used).
    MODEL_SIZE_MAP: dict[str, str] = {
        "base": "time-series-foundation-models/lag-llama",
    }

    CHECKPOINT_FILENAME = "lag-llama.ckpt"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        seed: int = 42,
        context_length: int = 512,
        num_samples: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="lag_llama", device=device, seed=seed)
        if model_size not in self.MODEL_SIZE_MAP:
            raise ValueError(
                f"Unknown model_size '{model_size}'. " f"Choose from {list(self.MODEL_SIZE_MAP)}"
            )
        self.model_size = model_size
        self.context_length = context_length
        self.num_samples = num_samples
        self._extra_kwargs = kwargs
        self._model = None
        self._ckpt_path: str | None = None
        self._ckpt_model_kwargs: dict[str, Any] | None = None
        self._scaled_context: np.ndarray | None = None
        self._scaler = SeriesScaler(method="standard")
        # The actual ``LagLlamaEstimator`` is created in ``predict`` (it is
        # cheap to build); ``_loaded`` tracks checkpoint availability so
        # ``_ensure_model_loaded`` is satisfied once weights are downloaded.
        self._loaded = False

    def _load_model(self) -> None:
        """Download the Lag-Llama checkpoint and build the estimator.

        Requires the ``lag-llama`` package:

            pip install git+https://github.com/time-series-foundation-models/lag-llama.git

        The weights are cached by ``huggingface_hub`` on first access.
        """
        if self._ckpt_path is not None:
            return

        repo_id = self.MODEL_SIZE_MAP[self.model_size]
        logger.info("Downloading Lag-Llama checkpoint from %s...", repo_id)
        t0 = time.time()

        try:
            from huggingface_hub import hf_hub_download

            self._ckpt_path = hf_hub_download(
                repo_id=repo_id,
                filename=self.CHECKPOINT_FILENAME,
            )
        except ImportError as exc:
            raise ImportError(
                "Lag-Llama requires 'huggingface_hub' (install via "
                "pip install huggingface_hub)."
            ) from exc
        except Exception as exc:
            # e.g. the checkpoint is gated and requires a token
            raise RuntimeError(
                f"Failed to download Lag-Llama checkpoint from '{repo_id}': {exc}. "
                "Run `huggingface-cli login` if the model is gated."
            ) from exc

        # Recover the exact trained architecture (embedding dims, scaler,
        # time features and the trained lag sequence) from the checkpoint
        # itself.  Recomputing lags via ``get_lags_for_frequency`` is
        # unreliable on pandas 2.x (legacy aliases raise "invalid
        # frequency"), so the checkpoint is the ground truth.  Loading it
        # once here means ``predict`` does not re-read the ~200 MB file.
        try:
            ckpt = torch.load(self._ckpt_path, map_location="cpu", weights_only=False)
            self._ckpt_model_kwargs = ckpt["hyper_parameters"]["model_kwargs"]
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read Lag-Llama checkpoint '{self._ckpt_path}': {exc}. "
                "The checkpoint format may have changed upstream."
            ) from exc

        logger.info("Lag-Llama checkpoint downloaded in %.2fs", time.time() - t0)
        self._loaded = True

    def _ensure_model_loaded(self) -> None:
        """Download the checkpoint if needed (the estimator itself is
        constructed lazily inside :meth:`predict`)."""
        if not self._loaded:
            self._load_model()
        if not self._loaded:
            from uniftsm.core.exceptions import ModelNotLoadedError

            raise ModelNotLoadedError(self.model_name)

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> LagLlamaForecaster:
        """Prepare Lag-Llama for prediction."""
        if isinstance(y, pd.DataFrame):
            if y.shape[1] > 1:
                logger.warning(
                    "Lag-Llama does not support multivariate forecasting. "
                    "Using the first column only."
                )
            y = y.iloc[:, 0]

        self._validate_inputs(y)

        if freq is None:
            from uniftsm.pipeline.frequency import FrequencyDetector

            detected = FrequencyDetector.detect(y)
            self._freq = detected
        else:
            self._freq = freq

        scaled = self._scaler.fit_transform(y)
        if len(scaled) > self.context_length:
            scaled = scaled[-self.context_length :]

        self._scaled_context = scaled
        self._context = y
        self._fitted = True
        logger.debug("Lag-Llama fit complete. Context length: %d", len(scaled))
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
            from gluonts.dataset.field_names import FieldName
            from gluonts.dataset.pandas import PandasDataset
            from lag_llama.gluon.estimator import LagLlamaEstimator
        except ImportError as exc:
            raise ImportError(
                "Lag-Llama inference requires the 'lag-llama' and 'gluonts' "
                "packages. Install with: pip install git+https://github.com/"
                "time-series-foundation-models/lag-llama.git"
            ) from exc

        try:
            assert self._ckpt_path is not None
            assert self._ckpt_model_kwargs is not None
            ckpt_model_kwargs = self._ckpt_model_kwargs
            lags = list(ckpt_model_kwargs["lags_seq"])

            # The lag features reach back up to ``max(lags)`` steps, so the
            # model needs ``context_length + max(lags)`` observations.
            # Anything shorter is zero-padded by GluonTS, which silently
            # corrupts the lag features; reject it instead.
            if len(self._scaled_context) < self.context_length:
                raise ValueError(
                    f"Lag-Llama requires at least context_length={self.context_length} "
                    f"observations (got {len(self._scaled_context)})."
                )

            estimator = LagLlamaEstimator(
                ckpt_path=self._ckpt_path,
                prediction_length=horizon,
                context_length=self.context_length,
                input_size=ckpt_model_kwargs.get("input_size", 1),
                distr_output="studentT",
                scaling=ckpt_model_kwargs.get("scaling", "mean"),
                num_parallel_samples=self.num_samples,
                time_feat=ckpt_model_kwargs.get("time_feat", True),
                n_layer=ckpt_model_kwargs.get("n_layer", 1),
                n_embd_per_head=ckpt_model_kwargs.get("n_embd_per_head", 32),
                n_head=ckpt_model_kwargs.get("n_head", 4),
                max_context_length=ckpt_model_kwargs.get("max_context_length", 2048),
                lags_seq=["D"],  # placeholder; replaced below from the ckpt
            )
            # The estimator's own lag derivation relies on pandas aliases
            # that changed in pandas 2.x — use the trained lags directly.
            estimator.lags_seq = lags
            # The git-pinned ``lag-llama`` package predates the high-level
            # ``get_predictor``/``predict`` helpers — the supported path is
            # to assemble the predictor from its parts.
            module = _load_lagllama_module(estimator)
            transformation = estimator.create_transformation()
            predictor = estimator.create_predictor(transformation, module)

            # Build a single-item GluonTS dataset from the scaled context
            # (truncated to the last ``context_length`` steps during fit,
            # so the timestamps must be truncated identically).
            n_ctx = len(self._scaled_context)
            context = self._context
            if context is not None and isinstance(context, pd.Series):
                timestamps = context.index[-n_ctx:]
            else:
                timestamps = pd.RangeIndex(n_ctx)
            ds = PandasDataset.from_long_dataframe(
                pd.DataFrame(
                    {
                        FieldName.TARGET: np.asarray(self._scaled_context, dtype=np.float32),
                        "timestamp": timestamps,
                        FieldName.ITEM_ID: "series_0",
                    }
                ).reset_index(drop=True),
                item_id=FieldName.ITEM_ID,
                timestamp="timestamp",
                target=FieldName.TARGET,
            )

            forecast = next(predictor.predict(ds, num_samples=self.num_samples))
            samples = np.asarray(forecast.samples, dtype=np.float64)
        except Exception as exc:
            raise PredictionError(self.model_name, str(exc)) from exc

        # Inverse scale back to the original units
        n_samples = samples.shape[0]
        samples_orig = np.zeros_like(samples)
        for j in range(n_samples):
            samples_orig[j] = self._scaler.inverse_transform(samples[j])

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
