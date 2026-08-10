"""Salesforce MOIRAI wrapper — unified time series pretraining.

MOIRAI is a transformer trained on the LOTSA dataset with a unified
input representation that handles multiple frequencies and
multivariate series via a patch-wise encoding.

Reference: Woo et al. 2024, https://arxiv.org/abs/2402.02592
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
    "moirai",
    metadata={
        "family": "moirai",
        "supports_multivariate": True,
        "supports_probabilistic": True,
        "min_length": 32,
        "max_context_length": 4096,
        "pretrained_sizes": ["small", "base", "large"],
        "paper": "MOIRAI: Unified Time Series Pretraining",
        "paper_url": "https://arxiv.org/abs/2402.02592",
    },
)
class MoiraiForecaster(BaseForecaster):
    """Salesforce MOIRAI wrapper — multivariate probabilistic forecasting.

    Parameters
    ----------
    model_size:
        One of ``"small"``, ``"base"``, ``"large"``.
    device:
        Torch device string.
    seed:
        Random seed.
    context_length:
        Maximum number of time steps to use as context.
    num_samples:
        Number of sample trajectories for probabilistic forecasts.
    **kwargs:
        Additional keyword arguments for the MOIRAI model.
    """

    MODEL_SIZE_MAP: dict[str, str] = {
        "small": "Salesforce/moirai-small",
        "base": "Salesforce/moirai-base",
        "large": "Salesforce/moirai-large",
    }

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        seed: int = 42,
        context_length: int = 512,
        num_samples: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name="moirai", device=device, seed=seed)
        if model_size not in self.MODEL_SIZE_MAP:
            raise ValueError(
                f"Unknown model_size '{model_size}'. " f"Choose from {list(self.MODEL_SIZE_MAP)}"
            )
        self.model_size = model_size
        self.context_length = context_length
        self.num_samples = num_samples
        self._extra_kwargs = kwargs
        self._model = None
        self._module: Any = None
        self._scaled_context: np.ndarray | None = None
        self._scaler: SeriesScaler | dict[str, SeriesScaler] = SeriesScaler(method="standard")
        self._is_multivariate = False

    def _load_model(self) -> None:
        """Download and initialise the MOIRAI module.

        Loads the weights through ``MoiraiModule.from_pretrained`` from the
        ``uni2ts`` package.  A ``MoiraiForecast`` wrapper (built lazily in
        :meth:`predict`) turns the module into a GluonTS predictor.
        """
        if self._module is not None:
            return

        repo_id = self.MODEL_SIZE_MAP[self.model_size]
        logger.info("Loading MOIRAI model '%s' from %s...", self.model_size, repo_id)
        t0 = time.time()

        try:
            from uni2ts.model.moirai import MoiraiModule
        except ImportError as exc:
            raise ImportError(
                "MOIRAI requires the 'uni2ts' library. " "Install with: pip install uni2ts"
            ) from exc

        self._module = MoiraiModule.from_pretrained(repo_id)
        self._module.to(self.device)
        self._module.eval()
        # ``_ensure_model_loaded`` (inherited from ``BaseForecaster``) checks
        # ``self._model``, so keep both references in sync.
        self._model = self._module
        logger.info("MOIRAI model loaded in %.2fs", time.time() - t0)

    def fit(
        self,
        y: pd.Series | pd.DataFrame,
        X: pd.DataFrame | None = None,
        freq: str | None = None,
        **kwargs: Any,
    ) -> MoiraiForecaster:
        """Prepare MOIRAI for prediction."""
        self._is_multivariate = isinstance(y, pd.DataFrame) and y.shape[1] > 1

        if isinstance(y, pd.DataFrame):
            scaled = np.zeros_like(y.values, dtype=np.float64)
            if self._is_multivariate:
                scaler_map: dict[str, SeriesScaler] = {}
                for col_idx, col in enumerate(y.columns):
                    s = SeriesScaler(method="standard")
                    scaled[:, col_idx] = s.fit_transform(y[col])
                    scaler_map[col] = s
                self._scaler = scaler_map
                self._scaled_context = scaled
            else:
                y = y.iloc[:, 0]

        if not self._is_multivariate:
            self._validate_inputs(y)
            scaler = self._scaler
            assert isinstance(scaler, SeriesScaler)  # univariate path uses a single scaler
            scaled = scaler.fit_transform(y)
            if len(scaled) > self.context_length:
                scaled = scaled[-self.context_length :]
            self._scaled_context = scaled

        # Detect frequency
        if freq is None:
            from uniftsm.pipeline.frequency import FrequencyDetector

            detected = FrequencyDetector.detect(y)
            self._freq = detected
        else:
            self._freq = freq

        self._context = y
        self._fitted = True
        logger.debug(
            "MOIRAI fit complete. Multivariate: %s, Context length: %d",
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
            from gluonts.dataset.common import ListDataset
            from uni2ts.model.moirai import MoiraiForecast

            context = np.asarray(self._scaled_context, dtype=np.float32)
            # GluonTS expects a 2-D target (context, channels) for
            # multivariate series and a 1-D target otherwise.
            target_dim = context.shape[1] if self._is_multivariate else 1

            model = MoiraiForecast(
                module=self._module,
                prediction_length=horizon,
                target_dim=target_dim,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                context_length=context.shape[0],
                patch_size="auto",
                num_samples=self.num_samples,
            )
            predictor = model.create_predictor(batch_size=1, device=str(self.device))

            start = (
                self._context.index[0]
                if isinstance(self._context, pd.Series)
                else pd.Timestamp("2000-01-01")
            )
            freq = self._freq or "D"
            ds = ListDataset([{"target": context, "start": start}], freq=freq)
            forecast = next(predictor.predict(ds))
            samples = np.asarray(forecast.samples, dtype=np.float64)
            # samples: (num_samples, horizon, target_dim) → drop the channel
            # axis for univariate series
            if samples.ndim == 3 and target_dim == 1:
                samples = samples[:, :, 0]
        except Exception as exc:
            raise PredictionError(self.model_name, str(exc)) from exc

        # Inverse scale
        n_samples, h = samples.shape if samples.ndim == 2 else (samples.shape[0], horizon)
        samples_orig = np.zeros_like(samples)
        if isinstance(self._scaler, dict):
            # samples shape: (n_samples, horizon, n_channels) after batch squeeze
            for col_idx, scaler in enumerate(self._scaler.values()):
                for j in range(n_samples):
                    samples_orig[j, :, col_idx] = scaler.inverse_transform(samples[j, :, col_idx])
        else:
            for j in range(n_samples):
                samples_orig[j] = self._scaler.inverse_transform(samples[j])

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
