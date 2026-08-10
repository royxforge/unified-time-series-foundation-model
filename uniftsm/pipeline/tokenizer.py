"""Quantization-based tokeniser for Chronos-style models.

Maps continuous time series values into discrete tokens via
learnable or fixed quantile binning.  This is the key preprocessing
step for language-model-based forecasting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

BinStrategy = Literal["uniform", "quantile", "log"]


class QuantileTokenizer:
    """Continuous-value → discrete-token mapping via binning.

    Parameters
    ----------
    num_bins:
        Number of quantisation bins (vocabulary size).
    strategy:
        Bin boundary placement strategy:

        - ``"uniform"`` — evenly spaced between observed min/max
        - ``"quantile"`` — equally-populated bins (data-adaptive)
        - ``"log"`` — logarithmic spacing (good for positive-heavy series)
    dtype:
        Output integer type for the tokens.

    Examples
    --------
    >>> tokenizer = QuantileTokenizer(num_bins=4096, strategy="quantile")
    >>> tokenizer.fit(series)
    >>> tokens = tokenizer.transform(series)
    >>> reconstructed = tokenizer.inverse_transform(tokens)
    """

    def __init__(
        self,
        num_bins: int = 4096,
        strategy: BinStrategy = "quantile",
        dtype: type = np.int32,
    ) -> None:
        self.num_bins = num_bins
        self.strategy = strategy
        self.dtype = dtype
        self._bin_edges: np.ndarray | None = None
        self._fitted = False

    def fit(self, y: pd.Series | np.ndarray) -> QuantileTokenizer:
        """Learn bin boundaries from the data.

        Args:
            y: Input series to learn bin boundaries from.

        Returns:
            ``self`` for method chaining.
        """
        values = self._to_array(y)

        if self.strategy == "uniform":
            vmin, vmax = float(values.min()), float(values.max())
            if vmax == vmin:
                vmax = vmin + 1.0
            self._bin_edges = np.linspace(vmin, vmax, self.num_bins + 1)

        elif self.strategy == "quantile":
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            self._bin_edges = np.percentile(values, percentiles)
            # Ensure unique edges (collapse duplicates)
            self._bin_edges = np.unique(self._bin_edges)
            if len(self._bin_edges) < 2:
                self._bin_edges = np.array([0.0, 1.0])

        elif self.strategy == "log":
            pos_values = values[values > 0]
            if len(pos_values) == 0:
                pos_values = np.array([1.0])
            log_min = np.log(pos_values.min())
            log_max = np.log(pos_values.max())
            if log_max == log_min:
                log_max = log_min + 1.0
            self._bin_edges = np.exp(np.linspace(log_min, log_max, self.num_bins + 1))

        else:
            raise ValueError(f"Unknown binning strategy: '{self.strategy}'")

        self._fitted = True
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Map continuous values to discrete token IDs.

        Args:
            y: Input series.

        Returns:
            Integer token IDs of shape ``(n,)``.
        """
        self._check_fitted()
        assert self._bin_edges is not None
        values = self._to_array(y)
        indices = np.digitize(values, self._bin_edges, right=False) - 1
        # Clip to valid range
        indices = np.clip(indices, 0, len(self._bin_edges) - 2)
        return np.asarray(indices, dtype=self.dtype)

    def inverse_transform(self, tokens: np.ndarray) -> np.ndarray:
        """Map token IDs back to continuous bin centres.

        Args:
            tokens: Integer token IDs.

        Returns:
            Continuous values (bin midpoints).
        """
        self._check_fitted()
        assert self._bin_edges is not None
        edges = np.asarray(self._bin_edges, dtype=np.dtype(np.float64))
        tokens = np.asarray(tokens, dtype=np.int64)
        tokens = np.clip(tokens, 0, len(edges) - 2)
        # Bin centre = average of left and right edges
        left = edges[tokens]
        right = edges[tokens + 1]
        return np.asarray((left + right) / 2.0, dtype=np.dtype(np.float64))

    def fit_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Convenience: fit then transform."""
        return self.fit(y).transform(y)

    @property
    def vocab_size(self) -> int:
        """Number of discrete tokens (columns in the embedding table)."""
        if self._bin_edges is None:
            return self.num_bins
        return len(self._bin_edges) - 1

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_array(y: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(y, pd.Series):
            return np.asarray(y.to_numpy(dtype=np.float64), dtype=np.dtype(np.float64))
        return np.asarray(y, dtype=np.dtype(np.float64))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("QuantileTokenizer has not been fitted. Call .fit() first.")

    def __repr__(self) -> str:
        return (
            f"QuantileTokenizer(num_bins={self.num_bins}, "
            f"strategy='{self.strategy}', fitted={self._fitted})"
        )
