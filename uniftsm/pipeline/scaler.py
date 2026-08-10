"""Per-series scaling and normalisation module.

TSFMs are typically trained on normalised data.  This module provides
reversible scalers that can be fitted to a training series and applied
to a test / forecast horizon.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

ScaleMethod = Literal["standard", "minmax", "robust", "identity"]


class SeriesScaler:
    """Per-series scaling with invertible transforms.

    Parameters
    ----------
    method:
        Scaling method:

        - ``"standard"`` — zero-mean, unit-variance (Z-score)
        - ``"minmax"`` — scale to ``[0, 1]``
        - ``"robust"`` — centre by median, scale by IQR
        - ``"identity"`` — no transformation

    Examples
    --------
    >>> scaler = SeriesScaler("standard")
    >>> scaler.fit(series)
    >>> scaled = scaler.transform(series)
    >>> back = scaler.inverse_transform(scaled)
    """

    def __init__(self, method: ScaleMethod = "standard") -> None:
        self.method = method
        self._mean: float = 0.0
        self._std: float = 1.0
        self._min: float = 0.0
        self._max: float = 1.0
        self._median: float = 0.0
        self._iqr: float = 1.0
        self._fitted: bool = False

    def fit(self, y: pd.Series | np.ndarray) -> SeriesScaler:
        """Compute scaling statistics from the input series.

        Args:
            y: Input time series.

        Returns:
            ``self`` for method chaining.
        """
        values = self._to_array(y)

        if self.method == "standard":
            self._mean = float(np.mean(values))
            self._std = float(np.std(values, ddof=1)) or 1.0
        elif self.method == "minmax":
            self._min = float(np.min(values))
            self._max = float(np.max(values))
            if self._max == self._min:
                self._max = self._min + 1.0
        elif self.method == "robust":
            self._median = float(np.median(values))
            q75, q25 = np.percentile(values, [75, 25])
            iqr = float(q75 - q25)
            # Degenerate or subnormal spreads (e.g. a single outlier among
            # many identical values) would overflow to ``inf`` when used as
            # a divisor.  Floor the scale so the transform stays finite and
            # invertible for any float64 input.
            self._iqr = max(iqr, 1e-12) if iqr else 1.0
        elif self.method == "identity":
            pass
        else:
            raise ValueError(f"Unknown scaling method: '{self.method}'")

        self._fitted = True
        return self

    def transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Apply the scaling transform.

        Args:
            y: Series to transform (must have same dimensions as fit data).

        Returns:
            Scaled array.
        """
        self._check_fitted()
        values = self._to_array(y)

        if self.method == "standard":
            return np.asarray((values - self._mean) / self._std, dtype=np.dtype(np.float64))
        if self.method == "minmax":
            return np.asarray(
                (values - self._min) / (self._max - self._min),
                dtype=np.dtype(np.float64),
            )
        if self.method == "robust":
            return np.asarray((values - self._median) / self._iqr, dtype=np.dtype(np.float64))
        return np.asarray(values, dtype=np.dtype(np.float64))  # identity

    def inverse_transform(
        self,
        y_scaled: pd.Series | np.ndarray,
    ) -> np.ndarray:
        """Reverse the scaling transform.

        Args:
            y_scaled: Scaled values to restore to original scale.

        Returns:
            Array in the original data scale.
        """
        self._check_fitted()
        values = self._to_array(y_scaled)

        if self.method == "standard":
            return np.asarray(values * self._std + self._mean, dtype=np.dtype(np.float64))
        if self.method == "minmax":
            return np.asarray(
                values * (self._max - self._min) + self._min,
                dtype=np.dtype(np.float64),
            )
        if self.method == "robust":
            return np.asarray(values * self._iqr + self._median, dtype=np.dtype(np.float64))
        return np.asarray(values, dtype=np.dtype(np.float64))  # identity

    def fit_transform(self, y: pd.Series | np.ndarray) -> np.ndarray:
        """Convenience: fit then transform in one call."""
        return self.fit(y).transform(y)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> dict:
        """Return scaling parameters for serialisation."""
        return {
            "method": self.method,
            "mean": self._mean,
            "std": self._std,
            "min": self._min,
            "max": self._max,
            "median": self._median,
            "iqr": self._iqr,
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_array(y: pd.Series | np.ndarray) -> np.ndarray:
        if isinstance(y, pd.Series):
            return np.asarray(y.to_numpy(dtype=np.float64), dtype=np.dtype(np.float64))
        return np.asarray(y, dtype=np.dtype(np.float64))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("SeriesScaler has not been fitted. Call .fit() first.")

    def __repr__(self) -> str:
        return f"SeriesScaler(method='{self.method}', fitted={self._fitted})"
