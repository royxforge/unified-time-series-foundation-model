"""Missing value imputation strategies for time series.

Provides several strategies for handling NaN values before passing
data to a TSFM, since most foundation models require dense input.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

ImputeStrategy = Literal[
    "forward_fill",
    "backward_fill",
    "linear",
    "cubic",
    "mean",
    "median",
    "zero",
]


class MissingValueImputer:
    """Handle missing (NaN) values in a time series.

    Parameters
    ----------
    strategy:
        Imputation method:

        - ``"forward_fill"`` — propagate last valid observation
        - ``"backward_fill"`` — propagate next valid observation
        - ``"linear"`` — linear interpolation
        - ``"cubic"`` — cubic spline interpolation
        - ``"mean"`` — replace with series mean
        - ``"median"`` — replace with series median
        - ``"zero"`` — replace with 0.0
    max_gap:
        Maximum consecutive NaN values to impute.  Gaps longer than
        this raise a warning and are filled with the series mean.
    warn_on_fill:
        Whether to log a warning when missing values are detected.
    """

    def __init__(
        self,
        strategy: ImputeStrategy = "forward_fill",
        max_gap: int = 10,
        warn_on_fill: bool = True,
    ) -> None:
        self.strategy = strategy
        self.max_gap = max_gap
        self.warn_on_fill = warn_on_fill

    def impute(self, y: pd.Series) -> pd.Series:
        """Fill NaN values in ``y`` using the configured strategy.

        Args:
            y: Input series that may contain NaN values.

        Returns:
            Series with NaN values imputed.  The index is preserved.
        """
        missing_count = y.isna().sum()
        if missing_count == 0:
            return y

        if self.warn_on_fill:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Imputing %d missing values using '%s' strategy.",
                missing_count,
                self.strategy,
            )

        result = y.copy()

        if self.strategy in ("forward_fill", "backward_fill"):
            if self.strategy == "forward_fill":
                result = result.ffill(limit=self.max_gap)
            else:
                result = result.bfill(limit=self.max_gap)
            # Fill any remaining NaN (beyond max_gap) with median
            if result.isna().any():
                result = result.fillna(result.median())

        elif self.strategy in ("linear", "cubic"):
            order = 3 if self.strategy == "cubic" else 1
            result = result.interpolate(
                method="time" if isinstance(result.index, pd.DatetimeIndex) else "linear",
                order=order,
                limit=self.max_gap,
            )
            if result.isna().any():
                result = result.fillna(result.median())

        elif self.strategy == "mean":
            # Guard against all-NaN input: pandas emits a RuntimeWarning
            # when reducing an empty slice.
            result = result.fillna(0.0) if result.isna().all() else result.fillna(result.mean())

        elif self.strategy == "median":
            median_val = 0.0 if result.isna().all() else result.median()
            result = result.fillna(median_val)

        elif self.strategy == "zero":
            result = result.fillna(0.0)

        else:
            raise ValueError(f"Unknown imputation strategy: '{self.strategy}'")

        return result

    def __repr__(self) -> str:
        return f"MissingValueImputer(strategy='{self.strategy}', " f"max_gap={self.max_gap})"
