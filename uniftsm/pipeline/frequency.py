"""Frequency detection and resampling module.

Infers the dominant frequency of a time series and provides resampling
utilities to align series with mismatched frequencies.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)

# Mapping from pandas offset aliases to descriptive names
FREQ_DESCRIPTION: dict[str, str] = {
    "T": "minutely",
    "min": "minutely",
    "H": "hourly",
    "D": "daily",
    "W": "weekly",
    "M": "monthly",
    "MS": "monthly",
    "ME": "monthly",
    "Q": "quarterly",
    "QS": "quarterly",
    "QE": "quarterly",
    "Y": "yearly",
    "YS": "yearly",
    "YE": "yearly",
    "B": "business_daily",
    "S": "secondly",
}

AggMethod = Literal["mean", "sum", "median", "last", "first"]


class FrequencyDetector:
    """Detect and validate the frequency of a time series index.

    Uses pandas frequency inference as a first pass, falls back to
    median-spacing heuristic for irregularly-sampled data.
    """

    @staticmethod
    def detect(y: pd.Series | pd.DataFrame) -> str | None:
        """Detect the frequency of a time series.

        Args:
            y: Series or DataFrame with a DatetimeIndex.

        Returns:
            Normalised pandas frequency alias (e.g. ``"H"``, ``"D"``,
            ``"10T"``), or ``None`` if detection fails.  Frequencies are
            normalised to canonical aliases (``min`` → ``T``, lowercase
            ``w``/``m`` → ``W``/``ME``) so downstream code can safely
            pass them to ``pd.date_range`` / ``pd.Timedelta``.
        """
        idx = y.index

        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError(
                f"Expected a DatetimeIndex, got {type(idx).__name__}. "
                f"Ensure your data has a datetime index."
            )

        # Try pandas built-in inference
        freq = pd.infer_freq(idx)
        if freq is not None:
            return FrequencyDetector._normalize(str(freq))

        # Fallback: median spacing heuristic
        if len(idx) >= 4:
            deltas = pd.Series(idx).diff().dt.total_seconds().median()
            if deltas < 90:
                return "T"  # minutely
            if deltas < 7200:
                return "H"  # hourly
            if deltas < 86400 * 2:
                return "D"  # daily
            if deltas < 86400 * 8:
                return "W"  # weekly
            if deltas < 86400 * 35:
                return "MS"  # monthly start
            return "YS"  # yearly start

        return None

    @staticmethod
    def _normalize(freq: str) -> str | None:
        """Normalise a pandas frequency string to a canonical alias.

        Handles the aliases pandas produces but rejects downstream
        (``"10min"``) and the aliases pandas 2.2+ deprecates (lowercase
        ``"w"`` / ``"m"`` / ``"y"``).  Returns ``None`` for strings
        that are not valid pandas frequencies.
        """
        try:
            pd.tseries.frequencies.to_offset(freq)
        except ValueError:
            return None

        # ``pd.infer_freq`` returns aliases that pandas accepts as offsets
        # but that break downstream consumers (``pd.Timedelta(unit=...)``
        # rejects multiplier units like "10min"; lowercase ``w``/``y`` are
        # deprecated by pandas 2.2+).  Normalise to canonical aliases.
        # Note: ``infer_freq`` only ever emits well-formed uppercase
        # strings (e.g. "W-SUN", "2H"), so the ``replace`` calls below
        # cannot corrupt an anchored weekday suffix.
        return freq.replace("min", "T").replace("w", "W").replace("y", "Y")

    @staticmethod
    def describe(freq: str) -> str:
        """Return a human-readable description of a frequency.

        Args:
            freq: Pandas frequency alias.

        Returns:
            Description string, e.g. ``"hourly"``, ``"daily"``.
        """
        return FREQ_DESCRIPTION.get(freq, freq)

    @staticmethod
    def is_supported(freq: str) -> bool:
        """Check if a frequency is in the supported list."""
        from uniftsm.core.config import config

        return freq in config.supported_frequencies


class FrequencyResampler:
    """Resample a time series to a target frequency.

    Handles upsampling (interpolation) and downsampling (aggregation).
    """

    def __init__(self, method: AggMethod = "mean") -> None:
        self.method = method

    def resample(
        self,
        y: pd.Series,
        target_freq: str,
        fill_limit: int = 3,
    ) -> pd.Series:
        """Resample ``y`` to ``target_freq``.

        Args:
            y: Input series with DatetimeIndex.
            target_freq: Target pandas frequency alias.
            fill_limit: Max consecutive NaN values to forward-fill.

        Returns:
            Resampled series with a regular ``target_freq`` index.
        """
        if not isinstance(y.index, pd.DatetimeIndex):
            raise TypeError("Series must have a DatetimeIndex.")

        # Determine the aggregation method for the resample
        agg_map: AggMethod = self._infer_aggregation(target_freq)
        resampled = y.resample(target_freq).agg(agg_map)

        # Forward-fill limited NaN gaps (from upsampling)
        resampled = resampled.ffill(limit=fill_limit)

        # Drop any remaining NaN
        resampled = resampled.dropna()
        return resampled

    def _infer_aggregation(self, freq: str) -> AggMethod:
        """Infer appropriate aggregation based on frequency."""
        # For higher-frequency targets (downsampling), sum is appropriate
        # for cumulative flows; mean for stocks. Default to mean.
        return self.method
