"""Meta-feature extraction for automatic model selection.

This module implements **Novel Contribution #3**: a set of meta-features
characterising a time series that feed into a lightweight classifier to
predict the best-performing TSFM without running any model.

Features inspired by: Lemke & Gabrys (2010) "Meta-learning for time
series forecasting" with modern additions for the TSFM era.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class MetaFeatureExtractor:
    """Extract characteristics of a time series for model selection.

    These ~15 meta-features capture the statistical properties that
    influence which TSFM architecture will perform best.

    Raises
    ------
    ValueError
        If the series has fewer than 10 observations.
    """

    @staticmethod
    def extract(y: pd.Series) -> dict[str, float]:
        """Extract meta-features from a time series.

        Args:
            y: Input time series with at least 10 observations.

        Returns:
            Dictionary mapping feature names to values.
        """
        values = y.dropna().values
        n = len(values)

        if n < 10:
            raise ValueError(f"Series too short for reliable feature extraction: {n} obs.")

        features: dict[str, float] = {}

        # 1. Length & density
        features["length"] = float(n)
        features["missing_ratio"] = float(y.isna().mean())

        # 2. Stationarity (ADF test)
        features["adf_pvalue"] = MetaFeatureExtractor._adf_test(values)

        # 3. Seasonality strength
        features["seasonal_strength"] = MetaFeatureExtractor._seasonal_strength(values)

        # 4. Trend characteristics
        trend_slope, trend_magnitude = MetaFeatureExtractor._trend(values)
        features["trend_slope"] = trend_slope
        features["trend_magnitude"] = trend_magnitude

        # 5. Noise characteristics
        features["snr"] = MetaFeatureExtractor._signal_to_noise(values)
        features["coefficient_of_variation"] = MetaFeatureExtractor._cv(values)

        # 6. Complexity (entropy)
        features["sample_entropy"] = MetaFeatureExtractor._sample_entropy(values)

        # 7. Autocorrelation
        features["autocorr_lag1"] = MetaFeatureExtractor._autocorr(values, lag=1)

        # 8. Frequency
        features["frequency"] = MetaFeatureExtractor._encode_frequency(y)

        # 9. Distribution shape
        # scipy's moment computation loses precision on (near-)constant data;
        # fall back to 0.0 (symmetric, mesokurtic) in that degenerate case.
        if np.std(values) < 1e-10:
            features["skewness"] = 0.0
            features["kurtosis"] = 0.0
        else:
            features["skewness"] = float(stats.skew(values))
            features["kurtosis"] = float(stats.kurtosis(values))

        # 10. Stable flag
        features["is_multivariate"] = 0.0

        return features

    # ── Helper methods ───────────────────────────────────────────────────

    @staticmethod
    def _adf_test(values: np.ndarray) -> float:
        """Augmented Dickey-Fuller test p-value."""
        try:
            from statsmodels.tsa.stattools import adfuller

            result = adfuller(values, maxlag=min(len(values) // 4, 20), autolag="AIC")
            return float(result[1])
        except Exception:
            return 0.5

    @staticmethod
    def _seasonal_strength(values: np.ndarray) -> float:
        """Seasonal strength via decomposition (0 = none, 1 = strong)."""
        n = len(values)
        if n < 24:
            return 0.0
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose

            period = min(12, n // 2)
            decomp = seasonal_decompose(values, model="additive", period=period)
            resid_var = float(np.nanvar(decomp.resid))
            total_var = float(np.var(values))
            if total_var < 1e-8:
                return 0.0
            strength = 1.0 - resid_var / total_var
            return float(np.clip(strength, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _trend(values: np.ndarray) -> tuple[float, float]:
        """Return (slope, magnitude) of linear trend."""
        n = len(values)
        if n < 4:
            return 0.0, 0.0
        x = np.arange(n, dtype=np.float64)
        slope = np.polyfit(x, values, 1)[0]
        mag = abs(slope) / (np.std(values) + 1e-8)
        return float(slope), float(mag)

    @staticmethod
    def _signal_to_noise(values: np.ndarray) -> float:
        """Signal-to-noise ratio (mean / std of differenced series)."""
        diff_std = np.std(np.diff(values))
        if diff_std < 1e-8:
            return 100.0
        return float(np.mean(values) / diff_std)

    @staticmethod
    def _cv(values: np.ndarray) -> float:
        """Coefficient of variation (std / |mean|)."""
        mean_abs = float(np.mean(np.abs(values)))
        if mean_abs < 1e-8:
            return 0.0
        return float(np.std(values) / mean_abs)

    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
        """Sample entropy — measures complexity of the series."""
        r = r_factor * np.std(x)
        if r < 1e-8:
            return 0.0
        try:

            def _count_matches(template_length: int) -> int:
                count = 0
                n = len(x)
                for i in range(n - template_length):
                    for j in range(i + 1, n - template_length):
                        if (
                            np.max(np.abs(x[i : i + template_length] - x[j : j + template_length]))
                            < r
                        ):
                            count += 1
                return count

            A = _count_matches(m + 1)
            B = _count_matches(m)
            # No matching templates → undefined entropy; return a bounded
            # fallback so the feature vector stays finite for XGBoost.
            if A == 0 or B == 0:
                return 0.0
            return float(-np.log(A / B))
        except Exception:
            return 0.0

    @staticmethod
    def _autocorr(values: np.ndarray, lag: int = 1) -> float:
        """Pearson autocorrelation at a given lag."""
        n = len(values)
        if n <= lag:
            return 0.0
        x = values[: n - lag]
        y = values[lag:]
        if np.std(x) < 1e-8 or np.std(y) < 1e-8:
            return 0.0
        result = np.corrcoef(x, y)
        return float(result[0, 1])

    @staticmethod
    def _encode_frequency(y: pd.Series) -> float:
        """Encode the dominant frequency as a numeric value.

        Returns:
            - 0 = irregular / unknown
            - 1 = yearly
            - 2 = quarterly
            - 3 = monthly
            - 4 = weekly
            - 5 = daily
            - 6 = hourly
            - 7 = minutely
        """
        from uniftsm.pipeline.frequency import FrequencyDetector

        freq = FrequencyDetector.detect(y)
        if freq is None:
            return 0.0
        freq_map = {
            "Y": 1,
            "YS": 1,
            "YE": 1,
            "Q": 2,
            "QS": 2,
            "QE": 2,
            "M": 3,
            "MS": 3,
            "ME": 3,
            "W": 4,
            "D": 5,
            "B": 5,
            "H": 6,
            "h": 6,
            "T": 7,
            "min": 7,
            "S": 8,
        }
        # Strip any suffix (e.g., "W-SUN" -> "W")
        base = freq.split("-")[0] if freq else ""
        return float(freq_map.get(base, 0))
