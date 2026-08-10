"""Synthetic data generator for controlled test cases.

Generates time series with known properties (trend, seasonality, noise)
for evaluation, unit testing, and ablation studies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class SyntheticDatasetGenerator:
    """Generate synthetic time series with controlled properties.

    Useful for testing, ablation studies, and validating model behaviour
    under known ground-truth conditions.
    """

    @staticmethod
    def linear_trend(
        n: int = 100,
        slope: float = 0.1,
        intercept: float = 0.0,
        noise_std: float = 0.05,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.Series:
        """Generate a series with linear trend.

        Args:
            n: Number of time steps.
            slope: Linear trend slope.
            intercept: Intercept.
            noise_std: Standard deviation of Gaussian noise.
            freq: Pandas frequency alias.
            seed: Random seed.

        Returns:
            Synthetic time series.
        """
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=np.float64)
        values = intercept + slope * x + rng.normal(0, noise_std, n)
        return pd.Series(
            values,
            index=pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq),
        )

    @staticmethod
    def seasonal(
        n: int = 100,
        period: int = 12,
        amplitude: float = 1.0,
        noise_std: float = 0.1,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.Series:
        """Generate a seasonal series.

        Args:
            n: Number of time steps.
            period: Seasonal period.
            amplitude: Seasonal amplitude.
            noise_std: Standard deviation of Gaussian noise.
            freq: Pandas frequency alias.
            seed: Random seed.

        Returns:
            Synthetic time series.
        """
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=np.float64)
        values = amplitude * np.sin(2 * np.pi * x / period) + rng.normal(0, noise_std, n)
        return pd.Series(
            values,
            index=pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq),
        )

    @staticmethod
    def trend_seasonal(
        n: int = 200,
        slope: float = 0.05,
        period: int = 12,
        amplitude: float = 1.0,
        noise_std: float = 0.1,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.Series:
        """Generate a series with both trend and seasonality.

        Args:
            n: Number of time steps.
            slope: Trend slope.
            period: Seasonal period.
            amplitude: Seasonal amplitude.
            noise_std: Standard deviation of Gaussian noise.
            freq: Pandas frequency alias.
            seed: Random seed.

        Returns:
            Synthetic time series.
        """
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=np.float64)
        trend = slope * x
        season = amplitude * np.sin(2 * np.pi * x / period)
        noise = rng.normal(0, noise_std, n)
        values = trend + season + noise
        return pd.Series(
            values,
            index=pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq),
        )

    @staticmethod
    def random_walk(
        n: int = 100,
        std: float = 0.1,
        drift: float = 0.0,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.Series:
        """Generate a random walk.

        Args:
            n: Number of time steps.
            std: Step standard deviation.
            drift: Drift term per step.
            freq: Pandas frequency alias.
            seed: Random seed.

        Returns:
            Synthetic time series.
        """
        rng = np.random.default_rng(seed)
        steps = rng.normal(drift, std, n)
        values = np.cumsum(steps)
        return pd.Series(
            values,
            index=pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq),
        )

    @staticmethod
    def complex(
        n: int = 300,
        freq: str = "h",
        seed: int = 42,
    ) -> pd.Series:
        """Generate a complex series with multiple seasonal components,
        trend, and autoregressive structure.

        Args:
            n: Number of time steps.
            freq: Pandas frequency alias.
            seed: Random seed.

        Returns:
            Synthetic time series.
        """
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=np.float64)

        trend = 0.02 * x
        daily_season = 2.0 * np.sin(2 * np.pi * x / 24)
        weekly_season = 1.5 * np.sin(2 * np.pi * x / 168)
        yearly_season = 0.5 * np.sin(2 * np.pi * x / (365 * 24))
        noise = rng.normal(0, 0.3, n)

        values = trend + daily_season + weekly_season + yearly_season + noise
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            index = pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq)
        return pd.Series(values, index=index)

    @staticmethod
    def multivariate(
        n: int = 100,
        n_channels: int = 3,
        freq: str = "D",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Generate a multivariate time series with correlated channels.

        Args:
            n: Number of time steps per channel.
            n_channels: Number of channels.
            freq: Pandas frequency alias.
            seed: Random seed.

        Returns:
            DataFrame with columns ``f"channel_{i}"``.
        """
        rng = np.random.default_rng(seed)
        x = np.arange(n, dtype=np.float64)

        data = {}
        for i in range(n_channels):
            trend = rng.uniform(-0.05, 0.05) * x
            season = rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * x / rng.integers(6, 24))
            noise = rng.normal(0, 0.2, n)
            data[f"channel_{i}"] = trend + season + noise

        return pd.DataFrame(
            data,
            index=pd.date_range(end=pd.Timestamp.now(), periods=n, freq=freq),
        )
