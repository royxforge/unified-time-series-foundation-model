"""M3 and M4 Competition dataset loaders.

Loads time series from the M3 and M4 forecasting competitions.

References:
    - Makridakis & Hibon 2000 (M3)
    - Makridakis et al. 2020 (M4)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class M3M4DatasetLoader:
    """Load M3 or M4 competition datasets.

    Datasets are downloaded from the M-competitions GitHub repository
    on first access.

    Parameters
    ----------
    cache_dir:
        Directory to cache downloaded datasets.
    """

    M4_URL = "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset"
    M3_URL = "https://raw.githubusercontent.com/Mcompetitions/M3-methods/master/Dataset"

    M4_FREQUENCIES = {
        "Yearly": "YS",
        "Quarterly": "QS",
        "Monthly": "MS",
        "Weekly": "W",
        "Daily": "D",
        "Hourly": "H",
    }

    def __init__(self, cache_dir: str | None = None) -> None:
        from uniftsm.core.config import config

        self._cache_dir = Path(cache_dir or config.data_dir).expanduser() / "m_competitions"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def load_m4(
        self,
        frequency: str = "Hourly",
        max_series: int | None = None,
    ) -> dict[str, pd.Series]:
        """Load M4 competition data for a given frequency.

        Args:
            frequency: One of ``"Yearly"``, ``"Quarterly"``, ``"Monthly"``,
                ``"Weekly"``, ``"Daily"``, ``"Hourly"``.
            max_series: Limit to the first N series.

        Returns:
            Dict mapping series ID → pandas Series.
        """
        if frequency not in self.M4_FREQUENCIES:
            raise ValueError(
                f"Unknown frequency '{frequency}'. " f"Choose from {list(self.M4_FREQUENCIES)}."
            )

        # M4 data is in CSV format
        filename = f"{frequency.lower()}-train.csv"
        filepath = self._cache_dir / filename

        if not filepath.exists():
            import urllib.request

            url = f"{self.M4_URL}/{filename}"
            logger.info("Downloading M4 %s data from %s...", frequency, url)
            urllib.request.urlretrieve(url, filepath)

        df = pd.read_csv(filepath)
        freq_alias = self.M4_FREQUENCIES[frequency]

        series_dict: dict[str, pd.Series] = {}
        count = 0
        for _, row in df.iterrows():
            if max_series is not None and count >= max_series:
                break
            series_id = str(row.iloc[0])
            values = row.iloc[1:].dropna().values.astype(float)
            if len(values) < 4:
                continue
            series_dict[series_id] = pd.Series(
                values,
                index=pd.date_range(
                    end=pd.Timestamp.now(),
                    periods=len(values),
                    freq=freq_alias,
                ),
                name=series_id,
            )
            count += 1

        logger.info(
            "Loaded %d series from M4-%s.",
            count,
            frequency,
        )
        return series_dict

    def load_m3(
        self,
        frequency: str = "Monthly",
        max_series: int | None = None,
    ) -> dict[str, pd.Series]:
        """Load M3 competition data.

        Args:
            frequency: One of ``"Monthly"``, ``"Quarterly"``, ``"Yearly"``.
            max_series: Limit to the first N series.

        Returns:
            Dict mapping series ID → pandas Series.
        """
        filepath = self._cache_dir / f"M3{frequency}.csv"

        if not filepath.exists():
            import urllib.request

            url = f"{self.M3_URL}/{filepath.name}"
            logger.info("Downloading M3 %s data from %s...", frequency, url)
            urllib.request.urlretrieve(url, filepath)

        df = pd.read_csv(filepath)

        series_dict: dict[str, pd.Series] = {}
        count = 0
        for _, row in df.iterrows():
            if max_series is not None and count >= max_series:
                break
            series_id = str(row.iloc[0])
            values = row.iloc[:, 1:].dropna().values.astype(float)
            if len(values) < 4:
                continue
            series_dict[series_id] = pd.Series(
                values,
                name=series_id,
            )
            count += 1

        logger.info("Loaded %d series from M3-%s.", count, frequency)
        return series_dict
