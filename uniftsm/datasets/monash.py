"""Monash Forecasting Archive dataset loader.

Loads datasets from the Monash Time Series Forecasting Archive,
a standardised benchmark for time series forecasting evaluation.

Reference: Godahewa et al. 2021, https://arxiv.org/abs/2105.06643
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class MonashDatasetLoader:
    """Load datasets from the Monash Forecasting Archive.

    Datasets are downloaded from the Monash repository on first access
    and cached locally.

    Parameters
    ----------
    cache_dir:
        Directory to cache downloaded datasets.
    """

    # Maps dataset name -> Zenodo record ID, zip file name, and metadata.
    # The `tsf_pattern` is a glob-like pattern to discover the .tsf file
    # inside the zip dynamically (some archives use non-standard names).
    DATASETS: dict[str, dict[str, Any]] = {
        "m4_hourly": {
            "zip": "m4_hourly_dataset.zip",
            "tsf_pattern": "*m4_hourly*",
            "record_id": "4656589",
            "frequency": "h",
            "description": "M4 Competition (Hourly)",
        },
        "electricity": {
            "zip": "electricity_hourly_dataset.zip",
            "tsf_pattern": "*electricity_hourly*",
            "record_id": "3898439",
            "frequency": "h",
            "description": "Electricity Load (Hourly)",
        },
        "traffic": {
            "zip": "traffic_hourly_dataset.zip",
            "tsf_pattern": "*traffic_hourly*",
            "record_id": "4656132",
            "frequency": "h",
            "description": "Traffic (Hourly)",
        },
        "weather": {
            "zip": "weather_dataset.zip",
            "tsf_pattern": "*weather*",
            "record_id": "4654822",
            "frequency": "d",
            "description": "Weather (Daily)",
        },
        "solar": {
            "zip": "solar_10_minutes_dataset.zip",
            "tsf_pattern": "*solar*",
            "record_id": "4656144",
            "frequency": "10min",
            "description": "Solar Power (10-minutely)",
        },
        "exchange_rate": {
            "zip": "fred_md_dataset.zip",
            "tsf_pattern": "*fred*",
            "record_id": "4654833",
            "frequency": "d",
            "description": "Exchange Rate (Daily)",
        },
        "influenza": {
            "zip": "nn5_weekly_dataset.zip",
            "tsf_pattern": "*nn5_weekly*",
            "record_id": "4656125",
            "frequency": "W",
            "description": "Influenza (NN5 Weekly)",
        },
    }

    def __init__(self, cache_dir: str | None = None) -> None:
        from uniftsm.core.config import config

        self._cache_dir = Path(cache_dir or config.data_dir).expanduser() / "monash"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def list_datasets(self) -> list[str]:
        """Return available dataset names."""
        return sorted(self.DATASETS)

    def load(
        self,
        dataset_name: str,
        max_series: int | None = None,
    ) -> dict[str, pd.Series]:
        """Load a specific dataset from the Monash archive.

        Args:
            dataset_name: One of the names from :meth:`list_datasets`.
            max_series: Limit to the first N series (useful for testing).

        Returns:
            Dict mapping series ID → time series as a pandas Series.

        Raises
        ------
        ValueError
            If the dataset name is unknown.
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. " f"Available: {self.list_datasets()}"
            )
        return self._load_tsf_dataset(dataset_name, max_series)

    def _find_tsf_in_zip(self, zip_path: Path, tsf_pattern: str) -> str | None:
        """Find a .tsf file inside a zip archive matching a pattern."""
        import fnmatch
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            for n in names:
                if fnmatch.fnmatch(n, tsf_pattern) and (n.endswith(".tsf") or n.endswith(".ts")):
                    return n
        return None

    def _load_tsf_dataset(
        self,
        name: str,
        max_series: int | None = None,
    ) -> dict[str, pd.Series]:
        """Load a .tsf format dataset (downloading zip + extracting if needed)."""
        import zipfile

        info = self.DATASETS[name]
        zip_path = self._cache_dir / info["zip"]

        # Download the zip if needed
        if not zip_path.exists():
            logger.info("Downloading Monash dataset '%s'...", name)
            self._download(name)

        # Look for previously extracted .tsf files matching the pattern
        tsf_name = self._find_tsf_in_zip(zip_path, info["tsf_pattern"])
        if tsf_name is None:
            raise FileNotFoundError(
                f"No .tsf file matching '{info['tsf_pattern']}' " f"found in {zip_path}."
            )

        tsf_path = self._cache_dir / tsf_name

        # Extract if needed
        if not tsf_path.exists():
            logger.info("Extracting %s from %s...", tsf_name, zip_path.name)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extract(tsf_name, self._cache_dir)

        return self._parse_tsf(tsf_path, name, max_series)

    def _download(self, name: str) -> None:
        """Download a dataset zip file from its individual Zenodo record."""
        import urllib.request

        info = self.DATASETS[name]
        record_id = info.get("record_id", "4654833")
        url = f"https://zenodo.org/records/{record_id}/files/{info['zip']}?download=1"
        dest = self._cache_dir / info["zip"]

        urllib.request.urlretrieve(url, dest)
        logger.info("Downloaded %s to %s", url, dest)

    def _parse_tsf(
        self,
        filepath: Path,
        name: str,
        max_series: int | None = None,
    ) -> dict[str, pd.Series]:
        """Parse a .tsf file from the Monash archive."""
        series_dict: dict[str, pd.Series] = {}
        frequency = self.DATASETS[name]["frequency"]

        with open(filepath) as f:
            lines = f.readlines()

        # Skip metadata / header lines
        data_lines = [
            line.strip()
            for line in lines
            if line.strip() and not line.startswith("#") and not line.startswith("@")
        ]

        count = 0
        for line in data_lines:
            if max_series is not None and count >= max_series:
                break

            parts = line.split(":")
            if len(parts) < 2:
                continue

            series_id = parts[0].strip().strip("'\"")
            # Value sequence is the last part (semicolon-separated)
            values_str = parts[-1].strip().strip("'\"")
            values = [float(v) for v in values_str.split(",") if v]

            if len(values) < 4:
                continue  # skip very short series

            series_dict[series_id] = pd.Series(
                values,
                index=pd.date_range(
                    end=pd.Timestamp.now(),
                    periods=len(values),
                    freq=frequency,
                ),
                name=series_id,
            )
            count += 1

        logger.info("Loaded %d series from Monash dataset '%s'.", count, name)
        return series_dict
