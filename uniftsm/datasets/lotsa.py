"""LOTSA dataset loader — Large-scale Open Time Series Archive.

LOTSA is the pre-training corpus used by MOIRAI.  This loader provides
access to a curated subset for benchmarking.

Reference: Woo et al. 2024, https://arxiv.org/abs/2402.02592
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class LOTSADatasetLoader:
    """Load the LOTSA dataset for pre-training / benchmarking.

    LOTSA is available via HuggingFace Datasets.

    Parameters
    ----------
    cache_dir:
        Directory to cache downloaded datasets.
    """

    HF_REPO = "salesforce/lotsa"

    def __init__(self, cache_dir: str | None = None) -> None:
        from uniftsm.core.config import config

        self._cache_dir = Path(cache_dir or config.data_dir).expanduser() / "lotsa"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def list_subsets(self) -> list[str]:
        """Return available LOTSA subsets (e.g., ``"monash"``, ``"others"``)."""
        return [
            "monash",
            "others",
            "solar",
            "electricity",
            "traffic",
            "weather",
        ]

    def load(
        self,
        subset: str = "monash",
        max_series: int | None = None,
        split: str = "train",
    ) -> dict[str, pd.Series]:
        """Load a LOTSA subset.

        Args:
            subset: One of the subsets from :meth:`list_subsets`.
            max_series: Limit to the first N series.
            split: ``"train"`` or ``"test"``.

        Returns:
            Dict mapping series ID → pandas Series.
        """
        try:
            from datasets import load_dataset

            dataset = load_dataset(
                self.HF_REPO,
                subset,
                split=split,
                streaming=max_series is not None,
            )
        except ImportError:
            logger.error(
                "The 'datasets' library is required for LOTSA. "
                "Install with: pip install datasets"
            )
            return {}
        except Exception as exc:
            logger.warning("Failed to load LOTSA subset '%s': %s", subset, exc)
            return {}

        series_dict: dict[str, pd.Series] = {}
        count = 0
        for i, example in enumerate(dataset):
            if max_series is not None and count >= max_series:
                break

            target = example.get("target", [])
            if not target:
                continue

            freq = example.get("freq", "H")

            series_dict[f"{subset}_{i}"] = pd.Series(
                target,
                index=pd.date_range(
                    start=pd.Timestamp.now() - pd.Timedelta(len(target), unit=freq),
                    periods=len(target),
                    freq=freq,
                ),
            )
            count += 1

        logger.info("Loaded %d series from LOTSA subset '%s'.", count, subset)
        return series_dict
