"""Tests for dataset loaders (synthetic and external)."""

import pandas as pd
import pytest

from uniftsm.datasets.synthetic import SyntheticDatasetGenerator


class TestSyntheticData:
    def test_linear_trend(self):
        series = SyntheticDatasetGenerator.linear_trend(n=100, slope=0.1, seed=42)
        assert len(series) == 100
        assert isinstance(series, pd.Series)
        assert isinstance(series.index, pd.DatetimeIndex)

    def test_seasonal(self):
        series = SyntheticDatasetGenerator.seasonal(n=120, period=12, seed=42)
        assert len(series) == 120

    def test_trend_seasonal(self):
        series = SyntheticDatasetGenerator.trend_seasonal(n=200, seed=42)
        assert len(series) == 200

    def test_random_walk(self):
        series = SyntheticDatasetGenerator.random_walk(n=100, seed=42)
        assert len(series) == 100

    def test_complex(self):
        series = SyntheticDatasetGenerator.complex(n=300, seed=42)
        assert len(series) == 300

    def test_multivariate(self):
        df = SyntheticDatasetGenerator.multivariate(n=100, n_channels=3, seed=42)
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 3
        assert len(df) == 100
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_different_frequencies(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = SyntheticDatasetGenerator.linear_trend(n=50, freq="h", seed=42)
        assert len(series) == 50


class TestMonashLoader:
    def test_list_datasets(self):
        from uniftsm.datasets.monash import MonashDatasetLoader

        loader = MonashDatasetLoader()
        datasets = loader.list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0

    def test_load_nonexistent_raises(self):
        from uniftsm.datasets.monash import MonashDatasetLoader

        loader = MonashDatasetLoader()
        with pytest.raises(ValueError, match="Unknown dataset"):
            loader.load("nonexistent_dataset_xyz")


class TestLOTSALoader:
    def test_list_datasets(self):
        from uniftsm.datasets.lotsa import LOTSADatasetLoader

        loader = LOTSADatasetLoader()
        subsets = loader.list_subsets()
        assert isinstance(subsets, list)
        assert len(subsets) > 0


class TestM3M4Loader:
    def test_load_m4_unknown_freq_raises(self):
        from uniftsm.datasets.m3_m4 import M3M4DatasetLoader

        loader = M3M4DatasetLoader()
        with pytest.raises(ValueError, match="Unknown frequency"):
            loader.load_m4(frequency="InvalidFreq")
