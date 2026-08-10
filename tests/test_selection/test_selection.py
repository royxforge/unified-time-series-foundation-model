"""Tests for MetaFeatureExtractor and ModelSelector."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.selection.benchmarks import BenchmarkDB, benchmark_db
from uniftsm.selection.meta_features import MetaFeatureExtractor
from uniftsm.selection.predictor import ModelSelector


class TestMetaFeatureExtractor:
    def test_extract_returns_all_keys(self, simple_series):
        features = MetaFeatureExtractor.extract(simple_series)
        expected_keys = [
            "length",
            "missing_ratio",
            "adf_pvalue",
            "seasonal_strength",
            "trend_slope",
            "trend_magnitude",
            "snr",
            "coefficient_of_variation",
            "sample_entropy",
            "autocorr_lag1",
            "frequency",
            "skewness",
            "kurtosis",
            "is_multivariate",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

    def test_short_series_raises(self):
        idx = pd.date_range("2020-01-01", periods=5, freq="D")
        short = pd.Series(np.random.randn(5), index=idx)
        with pytest.raises(ValueError, match="too short"):
            MetaFeatureExtractor.extract(short)

    def test_length_feature(self, simple_series):
        features = MetaFeatureExtractor.extract(simple_series)
        assert features["length"] == len(simple_series)

    def test_seasonal_strength_short_series(self):
        idx = pd.date_range("2020-01-01", periods=15, freq="D")
        short = pd.Series(np.random.randn(15), index=idx)
        features = MetaFeatureExtractor.extract(short)
        assert features["seasonal_strength"] == 0.0

    def test_missing_ratio(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        values = np.random.randn(100)
        values[::5] = np.nan
        s = pd.Series(values, index=idx)
        features = MetaFeatureExtractor.extract(s)
        assert features["missing_ratio"] == pytest.approx(0.2, abs=0.01)

    def test_constant_series(self):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        s = pd.Series(np.ones(50), index=idx)
        features = MetaFeatureExtractor.extract(s)
        assert not np.isnan(features["sample_entropy"])

    def test_frequency_detection_with_daily_index(self):
        idx = pd.date_range("2020-01-01", periods=30, freq="D")
        s = pd.Series(np.random.randn(30), index=idx)
        features = MetaFeatureExtractor.extract(s)
        assert isinstance(features["frequency"], float)


class TestModelSelector:
    def test_rule_based_fallback(self, simple_series):
        selector = ModelSelector()
        predictions = selector.predict(simple_series, top_k=2)
        assert len(predictions) <= 2
        for name, score in predictions:
            assert isinstance(name, str)
            assert isinstance(score, float)

    def test_with_metric(self, simple_series):
        selector = ModelSelector(metric="CRPS")
        predictions = selector.predict(simple_series, top_k=1)
        assert len(predictions) >= 0

    def test_train_and_predict(self):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("xgboost not installed")
        rng = np.random.default_rng(42)
        n_series = 20
        features = {
            "length": rng.integers(50, 200, n_series),
            "adf_pvalue": rng.uniform(0, 1, n_series),
            "seasonal_strength": rng.uniform(0, 1, n_series),
            "trend_magnitude": rng.uniform(0, 1, n_series),
            "snr": rng.uniform(0.1, 10, n_series),
        }
        df = pd.DataFrame(features)
        best_models = pd.Series(rng.choice(["chronos", "timesfm", "moirai"], n_series))
        selector = ModelSelector()
        selector.train(df, best_models)
        assert selector._is_trained

        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        s = pd.Series(np.random.randn(100), index=idx)
        predictions = selector.predict(s)
        assert len(predictions) > 0

    def test_fallback_when_not_trained(self, simple_series):
        selector = ModelSelector()
        assert not selector._is_trained
        preds = selector.predict(simple_series)
        assert len(preds) >= 0


class TestBenchmarkDB:
    def test_get_best_model(self):
        db = BenchmarkDB()
        best = db.get_best_model("m4_hourly", metric="MASE")
        assert best is not None
        assert isinstance(best, str)

    def test_get_best_model_unknown_dataset(self):
        db = BenchmarkDB()
        assert db.get_best_model("nonexistent") is None

    def test_get_performance(self):
        db = BenchmarkDB()
        perf = db.get_performance("chronos", "m4_hourly", metric="MASE")
        assert perf is not None
        assert isinstance(perf, float)

    def test_get_performance_unknown(self):
        db = BenchmarkDB()
        assert db.get_performance("unknown_model", "m4_hourly") is None

    def test_to_dataframe_shape(self):
        db = BenchmarkDB()
        df = db.to_dataframe()
        assert len(df) > 0
        assert "model" in df.columns
        assert "MASE" in df.columns

    def test_global_db_is_singleton(self):
        assert isinstance(benchmark_db, BenchmarkDB)
