"""Full end-to-end integration tests for the UniTSFM workflow."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import MockForecaster


class TestSelectionIntegration:
    """Meta-features and model selection."""

    def test_extract_features(self, simple_series):
        from uniftsm.selection.meta_features import MetaFeatureExtractor

        features = MetaFeatureExtractor.extract(simple_series)
        assert "length" in features
        assert "seasonal_strength" in features
        assert features["length"] == len(simple_series)

    def test_short_series_raises(self):
        from uniftsm.selection.meta_features import MetaFeatureExtractor

        short = pd.Series(np.arange(5), index=pd.date_range("2020-01-01", periods=5, freq="D"))
        with pytest.raises((ValueError, Exception)):
            MetaFeatureExtractor.extract(short)

    def test_registry_suggest_returns_models(self):
        import uniftsm.models.chronos  # noqa: F401
        import uniftsm.models.lag_llama  # noqa: F401
        import uniftsm.models.moirai  # noqa: F401
        import uniftsm.models.timesfm  # noqa: F401
        import uniftsm.models.ttm  # noqa: F401
        from uniftsm.core.registry import registry

        suggestions = registry.suggest(
            series_length=200, frequency="D", is_multivariate=False, needs_uncertainty=True
        )
        assert len(suggestions) > 0
        for name, score in suggestions:
            assert isinstance(name, str)
            assert isinstance(score, float)


class TestBacktestingIntegration:
    """Backtesting with mock models."""

    def test_backtesting_produces_results(self, hourly_series):
        from uniftsm.evaluation.backtesting import BacktestingEngine, RollingWindowSplit
        from uniftsm.evaluation.metrics import compute_metrics

        splitter = RollingWindowSplit(train_size=100, test_size=20, step_size=30)
        engine = BacktestingEngine(splitter)
        model = MockForecaster(model_name="test")
        results = engine.run(hourly_series, model)
        assert len(results) > 0
        for r in results:
            metrics = compute_metrics(
                y_true=r["y_test"].values,
                y_pred=r["y_pred_mean"],
                mean=r["y_pred_mean"],
                std=r["y_pred_std"],
            )
            assert "MASE" in metrics

    def test_backtesting_multiple_models(self, hourly_series):
        from uniftsm.evaluation.backtesting import BacktestingEngine, RollingWindowSplit

        splitter = RollingWindowSplit(train_size=100, test_size=20, step_size=50)
        engine = BacktestingEngine(splitter)
        models = {"a": MockForecaster(model_name="a"), "b": MockForecaster(model_name="b")}
        results = engine.run_multiple(hourly_series, models)
        assert set(results.keys()) == {"a", "b"}
        assert len(results["a"]) > 0


class TestStatisticalSignificance:
    """Statistical tests with forecast errors."""

    def test_dm_test(self):
        from uniftsm.evaluation.significance import diebold_mariano_test

        result = diebold_mariano_test(
            np.random.normal(0, 1, 100), np.random.normal(0.5, 1, 100), h=1
        )
        assert "dm_stat" in result
        assert "p_value" in result

    def test_wilcoxon_test(self):
        from uniftsm.evaluation.significance import wilcoxon_signed_rank_test

        result = wilcoxon_signed_rank_test(
            np.random.normal(0, 1, 50), np.random.normal(0.4, 1, 50)
        )
        assert "p_value" in result


class TestDatasetsIntegration:
    """Synthetic dataset generation."""

    def test_linear_trend(self):
        from uniftsm.datasets.synthetic import SyntheticDatasetGenerator

        s = SyntheticDatasetGenerator.linear_trend(n=200, slope=0.1, freq="D")
        assert len(s) == 200
        assert isinstance(s.index, pd.DatetimeIndex)

    def test_seasonal(self):
        from uniftsm.datasets.synthetic import SyntheticDatasetGenerator

        s = SyntheticDatasetGenerator.seasonal(n=120, period=12, freq="D")
        assert not s.isna().any()

    def test_random_walk(self):
        from uniftsm.datasets.synthetic import SyntheticDatasetGenerator

        s = SyntheticDatasetGenerator.random_walk(n=100, freq="D")
        assert len(s) == 100

    def test_complex(self):
        from uniftsm.datasets.synthetic import SyntheticDatasetGenerator

        s = SyntheticDatasetGenerator.complex(n=200, freq="D")
        assert len(s) == 200
