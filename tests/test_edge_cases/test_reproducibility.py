"""Tests for reproducibility, seeding, and serialization."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestSeedReproducibility:
    """Tests that random seeds produce deterministic behavior."""

    def test_base_forecaster_seed_consistency(self):
        from uniftsm.core.base import BaseForecaster

        class TestModel(BaseForecaster):
            def _load_model(self):
                pass

            def fit(self, y, X=None, freq=None, **kwargs):
                self._fitted = True
                self._context = y
                return self

            def predict(self, horizon, **kwargs):
                return pd.DataFrame({"mean": np.zeros(horizon), "std": np.ones(horizon)})

        m1 = TestModel(model_name="test", seed=42)
        m2 = TestModel(model_name="test", seed=42)
        assert m1.seed == m2.seed
        assert m1.get_params()["seed"] == m2.get_params()["seed"]


class TestSerialization:
    """Tests for model parameter serialization."""

    def test_scaler_get_params(self):
        from uniftsm.pipeline.scaler import SeriesScaler

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scaler = SeriesScaler("standard")
        scaler.fit(data)
        params = scaler.get_params()
        assert params["method"] == "standard"
        assert abs(params["mean"] - 3.0) < 1e-10

    def test_config_serialization(self):
        from uniftsm.core.config import UniTSFMConfig

        cfg = UniTSFMConfig(seed=123, device="cpu")
        d = cfg.model_dump()
        assert d["seed"] == 123

    def test_config_env_prefix(self):
        import os

        os.environ["UNITSFM_SEED"] = "999"
        try:
            from uniftsm.core.config import UniTSFMConfig

            cfg = UniTSFMConfig()
            assert cfg.seed == 999
        finally:
            del os.environ["UNITSFM_SEED"]


class TestDeterministicMetrics:
    """Tests that metrics are deterministic."""

    def test_mase_deterministic(self):
        from uniftsm.evaluation.metrics import mase

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()
        y_insample = np.array([1.0, 2.0, 3.0])
        assert mase(y_true, y_pred, y_insample) == mase(y_true, y_pred, y_insample)

    def test_smape_deterministic(self):
        from uniftsm.evaluation.metrics import smape

        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.9])
        assert smape(y_true, y_pred) == smape(y_true, y_pred)


class TestModelSelectionReproducibility:
    """Tests that model selection is reproducible."""

    def test_meta_features_deterministic(self, simple_series):
        from uniftsm.selection.meta_features import MetaFeatureExtractor

        f1 = MetaFeatureExtractor.extract(simple_series)
        f2 = MetaFeatureExtractor.extract(simple_series)
        assert f1 == f2

    def test_registry_suggest_consistent(self):
        # Import model modules to register them
        import uniftsm.models.chronos  # noqa: F401
        import uniftsm.models.lag_llama  # noqa: F401
        import uniftsm.models.moirai  # noqa: F401
        import uniftsm.models.timesfm  # noqa: F401
        import uniftsm.models.ttm  # noqa: F401
        from uniftsm.core.registry import registry

        s1 = registry.suggest(
            series_length=200, frequency="D", is_multivariate=False, needs_uncertainty=True
        )
        s2 = registry.suggest(
            series_length=200, frequency="D", is_multivariate=False, needs_uncertainty=True
        )
        assert s1 == s2
        assert len(s1) > 0


class TestBenchmarkDB:
    """Benchmark database operations."""

    def test_db_initializes(self):
        from uniftsm.selection.benchmarks import BenchmarkDB

        assert BenchmarkDB() is not None

    def test_get_best_model(self):
        from uniftsm.selection.benchmarks import BenchmarkDB

        db = BenchmarkDB()
        best = db.get_best_model("electricity", metric="MASE")
        assert best is not None
        assert isinstance(best, str)

    def test_get_performance(self):
        from uniftsm.selection.benchmarks import BenchmarkDB

        db = BenchmarkDB()
        perf = db.get_performance("chronos", "electricity", metric="MASE")
        assert perf is not None
        assert isinstance(perf, float)

    def test_to_dataframe(self):
        from uniftsm.selection.benchmarks import BenchmarkDB

        db = BenchmarkDB()
        df = db.to_dataframe()
        assert len(df) > 0
        assert "model" in df.columns
        assert "MASE" in df.columns
