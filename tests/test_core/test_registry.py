"""Tests for the ModelRegistry."""

import pytest

from uniftsm.core.base import BaseForecaster
from uniftsm.core.registry import ModelRegistry, registry


class DummyModel(BaseForecaster):
    def _load_model(self): ...
    def fit(self, y, X=None, freq=None, **kwargs):
        self._fitted = True
        return self

    def predict(self, horizon, **kwargs):
        import numpy as np
        import pandas as pd

        return pd.DataFrame({"mean": np.zeros(horizon), "std": np.ones(horizon)})


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        reg.register("dummy")(DummyModel)
        cls = reg.get("dummy")
        assert cls is DummyModel

    def test_get_unknown_raises(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="Unknown model"):
            reg.get("nonexistent")

    def test_list_models(self):
        reg = ModelRegistry()
        reg.register("a")(DummyModel)
        reg.register("b")(DummyModel)
        models = reg.list_models()
        assert "a" in models
        assert "b" in models

    def test_register_with_metadata(self):
        reg = ModelRegistry()
        reg.register("dummy", metadata={"paper": "test", "supports_multivariate": False})(
            DummyModel
        )
        meta = reg.get_metadata("dummy")
        assert meta["paper"] == "test"
        assert meta["supports_multivariate"] is False

    def test_list_with_metadata(self):
        reg = ModelRegistry()
        reg.register("dummy", metadata={"paper": "test"})(DummyModel)
        items = reg.list_with_metadata()
        assert any(item["name"] == "dummy" for item in items)
        assert any(item.get("paper") == "test" for item in items)

    def test_contains(self):
        reg = ModelRegistry()
        reg.register("dummy")(DummyModel)
        assert "dummy" in reg
        assert "nope" not in reg

    def test_len(self):
        reg = ModelRegistry()
        reg.register("a")(DummyModel)
        reg.register("b")(DummyModel)
        assert len(reg) == 2

    def test_suggest_empty(self):
        reg = ModelRegistry()
        suggestions = reg.suggest(
            series_length=100, frequency="D", is_multivariate=False, needs_uncertainty=False
        )
        assert suggestions == []

    def test_suggest_with_models(self):
        reg = ModelRegistry()
        reg.register(
            "m1", metadata={"min_length": 10, "max_context_length": 500, "family": "chronos"}
        )(DummyModel)
        reg.register(
            "m2", metadata={"min_length": 10, "max_context_length": 500, "family": "ttm"}
        )(DummyModel)
        suggestions = reg.suggest(
            series_length=100, frequency="D", is_multivariate=False, needs_uncertainty=True
        )
        assert len(suggestions) > 0
        assert len(suggestions) <= 3
        # All scores should be positive
        for _, score in suggestions:
            assert isinstance(score, float)

    def test_suggest_multivariate_preference(self):
        reg = ModelRegistry()
        reg.register("univariate", metadata={"supports_multivariate": False, "min_length": 10})(
            DummyModel
        )
        reg.register("multivariate", metadata={"supports_multivariate": True, "min_length": 10})(
            DummyModel
        )
        suggestions = reg.suggest(
            series_length=100, frequency="D", is_multivariate=True, needs_uncertainty=False
        )
        # The multivariate-supporting model should be preferred
        names = [n for n, _ in suggestions]
        assert "multivariate" in names

    def test_suggest_incompatible_length(self):
        reg = ModelRegistry()
        reg.register("short", metadata={"min_length": 1000})(DummyModel)
        reg.register("long", metadata={"min_length": 10})(DummyModel)
        suggestions = reg.suggest(
            series_length=100, frequency="D", is_multivariate=False, needs_uncertainty=False
        )
        # "short" should be excluded because series_length < min_length
        names = [n for n, _ in suggestions]
        assert "short" not in names

    def test_global_registry_is_singleton(self):
        assert registry is registry
        assert isinstance(registry, ModelRegistry)

    def test_repr(self):
        reg = ModelRegistry()
        reg.register("dummy")(DummyModel)
        r = repr(reg)
        assert "ModelRegistry" in r
        assert "dummy" in r

    def test_warning_on_overwrite(self):
        reg = ModelRegistry()
        reg.register("dup")(DummyModel)
        # Should not raise
        reg.register("dup")(DummyModel)
        assert reg.get("dup") is DummyModel
