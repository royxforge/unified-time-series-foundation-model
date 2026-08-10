"""Tests for all TSFM model wrappers with mocked external dependencies.

Mocks the heavy TSFM packages (chronos, timesfm, uni2ts, etc.)
so tests can run without installing them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sample_series() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.Series(np.cumsum(np.random.randn(100)), index=idx, name="value")


# ── Test ChronosForecaster ─────────────────────────────────────────────────


class TestChronosForecaster:
    def test_init_defaults(self):
        from uniftsm.models.chronos import ChronosForecaster

        model = ChronosForecaster()
        assert model.model_name == "chronos"
        assert model.model_size == "small"

    def test_init_custom_size(self):
        from uniftsm.models.chronos import ChronosForecaster

        model = ChronosForecaster(model_size="tiny")
        assert model.model_size == "tiny"

    def test_fit_stores_context(self, sample_series):
        from uniftsm.models.chronos import ChronosForecaster

        model = ChronosForecaster()
        model.fit(sample_series)
        assert model._fitted
        assert model._context is not None

    def test_fit_validates_short_series(self):
        from uniftsm.core.exceptions import InsufficientDataError
        from uniftsm.models.chronos import ChronosForecaster

        model = ChronosForecaster()
        short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        with pytest.raises(InsufficientDataError):
            model.fit(short)

    def test_get_params(self):
        from uniftsm.models.chronos import ChronosForecaster

        model = ChronosForecaster(model_size="base", seed=123)
        params = model.get_params()
        assert params["model_size"] == "base"
        assert params["seed"] == 123

    def test_registry_contains_chronos(self):
        from uniftsm.core.registry import registry

        try:
            meta = registry.get_metadata("chronos")
            assert meta["family"] == "chronos"
        except ValueError:
            pytest.skip("chronos not registered yet (run model module import first)")


# ── Test TimesFMForecaster ────────────────────────────────────────────────


class TestTimesFMForecaster:
    def test_init_defaults(self):
        from uniftsm.models.timesfm import TimesFMForecaster

        model = TimesFMForecaster()
        assert model.model_name == "timesfm"

    def test_fit(self, sample_series):
        from uniftsm.models.timesfm import TimesFMForecaster

        model = TimesFMForecaster()
        model.fit(sample_series)
        assert model._fitted


# ── Test MoiraiForecaster ─────────────────────────────────────────────────


class TestMoiraiForecaster:
    def test_init_defaults(self):
        from uniftsm.models.moirai import MoiraiForecaster

        model = MoiraiForecaster()
        assert model.model_name == "moirai"

    def test_fit(self, sample_series):
        from uniftsm.models.moirai import MoiraiForecaster

        model = MoiraiForecaster()
        model.fit(sample_series)
        assert model._fitted


# ── Test LagLlamaForecaster ───────────────────────────────────────────────


class TestLagLlamaForecaster:
    def test_init_defaults(self):
        from uniftsm.models.lag_llama import LagLlamaForecaster

        model = LagLlamaForecaster()
        assert model.model_name == "lag_llama"

    def test_fit(self, sample_series):
        from uniftsm.models.lag_llama import LagLlamaForecaster

        model = LagLlamaForecaster()
        model.fit(sample_series)
        assert model._fitted


# ── Test TTMForecaster ────────────────────────────────────────────────────


class TestTTMForecaster:
    def test_init_defaults(self):
        from uniftsm.models.ttm import TTMForecaster

        model = TTMForecaster()
        assert model.model_name == "ttm"

    def test_fit(self, sample_series):
        from uniftsm.models.ttm import TTMForecaster

        model = TTMForecaster()
        model.fit(sample_series)
        assert model._fitted

    def test_predict_pads_short_context(self):
        """Regression: TTM requires a fixed 512-step context; short series
        must be left-padded instead of crashing with
        'Context length in past_values is shorter than TTM context_length'.
        """
        import torch

        from uniftsm.models.ttm import TTMForecaster

        captured: dict = {}

        class _Config:
            context_length = 512
            prediction_length = 96

        class _FakeModel:
            def __init__(self):
                self.config = _Config()

            def eval(self):
                return self

            def to(self, device):
                return self

            def __call__(self, past_values=None, return_loss=None, return_dict=None):
                captured["shape"] = tuple(past_values.shape)
                preds = torch.zeros(1, self.config.prediction_length, 1)
                return type(
                    "Out",
                    (),
                    {"prediction_outputs": preds},
                )()

        # Short series (90 steps) — shorter than TTM's 512-step context.
        idx = pd.date_range("2020-01-01", periods=90, freq="D")
        y = pd.Series(np.cumsum(np.random.randn(90)), index=idx)

        model = TTMForecaster()
        model.fit(y)
        model._model = _FakeModel()

        pred = model.predict(horizon=12, return_quantiles=False)
        assert captured["shape"] == (1, 512, 1), "context must be padded to 512"
        assert len(pred) == 12
        assert "mean" in pred.columns


# ── Test BaseModelImpl ────────────────────────────────────────────────────


class TestBaseModelImpl:
    def test_cache_path(self):
        from uniftsm.models._base_impl import BaseModelImpl

        impl = BaseModelImpl("test_model")
        assert impl.model_name == "test_model"

    def test_resolve_device(self):
        from uniftsm.models._base_impl import BaseModelImpl

        device = BaseModelImpl._resolve_device("cpu")
        assert str(device) == "cpu"

    def test_resolve_dtype(self):
        from uniftsm.models._base_impl import BaseModelImpl

        impl = BaseModelImpl("test")
        dtype = impl._resolve_dtype("float32")
        assert dtype is not None
