"""Property-based tests for ensemble strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import MockForecaster


class TestEnsembleInvariants:
    """Invariants that all ensemble strategies must satisfy."""

    @pytest.fixture
    def models(self, simple_series):
        horizon = 15
        models = [
            MockForecaster(
                model_name="a",
                mean_values=np.ones(horizon) * 10,
                std_values=np.ones(horizon) * 0.5,
            ),
            MockForecaster(
                model_name="b",
                mean_values=np.ones(horizon) * 12,
                std_values=np.ones(horizon) * 1.0,
            ),
        ]
        for m in models:
            m.fit(simple_series)
        return models

    def test_averaging_in_bounds(self, models):
        from uniftsm.ensemble.averaging import AverageEnsemble

        horizon = 12
        ensemble = AverageEnsemble(models)
        pred = ensemble.predict(horizon)

        assert len(pred) == horizon
        assert np.all(np.isfinite(pred["mean"]))
        assert np.all(pred["std"] >= 0)

    def test_uncertainty_weighted_bounds(self, models):
        from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble

        horizon = 12
        ensemble = UncertaintyWeightedEnsemble(models)
        pred = ensemble.predict(horizon)

        assert len(pred) == horizon
        assert np.all(np.isfinite(pred["mean"]))
        assert np.all(pred["std"] >= 0)

    def test_quantiles_maintain_order(self, simple_series):
        from uniftsm.ensemble.averaging import AverageEnsemble

        horizon = 10
        models = [
            MockForecaster(
                model_name="a", mean_values=np.ones(horizon) * 10, std_values=np.ones(horizon)
            ),
            MockForecaster(
                model_name="b",
                mean_values=np.ones(horizon) * 11,
                std_values=np.ones(horizon) * 1.5,
            ),
        ]
        for m in models:
            m.fit(simple_series)

        ensemble = AverageEnsemble(models)
        result = ensemble.predict(horizon, return_quantiles=True)
        assert isinstance(result, dict)
        for q10, q50, q90 in zip(
            result["q_0.1"].values.ravel(),
            result["q_0.5"].values.ravel(),
            result["q_0.9"].values.ravel(),
            strict=True,
        ):
            assert q10 <= q50 <= q90

    def test_single_model_fallback(self, simple_series):
        from uniftsm.ensemble.averaging import AverageEnsemble
        from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble

        horizon = 10
        model = MockForecaster(
            model_name="only", mean_values=np.ones(horizon), std_values=np.ones(horizon)
        )
        model.fit(simple_series)

        for EnsembleCls in [AverageEnsemble, UncertaintyWeightedEnsemble]:
            ensemble = EnsembleCls([model])
            pred = ensemble.predict(horizon)
            assert len(pred) == horizon

    def test_empty_model_list_raises(self):
        from uniftsm.core.exceptions import EnsembleError
        from uniftsm.ensemble.averaging import AverageEnsemble

        with pytest.raises(EnsembleError):
            AverageEnsemble([])

    def test_models_must_be_fitted(self):
        from uniftsm.core.exceptions import EnsembleError
        from uniftsm.ensemble.base import BaseEnsemble

        hor = 10
        m = MockForecaster(model_name="unfitted", mean_values=np.ones(hor))
        with pytest.raises(EnsembleError):
            ensemble_type = type(
                "TestEnsemble", (BaseEnsemble,), {"predict": lambda self, h, **kw: pd.DataFrame()}
            )
            ensemble_type([m])


class TestConformalProperties:
    """Property tests for conformal prediction."""

    def test_conformal_intervals(self, simple_series):
        from uniftsm.ensemble.conformal import ConformalPrediction

        horizon = 10
        model = MockForecaster(
            model_name="a", mean_values=np.ones(horizon), std_values=np.ones(horizon) * 0.5
        )
        model.fit(simple_series)

        calib_series = pd.Series(np.random.normal(0, 0.5, 100))
        conformal = ConformalPrediction(model, alpha=0.1)
        conformal.calibrate(calib_series, horizon=horizon)

        lower, upper = conformal.predict_interval(horizon)
        assert len(lower) == horizon
        assert len(upper) == horizon
        assert (upper.values > lower.values).all()


class TestStackingProperties:
    """Property tests for stacking ensemble."""

    def test_stacking_train_and_predict(self, simple_series):
        from uniftsm.ensemble.stacking import StackingEnsemble

        horizon = 10
        train_len = len(simple_series) // 2
        y_train = simple_series.iloc[:train_len]

        models = [
            MockForecaster(
                model_name="a", mean_values=np.ones(horizon) * 10, std_values=np.ones(horizon)
            ),
            MockForecaster(
                model_name="b", mean_values=np.ones(horizon) * 12, std_values=np.ones(horizon)
            ),
        ]
        for m in models:
            m.fit(y_train)

        ensemble = StackingEnsemble(models)
        ensemble.train(y_train=y_train, horizon=horizon)
        pred = ensemble.predict(horizon)
        assert np.all(np.isfinite(pred["mean"]))
        assert np.all(pred["std"] >= 0)
