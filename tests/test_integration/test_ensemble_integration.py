"""Integration tests: Ensemble strategies working with mock forecasters."""

from __future__ import annotations

import numpy as np

from conftest import MockForecaster


class TestEnsembleWithEvaluation:
    """Ensemble -> metrics evaluation integration."""

    def test_ensemble_predict_then_evaluate(self, simple_series):
        from uniftsm.ensemble.averaging import AverageEnsemble
        from uniftsm.evaluation.metrics import compute_metrics

        horizon = 20
        models = [
            MockForecaster(
                model_name="a", mean_values=np.ones(horizon) * 10, std_values=np.ones(horizon)
            ),
            MockForecaster(
                model_name="b", mean_values=np.ones(horizon) * 11, std_values=np.ones(horizon) * 2
            ),
        ]
        for m in models:
            m.fit(simple_series)

        pred = AverageEnsemble(models).predict(horizon)
        assert "mean" in pred.columns and "std" in pred.columns
        assert len(pred) == horizon

        metrics = compute_metrics(
            y_true=np.ones(horizon) * 10.5,
            y_pred=pred["mean"].values,
            mean=pred["mean"].values,
            std=pred["std"].values,
        )
        assert "MASE" in metrics and "CRPS" in metrics


class TestUncertaintyWeightedIntegration:
    """Uncertainty-weighted ensemble with varied forecasters."""

    def test_predict_and_quantiles(self, simple_series):
        from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble

        horizon = 20
        models = [
            MockForecaster(
                model_name="confident",
                mean_values=np.ones(horizon) * 10,
                std_values=np.ones(horizon) * 0.1,
            ),
            MockForecaster(
                model_name="uncertain",
                mean_values=np.ones(horizon) * 15,
                std_values=np.ones(horizon) * 5.0,
            ),
        ]
        for m in models:
            m.fit(simple_series)

        ensemble = UncertaintyWeightedEnsemble(models, temperature=2.0)
        pred = ensemble.predict(horizon)
        assert np.all(np.isfinite(pred["mean"]))

        result = ensemble.predict(horizon, return_quantiles=True)
        assert isinstance(result, dict) and "q_0.5" in result

    def test_temperature_affects_output(self, simple_series):
        from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble

        horizon = 10
        models = [
            MockForecaster(
                model_name="a",
                mean_values=np.ones(horizon) * 10,
                std_values=np.ones(horizon) * 0.5,
            ),
            MockForecaster(
                model_name="b",
                mean_values=np.ones(horizon) * 12,
                std_values=np.ones(horizon) * 2.0,
            ),
        ]
        for m in models:
            m.fit(simple_series)

        hot = UncertaintyWeightedEnsemble(models, temperature=5.0).predict(horizon)
        cold = UncertaintyWeightedEnsemble(models, temperature=0.1).predict(horizon)
        assert not np.allclose(hot["mean"].values, cold["mean"].values)


class TestStackingEnsembleIntegration:
    """Stacking ensemble with meta-model training."""

    def test_stacking_train_and_predict(self, simple_series):
        from uniftsm.ensemble.stacking import StackingEnsemble

        horizon = 10
        train_len = 80
        y_train = simple_series.iloc[:train_len]

        models = [
            MockForecaster(model_name="a", mean_values=np.linspace(0, 5, horizon)),
            MockForecaster(model_name="b", mean_values=np.linspace(2, 7, horizon)),
        ]
        for m in models:
            m.fit(y_train)

        ensemble = StackingEnsemble(models)
        ensemble.train(y_train=y_train, horizon=horizon)  # Pass pd.Series, not numpy array
        pred = ensemble.predict(horizon)
        assert "mean" in pred.columns
        assert len(pred) == horizon


class TestCrossEnsemble:
    """Different ensemble strategies on same models."""

    def test_consistent_shapes(self, simple_series):
        from uniftsm.ensemble.averaging import AverageEnsemble
        from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble

        horizon = 12
        models = [
            MockForecaster(model_name="a", mean_values=np.sin(np.linspace(0, 2 * np.pi, horizon))),
            MockForecaster(model_name="b", mean_values=np.cos(np.linspace(0, 2 * np.pi, horizon))),
        ]
        for m in models:
            m.fit(simple_series)

        avg = AverageEnsemble(models).predict(horizon)
        uw = UncertaintyWeightedEnsemble(models).predict(horizon)
        assert avg.shape == uw.shape == (horizon, 2)
