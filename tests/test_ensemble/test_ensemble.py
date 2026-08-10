"""Tests for ensemble base class and strategies."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.core.exceptions import EnsembleError
from uniftsm.ensemble.averaging import AverageEnsemble, WeightedAverageEnsemble
from uniftsm.ensemble.base import BaseEnsemble
from uniftsm.ensemble.conformal import ConformalPrediction
from uniftsm.ensemble.stacking import StackingEnsemble
from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble


class TestBaseEnsemble:
    def test_empty_models_raises(self):
        with pytest.raises(EnsembleError):

            class TestEnsemble(BaseEnsemble):
                def predict(self, *a, **kw): ...

            TestEnsemble([])

    def test_unfitted_models_raises(self, mock_forecasters):
        for m in mock_forecasters:
            m._fitted = False
        with pytest.raises(EnsembleError):

            class TestEnsemble(BaseEnsemble):
                def predict(self, *a, **kw): ...

            TestEnsemble(mock_forecasters)

    def test_predict_interval(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = AverageEnsemble(mock_forecasters)
        lower, upper = ensemble.predict_interval(horizon=10, alpha=0.05)
        assert len(lower) == 10
        assert len(upper) == 10


class TestAverageEnsemble:
    def test_predict_shape(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = AverageEnsemble(mock_forecasters)
        pred = ensemble.predict(horizon=20)
        assert "mean" in pred.columns
        assert "std" in pred.columns
        assert len(pred) == 20

    def test_predict_quantiles(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = AverageEnsemble(mock_forecasters)
        result = ensemble.predict(horizon=20, return_quantiles=True)
        assert isinstance(result, dict)
        assert "q_0.1" in result
        assert "q_0.5" in result
        assert "q_0.9" in result


class TestWeightedAverageEnsemble:
    def test_custom_weights(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = WeightedAverageEnsemble(mock_forecasters, weights=[0.5, 0.3, 0.2])
        pred = ensemble.predict(horizon=20)
        assert len(pred) == 20

    def test_weight_mismatch_raises(self, mock_forecasters):
        with pytest.raises(EnsembleError):
            WeightedAverageEnsemble(mock_forecasters, weights=[0.5, 0.5])

    def test_uniform_weights_default(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = WeightedAverageEnsemble(mock_forecasters)
        assert len(ensemble._weights) == 3
        assert np.allclose(ensemble._weights.sum(), 1.0)


class TestUncertaintyWeightedEnsemble:
    def test_predict_shape(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = UncertaintyWeightedEnsemble(mock_forecasters)
        pred = ensemble.predict(horizon=20)
        assert "mean" in pred.columns
        assert "std" in pred.columns
        assert len(pred) == 20

    def test_temperature_zero(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = UncertaintyWeightedEnsemble(mock_forecasters, temperature=0.0)
        pred = ensemble.predict(horizon=20)
        assert len(pred) == 20

    def test_get_weights(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = UncertaintyWeightedEnsemble(mock_forecasters)
        weights = ensemble.get_weights(horizon=20)
        assert weights.shape == (20, 3)

    def test_predict_with_quantiles(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = UncertaintyWeightedEnsemble(mock_forecasters)
        result = ensemble.predict(horizon=20, return_quantiles=True)
        assert isinstance(result, dict)
        assert "q_0.5" in result


class TestStackingEnsemble:
    def test_train_and_predict(self, mock_forecasters):
        y_train = pd.Series(np.random.randn(50))
        for m in mock_forecasters:
            m.fit(y_train)
        ensemble = StackingEnsemble(mock_forecasters)
        # Only train with horizon <= fit length
        ensemble.train(y_train, horizon=10)
        pred = ensemble.predict(horizon=10)
        assert "mean" in pred.columns
        assert len(pred) == 10

    def test_not_trained_raises(self, mock_forecasters):
        for m in mock_forecasters:
            m.fit(pd.Series(np.random.randn(50)))
        ensemble = StackingEnsemble(mock_forecasters)
        with pytest.raises(EnsembleError, match="not been trained"):
            ensemble.predict(horizon=20)


class TestConformalPrediction:
    def test_calibrate_and_predict(self, mock_forecaster):
        mock_forecaster.fit(pd.Series(np.random.randn(50)))
        conf = ConformalPrediction(mock_forecaster, alpha=0.05)
        y_holdout = pd.Series(np.random.randn(20))
        conf.calibrate(y_holdout, horizon=20)
        lower, upper = conf.predict_interval(horizon=20)
        assert len(lower) == 20
        assert len(upper) == 20

    def test_not_calibrated_raises(self, mock_forecaster):
        mock_forecaster.fit(pd.Series(np.random.randn(50)))
        conf = ConformalPrediction(mock_forecaster)
        with pytest.raises(EnsembleError, match="must be calibrated"):
            conf.predict_interval(horizon=20)

    def test_unfitted_model_raises(self, mock_forecaster):
        mock_forecaster._fitted = False
        with pytest.raises(EnsembleError, match="must be fitted"):
            ConformalPrediction(mock_forecaster)
