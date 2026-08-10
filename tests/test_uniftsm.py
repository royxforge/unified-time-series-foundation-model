"""Tests for the UniTSFM convenience class."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


class TestUniTSFM:
    def test_init_defaults(self):
        from uniftsm import UniTSFM

        model = UniTSFM()
        assert model.ensemble is False
        assert model.auto_select is False
        # model_name defaults to None (chronos fallback happens in fit())
        assert model.model_name is None

    def test_init_with_ensemble(self):
        from uniftsm import UniTSFM

        model = UniTSFM(ensemble=True)
        assert model.ensemble is True

    def test_init_with_auto_select(self):
        from uniftsm import UniTSFM

        model = UniTSFM(auto_select=True)
        assert model.auto_select is True

    def test_init_with_custom_model(self):
        from uniftsm import UniTSFM

        model = UniTSFM(model_name="timesfm")
        assert model.model_name == "timesfm"

    def test_fit_single_model(self, simple_series):
        from uniftsm import UniTSFM

        model = UniTSFM(model_name="chronos")
        model.fit(simple_series)
        assert model._fitted

    def test_fit_and_predict_single_model(self, simple_series):
        from uniftsm import UniTSFM

        model = UniTSFM(model_name="chronos")
        model.fit(simple_series)
        with patch.object(model._forecaster, "predict") as mock_predict:
            mock_predict.return_value = pd.DataFrame({"mean": np.zeros(10), "std": np.ones(10)})
            pred = model.predict(horizon=10)
            assert "mean" in pred.columns
            assert "std" in pred.columns

    def test_get_params(self):
        from uniftsm import UniTSFM

        model = UniTSFM(model_name="chronos", ensemble=False, auto_select=False)
        params = model.get_params()
        assert params["model_name"] == "chronos"

    def test_repr(self):
        from uniftsm import UniTSFM

        model = UniTSFM()
        r = repr(model)
        assert "UniTSFM" in r

    def test_is_probabilistic(self, simple_series):
        from uniftsm import UniTSFM

        model = UniTSFM(model_name="chronos")
        model.fit(simple_series)
        assert isinstance(model.is_probabilistic, bool)

    def test_short_series_raises(self):
        from uniftsm import UniTSFM
        from uniftsm.core.exceptions import InsufficientDataError

        model = UniTSFM(model_name="chronos")
        short = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
        with pytest.raises(InsufficientDataError):
            model.fit(short)
