"""Tests for UniTSFM custom exceptions."""

import pytest

from uniftsm.core.exceptions import (
    BacktestingError,
    ConfigurationError,
    DatasetNotFoundError,
    EnsembleError,
    InsufficientDataError,
    InvalidFrequencyError,
    ModelNotLoadedError,
    PredictionError,
    UniTSFMError,
)


class TestUniTSFMExceptions:
    def test_all_inherit_from_base(self):
        assert issubclass(ModelNotLoadedError, UniTSFMError)
        assert issubclass(InvalidFrequencyError, UniTSFMError)
        assert issubclass(InsufficientDataError, UniTSFMError)
        assert issubclass(PredictionError, UniTSFMError)
        assert issubclass(ConfigurationError, UniTSFMError)
        assert issubclass(BacktestingError, UniTSFMError)
        assert issubclass(DatasetNotFoundError, UniTSFMError)
        assert issubclass(EnsembleError, UniTSFMError)

    def test_model_not_loaded(self):
        e = ModelNotLoadedError("chronos")
        assert "chronos" in str(e)
        assert isinstance(e.details, dict)

    def test_invalid_frequency(self):
        e = InvalidFrequencyError("XYZ")
        assert "XYZ" in str(e)

        e_with_list = InvalidFrequencyError("XYZ", valid_freqs=["H", "D", "W"])
        assert "H" in str(e_with_list)
        assert "D" in str(e_with_list)

    def test_insufficient_data(self):
        e = InsufficientDataError(required=10, actual=3)
        assert "10" in str(e)
        assert "3" in str(e)

        e_with_model = InsufficientDataError(required=10, actual=3, model_name="test")
        assert "test" in str(e_with_model)

    def test_prediction_error(self):
        e = PredictionError(model_name="test", reason="NaN values")
        assert "test" in str(e)
        assert "NaN" in str(e)

    def test_configuration_error(self):
        e = ConfigurationError(field="device", reason="invalid")
        assert "device" in str(e)
        assert "invalid" in str(e)

    def test_backtesting_error(self):
        e = BacktestingError(strategy="rolling", reason="window too small")
        assert "rolling" in str(e)
        assert "window too small" in str(e)

    def test_dataset_not_found(self):
        e = DatasetNotFoundError("monash_energy")
        assert "monash_energy" in str(e)

        e_with_source = DatasetNotFoundError("test", source="huggingface")
        assert "huggingface" in str(e_with_source)

    def test_ensemble_error(self):
        e = EnsembleError(strategy="weighted", reason="no models")
        assert "weighted" in str(e)
        assert "no models" in str(e)

    def test_catch_all_with_base(self):
        try:
            raise ModelNotLoadedError("test")
        except UniTSFMError:
            pass  # Expected
        else:
            pytest.fail("ModelNotLoadedError not caught by UniTSFMError")
