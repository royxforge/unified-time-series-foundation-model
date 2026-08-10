"""Custom exceptions for the UniTSFM framework.

All framework-specific exceptions inherit from UniTSFMError,
allowing users to catch all library errors with a single base class.
"""

from typing import Any


class UniTSFMError(Exception):
    """Base exception for all UniTSFM errors."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.details = details or {}
        super().__init__(message)


class ModelNotLoadedError(UniTSFMError):
    """Raised when trying to predict before loading model weights."""

    def __init__(self, model_name: str) -> None:
        super().__init__(
            f"Model '{model_name}' has not been loaded. Call fit() first, "
            f"or ensure the model weights are properly initialized."
        )


class InvalidFrequencyError(UniTSFMError):
    """Raised when the frequency string is unrecognised or unsupported."""

    def __init__(self, freq: str, valid_freqs: list[str] | None = None) -> None:
        msg = f"Unrecognised frequency: '{freq}'."
        if valid_freqs:
            msg += f" Valid options: {', '.join(sorted(valid_freqs))}"
        super().__init__(msg)


class InsufficientDataError(UniTSFMError):
    """Raised when the input series is too short for the requested model."""

    def __init__(self, required: int, actual: int, model_name: str = "") -> None:
        model_tag = f" for model '{model_name}'" if model_name else ""
        super().__init__(
            f"Insufficient data{model_tag}: "
            f"required at least {required} observations, got {actual}."
        )


class PredictionError(UniTSFMError):
    """Raised when forecasting fails unexpectedly."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(f"Prediction failed for model '{model_name}': {reason}")


class ConfigurationError(UniTSFMError):
    """Raised when the global configuration is invalid."""

    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"Configuration error in '{field}': {reason}")


class BacktestingError(UniTSFMError):
    """Raised when a backtesting strategy encounters an error."""

    def __init__(self, strategy: str, reason: str) -> None:
        super().__init__(f"Backtesting strategy '{strategy}' failed: {reason}")


class DatasetNotFoundError(UniTSFMError):
    """Raised when a requested dataset cannot be found or downloaded."""

    def __init__(self, dataset_name: str, source: str = "") -> None:
        msg = f"Dataset '{dataset_name}' not found."
        if source:
            msg += f" Attempted source: {source}"
        super().__init__(msg)


class EnsembleError(UniTSFMError):
    """Raised when an ensemble operation fails."""

    def __init__(self, strategy: str, reason: str) -> None:
        super().__init__(f"Ensemble strategy '{strategy}' failed: {reason}")
