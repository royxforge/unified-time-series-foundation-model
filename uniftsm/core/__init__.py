"""Core framework — abstract base classes, plugin registry, configuration, and exceptions."""

from uniftsm.core.base import BaseForecaster
from uniftsm.core.config import UniTSFMConfig
from uniftsm.core.exceptions import (
    InsufficientDataError,
    InvalidFrequencyError,
    ModelNotLoadedError,
    PredictionError,
    UniTSFMError,
)
from uniftsm.core.registry import ModelRegistry, registry

__all__ = [
    "BaseForecaster",
    "ModelRegistry",
    "registry",
    "UniTSFMConfig",
    "UniTSFMError",
    "ModelNotLoadedError",
    "InvalidFrequencyError",
    "InsufficientDataError",
    "PredictionError",
]
