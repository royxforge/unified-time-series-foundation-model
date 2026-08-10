"""TSFM model wrappers — each registered as a plugin via @registry.register()."""

from uniftsm.models._base_impl import BaseModelImpl
from uniftsm.models.chronos import ChronosForecaster
from uniftsm.models.lag_llama import LagLlamaForecaster
from uniftsm.models.moirai import MoiraiForecaster
from uniftsm.models.timesfm import TimesFMForecaster
from uniftsm.models.ttm import TTMForecaster

__all__ = [
    "ChronosForecaster",
    "TimesFMForecaster",
    "MoiraiForecaster",
    "LagLlamaForecaster",
    "TTMForecaster",
    "BaseModelImpl",
]
