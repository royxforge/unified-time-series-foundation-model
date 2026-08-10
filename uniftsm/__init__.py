"""UniTSFM: Unified Time Series Foundation Model Toolkit.

A unified, modular, PyTorch-based toolkit that wraps every major Time Series
Foundation Model (TimesFM, Chronos, MOIRAI, Lag-Llama, TinyTimeMixers) behind
a single ``model.fit(X) → model.predict(h)`` API with automatic model selection,
cross-model ensembling, uncertainty calibration, and rigorous evaluation.

Key features:
    - 5+ TSFMs, 1 unified API
    - Adaptive patch size optimization (differentiable gating)
    - Cross-model uncertainty-weighted ensembling
    - Meta-learning for automatic model selection
    - Comprehensive evaluation (MASE, sMAPE, CRPS, DM test)
    - CLI-first interface
    - LaTeX-ready comparison reports

Example::

    from uniftsm import UniTSFM
    import pandas as pd

    # Load your data
    electricity = pd.read_csv("electricity.csv", index_col=0, parse_dates=True)

    # Auto-select, ensemble, and forecast in 3 lines
    model = UniTSFM(auto_select=True, ensemble=True)
    model.fit(electricity["load"])
    forecast = model.predict(horizon=168, return_quantiles=True)

    # forecast["q_0.5"]["forecast"] - ready to plot
"""

# Importing ``uniftsm.models`` triggers the ``@registry.register()``
# decorators that register every TSFM wrapper with the global registry.
# This MUST happen at package import time so that ``UniTSFM(model_name=...)``,
# auto-selection, and ensembling work out of the box.
from uniftsm import models as _models  # noqa: F401  (registers model plugins)
from uniftsm._uniftsm import UniTSFM
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
    "registry",
    "ModelRegistry",
    "UniTSFMConfig",
    "UniTSFM",
    "UniTSFMError",
    "ModelNotLoadedError",
    "InvalidFrequencyError",
    "InsufficientDataError",
    "PredictionError",
]

# NOTE: `_models` is intentionally excluded from ``__all__``: it is
# imported purely for its registration side effect.

__version__ = "0.1.0"
