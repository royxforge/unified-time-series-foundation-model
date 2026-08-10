"""Cross-model ensembling strategies — averaging, uncertainty-weighted, stacking, and conformal prediction."""

from uniftsm.ensemble.averaging import AverageEnsemble, WeightedAverageEnsemble
from uniftsm.ensemble.base import BaseEnsemble
from uniftsm.ensemble.conformal import ConformalPrediction
from uniftsm.ensemble.stacking import StackingEnsemble
from uniftsm.ensemble.uncertainty_weighted import UncertaintyWeightedEnsemble

__all__ = [
    "BaseEnsemble",
    "AverageEnsemble",
    "WeightedAverageEnsemble",
    "UncertaintyWeightedEnsemble",
    "StackingEnsemble",
    "ConformalPrediction",
]
