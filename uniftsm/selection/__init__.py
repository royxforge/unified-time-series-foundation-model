"""Automatic model selection — meta-feature extraction, meta-classifier, and precomputed benchmarks."""

from uniftsm.selection.meta_features import MetaFeatureExtractor
from uniftsm.selection.predictor import ModelSelector

__all__ = [
    "MetaFeatureExtractor",
    "ModelSelector",
]
