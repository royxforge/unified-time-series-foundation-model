"""Benchmark dataset loaders — Monash, LOTSA, M3/M4, and synthetic data generators."""

from uniftsm.datasets.m3_m4 import M3M4DatasetLoader
from uniftsm.datasets.monash import MonashDatasetLoader
from uniftsm.datasets.synthetic import SyntheticDatasetGenerator

__all__ = [
    "MonashDatasetLoader",
    "M3M4DatasetLoader",
    "SyntheticDatasetGenerator",
]
