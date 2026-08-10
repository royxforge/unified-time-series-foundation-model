"""Backtesting engines for time series cross-validation.

Provides rolling-window, expanding-window, and custom cross-validation
splitters specifically designed for time series evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class TrainTestSplit:
    """A single train/test split for time series evaluation."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    split_index: int


class BaseSplitter(ABC):
    """Abstract base for time series split strategies."""

    @abstractmethod
    def split(
        self,
        n_observations: int,
    ) -> Generator[TrainTestSplit, None, None]:
        """Yield train/test split indices."""

    @abstractmethod
    def __repr__(self) -> str: ...


class RollingWindowSplit(BaseSplitter):
    """Rolling window cross-validation.

    The training window has a fixed size and slides forward.

    Parameters
    ----------
    train_size:
        Number of observations in each training window.
    test_size:
        Number of observations in each test window.
    step_size:
        Number of steps to slide the window forward.  Defaults to
        ``test_size`` (non-overlapping).
    """

    def __init__(
        self,
        train_size: int,
        test_size: int,
        step_size: int | None = None,
    ) -> None:
        if train_size < 1:
            raise ValueError("train_size must be >= 1")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size

    def split(
        self,
        n_observations: int,
    ) -> Generator[TrainTestSplit, None, None]:
        split_idx = 0
        start = 0
        while start + self.train_size + self.test_size <= n_observations:
            train_end = start + self.train_size
            test_start = train_end
            test_end = test_start + self.test_size
            yield TrainTestSplit(
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                split_index=split_idx,
            )
            start += self.step_size
            split_idx += 1

    def __repr__(self) -> str:
        return (
            f"RollingWindowSplit(train_size={self.train_size}, "
            f"test_size={self.test_size}, step={self.step_size})"
        )


class ExpandingWindowSplit(BaseSplitter):
    """Expanding window cross-validation.

    The training window grows with each split — all past data is used
    up to the test start.

    Parameters
    ----------
    min_train_size:
        Minimum number of observations required in the first training set.
    test_size:
        Number of observations in each test window.
    step_size:
        Number of steps between test windows.  Defaults to ``test_size``.
    """

    def __init__(
        self,
        min_train_size: int,
        test_size: int,
        step_size: int | None = None,
    ) -> None:
        if min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.step_size = step_size or test_size

    def split(
        self,
        n_observations: int,
    ) -> Generator[TrainTestSplit, None, None]:
        split_idx = 0
        train_end = self.min_train_size
        while train_end + self.test_size <= n_observations:
            test_start = train_end
            test_end = test_start + self.test_size
            yield TrainTestSplit(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                split_index=split_idx,
            )
            train_end += self.step_size
            split_idx += 1

    def __repr__(self) -> str:
        return (
            f"ExpandingWindowSplit(min_train_size={self.min_train_size}, "
            f"test_size={self.test_size}, step={self.step_size})"
        )


class TimeSeriesCV(BaseSplitter):
    """Time-series cross-validation with configurable gap.

    Parameters
    ----------
    n_splits:
        Number of folds.
    test_size:
        Number of test observations per fold.
    gap:
        Gap between train and test sets (to prevent leakage).
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 24,
        gap: int = 0,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if test_size < 1:
            raise ValueError("test_size must be >= 1")
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        n_observations: int,
    ) -> Generator[TrainTestSplit, None, None]:
        total_needed = (self.test_size + self.gap) * self.n_splits + self.n_splits
        if n_observations < total_needed:
            raise ValueError(
                f"Need at least {total_needed} observations for "
                f"{self.n_splits} splits, got {n_observations}. "
                f"Reduce n_splits or test_size."
            )

        for i in range(self.n_splits):
            test_end = n_observations - i * self.test_size
            test_start = test_end - self.test_size
            train_end = test_start - self.gap
            yield TrainTestSplit(
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                split_index=i,
            )

    def __repr__(self) -> str:
        return (
            f"TimeSeriesCV(n_splits={self.n_splits}, "
            f"test_size={self.test_size}, gap={self.gap})"
        )


class BacktestingEngine:
    """Run backtesting for one or more models.

    Parameters
    ----------
    splitter:
        A :class:`BaseSplitter` instance defining the CV scheme.
    """

    def __init__(self, splitter: BaseSplitter) -> None:
        self.splitter = splitter

    def run(
        self,
        y: pd.Series,
        model,
        **predict_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run backtesting for a single model.

        Args:
            y: Full time series.
            model: A fitted :class:`BaseForecaster` instance.
            **predict_kwargs: Keyword arguments for ``model.predict()``.

        Returns:
            List of result dicts, one per split, containing
            ``split_index``, ``train``, ``test`` segments, and
            ``predictions``.
        """
        results = []
        for split in self.splitter.split(len(y)):
            y_train = y.iloc[split.train_start : split.train_end]
            y_test = y.iloc[split.test_start : split.test_end]

            model.fit(y_train)
            horizon = len(y_test)
            pred = model.predict(horizon, **predict_kwargs)

            results.append(
                {
                    "split_index": split.split_index,
                    "train_start": split.train_start,
                    "train_end": split.train_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_pred_mean": pred["mean"].values,
                    "y_pred_std": pred["std"].values,
                }
            )
        return results

    def run_multiple(
        self,
        y: pd.Series,
        models: dict[str, Any],
        **predict_kwargs: Any,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run backtesting for multiple models.

        Args:
            y: Full time series.
            models: Dict mapping model name to fitted forecaster.
            **predict_kwargs: Keyword args for ``model.predict()``.

        Returns:
            Dict mapping model name to list of split results.
        """
        return {name: self.run(y, model, **predict_kwargs) for name, model in models.items()}
