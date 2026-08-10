"""Tests for backtesting strategies."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.evaluation.backtesting import (
    BacktestingEngine,
    ExpandingWindowSplit,
    RollingWindowSplit,
    TimeSeriesCV,
    TrainTestSplit,
)


class TestRollingWindowSplit:
    def test_split_generates_windows(self):
        splitter = RollingWindowSplit(train_size=50, test_size=10, step_size=10)
        splits = list(splitter.split(200))
        assert len(splits) >= 1
        for split in splits:
            assert split.test_end - split.test_start == 10
            assert split.train_end - split.train_start == 50

    def test_max_windows_via_step(self):
        splitter = RollingWindowSplit(train_size=50, test_size=10, step_size=50)
        splits = list(splitter.split(200))
        assert len(splits) >= 1

    def test_repr(self):
        splitter = RollingWindowSplit(train_size=50, test_size=10)
        r = repr(splitter)
        assert "RollingWindowSplit" in r


class TestExpandingWindowSplit:
    def test_split_generates_windows(self):
        splitter = ExpandingWindowSplit(min_train_size=30, test_size=10, step_size=10)
        splits = list(splitter.split(200))
        assert len(splits) >= 1
        for split in splits:
            assert split.test_end - split.test_start == 10

    def test_train_length_increases(self):
        splitter = ExpandingWindowSplit(min_train_size=30, test_size=10, step_size=10)
        splits = list(splitter.split(200))
        train_ends = [s.train_end for s in splits]
        assert all(train_ends[i] <= train_ends[i + 1] for i in range(len(train_ends) - 1))

    def test_repr(self):
        splitter = ExpandingWindowSplit(min_train_size=30, test_size=10)
        r = repr(splitter)
        assert "ExpandingWindowSplit" in r


class TestTimeSeriesCV:
    def test_split_generates_folds(self):
        splitter = TimeSeriesCV(n_splits=5, test_size=10)
        splits = list(splitter.split(500))
        assert len(splits) == 5

    def test_non_overlapping(self):
        splitter = TimeSeriesCV(n_splits=5, test_size=10)
        splits = list(splitter.split(500))
        for split in splits:
            assert split.test_end - split.test_start == 10

    def test_insufficient_data_raises(self):
        splitter = TimeSeriesCV(n_splits=5, test_size=10)
        with pytest.raises(ValueError, match="at least"):
            list(splitter.split(50))

    def test_repr(self):
        splitter = TimeSeriesCV(n_splits=5, test_size=10)
        r = repr(splitter)
        assert "TimeSeriesCV" in r


class TestBacktestingEngine:
    def test_run_with_mock(self, mock_forecaster):
        y = pd.Series(np.random.randn(200))
        mock_forecaster.fit(y)
        splitter = RollingWindowSplit(train_size=50, test_size=10, step_size=50)
        engine = BacktestingEngine(splitter)
        results = engine.run(y, mock_forecaster)
        assert len(results) > 0
        assert "y_pred_mean" in results[0]
        assert "y_test" in results[0]

    def test_run_multiple(self, mock_forecasters):
        y = pd.Series(np.random.randn(200))
        for m in mock_forecasters:
            m.fit(y)
        models = {m.model_name: m for m in mock_forecasters}
        splitter = RollingWindowSplit(train_size=50, test_size=10, step_size=50)
        engine = BacktestingEngine(splitter)
        results = engine.run_multiple(y, models)
        assert len(results) == 3
        for name in results:
            assert len(results[name]) > 0

    def test_train_test_split_dataclass(self):
        split = TrainTestSplit(
            train_start=0, train_end=10, test_start=10, test_end=20, split_index=0
        )
        assert split.train_start == 0
        assert split.train_end == 10
        assert split.test_start == 10
        assert split.test_end == 20
        assert split.split_index == 0
