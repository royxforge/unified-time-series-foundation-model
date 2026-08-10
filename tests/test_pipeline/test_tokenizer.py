"""Tests for QuantileTokenizer."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.pipeline.tokenizer import QuantileTokenizer


class TestQuantileTokenizer:
    def test_uniform_roundtrip(self, simple_series):
        tok = QuantileTokenizer(num_bins=100, strategy="uniform")
        tokens = tok.fit_transform(simple_series)
        assert tokens.dtype == np.int32
        assert tokens.min() >= 0
        assert tokens.max() < tok.vocab_size
        restored = tok.inverse_transform(tokens)
        assert len(restored) == len(simple_series)

    def test_quantile_roundtrip(self, simple_series):
        tok = QuantileTokenizer(num_bins=100, strategy="quantile")
        tokens = tok.fit_transform(simple_series)
        restored = tok.inverse_transform(tokens)
        assert len(restored) == len(simple_series)

    def test_log_strategy_positive_only(self):
        series = pd.Series(np.abs(np.random.randn(200)) + 0.1)
        tok = QuantileTokenizer(num_bins=100, strategy="log")
        tokens = tok.fit_transform(series)
        assert tokens.min() >= 0

    def test_log_with_zeros(self):
        series = pd.Series(np.array([0.0, 0.0, 0.0]))
        tok = QuantileTokenizer(num_bins=10, strategy="log")
        tok.fit(series)  # Should not crash
        assert tok.is_fitted

    def test_not_fitted_raises(self):
        tok = QuantileTokenizer()
        with pytest.raises(RuntimeError, match="not been fitted"):
            tok.transform(np.array([1.0]))

    def test_vocab_size(self, simple_series):
        tok = QuantileTokenizer(num_bins=50, strategy="quantile")
        assert tok.vocab_size == 50  # Before fitting, returns num_bins
        tok.fit(simple_series)
        assert tok.vocab_size <= 50

    def test_constant_series(self):
        series = pd.Series(np.ones(100))
        tok = QuantileTokenizer(num_bins=10, strategy="uniform")
        tokens = tok.fit_transform(series)
        assert len(np.unique(tokens)) <= 2  # Should map to ~1-2 bins

    def test_numpy_input(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tok = QuantileTokenizer(num_bins=10, strategy="uniform")
        tokens = tok.fit_transform(arr)
        assert len(tokens) == 5

    def test_invalid_strategy(self):
        with pytest.raises(ValueError):
            QuantileTokenizer(strategy="invalid").fit(np.array([1.0, 2.0]))

    def test_repr(self):
        tok = QuantileTokenizer(num_bins=4096, strategy="quantile")
        r = repr(tok)
        assert "4096" in r
        assert "quantile" in r
