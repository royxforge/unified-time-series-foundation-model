"""Integration tests: Pipeline components working together."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch


class TestScalerPatcherIntegration:
    """Scaler -> Patcher end-to-end with even-length input."""

    def test_scale_then_patch_produces_output(self, torch_device):
        from uniftsm.pipeline.patcher import AdaptivePatcher
        from uniftsm.pipeline.scaler import SeriesScaler

        # Use length divisible by all patch sizes (LCM of 8,16,32,64 = 64)
        idx = pd.date_range("2020-01-01", periods=64, freq="D")
        series = pd.Series(np.cumsum(np.random.randn(64)), index=idx)

        scaler = SeriesScaler("standard")
        scaled = scaler.fit_transform(series)

        tensor = torch.from_numpy(scaled).float().unsqueeze(0).to(torch_device)
        patcher = AdaptivePatcher(input_length=64, hidden_dim=16, patch_sizes=[8, 16, 32]).to(
            torch_device
        )
        output, weights = patcher(tensor)

        assert output.ndim == 3
        assert output.shape[0] == 1
        assert output.shape[2] == 16
        assert weights.ndim == 2
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, device=torch_device))

    def test_minmax_roundtrip(self, simple_series):
        from uniftsm.pipeline.scaler import SeriesScaler

        scaler = SeriesScaler("minmax")
        scaled = scaler.fit_transform(simple_series)
        assert np.min(scaled) >= 0.0 - 1e-6
        assert np.max(scaled) <= 1.0 + 1e-6
        np.testing.assert_allclose(
            scaler.inverse_transform(scaled), simple_series.values, rtol=1e-5
        )

    def test_robust_roundtrip(self, simple_series):
        from uniftsm.pipeline.scaler import SeriesScaler

        scaler = SeriesScaler("robust")
        scaled = scaler.fit_transform(simple_series)
        np.testing.assert_allclose(
            scaler.inverse_transform(scaled), simple_series.values, rtol=1e-5
        )


class TestTokenizerScalerFlow:
    """Scaler -> Tokenizer integration."""

    def test_tokenizer_quantizes(self, simple_series):
        from uniftsm.pipeline.scaler import SeriesScaler
        from uniftsm.pipeline.tokenizer import QuantileTokenizer

        scaler = SeriesScaler("standard")
        scaled = scaler.fit_transform(simple_series)

        tokenizer = QuantileTokenizer(num_bins=256, strategy="quantile")
        tokens = tokenizer.fit_transform(scaled)

        reconstructed = scaler.inverse_transform(tokenizer.inverse_transform(tokens))
        relative_error = np.mean(np.abs(reconstructed - simple_series.values)) / (
            np.mean(np.abs(simple_series.values)) + 1e-8
        )
        assert relative_error < 0.5

    def test_tokenizer_rejects_unfitted(self, simple_series):
        from uniftsm.pipeline.tokenizer import QuantileTokenizer

        with pytest.raises(RuntimeError, match="not been fitted"):
            QuantileTokenizer().transform(simple_series)


class TestImputeThenScale:
    """Missing value imputation -> scaling integration."""

    def test_impute_then_scale(self, series_with_nans):
        from uniftsm.pipeline.missing import MissingValueImputer
        from uniftsm.pipeline.scaler import SeriesScaler

        imputer = MissingValueImputer("linear", max_gap=5)
        cleaned = imputer.impute(series_with_nans)
        assert not cleaned.isna().any()

        scaled = SeriesScaler("robust").fit_transform(cleaned)
        assert np.all(np.isfinite(scaled))

    def test_all_strategies_finite(self, series_with_nans):
        from uniftsm.pipeline.missing import MissingValueImputer
        from uniftsm.pipeline.scaler import SeriesScaler

        for strategy in ["forward_fill", "backward_fill", "linear", "mean", "median", "zero"]:
            cleaned = MissingValueImputer(strategy, max_gap=5, warn_on_fill=False).impute(
                series_with_nans
            )
            assert not cleaned.isna().any(), f"{strategy}"
            assert np.all(
                np.isfinite(SeriesScaler("standard").fit_transform(cleaned))
            ), f"{strategy}"


class TestFrequencyPipeline:
    """Frequency detection -> scaling."""

    def test_daily_standard_scaler(self):
        from uniftsm.pipeline.scaler import SeriesScaler

        idx = pd.date_range("2020-01-01", periods=100, freq="D")
        series = pd.Series(np.cumsum(np.random.randn(100)), index=idx)
        scaled = SeriesScaler("standard").fit_transform(series)
        assert abs(float(np.mean(scaled))) < 1e-10

    def test_detect_fails_on_int_index(self):
        from uniftsm.pipeline.frequency import FrequencyDetector

        with pytest.raises(TypeError, match="DatetimeIndex"):
            FrequencyDetector.detect(pd.Series(np.arange(10.0)))


class TestFullPipelineRoundtrip:
    """Complete pipeline: impute -> scale -> inverse."""

    def test_impute_scale_inverse(self, series_with_nans):
        from uniftsm.pipeline.missing import MissingValueImputer
        from uniftsm.pipeline.scaler import SeriesScaler

        cleaned = MissingValueImputer("linear", max_gap=10, warn_on_fill=False).impute(
            series_with_nans
        )
        scaler = SeriesScaler("robust")
        scaled = scaler.fit_transform(cleaned)
        np.testing.assert_allclose(scaler.inverse_transform(scaled), cleaned.values, rtol=1e-5)

    def test_all_zero_series(self):
        from uniftsm.pipeline.scaler import SeriesScaler
        from uniftsm.pipeline.tokenizer import QuantileTokenizer

        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        zeros = pd.Series(np.zeros(50), index=idx)
        assert np.allclose(SeriesScaler("standard").fit_transform(zeros), 0.0)
        tokens = QuantileTokenizer(num_bins=16, strategy="uniform").fit_transform(zeros)
        assert np.all(tokens == 0)
