"""Tests for FrequencyDetector and FrequencyResampler."""

import numpy as np
import pandas as pd
import pytest

from uniftsm.pipeline.frequency import FrequencyDetector, FrequencyResampler


class TestFrequencyDetector:
    def test_detect_daily(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        s = pd.Series(np.arange(10.0), index=idx)
        freq = FrequencyDetector.detect(s)
        assert freq is not None

    def test_detect_returns_string(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="h")
        s = pd.Series(np.arange(10.0), index=idx)
        freq = FrequencyDetector.detect(s)
        assert isinstance(freq, str)

    def test_detect_none_for_ints(self):
        s = pd.Series(np.arange(10.0))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            FrequencyDetector.detect(s)

    def test_describe(self):
        assert FrequencyDetector.describe("H") == "hourly"
        assert FrequencyDetector.describe("D") == "daily"
        assert FrequencyDetector.describe("W") == "weekly"

    def test_is_supported(self):
        assert isinstance(FrequencyDetector.is_supported("D"), bool)
        assert isinstance(FrequencyDetector.is_supported("XYZ"), bool)

    def test_dataframe_input(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        df = pd.DataFrame({"a": np.arange(10.0)}, index=idx)
        assert FrequencyDetector.detect(df) == "D"

    def test_detect_minutely_multiplier_normalised(self):
        # Regression: ``pd.infer_freq`` returns "10min", which is not a
        # valid ``pd.Timedelta`` unit.  It must be normalised to "10T".
        idx = pd.date_range("2020-01-01", periods=50, freq="10min")
        s = pd.Series(np.arange(50.0), index=idx)
        assert FrequencyDetector.detect(s) == "10T"

    def test_detect_weekly_anchored(self):
        idx = pd.date_range("2020-01-05", periods=12, freq="W-SUN")
        s = pd.Series(np.arange(12.0), index=idx)
        assert FrequencyDetector.detect(s) == "W-SUN"

    def test_normalise_legacy_aliases(self):
        # Lowercase single-letter aliases are deprecated by pandas 2.2+;
        # they must be canonicalised to the uppercase forms.
        assert FrequencyDetector._normalize("min") == "T"
        assert FrequencyDetector._normalize("10min") == "10T"
        assert FrequencyDetector._normalize("w") == "W"
        assert FrequencyDetector._normalize("D") == "D"
        assert FrequencyDetector._normalize("2H") == "2H"


class TestFrequencyResampler:
    def test_downsample_daily_to_weekly(self):
        idx = pd.date_range("2020-01-01", periods=70, freq="D")
        s = pd.Series(np.arange(70.0), index=idx)
        resampler = FrequencyResampler(method="mean")
        result = resampler.resample(s, "W")
        assert len(result) < 70
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_upsample(self):
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        s = pd.Series(np.arange(10.0), index=idx)
        resampler = FrequencyResampler(method="last")
        result = resampler.resample(s, "h", fill_limit=3)
        assert len(result) > 10

    def test_non_datetime_index_raises(self):
        s = pd.Series(np.arange(10.0))
        resampler = FrequencyResampler()
        with pytest.raises(TypeError, match="DatetimeIndex"):
            resampler.resample(s, "D")
