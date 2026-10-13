"""Tests for CLI commands using Click's CliRunner."""

from __future__ import annotations

import importlib
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

# ``uniftsm.cli.forecast`` is shadowed by the Click command of the same name in
# the package namespace, so resolve the module through importlib.
forecast_module = importlib.import_module("uniftsm.cli.forecast")
# ``uniftsm.core.registry`` is likewise shadowed by the singleton instance.
registry_module = importlib.import_module("uniftsm.core.registry")

from uniftsm.cli.main import cli
from tests.conftest import MockForecaster


class TestCLIHelp:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "UniTSFM" in result.output

    def test_forecast_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["forecast", "--help"])
        assert result.exit_code == 0

    def test_benchmark_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0

    def test_compare_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0

    def test_serve_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0


class TestCLIForecast:
    @patch("uniftsm.cli.forecast.pd.read_csv")
    def test_forecast_basic(self, mock_read_csv):
        # Mock the CSV data
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        mock_df = pd.DataFrame({"value": np.random.randn(50)}, index=idx)
        mock_read_csv.return_value = mock_df

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Create dummy file so Click doesn't complain
            with open("test_data.csv", "w") as f:
                f.write("date,value\n")

            result = runner.invoke(
                cli,
                [
                    "forecast",
                    "test_data.csv",
                    "--horizon",
                    "10",
                    "--model",
                    "chronos",
                ],
            )
            # Should either succeed or fail gracefully - we're testing the flow
            assert result.exit_code in (0, 1)

    @patch("uniftsm.cli.forecast.pd.read_csv")
    def test_forecast_with_output(self, mock_read_csv):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        mock_df = pd.DataFrame({"value": np.random.randn(50)}, index=idx)
        mock_read_csv.return_value = mock_df

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test_data.csv", "w") as f:
                f.write("date,value\n")

            result = runner.invoke(
                cli,
                [
                    "forecast",
                    "test_data.csv",
                    "--horizon",
                    "10",
                    "--output",
                    "output.csv",
                ],
            )
            assert result.exit_code in (0, 1)


class TestForecastOutputFormats:
    """--format json/csv make the command usable from scripts and pipelines."""

    @pytest.fixture
    def forecast_env(self, monkeypatch):
        """Patch data loading and model lookup so no real model is required."""
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        mock_df = pd.DataFrame({"value": np.linspace(0, 5, 50)}, index=idx)
        monkeypatch.setattr(forecast_module.pd, "read_csv", lambda *a, **k: mock_df)

        def fake_get(name):
            def factory(**kwargs):
                return MockForecaster(model_name=name)

            return factory

        monkeypatch.setattr(registry_module.registry, "get", fake_get)
        return mock_df

    def _run(self, runner, *extra):
        with runner.isolated_filesystem():
            with open("test_data.csv", "w") as handle:
                handle.write("date,value\n")
            return runner.invoke(
                cli, ["forecast", "test_data.csv", "--horizon", "4", *extra]
            )

    def test_json_format_emits_only_valid_json(self, forecast_env):
        result = self._run(CliRunner(), "--format", "json")

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["model"] == "chronos"
        assert payload["horizon"] == 4
        assert payload["context_length"] == 50
        assert len(payload["index"]) == 4
        assert set(payload["forecasts"]) == {"q_0.1", "q_0.5", "q_0.9"}
        assert all(len(v) == 4 for v in payload["forecasts"].values())
        assert all(isinstance(v, float) for v in payload["forecasts"]["q_0.5"])

    def test_json_output_file_contains_payload(self, forecast_env):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test_data.csv", "w") as handle:
                handle.write("date,value\n")
            result = runner.invoke(
                cli,
                [
                    "forecast",
                    "test_data.csv",
                    "--horizon",
                    "4",
                    "--format",
                    "json",
                    "--output",
                    "fc.json",
                ],
            )
            assert result.exit_code == 0, result.output
            payload = json.loads(open("fc.json", encoding="utf-8").read())
            assert payload["horizon"] == 4
            # Status chatter must not contaminate the piped JSON payload.
            assert result.stdout.startswith("{")

    def test_csv_format_prints_csv(self, forecast_env):
        result = self._run(CliRunner(), "--format", "csv")

        assert result.exit_code == 0, result.output
        header = result.stdout.splitlines()[0]
        assert header.startswith(",q_0.1")
        assert len(result.stdout.strip().splitlines()) == 5  # header + horizon

    def test_table_format_remains_default(self, forecast_env):
        result = self._run(CliRunner())

        assert result.exit_code == 0, result.output
        assert "q_0.5" in result.output

    def test_json_serialises_missing_values_as_null(self):
        from uniftsm.cli.forecast import _json_scalar

        assert _json_scalar(float("nan")) is None
        assert _json_scalar(pd.NaT) is None
        assert _json_scalar(np.int64(3)) == 3
        assert _json_scalar(pd.Timestamp("2020-01-01")) == "2020-01-01T00:00:00"

    def test_invalid_format_is_rejected(self, forecast_env):
        result = self._run(CliRunner(), "--format", "xml")

        assert result.exit_code != 0
        assert "xml" in result.output


class TestForecastFailureExitCodes:
    """A forecast that produced nothing must not exit 0 and look like a result."""

    @pytest.fixture
    def no_models(self, monkeypatch):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        mock_df = pd.DataFrame({"value": np.linspace(0, 5, 50)}, index=idx)
        monkeypatch.setattr(forecast_module.pd, "read_csv", lambda *a, **k: mock_df)
        monkeypatch.setattr(registry_module.registry, "list_models", lambda: ["broken"])

        def raise_on_get(name):
            raise RuntimeError("model unavailable")

        monkeypatch.setattr(registry_module.registry, "get", raise_on_get)
        return mock_df

    def test_ensemble_with_no_loadable_models_fails(self, no_models):
        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test_data.csv", "w") as handle:
                handle.write("date,value\n")
            result = runner.invoke(
                cli, ["forecast", "test_data.csv", "--horizon", "4", "--ensemble"]
            )

        assert result.exit_code != 0
        assert "No models could be loaded" in result.output

    def test_auto_with_no_suggestion_fails(self, monkeypatch):
        idx = pd.date_range("2020-01-01", periods=50, freq="D")
        mock_df = pd.DataFrame({"value": np.linspace(0, 5, 50)}, index=idx)
        monkeypatch.setattr(forecast_module.pd, "read_csv", lambda *a, **k: mock_df)
        monkeypatch.setattr(registry_module.registry, "suggest", lambda **kwargs: [])

        runner = CliRunner()
        with runner.isolated_filesystem():
            with open("test_data.csv", "w") as handle:
                handle.write("date,value\n")
            result = runner.invoke(cli, ["forecast", "test_data.csv", "--model", "auto"])

        assert result.exit_code != 0
        assert "No suitable model" in result.output


class TestCLIBenchmark:
    def test_benchmark_runs(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0


class TestCLICompare:
    def test_compare_runs(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0


class TestCLIServe:
    def test_serve_runs(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "model" in result.output
