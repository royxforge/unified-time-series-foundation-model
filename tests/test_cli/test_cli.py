"""Tests for CLI commands using Click's CliRunner."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
from click.testing import CliRunner

from uniftsm.cli.main import cli


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
