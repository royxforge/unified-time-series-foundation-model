"""Integration tests for CLI modules using Click's CliRunner."""

from __future__ import annotations

from click.testing import CliRunner


class TestCLIImports:
    """Test that CLI modules import correctly."""

    def test_forecast_module(self):
        from uniftsm.cli import forecast

        assert forecast is not None

    def test_benchmark_module(self):
        from uniftsm.cli import benchmark

        assert benchmark is not None

    def test_compare_module(self):
        from uniftsm.cli import compare

        assert compare is not None

    def test_serve_module(self):
        from uniftsm.cli import serve

        assert serve is not None

    def test_main_module(self):
        from uniftsm.cli.main import main

        assert callable(main)


class TestCLICommands:
    """Test CLI command invocation with CliRunner."""

    def test_forecast_help(self):
        from uniftsm.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["forecast", "--help"])
        assert result.exit_code == 0
        assert "horizon" in result.output.lower()

    def test_benchmark_help(self):
        from uniftsm.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output.lower()

    def test_compare_help(self):
        from uniftsm.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", "--help"])
        assert result.exit_code == 0
        assert "models" in result.output.lower()

    def test_serve_help(self):
        from uniftsm.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "port" in result.output.lower()

    def test_main_help(self):
        from uniftsm.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "forecast" in result.output
        assert "benchmark" in result.output
        assert "compare" in result.output
        assert "serve" in result.output

    def test_version(self):
        from uniftsm.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower() or "uniftsm" in result.output.lower()
