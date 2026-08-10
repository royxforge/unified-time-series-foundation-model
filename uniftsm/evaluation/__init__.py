"""Rigorous benchmarking and evaluation — metrics, backtesting, statistical baselines, significance tests, and LaTeX reports."""

from uniftsm.evaluation.backtesting import (
    BacktestingEngine,
    ExpandingWindowSplit,
    RollingWindowSplit,
    TimeSeriesCV,
)
from uniftsm.evaluation.metrics import (
    compute_metrics,
    coverage,
    crps,
    mase,
    mrae,
    nmae,
    owa,
    smape,
)
from uniftsm.evaluation.report import (
    ComparisonReport,
    latex_table,
)
from uniftsm.evaluation.significance import (
    diebold_mariano_test,
    paired_bootstrap_test,
    wilcoxon_signed_rank_test,
)
from uniftsm.evaluation.statistical_baselines import (
    ARIMAForecaster,
    ETSForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
)

__all__ = [
    "compute_metrics",
    "mase",
    "smape",
    "crps",
    "nmae",
    "owa",
    "mrae",
    "coverage",
    "BacktestingEngine",
    "RollingWindowSplit",
    "ExpandingWindowSplit",
    "TimeSeriesCV",
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "ETSForecaster",
    "ARIMAForecaster",
    "ThetaForecaster",
    "diebold_mariano_test",
    "wilcoxon_signed_rank_test",
    "paired_bootstrap_test",
    "ComparisonReport",
    "latex_table",
]
