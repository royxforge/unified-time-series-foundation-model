"""LaTeX-ready comparison reports and result tables.

Generates publication-quality comparison tables from evaluation results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ComparisonReport:
    """Aggregates model comparison results into a structured report.

    Parameters
    ----------
    results:
        List of dicts with keys ``model``, ``dataset``, and metric names.
    metric_names:
        Ordered list of metric names to include in reports.
    """

    results: list[dict[str, Any]]
    metric_names: list[str] = field(default_factory=lambda: ["MASE", "sMAPE", "CRPS", "RMSE"])

    def __post_init__(self) -> None:
        self._df = pd.DataFrame(self.results)

    def summary(self) -> pd.DataFrame:
        """Average metrics per model, across all datasets.

        Returns:
            DataFrame with models as rows and metric averages as columns.
        """
        return self._df.groupby("model")[self.metric_names].mean()

    def best_model_per_dataset(self, metric: str = "MASE") -> pd.Series:
        """For each dataset, which model is best on ``metric``?

        Args:
            metric: Metric to minimse.

        Returns:
            Series mapping dataset → best model name.
        """
        idx = self._df.groupby("dataset")[metric].idxmin()
        return self._df.loc[idx, ["dataset", "model"]].set_index("dataset")["model"]

    def rank_matrix(self, metric: str = "MASE") -> pd.DataFrame:
        """Rank each model on each dataset (lower is better).

        Returns:
            DataFrame with datasets as rows and models as columns.
        """
        pivot = self._df.pivot_table(
            values=metric, index="dataset", columns="model", aggfunc="mean"
        )
        return pivot.rank(axis=1)

    def average_rank(self, metric: str = "MASE") -> pd.Series:
        """Average rank across all datasets for each model."""
        return self.rank_matrix(metric).mean()

    def win_count(self, metric: str = "MASE") -> pd.Series:
        """Number of datasets where each model is the best."""
        ranks = self.rank_matrix(metric)
        best = ranks.idxmin(axis=1)
        return best.value_counts()


def latex_table(
    results: list[dict[str, Any]],
    metric_names: list[str] | None = None,
    caption: str = "Forecast accuracy comparison across datasets.",
    label: str = "tab:comparison",
    highlight_best: bool = True,
) -> str:
    """Generate a LaTeX-ready comparison table.

    Args:
        results: List of dicts with keys ``model``, ``dataset``, and
            metric names.
        metric_names: Metrics to include (columns).
        caption: LaTeX table caption.
        label: LaTeX label for cross-referencing.
        highlight_best: If ``True``, bold the best value in each
            dataset × metric cell.

    Returns:
        LaTeX table string.
    """
    if metric_names is None:
        metric_names = ["MASE", "sMAPE", "CRPS", "RMSE"]
    if not results:
        return "% No results to display.\n"
    df = pd.DataFrame(results)
    if df.empty or "model" not in df.columns or "dataset" not in df.columns:
        return "% No valid results to display.\n"
    models = sorted(df["model"].unique())
    # Per-dataset means, one value per (dataset, metric, model) cell.
    pivot = df.pivot_table(
        values=metric_names,
        index="dataset",
        columns="model",
        aggfunc="mean",
    )
    # ``pivot.columns`` is a MultiIndex (metric, model); keep only the
    # models that are actually present in the data.
    models = [m for m in models if m in pivot.columns.get_level_values(1)]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{ll" + "c" * len(models) + "}",
        r"\toprule",
        "Dataset & Metric & " + " & ".join(models) + r" \\",
        r"\midrule",
    ]

    for dataset in pivot.index:
        for metric in metric_names:
            if metric not in pivot.columns.get_level_values(0):
                continue
            vals = pivot.loc[dataset, metric]
            row_parts = [str(dataset), metric]
            if isinstance(vals, pd.Series):
                best_val = vals.min()
                for m in models:
                    v = vals.get(m, float("nan"))
                    if pd.isna(v):
                        row_parts.append("--")
                    elif v == best_val and highlight_best:
                        row_parts.append(rf"\textbf{{{v:.3f}}}")
                    else:
                        row_parts.append(f"{v:.3f}")
            else:
                row_parts.append(f"{float(vals):.3f}")
            lines.append(" & ".join(row_parts) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)
