#!/usr/bin/env python3
"""Plot benchmark results as faceted MASE bar charts.

Reads ``benchmarks/results/benchmark_results.csv`` (produced by
``run_all.py``) and renders one panel per dataset with a bar per
model.  The best model in each panel is highlighted and all bars are
value-labelled, so per-dataset winners and cross-model gaps are
readable even where scales differ wildly (e.g. exchange_rate).

Usage:
    python benchmarks/plot_results.py                 # MASE, default output
    python benchmarks/plot_results.py --metric CRPS   # any metric
    python benchmarks/plot_results.py --output my.png

Output is written to ``benchmarks/results/benchmark_mase.png`` by
default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
DEFAULT_CSV = RESULTS_DIR / "benchmark_results.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "benchmark_mase.png"

# Consistent, colour-blind-friendly model palette (matches the README badges).
MODEL_COLORS = {
    "chronos": "#6366f1",  # indigo
    "timesfm": "#14b8a6",  # teal
    "moirai": "#f59e0b",  # amber
    "lag_llama": "#ef4444",  # red
    "ttm": "#8b5cf6",  # violet
}
BEST_EDGE = "#0f172a"  # near-black edge for the winning bar
GRID_ALPHA = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="MASE",
        help="Metric to plot (column in the CSV)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path",
    )
    return parser.parse_args()


def _plot_panel(
    ax: plt.Axes,
    models: list[str],
    values: list[float],
    dataset: str,
    metric: str,
    best_model: str,
    show_ylabel: bool = True,
) -> None:
    """Render a single dataset panel: one bar per model."""
    colors = [MODEL_COLORS.get(m, "#94a3b8") for m in models]
    x = np.arange(len(models))
    bars = ax.bar(x, values, width=0.68, color=colors, zorder=3)

    # Highlight the best (lowest) model.
    best_idx = models.index(best_model)
    bars[best_idx].set_edgecolor(BEST_EDGE)
    bars[best_idx].set_linewidth(1.8)
    bars[best_idx].set_hatch("///")

    # Value labels on every bar (NaN cells render as "n/a").
    for xi, v in zip(x, values, strict=True):
        if pd.isna(v):
            continue
        ax.text(
            xi,
            v,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color="#334155",
        )

    ax.set_title(dataset.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7.5)
    if show_ylabel:
        ax.set_ylabel(metric, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=GRID_ALPHA, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    # Leave headroom for the value labels (NaN-safe).
    ymax = float(np.nanmax(values)) * 1.18
    ax.set_ylim(0, ymax)


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(
            f"{args.csv} not found. Run `python benchmarks/run_all.py --quick` "
            "first to produce benchmark results."
        )

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError(f"{args.csv.name} is empty; nothing to plot.")
    if args.metric not in df.columns:
        raise ValueError(
            f"Column '{args.metric}' not in {args.csv.name}. " f"Available: {list(df.columns)}"
        )

    pivot = df.pivot_table(index="dataset", columns="model", values=args.metric, aggfunc="mean")
    datasets = list(pivot.index)
    models = list(pivot.columns)
    best = pivot.idxmin(axis=1)

    n_datasets = len(datasets)
    n_cols = 4
    n_rows = int(np.ceil(n_datasets / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.4 * n_rows), squeeze=False)

    for i, (ax, dataset) in enumerate(zip(axes.flat, datasets, strict=False)):
        values = pivot.loc[dataset].tolist()
        _plot_panel(
            ax,
            models=models,
            values=values,
            dataset=dataset,
            metric=args.metric,
            best_model=best[dataset],
            show_ylabel=(i % n_cols == 0),
        )

    # Hide any unused panels.
    for ax in axes.flat[n_datasets:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Zero-shot {args.metric} by model and dataset " "(lower is better; hatched bar = best)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
