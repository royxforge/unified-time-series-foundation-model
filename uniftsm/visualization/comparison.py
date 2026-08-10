"""Side-by-side model comparison visualisation."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_comparison(
    y_history: pd.Series,
    forecasts: dict[str, pd.DataFrame],
    y_true: pd.Series | None = None,
    title: str = "Model Comparison",
    figsize: tuple[int, int] = (14, 6),
    save_path: str | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Compare multiple model forecasts on the same plot.

    Args:
        y_history: Historical (training) time series.
        forecasts: Dict mapping model name → forecast DataFrame with
            ``"mean"`` column.
        y_true: Optional ground truth.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.
        **kwargs: Additional plot arguments.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # History
    ax.plot(
        y_history.index,
        y_history.values,
        color="black",
        label="History",
        linewidth=1.5,
        **kwargs,
    )

    # Each model forecast
    colors = plt.cm.tab10(np.linspace(0, 1, len(forecasts)))
    for (model_name, forecast), color in zip(forecasts.items(), colors, strict=True):
        ax.plot(
            forecast.index,
            forecast["mean"].values,
            color=color,
            label=model_name,
            linewidth=2.0,
            linestyle="--",
            **kwargs,
        )

    # Ground truth
    if y_true is not None:
        ax.plot(
            y_true.index,
            y_true.values,
            color="green",
            linestyle=":",
            label="Ground Truth",
            linewidth=1.5,
        )

    ax.axvline(
        x=y_history.index[-1],
        color="gray",
        linestyle=":",
        alpha=0.7,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_rank_heatmap(
    rank_matrix: pd.DataFrame,
    title: str = "Model Rankings by Dataset",
    figsize: tuple[int, int] = (10, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot a heatmap of model rankings across datasets.

    This is the "No Free Lunch" map described in the paper.

    Args:
        rank_matrix: DataFrame with datasets as rows, models as columns,
            and ranks as values (1 = best).
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(rank_matrix.values, cmap="RdYlGn_r", aspect="auto", vmin=1)

    ax.set_xticks(range(len(rank_matrix.columns)))
    ax.set_xticklabels(rank_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(rank_matrix.index)))
    ax.set_yticklabels(rank_matrix.index)

    # Annotate cells
    for i in range(len(rank_matrix.index)):
        for j in range(len(rank_matrix.columns)):
            val = rank_matrix.values[i, j]
            ax.text(
                j,
                i,
                f"{val:.0f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="black" if 1.5 < val < rank_matrix.max().max() - 0.5 else "white",
            )

    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, label="Rank (1 = Best)")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
