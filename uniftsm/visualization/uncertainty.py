"""Uncertainty visualisation — fan charts, prediction intervals."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_uncertainty_fan(
    y_forecast: dict[str, pd.DataFrame],
    y_true: pd.Series | None = None,
    title: str = "Uncertainty Fan Chart",
    figsize: tuple[int, int] = (12, 5),
    save_path: str | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Plot a fan chart showing prediction intervals at multiple levels.

    Args:
        y_forecast: Dict mapping quantile strings (e.g. ``"q_0.1"``,
            ``"q_0.9"``) to DataFrames with a ``"forecast"`` column.
        y_true: Optional ground truth for comparison.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save the figure.
        **kwargs: Additional plot arguments.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Parse quantile levels
    quantile_data = {}
    for key, df in y_forecast.items():
        q_val = float(key.split("_")[1])
        quantile_data[q_val] = df["forecast"].values

    sorted_quantiles = sorted(quantile_data.keys())
    idx = list(y_forecast.values())[0].index

    # Plot median first
    median_q = 0.5
    if median_q in quantile_data:
        ax.plot(
            idx,
            quantile_data[median_q],
            color="red",
            linewidth=2.0,
            label="Median",
            **kwargs,
        )

    # Fill between symmetric quantile pairs (outer = lighter)
    n_levels = len(sorted_quantiles) // 2
    for i in range(n_levels):
        lower_q = sorted_quantiles[i]
        upper_q = sorted_quantiles[-(i + 1)]
        if lower_q >= median_q or upper_q <= median_q:
            continue
        alpha = 0.1 + 0.15 * (1 - (median_q - lower_q) / (median_q - sorted_quantiles[0]))
        ax.fill_between(
            idx,
            quantile_data[lower_q],
            quantile_data[upper_q],
            color="blue",
            alpha=min(alpha, 0.4),
            label=f"{(upper_q - lower_q) * 100:.0f}% CI",
        )

    # Ground truth
    if y_true is not None:
        ax.plot(
            y_true.index,
            y_true.values,
            color="green",
            linestyle="--",
            label="Ground Truth",
            linewidth=1.5,
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
