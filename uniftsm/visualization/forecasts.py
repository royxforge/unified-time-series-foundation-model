"""Plot forecast results against ground truth.

Provides publication-quality time series plots with history,
forecast mean, and optional confidence intervals.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def plot_forecast(
    y_history: pd.Series,
    y_forecast: pd.DataFrame,
    y_true: pd.Series | None = None,
    title: str = "Forecast",
    figsize: tuple[int, int] = (12, 5),
    history_label: str = "History",
    forecast_label: str = "Forecast",
    true_label: str = "Ground Truth",
    interval_alpha: float = 0.2,
    save_path: str | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Plot forecast results.

    Args:
        y_history: Historical (training) time series.
        y_forecast: DataFrame with at least ``"mean"`` column, and
            optionally ``"std"`` for confidence bands.
        y_true: Optional ground truth for comparison.
        title: Plot title.
        figsize: Figure size (width, height) in inches.
        history_label: Label for the historical data.
        forecast_label: Label for the forecast mean.
        true_label: Label for ground truth.
        interval_alpha: Transparency of confidence intervals.
        save_path: If provided, save the figure to this path.
        **kwargs: Additional keyword arguments for ``plt.plot``.

    Returns:
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot history
    ax.plot(
        y_history.index,
        y_history.values,
        color="black",
        label=history_label,
        linewidth=1.5,
        **kwargs,
    )

    # Plot forecast mean
    ax.plot(
        y_forecast.index,
        y_forecast["mean"].values,
        color="blue",
        label=forecast_label,
        linewidth=2.0,
        **kwargs,
    )

    # Confidence intervals from std
    if "std" in y_forecast.columns:
        std = y_forecast["std"].values
        mean = y_forecast["mean"].values
        idx = y_forecast.index
        ax.fill_between(
            idx,
            mean - 1.96 * std,
            mean + 1.96 * std,
            color="blue",
            alpha=interval_alpha,
            label="95% CI",
        )

    # Ground truth
    if y_true is not None:
        ax.plot(
            y_true.index,
            y_true.values,
            color="green",
            linestyle="--",
            label=true_label,
            linewidth=1.5,
            **kwargs,
        )

    ax.axvline(
        x=y_history.index[-1],
        color="gray",
        linestyle=":",
        alpha=0.7,
        label="Forecast start",
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
