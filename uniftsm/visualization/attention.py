"""Attention map visualisation for transformer-based TSFMs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_map(
    attention_weights: np.ndarray,
    title: str = "Attention Map",
    figsize: tuple[int, int] = (8, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot attention weight heatmap.

    Args:
        attention_weights: Attention matrix of shape
            ``(n_heads, n_tokens, n_tokens)`` or ``(n_tokens, n_tokens)``.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.
    """
    weights = np.asarray(attention_weights, dtype=np.float64)

    if weights.ndim == 3:
        # Average across heads
        weights = weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(weights, cmap="viridis", aspect="auto")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Key Position", fontsize=12)
    ax.set_ylabel("Query Position", fontsize=12)
    fig.colorbar(im, ax=ax, label="Attention Weight")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_attention_rollout(
    attention_matrices: list[np.ndarray],
    title: str = "Attention Rollout",
    figsize: tuple[int, int] = (12, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot attention rollout across layers (Abnar & Zuidema 2020).

    Shows how information flows from input to output through the
    attention layers.

    Args:
        attention_matrices: List of attention matrices, one per layer,
            each of shape ``(n_tokens, n_tokens)``.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        The matplotlib Figure object.
    """
    n_layers = len(attention_matrices)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)

    if n_layers == 1:
        axes = [axes]

    for layer_idx, attn in enumerate(attention_matrices):
        attn = np.asarray(attn, dtype=np.float64)
        if attn.ndim == 3:
            attn = attn.mean(axis=0)

        axes[layer_idx].imshow(attn, cmap="viridis", aspect="auto")
        axes[layer_idx].set_title(f"Layer {layer_idx + 1}", fontsize=10)
        axes[layer_idx].set_xlabel("Key")
        axes[layer_idx].set_ylabel("Query")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
