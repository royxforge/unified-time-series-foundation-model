"""Adaptive patching with differentiable patch-size selection.

This module implements the **first novel contribution** of UniTSFM:
a learnable gating mechanism that selects optimal patch scales for
each input series, replacing the fixed patch size used by existing TSFMs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePatcher(nn.Module):
    """Differentiable patching with learnable scale selection.

    Standard TSFM patching uses a fixed patch size (e.g., 32 for TimesFM,
    8 for TTM).  This module learns a soft combination across multiple
    patch scales via a lightweight gating network.

    The gate network observes the pooled series statistics and assigns
    attention weights to each candidate patch size.  The final
    representation is a weighted combination of all scales.

    Parameters
    ----------
    input_length:
        Length of the input time series (number of time steps).
    hidden_dim:
        Hidden dimension of the gating network and the output projection.
    patch_sizes:
        Candidate patch sizes to consider.  Each must divide
        ``input_length`` evenly for the unfold operation.
    """

    def __init__(
        self,
        input_length: int,
        hidden_dim: int = 64,
        patch_sizes: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.hidden_dim = hidden_dim
        self.patch_sizes = sorted(patch_sizes or [8, 16, 32, 64])

        # Filter to patch sizes that divide input_length
        self.patch_sizes = [ps for ps in self.patch_sizes if input_length % ps == 0]
        if not self.patch_sizes:
            raise ValueError(
                f"No patch size in {sorted(patch_sizes or [8, 16, 32, 64])} "
                f"divides input_length {input_length}."
            )

        # Gating network: series-level stats → attention over scales
        self.gate_network = nn.Sequential(
            nn.Linear(input_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(self.patch_sizes)),
            nn.Softmax(dim=-1),
        )

        # Per-scale projection heads: each maps (patch_size_i) → (hidden_dim)
        self.projections = nn.ModuleDict()
        for ps in self.patch_sizes:
            self.projections[str(ps)] = nn.Linear(ps, hidden_dim)

        # Determine target number of patches (use the median)
        self._num_patches_target = min(input_length // ps for ps in self.patch_sizes)

    @staticmethod
    def _unfold_series(x: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Extract non-overlapping patches from a 1D time series.

        Args:
            x: (batch, input_length)
            patch_size: size of each patch

        Returns:
            (batch, num_patches, patch_size)
        """
        batch_size, length = x.shape
        num_patches = length // patch_size
        return x.view(batch_size, num_patches, patch_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive patching with learned scale mixture.

        Args:
            x: Input tensor of shape ``(batch, input_length)`` for
               univariate or ``(batch, channels, input_length)`` for
               multivariate series.

        Returns:
            A tuple ``(patched_repr, gate_weights)`` where
            ``patched_repr`` has shape ``(batch, num_patches, hidden_dim)``
            and ``gate_weights`` has shape ``(batch, num_scales)``.
        """
        batch_size = x.shape[0]

        # Compute gate weights from pooled series statistics
        pooled = x.mean(dim=1) if x.dim() == 3 else x
        gate_weights: torch.Tensor = self.gate_network(pooled)

        # Build patched representations at each scale
        patched_scales: list[torch.Tensor] = []
        for i, patch_size in enumerate(self.patch_sizes):
            num_patches = self.input_length // patch_size

            if x.dim() == 3:
                # Multivariate: average across channels after patching
                channel_patches = []
                for c in range(x.shape[1]):
                    patches = self._unfold_series(x[:, c, :], patch_size)
                    channel_patches.append(patches)
                patches_3d = torch.stack(channel_patches).mean(dim=0)
            else:
                patches_3d = self._unfold_series(x, patch_size)

            # Project each patch to hidden_dim
            projected = self.projections[str(patch_size)](patches_3d)
            # → (batch, num_patches, hidden_dim)

            # Adaptive pool to target number of patches
            if num_patches != self._num_patches_target:
                projected = projected.permute(
                    0, 2, 1
                ).contiguous()  # (batch, hidden_dim, num_patches)
                projected = F.adaptive_avg_pool1d(
                    projected, self._num_patches_target
                )  # (batch, hidden_dim, target)
                projected = projected.permute(0, 2, 1).contiguous()  # (batch, target, hidden_dim)

            # Weight by gate scale
            weight = gate_weights[:, i].view(batch_size, 1, 1)
            patched_scales.append(projected * weight)

        # Sum across scales → (batch, target_patches, hidden_dim)
        aggregated = torch.stack(patched_scales).sum(dim=0)

        return aggregated, gate_weights

    def get_selected_scale(self, gate_weights: torch.Tensor) -> torch.Tensor:
        """Return the dominant patch scale index for each batch element.

        Useful for interpretability: which patch size does the model prefer?
        """
        return gate_weights.argmax(dim=-1)

    def extra_repr(self) -> str:
        return (
            f"input_length={self.input_length}, "
            f"hidden_dim={self.hidden_dim}, "
            f"patch_sizes={self.patch_sizes}"
        )
