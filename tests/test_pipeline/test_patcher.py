"""Tests for the AdaptivePatcher."""

import pytest
import torch

from uniftsm.pipeline.patcher import AdaptivePatcher


class TestAdaptivePatcher:
    def test_init_defaults(self):
        patcher = AdaptivePatcher(input_length=64)
        assert patcher.patch_sizes == [8, 16, 32, 64]
        assert patcher.hidden_dim == 64

    def test_forward_shape(self):
        batch, length = 4, 64
        x = torch.randn(batch, length)
        patcher = AdaptivePatcher(input_length=length)
        repr_out, weights = patcher(x)
        # Output has min(num_patches) = 64//64 = 1 patch
        assert repr_out.shape == (batch, 1, 64)
        assert weights.shape == (batch, 4)

    def test_gate_weights_sum_to_one(self):
        batch, length = 4, 64
        x = torch.randn(batch, length)
        patcher = AdaptivePatcher(input_length=length)
        _, weights = patcher(x)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch))

    def test_multivariate_input(self):
        batch, channels, length = 4, 3, 64
        x = torch.randn(batch, channels, length)
        patcher = AdaptivePatcher(input_length=length)
        repr_out, weights = patcher(x)
        assert repr_out.shape[0] == batch
        assert weights.shape == (batch, 4)

    def test_patch_size_divisor_check(self):
        with pytest.raises(ValueError, match="No patch size"):
            AdaptivePatcher(input_length=7, patch_sizes=[8, 16])

    def test_custom_patch_sizes(self):
        batch, length = 4, 96
        x = torch.randn(batch, length)
        patcher = AdaptivePatcher(input_length=length, patch_sizes=[12, 24, 48])
        repr_out, weights = patcher(x)
        assert weights.shape == (batch, 3)

    def test_get_selected_scale(self):
        batch, length = 4, 64
        x = torch.randn(batch, length)
        patcher = AdaptivePatcher(input_length=length)
        _, weights = patcher(x)
        scales = patcher.get_selected_scale(weights)
        assert scales.shape == (batch,)
        assert all(0 <= s.item() < 4 for s in scales)

    def test_extra_repr(self):
        patcher = AdaptivePatcher(input_length=64)
        r = patcher.extra_repr()
        assert "input_length=64" in r
        assert "hidden_dim=64" in r

    def test_gradient_flows(self):
        batch, length = 2, 64
        x = torch.randn(batch, length, requires_grad=True)
        patcher = AdaptivePatcher(input_length=length)
        out, _ = patcher(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for param in patcher.gate_network.parameters():
            assert param.grad is not None
