"""Shared download, cache, and device management for model wrappers.

Provides a common base with lazy weight loading and HuggingFace Hub
integration so that each model wrapper does not duplicate boilerplate.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from uniftsm.core.config import config

logger = logging.getLogger(__name__)


class BaseModelImpl:
    """Shared logic for model weight management.

    Handles device resolution, dtype casting, cache directory setup,
    and HuggingFace model loading — consistent across all TSFM wrappers.
    """

    def __init__(self, model_name: str, device: str = "auto", seed: int = 42) -> None:
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.seed = seed

        # Cache directory
        self._cache_dir = Path(config.model_cache_dir).expanduser() / model_name
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Device helpers ───────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype_str: str | None = None) -> torch.dtype:
        dtype_str = dtype_str or config.dtype
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        resolved = dtype_map.get(dtype_str, torch.float32)
        # bfloat16 is not available on all hardware
        if resolved == torch.bfloat16 and not (
            torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        ):
            logger.warning("bfloat16 not supported on this device; falling back to float32.")
            resolved = torch.float32
        return resolved

    @staticmethod
    def _to_device(
        obj: Any,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> Any:
        """Move a model or tensor to the target device."""
        if isinstance(obj, torch.nn.Module):
            return obj.to(device=device, dtype=dtype or next(obj.parameters()).dtype)
        if isinstance(obj, torch.Tensor):
            return obj.to(device=device, dtype=dtype or obj.dtype)
        return obj

    # ── HF Hub helpers ───────────────────────────────────────────────────

    @staticmethod
    def _hf_model_dir(repo_id: str) -> str:
        """Resolve the local cache directory for a HuggingFace model repo.

        Uses the ``HF_HUB_CACHE`` location (respecting ``HF_HOME`` /
        ``HF_HUB_CACHE`` environment variables) and the standard
        ``models--<org>--<name>`` cache layout.
        """
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            cache_root = Path(HF_HUB_CACHE)
        except ImportError:
            cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        return str(cache_root / f"models--{repo_id.replace('/', '--')}")

    # ── Logging ──────────────────────────────────────────────────────────

    def log_model_load(self, repo_id: str) -> None:
        logger.info("Loading model '%s' from %s...", self.model_name, repo_id)

    def log_model_loaded(self, elapsed: float) -> None:
        logger.info(
            "Model '%s' loaded in %.2fs on %s.",
            self.model_name,
            elapsed,
            self.device,
        )
