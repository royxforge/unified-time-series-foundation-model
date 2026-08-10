"""Global configuration for the UniTSFM framework.

Uses Pydantic for validation, serialization, and IDE-friendly autocomplete.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class UniTSFMConfig(BaseSettings):
    """Global configuration for UniTSFM.

    Loaded from environment variables (``UNITSFM_*`` prefix) or a YAML file
    passed via ``UNITSFM_CONFIG``.  Sensible defaults for local development.
    """

    # ── Device & compute ──────────────────────────────────────────────────
    device: str = Field(
        default="auto",
        description="Torch device. 'auto' → cuda if available else cpu.",
    )
    seed: int = Field(
        default=42,
        description="Global random seed for reproducibility.",
        ge=0,
    )
    num_threads: int = Field(
        default=4,
        description="Number of CPU threads for intra-op parallelism.",
        ge=1,
    )
    dtype: Literal["float32", "bfloat16", "float16"] = Field(
        default="float32",
        description="Default torch dtype for inference.",
    )

    # ── Model defaults ────────────────────────────────────────────────────
    default_context_length: int = Field(
        default=512,
        description="Default context length (look-back window) for TSFMs.",
        ge=32,
    )
    default_horizon: int = Field(
        default=24,
        description="Default forecast horizon in steps.",
        ge=1,
    )
    model_cache_dir: str = Field(
        default="~/.cache/uniftsm/models",
        description="Directory to cache downloaded model weights.",
    )

    # ── Patching ──────────────────────────────────────────────────────────
    patch_sizes: list[int] = Field(
        default=[8, 16, 32, 64],
        description="Candidate patch sizes for adaptive patching.",
        min_length=1,
    )
    patcher_hidden_dim: int = Field(
        default=64,
        description="Hidden dimension of the adaptive patcher gating network.",
        ge=16,
    )

    # ── Ensembling ────────────────────────────────────────────────────────
    ensemble_temperature: float = Field(
        default=1.0,
        description="Temperature for uncertainty-weighted ensemble (sharpness).",
        ge=0.1,
        le=10.0,
    )
    ensemble_method: Literal["average", "uncertainty_weighted", "stacking"] = Field(
        default="uncertainty_weighted",
        description="Default ensemble strategy.",
    )

    # ── Evaluation ────────────────────────────────────────────────────────
    n_bootstrap: int = Field(
        default=1000,
        description="Number of bootstrap samples for confidence intervals.",
        ge=10,
    )
    alpha: float = Field(
        default=0.05,
        description="Significance level for prediction intervals and tests.",
        gt=0.0,
        lt=1.0,
    )

    # ── Frequencies ───────────────────────────────────────────────────────
    supported_frequencies: list[str] = Field(
        default=["h", "D", "W", "ME", "QE", "YE", "min", "S", "B"],
        description="Supported pandas frequency aliases.",
    )

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: str = Field(
        default="~/.cache/uniftsm/data",
        description="Directory to store downloaded benchmark datasets.",
    )
    results_dir: str = Field(
        default="./benchmarks/results",
        description="Directory to write evaluation results.",
    )

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity.",
    )

    model_config = {"env_prefix": "UNITSFM_", "frozen": False}

    @model_validator(mode="after")
    def _resolve_paths(self) -> UniTSFMConfig:
        """Expand user home (``~``) in path fields."""
        for field_name in ("model_cache_dir", "data_dir", "results_dir"):
            current: str = getattr(self, field_name)
            resolved = str(Path(current).expanduser().resolve())
            object.__setattr__(self, field_name, resolved)
        return self

    @model_validator(mode="after")
    def _resolve_device(self) -> UniTSFMConfig:
        """Resolve 'auto' device to concrete device string."""
        if self.device == "auto":
            import torch

            object.__setattr__(self, "device", "cuda" if torch.cuda.is_available() else "cpu")
        return self


# Module-level singleton — importers get a live config.
config: UniTSFMConfig = UniTSFMConfig()


def reload_config(config_path: str | None = None) -> UniTSFMConfig:
    """Reload configuration, optionally from a YAML file.

    Args:
        config_path: Path to a YAML configuration file.  If ``None``,
                     reload from environment variables only.

    Returns:
        The new global :class:`UniTSFMConfig` instance.
    """
    global config
    if config_path is not None:
        import yaml

        path = Path(config_path).expanduser().resolve()
        with open(path) as fh:
            overrides = yaml.safe_load(fh) or {}
        config = UniTSFMConfig(**overrides)
    else:
        config = UniTSFMConfig()
    return config
