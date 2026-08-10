"""Tests for UniTSFMConfig and reload_config."""

from uniftsm.core.config import UniTSFMConfig, config, reload_config


class TestUniTSFMConfig:
    def test_default_values(self):
        cfg = UniTSFMConfig()
        assert cfg.seed == 42
        assert cfg.num_threads == 4
        assert cfg.default_context_length == 512
        assert cfg.default_horizon == 24
        assert cfg.ensemble_temperature == 1.0
        assert cfg.ensemble_method == "uncertainty_weighted"
        assert cfg.alpha == 0.05
        assert cfg.log_level == "INFO"

    def test_device_resolved(self):
        cfg = UniTSFMConfig(device="auto")
        assert cfg.device in ("cpu", "cuda")

    def test_path_expansion(self):
        cfg = UniTSFMConfig(model_cache_dir="~/test_uniftsm")
        assert "~" not in cfg.model_cache_dir

    def test_fields_override(self):
        cfg = UniTSFMConfig(seed=123, num_threads=8)
        assert cfg.seed == 123
        assert cfg.num_threads == 8

    def test_dtype_default(self):
        cfg = UniTSFMConfig()
        assert cfg.dtype == "float32"

    def test_patch_sizes_default(self):
        cfg = UniTSFMConfig()
        assert cfg.patch_sizes == [8, 16, 32, 64]

    def test_environment_prefix(self):
        import os

        os.environ["UNITSFM_SEED"] = "99"
        try:
            cfg = UniTSFMConfig()
            assert cfg.seed == 99
        finally:
            os.environ.pop("UNITSFM_SEED", None)

    def test_global_config_is_instance(self):
        assert isinstance(config, UniTSFMConfig)

    def test_reload_config(self):
        new_cfg = reload_config()
        assert isinstance(new_cfg, UniTSFMConfig)
