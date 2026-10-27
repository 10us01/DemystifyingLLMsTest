"""
Challenge Stage 01 Test (Opaque)
"""
import pytest


def test_stage_01():
    from config.model_config import ModelConfig

    try:
        cfg = ModelConfig()
    except Exception as e:
        pytest.fail(f"Stage 01 criteria not met: default model configuration invalid: {e}")

    assert cfg.d_model % cfg.n_heads == 0, (
        f"Stage 01 criteria not met: d_model ({cfg.d_model}) must be divisible by n_heads ({cfg.n_heads})"
    )
    assert cfg.d_model == 768, (
        f"Stage 01 criteria not met: expected standard d_model=768, got {cfg.d_model}"
    )
