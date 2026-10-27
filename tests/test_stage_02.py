"""
Challenge Stage 02 Test (Opaque)
"""
import pytest


def test_stage_02():
    from config.model_config import ModelConfig

    try:
        cfg = ModelConfig(d_model=768)
    except Exception as e:
        pytest.fail(f"Stage 02 criteria not met: configuration initialization error: {e}")

    assert cfg.vocab_size == 50257, (
        f"Stage 02 criteria not met: expected vocab_size=50257, got {cfg.vocab_size}"
    )
