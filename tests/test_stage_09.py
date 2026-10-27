"""
Challenge Stage 09 Test (Opaque)
"""
import pytest


def test_stage_09():
    from config.inference_config import InferenceConfig

    cfg = InferenceConfig()
    assert cfg.max_new_tokens > 0, (
        f"Stage 09 criteria not met: max_new_tokens is {cfg.max_new_tokens}, generating nothing"
    )
    assert cfg.max_new_tokens >= 64, (
        f"Stage 09 criteria not met: default generation capacity too low ({cfg.max_new_tokens})"
    )
