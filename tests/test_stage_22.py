"""
Challenge Stage 22 Test (Opaque)
"""
import pytest
import torch


def test_stage_22(tiny_config):
    from model.heads.lm_head import LMHead

    head = LMHead(config=tiny_config)
    batch_size, seq_len = 2, 4
    hidden = torch.randn(batch_size, seq_len, tiny_config.d_model)

    try:
        logits = head(hidden)
    except Exception as e:
        pytest.fail(f"Stage 22 criteria not met: LMHead forward execution failed: {e}")

    assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size), (
        f"Stage 22 criteria not met: expected logits shape ({batch_size}, {seq_len}, {tiny_config.vocab_size}), "
        f"got {logits.shape}"
    )
