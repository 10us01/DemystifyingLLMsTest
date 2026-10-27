"""
Challenge Stage 19 Test (Opaque)
"""
import pytest
import torch
from model.attention.multi_head import MultiHeadAttention


def test_stage_19(tiny_config):
    mha = MultiHeadAttention(config=tiny_config)
    batch_size, seq_len = 2, 4
    x = torch.randn(batch_size, seq_len, tiny_config.d_model)

    try:
        out = mha(x)
    except RuntimeError as e:
        pytest.fail(
            f"Stage 19 criteria not met: MultiHeadAttention failed during head combination: {e}. "
            f"Verify head concatenation dimension."
        )

    expected_shape = (batch_size, seq_len, tiny_config.d_model)
    assert out.shape == expected_shape, (
        f"Stage 19 criteria not met: expected output shape {expected_shape}, got {out.shape}. "
        f"Heads appear to be concatenated along the wrong dimension."
    )
