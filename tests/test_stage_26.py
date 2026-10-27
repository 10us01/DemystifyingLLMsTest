"""
Challenge Stage 26 Test (Opaque)
"""
import pytest
import torch
from model.blocks.transformer_block import TransformerBlock


def test_stage_26(tiny_config):
    block = TransformerBlock(config=tiny_config)

    execution_order = []

    block.attn.register_forward_hook(
        lambda mod, inp, out: execution_order.append("attention")
    )
    block.ffn.register_forward_hook(
        lambda mod, inp, out: execution_order.append("feedforward")
    )

    x = torch.randn(1, 2, tiny_config.d_model)
    try:
        block(x)
    except Exception:
        # If execution fails because sublayer outputs have shape bugs,
        # hooks have already recorded the invocation order
        pass

    assert execution_order and execution_order[0] != "feedforward", (
        "Stage 26 criteria not met: transformer sublayers executed in reverse order "
        "(FeedForward was invoked before MultiHeadAttention)"
    )

    assert len(execution_order) >= 2, (
        f"Stage 26 criteria not met: expected both attention and feedforward to execute, "
        f"got called: {execution_order}"
    )

    assert execution_order[:2] == ["attention", "feedforward"], (
        f"Stage 26 criteria not met: expected sublayer order ['attention', 'feedforward'], "
        f"got {execution_order[:2]}"
    )
