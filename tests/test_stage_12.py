"""
Challenge Stage 12 Test (Opaque)
"""
import pytest
import torch
from model.attention.causal_mask import create_causal_mask


def test_stage_12():
    seq_len = 4
    mask = create_causal_mask(seq_len=seq_len)

    # In causal autoregressive attention:
    # Row i represents the query position, Col j represents the key position.
    # Future positions (j > i) must be masked with -inf.
    # Past and current positions (j <= i) must be allowed (0.0).

    future_masked = torch.isneginf(mask[0, 0, 0, 1]).item()
    past_allowed = not torch.isneginf(mask[0, 0, 1, 0]).item()
    diag_allowed = not torch.isneginf(mask[0, 0, 0, 0]).item()

    assert future_masked, (
        "Stage 12 criteria not met: causal mask allows future positions (j > i) to be attended"
    )
    assert past_allowed, (
        "Stage 12 criteria not met: causal mask blocks past positions (j < i) instead of allowing them"
    )
    assert diag_allowed, (
        "Stage 12 criteria not met: causal mask diagonal position (current token) is masked"
    )
