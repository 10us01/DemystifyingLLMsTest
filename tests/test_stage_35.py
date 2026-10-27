"""
Challenge Stage 35 Test (Opaque)
"""
import pytest
import torch
from model.attention.multi_head import MultiHeadAttention


def test_stage_35(tiny_config):
    mha = MultiHeadAttention(config=tiny_config)

    # In MultiHeadAttention, Query, Key, and Value projections must be distinct parameter transformations:
    assert mha.q_proj is not mha.k_proj, (
        "Stage 35 criteria not met: q_proj and k_proj share the same linear layer instance"
    )
    assert mha.q_proj is not mha.v_proj, (
        "Stage 35 criteria not met: q_proj and v_proj share the same linear layer instance"
    )
    assert mha.k_proj is not mha.v_proj, (
        "Stage 35 criteria not met: k_proj and v_proj share the same linear layer instance"
    )

    assert mha.q_proj.weight is not mha.k_proj.weight, (
        "Stage 35 criteria not met: q_proj.weight and k_proj.weight reference identical parameter memory"
    )
    assert mha.q_proj.weight is not mha.v_proj.weight, (
        "Stage 35 criteria not met: q_proj.weight and v_proj.weight reference identical parameter memory"
    )
