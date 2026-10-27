"""
Challenge Stage 05 Test (Opaque)
"""
import pytest
from model.layers.feedforward import FeedForward


def test_stage_05(tiny_config):
    ffn = FeedForward(config=tiny_config)

    expected_d_ffn = tiny_config.d_model * 4
    assert ffn.d_ffn == expected_d_ffn, (
        f"Stage 05 criteria not met: feedforward hidden dimension expected {expected_d_ffn} "
        f"(4x expansion over d_model={tiny_config.d_model}), but got {ffn.d_ffn}"
    )
    assert ffn.linear1.out_features == expected_d_ffn, (
        f"Stage 05 criteria not met: linear1 out_features expected {expected_d_ffn}, "
        f"got {ffn.linear1.out_features}"
    )
