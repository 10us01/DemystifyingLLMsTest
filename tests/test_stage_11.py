"""
Challenge Stage 11 Test (Opaque)
"""
import math
import pytest
import torch
import model.attention.scaled_dot_product as sdp


def test_stage_11():
    captured_scores = []
    original_softmax = sdp.F.softmax

    def hook_softmax(x, *args, **kwargs):
        captured_scores.append(x.detach().clone())
        return original_softmax(x, *args, **kwargs)

    sdp.F.softmax = hook_softmax
    try:
        d_k = 64
        q = torch.ones(1, 1, 2, d_k)
        k = torch.ones(1, 1, 2, d_k)
        v = torch.randn(1, 1, 2, d_k)

        sdp.scaled_dot_product_attention(q, k, v)

        assert len(captured_scores) > 0, "Stage 11 criteria not met: softmax was not invoked"
        actual_score = captured_scores[0][0, 0, 0, 0].item()
        expected_score = 64.0 / math.sqrt(d_k)

        assert not math.isclose(actual_score, 64.0, abs_tol=1e-3), (
            "Stage 11 criteria not met: attention scores appear to lack 1/sqrt(d_k) scaling"
        )
        assert math.isclose(actual_score, expected_score, abs_tol=1e-3), (
            f"Stage 11 criteria not met: expected scaled score {expected_score:.4f}, got {actual_score:.4f}"
        )
    finally:
        sdp.F.softmax = original_softmax
