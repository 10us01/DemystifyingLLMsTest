"""
Challenge Stage 18 Test (Opaque)
"""
import pytest
import torch
from model.layers.residual import ResidualConnection


def test_stage_18():
    res = ResidualConnection(dropout=0.0)

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    sublayer_out = torch.tensor([[10.0, 20.0], [30.0, 40.0]])

    out = res(x, sublayer_out)

    expected_sum = x + sublayer_out
    buggy_duplicate = x + x

    assert not torch.allclose(out, buggy_duplicate, atol=1e-4), (
        "Stage 18 criteria not met: residual connection appears to add input to input (x + x), "
        "ignoring sublayer output"
    )
    assert torch.allclose(out, expected_sum, atol=1e-4), (
        f"Stage 18 criteria not met: expected residual output (x + sublayer(x))={expected_sum}, got {out}"
    )
