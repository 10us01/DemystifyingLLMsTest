"""
Challenge Stage 17 Test (Opaque)
"""
import pytest
import torch
from model.layers.layer_norm import LayerNorm


def test_stage_17(tiny_config):
    d_model = tiny_config.d_model
    ln = LayerNorm(config=tiny_config)

    # Construct input with different mean offsets across token positions
    x = torch.zeros(2, 2, d_model)
    x[0, 0] = torch.linspace(0, 10, d_model)
    x[0, 1] = torch.linspace(100, 200, d_model)
    x[1, 0] = torch.linspace(50, 60, d_model)
    x[1, 1] = torch.linspace(300, 400, d_model)

    out = ln(x)

    # In LayerNorm, each individual token vector must be normalized across its features (dim=-1)
    mean_dim_neg1 = out.mean(dim=-1)
    var_dim_neg1 = out.var(dim=-1, unbiased=False)

    expected_zeros = torch.zeros_like(mean_dim_neg1)
    expected_ones = torch.ones_like(var_dim_neg1)

    assert torch.allclose(mean_dim_neg1, expected_zeros, atol=1e-3), (
        f"Stage 17 criteria not met: LayerNorm feature mean along dim=-1 is not 0 (got {mean_dim_neg1}). "
        f"Normalization appears to be applied along the wrong dimension."
    )
    assert torch.allclose(var_dim_neg1, expected_ones, atol=1e-2), (
        f"Stage 17 criteria not met: LayerNorm feature variance along dim=-1 is not 1 (got {var_dim_neg1})."
    )
