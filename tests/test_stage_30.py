"""
Challenge Stage 30 Test (Opaque)
"""
import pytest
import torch
from model.attention.scaled_dot_product import scaled_dot_product_attention


def test_stage_30():
    # Pass rectangular sequence lengths: seq_len_q != seq_len_k
    batch_size, n_heads, seq_q, seq_k, d_k = 2, 2, 3, 5, 16

    q = torch.randn(batch_size, n_heads, seq_q, d_k)
    k = torch.randn(batch_size, n_heads, seq_k, d_k)
    v = torch.randn(batch_size, n_heads, seq_k, d_k)

    out, weights = scaled_dot_product_attention(q, k, v)

    # In standard attention, each query position defines a probability distribution over keys:
    # sum_{j in keys} A_{i, j} == 1.0 (along dim=-1)
    sums_across_keys = weights.sum(dim=-1)
    sums_across_queries = weights.sum(dim=-2)

    expected_ones = torch.ones(batch_size, n_heads, seq_q)

    assert not torch.allclose(sums_across_queries, torch.ones(batch_size, n_heads, seq_k), atol=1e-3), (
        "Stage 30 criteria not met: attention weights sum to 1 across query positions (dim=-2) "
        "instead of key positions (dim=-1)"
    )

    assert torch.allclose(sums_across_keys, expected_ones, atol=1e-3), (
        f"Stage 30 criteria not met: attention weights do not sum to 1.0 across keys (dim=-1). "
        f"Softmax appears to be applied along the wrong dimension."
    )
