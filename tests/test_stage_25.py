"""
Challenge Stage 25 Test (Opaque)
"""
import pytest
import torch


def test_stage_25():
    from model.embeddings.position_embedding import SinusoidalPositionEmbedding

    d_model = 64
    pos_emb = SinusoidalPositionEmbedding(d_model=d_model, max_seq_len=128)
    pe = pos_emb(seq_len=1)  # shape (1, 1, d_model)

    pos0 = pe[0, 0]  # (d_model,)
    even_dims = pos0[0::2]
    odd_dims = pos0[1::2]

    # At position 0, sin(0) = 0.0 for even dimensions
    assert torch.allclose(even_dims, torch.zeros_like(even_dims), atol=1e-5), (
        "Stage 25 criteria not met: even dimension positional encodings at position 0 must equal sin(0)=0. "
        "Cosine was applied to both even and odd dimensions."
    )

    # At position 0, cos(0) = 1.0 for odd dimensions
    assert torch.allclose(odd_dims, torch.ones_like(odd_dims), atol=1e-5), (
        "Stage 25 criteria not met: odd dimension positional encodings at position 0 must equal cos(0)=1."
    )
