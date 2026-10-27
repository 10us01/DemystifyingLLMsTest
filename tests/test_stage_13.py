"""
Challenge Stage 13 Test (Opaque)
"""
import pytest
import torch


def test_stage_13(tiny_config):
    from model.embeddings.embedding_layer import EmbeddingLayer

    layer = EmbeddingLayer(config=tiny_config, dropout=0.0)
    layer.eval()

    x = torch.tensor([[5, 10, 15, 20]], dtype=torch.long)
    with torch.no_grad():
        out = layer(x)
        tok = layer.token_embedding(x)
        pos = layer.position_embedding(seq_len=x.size(1), device=x.device)

        expected_add = tok + pos
        expected_mul = tok * pos

    assert not torch.allclose(out, expected_mul, atol=1e-4), (
        "Stage 13 criteria not met: embeddings appear to be combined via multiplication instead of addition"
    )
    assert torch.allclose(out, expected_add, atol=1e-4), (
        "Stage 13 criteria not met: embedding output does not match expected vector addition (token + position)"
    )
