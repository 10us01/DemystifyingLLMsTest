"""
Integration Stage 02 Bonus Test (Opaque)
"""

import torch
import pytest


def test_integration_02(tiny_config, sample_input, device):
    try:
        from model.blocks.block_stack import BlockStack
        from model.heads.lm_head import LMHead
        from model.embeddings.embedding_layer import EmbeddingLayer
    except ImportError as e:
        pytest.fail(f"Model submodules import failed: {e}")

    try:
        # Instantiate layers using tiny configuration
        embeddings = EmbeddingLayer()
        blocks = BlockStack()
        head = LMHead()

        # Check forward pass callable
        if not hasattr(embeddings, "forward") and not callable(embeddings):
            pytest.fail("EmbeddingLayer forward method not implemented")

        x = sample_input.to(device)
        batch_size, seq_len = x.shape

        h = embeddings(x)
        assert torch.is_tensor(h), "Embedding output must be a torch.Tensor"
        assert h.shape == (batch_size, seq_len, tiny_config.d_model), (
            f"Expected embedding shape ({batch_size}, {seq_len}, {tiny_config.d_model}), "
            f"got {h.shape}"
        )

        h = blocks(h)
        assert h.shape == (batch_size, seq_len, tiny_config.d_model), (
            f"Expected block stack output shape ({batch_size}, {seq_len}, {tiny_config.d_model}), "
            f"got {h.shape}"
        )

        logits = head(h)
        assert logits.shape == (batch_size, seq_len, tiny_config.vocab_size), (
            f"Expected logits shape ({batch_size}, {seq_len}, {tiny_config.vocab_size}), "
            f"got {logits.shape}"
        )

        # Numerics validation
        assert not torch.isnan(logits).any(), "Model logits contain NaN values"
        assert not torch.isinf(logits).any(), "Model logits contain Inf values"
    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"Model forward pass verification failed: {e}")
