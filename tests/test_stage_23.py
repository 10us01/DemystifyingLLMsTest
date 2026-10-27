"""
Challenge Stage 23 Test (Opaque)
"""
import pytest
import torch
from inference.generation.kv_cache import KVCache


def test_stage_23():
    cache = KVCache(num_layers=1)
    k1 = torch.randn(1, 4, 3, 16)
    v1 = torch.randn(1, 4, 3, 16)
    cache.update(0, k1, v1)
    assert cache.get_seq_length(0) == 3

    k2 = torch.randn(1, 4, 1, 16)
    v2 = torch.randn(1, 4, 1, 16)
    cached_k, cached_v = cache.update(0, k2, v2)
    assert cache.get_seq_length(0) == 4, f"Cache should accumulate sequence length, expected 4 got {cache.get_seq_length(0)}"
    assert cached_k.shape[-2] == 4
