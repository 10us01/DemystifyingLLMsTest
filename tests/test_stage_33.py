"""
Challenge Stage 33 Test (Opaque)
"""
import pytest
import torch
import torch.nn as nn
from inference.generation.autoregressive import autoregressive_generate


class CacheTrackingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 2
        self.call_shapes = []

    def forward(self, x, kv_cache=None):
        self.call_shapes.append(x.shape)
        batch_size, seq_len = x.shape
        logits = torch.zeros(batch_size, seq_len, 50)
        return logits


def test_stage_33():
    model = CacheTrackingModel()
    input_ids = torch.tensor([[1, 2, 3]])
    autoregressive_generate(model, input_ids, max_new_tokens=2, use_cache=True)
    # The subsequent calls with cache enabled should pass only the newly generated token: seq_len = 1
    assert len(model.call_shapes) >= 2
    for shape in model.call_shapes[1:]:
        assert shape[1] == 1, f"With KV cache active, only the new token (seq_len=1) should be passed, got shape {shape}"
