"""
Challenge Stage 16 Test (Opaque)
"""
import pytest
import torch
import torch.nn as nn
from inference.generation.autoregressive import autoregressive_generate


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 1

    def forward(self, x, kv_cache=None):
        # Always predicts token ID 99 at the next position
        batch_size, seq_len = x.shape
        logits = torch.zeros(batch_size, seq_len, 105)
        logits[:, :, 99] = 100.0
        return logits


def test_stage_16():
    model = DummyModel()
    input_ids = torch.tensor([[1, 2, 3]])
    generated = autoregressive_generate(model, input_ids, max_new_tokens=2, use_cache=False)
    assert generated[0, :3].tolist() == [1, 2, 3], f"Prompt tokens should be at the start, got {generated[0].tolist()}"
    assert generated[0, -1].item() == 99, "Generated tokens should be appended"
