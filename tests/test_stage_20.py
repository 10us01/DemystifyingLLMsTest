"""
Challenge Stage 20 Test (Opaque)
"""
import pytest
import torch
from inference.sampling.top_k import apply_top_k


def test_stage_20():
    logits = torch.tensor([[1.0, 50.0, 2.0, 100.0, 0.5]])
    # Top-2 highest should keep indices 1 (50.0) and 3 (100.0)
    filtered = apply_top_k(logits, k=2)
    assert filtered[0, 3] == 100.0, "Highest logit must be kept"
    assert filtered[0, 1] == 50.0, "Second highest logit must be kept"
    assert filtered[0, 0] == float("-inf"), "Lower logits must be masked"
    assert filtered[0, 4] == float("-inf"), "Lowest logit must be masked"
