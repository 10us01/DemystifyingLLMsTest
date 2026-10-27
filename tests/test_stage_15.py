"""
Challenge Stage 15 Test (Opaque)
"""
import pytest
import torch
from inference.sampling.top_p import apply_top_p


def test_stage_15():
    # If not sorted:
    # logits = [0.1, 10.0, 0.1]. The highest probability is at index 1.
    # Without sorting descending, cumsum on index 0 might be small, but if p=0.8,
    # sorting ensures index 1 (p ~ 0.99) is kept and index 0 & 2 are filtered out.
    logits = torch.tensor([[1.0, 10.0, 2.0]])
    filtered = apply_top_p(logits, p=0.8)
    assert filtered[0, 1] == 10.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 2] == float("-inf")
