"""
Challenge Stage 06 Test (Opaque)
"""
import pytest
import torch
from inference.sampling.temperature import apply_temperature


def test_stage_06():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    # Default temperature or standard temp should not divide by zero or result in inf/nan
    out = apply_temperature(logits)
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    assert out.shape == logits.shape
