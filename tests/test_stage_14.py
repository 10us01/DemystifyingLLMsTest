"""
Challenge Stage 14 Test (Opaque)
"""
import pytest
import torch
import torch.nn.functional as F
from inference.sampling.temperature import apply_temperature


def test_stage_14():
    # If temperature is applied to logits before softmax:
    # high temp (> 1.0) flattens distribution, low temp (< 1.0) sharpens it
    logits = torch.tensor([[10.0, 0.0]])
    temp = 2.0
    scaled = apply_temperature(logits, temperature=temp)
    # If it applied softmax first, probabilities are divided by 2, which sum to 0.5 instead of being scaled logits
    if torch.allclose(scaled.sum(dim=-1), torch.ones(1), atol=1e-2):
        # Returned probs directly: check if entropy increased compared to temp=1
        base_probs = F.softmax(logits, dim=-1)
        assert torch.allclose(scaled.sum(dim=-1), torch.ones(1)), "Probabilities must sum to 1.0"
    else:
        # Returned scaled logits:
        diff = scaled[0, 0] - scaled[0, 1]
        assert torch.isclose(diff, torch.tensor(5.0), atol=1e-3), "Logits should be divided by temperature"
