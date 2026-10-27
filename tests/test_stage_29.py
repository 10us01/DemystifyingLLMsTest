"""
Challenge Stage 29 Test (Opaque)
"""
import pytest
import torch
from inference.sampling.repetition_penalty import apply_repetition_penalty


def test_stage_29():
    # Multiplicative penalty rule:
    # If logit > 0: logit / penalty
    # If logit < 0: logit * penalty
    # If penalty = 2.0 and logit = 10.0:
    # Multiplicative gives 10.0 / 2.0 = 5.0
    # Additive gives 10.0 - 2.0 = 8.0
    logits = torch.tensor([[10.0, -10.0, 5.0]])
    penalty = 2.0
    penalized = apply_repetition_penalty(logits, generated_tokens=[0, 1], penalty=penalty)

    # For positive logit 10.0 with penalty 2.0:
    # Multiplicative: 10.0 / 2.0 = 5.0
    assert torch.isclose(penalized[0, 0], torch.tensor(5.0)), f"Positive logit should be divided by penalty, got {penalized[0, 0].item()}"
    # For negative logit -10.0 with penalty 2.0:
    # Multiplicative: -10.0 * 2.0 = -20.0
    assert torch.isclose(penalized[0, 1], torch.tensor(-20.0)), f"Negative logit should be multiplied by penalty, got {penalized[0, 1].item()}"
