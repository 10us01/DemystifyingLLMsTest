"""
Integration Stage 03 Bonus Test (Opaque)
"""

import torch
import pytest


def test_integration_03(sample_logits):
    try:
        from inference.sampling.sampler import Sampler
        from inference.sampling.temperature import apply_temperature
        from inference.sampling.top_p import apply_top_p
        from inference.sampling.top_k import apply_top_k
    except ImportError as e:
        pytest.fail(f"Sampling module import failed: {e}")

    try:
        logits = sample_logits.clone()

        # 1. Temperature scaling: high temp should increase entropy, low temp decrease entropy
        high_temp_probs = apply_temperature(logits, temperature=2.0)
        low_temp_probs = apply_temperature(logits, temperature=0.5)

        assert torch.is_tensor(high_temp_probs) and torch.is_tensor(low_temp_probs), (
            "apply_temperature must return torch.Tensor"
        )
        assert torch.allclose(high_temp_probs.sum(dim=-1), torch.ones(logits.size(0)), atol=1e-3), (
            "Probability distribution must sum to 1.0"
        )

        entropy_high = -(high_temp_probs * torch.log(high_temp_probs + 1e-9)).sum()
        entropy_low = -(low_temp_probs * torch.log(low_temp_probs + 1e-9)).sum()
        assert entropy_high > entropy_low, (
            "Higher temperature must produce higher entropy distribution than lower temperature"
        )

        # 2. Sampler end-to-end token generation
        sampler = Sampler(temperature=0.8, top_p=0.9, top_k=50)
        token = sampler.sample(logits)
        assert torch.is_tensor(token), "Sampler must return a torch.Tensor"
        assert token.numel() == logits.size(0), "Sampler must return one token ID per batch element"
    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"Sampling pipeline verification failed: {e}")
