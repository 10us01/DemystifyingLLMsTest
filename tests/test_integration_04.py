"""
Integration Stage 04 Bonus Test (Opaque)
"""

import torch
import pytest


def test_integration_04():
    try:
        from inference.generation.autoregressive import autoregressive_generate
        from config.inference_config import InferenceConfig
    except ImportError as e:
        pytest.fail(f"Autoregressive generation import failed: {e}")

    try:
        prompt_ids = torch.tensor([[101, 2054, 2003]], dtype=torch.long)
        prompt_len = prompt_ids.size(1)
        max_new_tokens = 5

        # Create a mock model forward function that predicts predictable next tokens
        def mock_forward(input_ids, **kwargs):
            batch, seq = input_ids.shape
            # Return synthetic logits where next token is seq + 100
            logits = torch.randn(batch, seq, 1000)
            logits[:, -1, seq + 100] = 50.0  # high confidence
            return logits

        output_ids = autoregressive_generate(
            model=mock_forward,
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            use_cache=False
        )

        assert torch.is_tensor(output_ids), "autoregressive_generate must return a torch.Tensor"
        expected_len = prompt_len + max_new_tokens
        assert output_ids.size(1) == expected_len, (
            f"Expected output sequence length {expected_len}, got {output_ids.size(1)}"
        )

        # Prefix integrity: prompt must remain at the START of the generated sequence
        prefix = output_ids[:, :prompt_len]
        assert torch.equal(prefix, prompt_ids), (
            f"Prompt prefix corrupted or appended in wrong order! Expected {prompt_ids}, got {prefix}"
        )
    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"Generation loop verification failed: {e}")
