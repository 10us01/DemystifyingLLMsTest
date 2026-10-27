"""Module implementation."""
import math
import torch


def apply_temperature(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError(f"temperature must be non-negative, got {temperature}")
    if temperature == 0:
        # A zero temperature denotes greedy decoding; leave argmax unchanged.
        return logits
    return logits / temperature
