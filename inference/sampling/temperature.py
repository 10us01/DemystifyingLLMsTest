"""Module implementation."""
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    """
    Apply temperature scaling to logits.
    """
    probs = F.softmax(logits, dim=-1)
    scaled = probs / temperature
    return scaled
