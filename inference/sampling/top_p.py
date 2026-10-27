"""Nucleus top-p sampling. Assigned: Sarthak. Bug: M5"""
import torch
import torch.nn.functional as F


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Nucleus sampling.
    """
    if p >= 1.0 or p <= 0.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    cumulative_probs = torch.cumsum(probs, dim=-1)

    remove_mask = cumulative_probs > p
    # Shift right to keep the first token that exceeds p
    remove_mask = torch.roll(remove_mask, 1, dims=-1)
    remove_mask[..., 0] = False

    filtered_logits = logits.clone()
    filtered_logits[remove_mask] = float("-inf")
    return filtered_logits
