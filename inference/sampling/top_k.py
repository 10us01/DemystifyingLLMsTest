"""Top-k sampling. Assigned: Sarthak. Bug: M10"""
import torch


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Mask logits outside top-k.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise ValueError("k must be an integer")
    if k <= 0 or k >= logits.size(-1):
        return logits

    val, _ = torch.topk(logits, k, largest=True, dim=-1)
    threshold = val[..., -1, None]
    filtered_logits = logits.clone()
    filtered_logits[filtered_logits < threshold] = float("-inf")
    return filtered_logits
