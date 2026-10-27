"""Repetition penalty. Assigned: Sarthak. Bug: H4"""
import math
import torch
from typing import List, Union


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: Union[List[int], torch.Tensor],
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to logits for generated tokens.
    Divide positive logits by the penalty and multiply negative logits by it.
    """
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError(f"penalty must be positive, got {penalty}")
    if penalty == 1.0:
        return logits

    logits = logits.clone().contiguous()
    if isinstance(generated_tokens, torch.Tensor):
        tokens = generated_tokens
    else:
        tokens = torch.as_tensor(generated_tokens, device=logits.device)

    tokens = tokens.to(device=logits.device, dtype=torch.long)
    vocab_size = logits.size(-1)
    if torch.any((tokens < 0) | (tokens >= vocab_size)):
        raise ValueError("generated token ID is outside the logits vocabulary")

    if logits.dim() >= 2 and tokens.dim() >= 2 and tokens.size(0) == logits.size(0):
        rows = logits.reshape(logits.size(0), -1, vocab_size)
        token_rows = tokens.reshape(tokens.size(0), -1)
        for row_idx, row_tokens in enumerate(token_rows):
            ids = torch.unique(row_tokens)
            values = rows[row_idx, :, ids]
            rows[row_idx, :, ids] = torch.where(values > 0, values / penalty, values * penalty)
    else:
        ids = torch.unique(tokens)
        values = logits[..., ids]
        logits[..., ids] = torch.where(values > 0, values / penalty, values * penalty)

    return logits
