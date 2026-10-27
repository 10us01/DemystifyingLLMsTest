"""
Dropout wrapper module.

Assigned to: Tanush
Status: Clean — Red Herring (Applies dropout to hidden states and attention weights,
        matching GPT-2 attention dropout specification).
"""

import torch
import torch.nn as nn
from typing import Optional


class Dropout(nn.Module):
    """
    Dropout layer supporting custom probability and activation scaling.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        return self.dropout(x)
