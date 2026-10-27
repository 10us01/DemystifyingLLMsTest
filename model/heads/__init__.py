"""
Model output heads package initialization.
"""

from .lm_head import LMHead
from .weight_tying import tie_weights, is_tied

__all__ = [
    "LMHead",
    "tie_weights",
    "is_tied",
]
