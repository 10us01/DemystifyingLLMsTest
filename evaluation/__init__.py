"""
Evaluation module — metrics, latency benchmarks, and evaluation runner.
"""

from .metrics import calculate_perplexity, cross_entropy_loss, token_accuracy, repetition_rate
from .benchmarks import benchmark_forward_latency, benchmark_generation_throughput
from .evaluate import ModelEvaluator

__all__ = [
    "calculate_perplexity",
    "cross_entropy_loss",
    "token_accuracy",
    "repetition_rate",
    "benchmark_forward_latency",
    "benchmark_generation_throughput",
    "ModelEvaluator",
]
