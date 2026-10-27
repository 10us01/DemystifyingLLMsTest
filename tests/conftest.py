import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def seed_everything():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture
def tiny_config():
    from config.model_config import ModelConfig
    return ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=1000,
        max_seq_len=128,
        ffn_multiplier=4,
        dropout=0.0,
    )


@pytest.fixture
def sample_input():
    return torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)


@pytest.fixture
def sample_logits():
    torch.manual_seed(42)
    return torch.randn(1, 1000)


@pytest.fixture
def device():
    return torch.device("cpu")
