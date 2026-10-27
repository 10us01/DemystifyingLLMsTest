"""
Challenge Stage 27 Test (Opaque)
"""
import pytest
import torch.nn as nn


def test_stage_27():
    from inference.engine.inference_engine import InferenceEngine
    from config.model_config import ModelConfig
    from config.inference_config import InferenceConfig

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x):
            return self.linear(x)

    model = MockModel()
    model.train()  # initial state

    cfg = ModelConfig(d_model=768, vocab_size=50257)
    inf_cfg = InferenceConfig(max_new_tokens=5)
    engine = InferenceEngine(model_config=cfg, inference_config=inf_cfg, model=model)

    assert hasattr(engine, "_prepare_model"), "Stage 27 criteria not met: _prepare_model method missing"
    engine._prepare_model()

    assert not model.training, (
        "Stage 27 criteria not met: model was configured in training mode (.train()) during inference preparation. "
        "Must be set to evaluation mode (.eval())."
    )
