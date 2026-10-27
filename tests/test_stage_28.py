"""
Challenge Stage 28 Test (Opaque)
"""
import pytest
import torch


def test_stage_28():
    from serving.pipeline import ServingPipeline

    class MockTokenizer:
        def encode(self, text):
            return [101, 202, 303]

        def decode(self, token_ids):
            return "decoded text"

    received_types = []

    class MockModel:
        def __call__(self, x, **kwargs):
            received_types.append(type(x))
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    f"Stage 28 criteria not met: model forward expected torch.Tensor, but received {type(x).__name__}"
                )
            return torch.randn(x.size(0), x.size(1), 10)

    tokenizer = MockTokenizer()
    model = MockModel()
    pipeline = ServingPipeline(model=model, tokenizer=tokenizer)

    try:
        pipeline.process("test input prompt")
    except TypeError as e:
        pytest.fail(str(e))
    except Exception as e:
        pytest.fail(f"Stage 28 execution failure: {e}")

    assert received_types and issubclass(received_types[0], torch.Tensor), (
        f"Stage 28 criteria not met: pipeline input was not converted to torch.Tensor: {received_types}"
    )
