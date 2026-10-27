"""
Integration Stage 05 Bonus Test (Opaque)
"""

import pytest


def test_integration_05(tiny_config):
    try:
        from inference.engine.inference_engine import InferenceEngine
        from config.inference_config import InferenceConfig
        from tokenizer import BPETokenizer
    except ImportError as e:
        pytest.fail(f"Full pipeline imports failed: {e}")

    try:
        inference_config = InferenceConfig(max_new_tokens=10, temperature=0.8)
        engine = InferenceEngine(model_config=tiny_config, inference_config=inference_config)
        tokenizer = BPETokenizer()

        prompt = "Once upon a time"

        if hasattr(engine, "generate"):
            result = engine.generate(prompt, tokenizer=tokenizer)
            assert isinstance(result, str), f"Expected string output from engine.generate, got {type(result)}"
            # Ensure the pipeline actually generated output without invoking fallback continuation
            assert "[generated continuation tokens]" not in result, (
                "Inference engine fell back to dummy continuation because core generation failed."
            )
            assert len(result.strip()) > len(prompt), (
                f"Generated text '{result}' did not produce new tokens beyond prompt '{prompt}'"
            )
        else:
            pytest.fail("InferenceEngine lacks 'generate' method")
    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"End-to-end integration pipeline verification failed: {e}")
