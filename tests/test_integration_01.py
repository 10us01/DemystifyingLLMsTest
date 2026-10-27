"""
Integration Stage 01 Bonus Test (Opaque)
"""

import pytest


def test_integration_01():
    try:
        from tokenizer import BPETokenizer
        from tokenizer.encoding import encode
        from tokenizer.decoding import decode
    except ImportError as e:
        pytest.fail(f"Tokenizer module import failed: {e}")

    sample_text = "The attention mechanism allows language models to capture context."

    try:
        tokenizer = BPETokenizer()
        # Test encode
        tokens = encode(sample_text)
        assert isinstance(tokens, list), f"Expected token list, got {type(tokens)}"
        assert len(tokens) > 0, "Encoded token list must not be empty"
        assert all(isinstance(t, int) for t in tokens), "All token IDs must be integers"

        # Test decode
        reconstructed = decode(tokens)
        assert isinstance(reconstructed, str), f"Expected string, got {type(reconstructed)}"
        assert len(reconstructed.strip()) > 0, "Decoded text must not be empty"

        # Roundtrip check: the reconstructed string should match or contain the original words
        for word in ["the", "attention", "mechanism", "language", "models", "context"]:
            assert word in reconstructed.lower(), (
                f"Tokenizer roundtrip corrupted word '{word}'. "
                f"Got decoded: '{reconstructed}'"
            )
    except AssertionError:
        raise
    except Exception as e:
        pytest.fail(f"Tokenizer integration pipeline threw error: {e}")
