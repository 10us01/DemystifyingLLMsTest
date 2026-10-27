"""
Challenge Stage 24 Test (Opaque)
"""
import pytest
from tokenizer.decoding import decode
from tokenizer.vocab import Vocabulary


def test_stage_24():
    vocab = Vocabulary()
    vocab.add_token("the", 10)
    vocab.add_token("cat", 11)
    vocab.add_token("sat", 12)
    # Decoding [10, 11, 12] should give "the cat sat"
    text = decode([10, 11, 12], vocab=vocab, skip_special_tokens=True)
    assert "the" in text, f"Decoder should not strip non-special first token, got: '{text}'"
    assert "sat" in text, f"Decoder should not strip non-special last token, got: '{text}'"
    assert text.strip() == "the cat sat"
