"""
Challenge Stage 04 Test (Opaque)
"""
import pytest
from tokenizer.vocab import Vocabulary


def test_stage_04():
    vocab = Vocabulary()
    assert vocab.pad_token_id >= 0, "pad_token_id must be non-negative"
    assert vocab.id_to_token(vocab.pad_token_id) is not None
