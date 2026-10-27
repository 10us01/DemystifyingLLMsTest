"""
Challenge Stage 21 Test (Opaque)
"""
import pytest
from tokenizer.encoding import encode
from tokenizer.bpe import BPETokenizer
from tokenizer.vocab import Vocabulary


def test_stage_21():
    vocab = Vocabulary()
    vocab.add_token("hello", 10)
    tokenizer = BPETokenizer(vocab=vocab)
    token_ids = encode("hello", tokenizer=tokenizer, add_bos=True)
    # Token IDs must start with vocab.bos_token_id
    assert len(token_ids) > 0
    assert token_ids[0] == vocab.bos_token_id, f"First token should be BOS ({vocab.bos_token_id}), got {token_ids}"
