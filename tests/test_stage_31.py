"""
Challenge Stage 31 Test (Opaque)
"""
import pytest
from tokenizer.bpe import BPETokenizer
from tokenizer.vocab import Vocabulary


def test_stage_31():
    # Merges defined with specific rank/priority order:
    # Let word = "abc"
    # Merge 1 (higher rank): ('b', 'c') -> "bc"
    # Merge 2 (lower rank): ('a', 'b') -> "ab"
    # Alphabetical order processes ('a', 'b') first, resulting in ["ab", "c"].
    # Rank order processes ('b', 'c') first, resulting in ["a", "bc"].
    merges = [("b", "c"), ("a", "b")]
    tokenizer = BPETokenizer(merges=merges)
    pieces = tokenizer.bpe("abc")
    assert pieces == ["a", "bc"], f"Merges should be applied by rank order, got {pieces}"
