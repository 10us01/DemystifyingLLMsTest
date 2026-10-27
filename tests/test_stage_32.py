"""
Challenge Stage 32 Test (Opaque)
"""
import pytest
from tokenizer.bpe import BPETokenizer


def test_stage_32():
    # Merge ("a", "b") -> "ab"
    # Word: "ab ab" or "abab" -> initial pieces: ["a", "b", "a", "b"]
    # With all occurrences merged: ["ab", "ab"]
    # With only first occurrence merged: ["ab", "a", "b"]
    merges = [("a", "b")]
    tokenizer = BPETokenizer(merges=merges)
    pieces = tokenizer.bpe("abab")
    assert pieces == ["ab", "ab"], f"All occurrences of merge pair should be merged, got {pieces}"
