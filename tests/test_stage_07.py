"""
Challenge Stage 07 Test (Opaque)
"""
import pytest
from inference.generation.stop_criteria import StopCriteria


def test_stage_07():
    criteria = StopCriteria(max_length=10)
    # At exactly max_length, generation should stop
    assert criteria.should_stop(current_length=10), "Should stop when current_length reaches max_length"
    assert not criteria.should_stop(current_length=9), "Should not stop before max_length"
