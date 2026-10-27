"""
Challenge Stage 03 Test (Opaque)
"""
import pytest


def test_stage_03():
    from config import constants

    assert hasattr(constants, "EOS_TOKEN_ID"), "Stage 03 criteria not met: EOS_TOKEN_ID missing"
    eos_id = constants.EOS_TOKEN_ID

    assert eos_id == 50256, (
        f"Stage 03 criteria not met: expected EOS_TOKEN_ID=50256, got {eos_id}"
    )
    assert 0 <= eos_id < 50257, (
        f"Stage 03 criteria not met: EOS_TOKEN_ID ({eos_id}) outside valid vocabulary range [0, 50257)"
    )
