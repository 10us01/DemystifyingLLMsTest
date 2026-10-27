"""
Challenge Stage 10 Test (Opaque)
"""
import pytest
import torch


def test_stage_10():
    from utils.tensor_utils import to_device

    t = torch.randn(2, 4)

    # Test transferring to meta device (supported natively in PyTorch on all platforms)
    target_device = "meta"
    t_moved = to_device(t, target_device)

    assert torch.is_tensor(t_moved), "Stage 10 criteria not met: return value must be a torch.Tensor"
    assert t_moved.device.type == "meta", (
        f"Stage 10 criteria not met: to_device ignored target device '{target_device}', "
        f"tensor remained on '{t_moved.device.type}'"
    )
