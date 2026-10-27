"""
Challenge Stage 08 Test (Opaque)
"""
import pytest


def test_stage_08():
    from serving.model_loader import ModelLoader

    loader = ModelLoader()
    filename = loader.weights_filename

    assert filename.endswith(".pt"), (
        f"Stage 08 criteria not met: weight file must use '.pt' extension, got '{filename}'"
    )
    assert not filename.endswith(".pth"), (
        f"Stage 08 criteria not met: weight file contains invalid '.pth' extension typo: '{filename}'"
    )
    assert filename == "model_weights.pt", (
        f"Stage 08 criteria not met: expected filename 'model_weights.pt', got '{filename}'"
    )
