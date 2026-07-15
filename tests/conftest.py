"""Auto-skip everything marked @pytest.mark.gpu unless a CUDA-capable torch
is actually importable and available. This machine has no GPU and no torch
install, so every GPU test collected here is skipped, never run — see
docs/GPU_VALIDATION_PLAN.md for how they're meant to be executed for real.
"""
import pytest


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _cuda_available():
        return
    skip_gpu = pytest.mark.skip(
        reason="requires a CUDA-capable GPU (torch not installed or no CUDA device on this machine)"
    )
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
