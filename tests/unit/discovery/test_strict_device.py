from types import SimpleNamespace

import pytest

from kvcot.discovery.strict_device import StrictDeviceError, verify_single_rtx3090


class FakeCuda:
    cudnn = SimpleNamespace(version=lambda: 8900)

    def __init__(self, *, count=1, name="NVIDIA GeForce RTX 3090"):
        self.count = count
        self.name = name

    def device_count(self):
        return self.count

    def current_device(self):
        return 0

    def get_device_properties(self, index):
        return SimpleNamespace(name=self.name, total_memory=24 * 1024**3)

    def get_device_capability(self, index):
        return (8, 6)


def test_strict_device_records_complete_single_3090_evidence():
    result = verify_single_rtx3090(
        FakeCuda(), torch_module=SimpleNamespace(version=SimpleNamespace(cuda="12.1")),
        driver_version_fn=lambda: "555.42",
    )
    assert result.visible_gpu_count == 1
    assert result.device_index == 0
    assert result.total_vram_bytes == 24 * 1024**3
    assert result.compute_capability == (8, 6)
    assert result.policy_satisfied


@pytest.mark.parametrize("count,name", [(0, ""), (2, "NVIDIA GeForce RTX 3090"), (1, "NVIDIA A100")])
def test_strict_device_fails_closed_for_wrong_visibility_or_model(count, name):
    with pytest.raises(StrictDeviceError):
        verify_single_rtx3090(
            FakeCuda(count=count, name=name),
            torch_module=SimpleNamespace(version=SimpleNamespace(cuda="12.1")),
            driver_version_fn=lambda: "555.42",
        )
