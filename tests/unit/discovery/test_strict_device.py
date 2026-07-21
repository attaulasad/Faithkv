from types import SimpleNamespace

import pytest

from kvcot.discovery.strict_device import (
    StrictDeviceError,
    verify_device_gate_from_raw_evidence,
    verify_single_rtx3090,
)


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


# --------------------------------------------------------------------------
# Independent-audit Gate H4.1/H4.2: `verify_device_gate_from_raw_evidence`
# must recompute the single-RTX-3090 policy from raw fields, never trust a
# bare worker-reported `verified=True`.
# --------------------------------------------------------------------------


def _valid_device_evidence(**overrides) -> dict:
    base = dict(
        verified=True, visible_gpu_count=1, gpu_name="NVIDIA GeForce RTX 3090", device_index=0,
        total_vram_bytes=24 * 1024**3, compute_capability=(8, 6), driver_version="555.42",
        cuda_runtime="12.1", cudnn_version="8900", policy_satisfied=True,
    )
    base.update(overrides)
    return base


def test_device_gate_passes_when_both_workers_agree_on_valid_raw_evidence():
    assert verify_device_gate_from_raw_evidence(_valid_device_evidence(), _valid_device_evidence()) is True


def test_device_gate_fails_on_worker_reported_verified_true_with_wrong_gpu_name():
    """A worker that sets `verified=True` but reports a non-3090 GPU name
    must still fail -- the gate never trusts the bare boolean."""
    bad = _valid_device_evidence(gpu_name="NVIDIA A100")
    assert verify_device_gate_from_raw_evidence(bad, _valid_device_evidence()) is False


def test_device_gate_fails_on_two_visible_gpus():
    bad = _valid_device_evidence(visible_gpu_count=2)
    assert verify_device_gate_from_raw_evidence(bad, _valid_device_evidence()) is False


def test_device_gate_fails_on_wrong_device_index():
    bad = _valid_device_evidence(device_index=1)
    assert verify_device_gate_from_raw_evidence(bad, _valid_device_evidence()) is False


@pytest.mark.parametrize("bad_vram", [16 * 1024**3, 48 * 1024**3, None, "24GB"])
def test_device_gate_fails_on_insufficient_or_malformed_vram(bad_vram):
    bad = _valid_device_evidence(total_vram_bytes=bad_vram)
    assert verify_device_gate_from_raw_evidence(bad, _valid_device_evidence()) is False


def test_device_gate_fails_on_mismatched_fullkv_rkv_device_evidence():
    """Both workers individually report `verified=True` and a plausible
    RTX 3090, but disagree on the ACTUAL GPU name/VRAM -- this must fail,
    never pass just because each self-reported success independently."""
    fullkv = _valid_device_evidence(gpu_name="NVIDIA GeForce RTX 3090")
    rkv = _valid_device_evidence(gpu_name="NVIDIA GeForce RTX 3090 Ti")
    assert verify_device_gate_from_raw_evidence(fullkv, rkv) is False

    fullkv2 = _valid_device_evidence(total_vram_bytes=24 * 1024**3)
    rkv2 = _valid_device_evidence(total_vram_bytes=23 * 1024**3)
    assert verify_device_gate_from_raw_evidence(fullkv2, rkv2) is False


def test_device_gate_fails_on_missing_driver_or_cuda_or_cudnn_version():
    for field in ("driver_version", "cuda_runtime", "cudnn_version"):
        bad = _valid_device_evidence(**{field: ""})
        assert verify_device_gate_from_raw_evidence(bad, _valid_device_evidence()) is False, field


def test_device_gate_fails_when_verified_flag_itself_is_false():
    bad = _valid_device_evidence(verified=False)
    assert verify_device_gate_from_raw_evidence(bad, _valid_device_evidence()) is False


def test_device_gate_passes_three_way_when_cli_preflight_agrees():
    """Independent-audit Gate H4.3: when a CLI preflight observation is
    supplied, it must be checked as a THIRD independent observation, never
    silently ignored."""
    assert verify_device_gate_from_raw_evidence(
        _valid_device_evidence(), _valid_device_evidence(), _valid_device_evidence()
    ) is True


def test_device_gate_fails_when_cli_preflight_disagrees_with_workers():
    """Both workers agree with each other but the CLI's OWN pre-launch
    observation reports different hardware -- this must fail, exactly like
    a FullKV/R-KV disagreement would."""
    cli = _valid_device_evidence(gpu_name="NVIDIA GeForce RTX 3090 Ti")
    assert verify_device_gate_from_raw_evidence(
        _valid_device_evidence(), _valid_device_evidence(), cli
    ) is False


def test_device_gate_fails_when_cli_preflight_itself_is_invalid():
    cli = _valid_device_evidence(verified=False)
    assert verify_device_gate_from_raw_evidence(
        _valid_device_evidence(), _valid_device_evidence(), cli
    ) is False


def test_device_gate_two_way_unaffected_when_cli_preflight_omitted():
    """Backward compatibility: omitting `cli_device_preflight` (the
    default, `None`) must behave exactly as the original two-way FullKV/
    R-KV-only check."""
    assert verify_device_gate_from_raw_evidence(_valid_device_evidence(), _valid_device_evidence()) is True
    assert verify_device_gate_from_raw_evidence(
        _valid_device_evidence(), _valid_device_evidence(), None
    ) is True


def test_device_gate_fails_on_cpu_disk_meta_offload_placement_evidence_absent_from_gate():
    """CPU/disk/meta placement is a SEPARATE check
    (`kvcot.discovery.runtime_evidence.derive_parameter_placement`'s
    `no_offload_verified`) -- this device gate does not duplicate it, but
    must still fail closed if the raw evidence dict is simply empty (e.g. a
    worker that never ran real device verification at all)."""
    assert verify_device_gate_from_raw_evidence({}, {}) is False
