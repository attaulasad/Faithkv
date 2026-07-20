"""CPU-only coordinator tests (B1B-R3 §11/§18) -- `subprocess_runner` is
injected so the entire coordination flow (temp dirs, launching both
workers, reading back JSON, schema validation, shared-identity checking,
cleanup) is exercised WITHOUT ever invoking a real Python subprocess,
torch, or CUDA."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcot.discovery.b2a_workers import (
    FullKVWorkerResult,
    RKVWorkerResult,
    WorkerFailedError,
    run_both_workers_via_subprocess,
    validate_shared_identity,
)

MANIFEST_HASH = "m" * 64
PROMPT_HASH = "p" * 64


def _fullkv_payload(**overrides) -> dict:
    payload = dict(
        role="fullkv", model_revision="modelrev", tokenizer_revision="tokrev",
        dataset_repo="HuggingFaceH4/MATH-500", dataset_revision="d" * 40,
        manifest_hash=MANIFEST_HASH, prompt_token_ids_sha256=PROMPT_HASH,
        natural_generated_token_ids=[1, 2, 3], natural_answer="42", natural_answer_status="correct",
        wall_seconds=1.5, peak_cuda_allocated_bytes=1000, peak_cuda_reserved_bytes=2000,
        every_parameter_on_cuda=True, batch_size=1, software_versions={"torch": "2.0"},
    )
    payload.update(overrides)
    return payload


def _rkv_payload(**overrides) -> dict:
    payload = dict(
        role="rkv", model_revision="modelrev", tokenizer_revision="tokrev",
        dataset_repo="HuggingFaceH4/MATH-500", dataset_revision="d" * 40,
        manifest_hash=MANIFEST_HASH, prompt_token_ids_sha256=PROMPT_HASH,
        rkv_upstream_revision="r" * 40, runtime_rkv_config_hash="h" * 64, frozen_rkv_config_hash="h" * 64,
        example_valid=True, event_count=3, observed_retention_ratio=0.5, no_op_numerical_parity=True,
        natural_answer_status="correct", wall_seconds_pass1=1.0, wall_seconds_pass2=1.0,
        wall_seconds_targeted_capture=0.1, wall_seconds_cache_clone_restore=0.05,
        wall_seconds_one_swap=0.01, wall_seconds_bridge_plus_48_scored=0.5,
        peak_cuda_allocated_bytes=1000, peak_cuda_reserved_bytes=2000, every_parameter_on_cuda=True,
        batch_size=1, software_versions={"torch": "2.0"},
    )
    payload.update(overrides)
    return payload


def _make_fake_runner(fullkv_payload: dict | None, rkv_payload: dict | None, *, fail_role: str | None = None):
    """Returns a fake `subprocess_runner(argv) -> CompletedProcess`-shaped
    callable that writes the requested payload to the `--output` path
    named in argv, simulating a successful worker subprocess -- never
    actually launching Python."""

    def runner(argv):
        role = argv[argv.index("--role") + 1]
        output_path = Path(argv[argv.index("--output") + 1])
        if role == fail_role:
            return SimpleNamespace(returncode=1, stdout="", stderr="simulated worker failure")
        payload = fullkv_payload if role == "fullkv" else rkv_payload
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return runner


def test_coordinator_combines_both_workers_successfully():
    runner = _make_fake_runner(_fullkv_payload(), _rkv_payload())
    result = run_both_workers_via_subprocess(
        "configs/discovery/llama8b_math500_b1024.yaml", "configs/discovery/b2a_one_example_manifest.json",
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert isinstance(result.fullkv, FullKVWorkerResult)
    assert isinstance(result.rkv, RKVWorkerResult)
    assert result.shared_identity_ok is True
    assert result.shared_identity_mismatches == ()


def test_coordinator_detects_shared_identity_mismatch():
    runner = _make_fake_runner(_fullkv_payload(manifest_hash=MANIFEST_HASH), _rkv_payload(manifest_hash="x" * 64))
    result = run_both_workers_via_subprocess(
        "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
    )
    assert result.shared_identity_ok is False
    assert any("manifest_hash" in m for m in result.shared_identity_mismatches)


def test_coordinator_raises_on_fullkv_worker_failure():
    runner = _make_fake_runner(_fullkv_payload(), _rkv_payload(), fail_role="fullkv")
    with pytest.raises(WorkerFailedError, match="fullkv worker exited"):
        run_both_workers_via_subprocess(
            "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
        )


def test_coordinator_raises_on_rkv_worker_failure():
    runner = _make_fake_runner(_fullkv_payload(), _rkv_payload(), fail_role="rkv")
    with pytest.raises(WorkerFailedError, match="rkv worker exited"):
        run_both_workers_via_subprocess(
            "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
        )


def test_coordinator_raises_if_worker_reports_success_but_writes_no_file():
    def runner(argv):
        return SimpleNamespace(returncode=0, stdout="", stderr="")  # never writes the output file

    with pytest.raises(WorkerFailedError, match="wrote no output file"):
        run_both_workers_via_subprocess(
            "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
        )


def test_coordinator_rejects_malformed_worker_output():
    def runner(argv):
        output_path = Path(argv[argv.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"role": "fullkv"}), encoding="utf-8")  # missing required fields
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(Exception):  # pydantic ValidationError
        run_both_workers_via_subprocess(
            "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
        )


def test_coordinator_cleans_up_temp_directory_after_success(tmp_path, monkeypatch):
    created_dirs: list[Path] = []
    import tempfile as tempfile_module

    real_mkdtemp = tempfile_module.mkdtemp

    def spying_mkdtemp(*args, **kwargs):
        d = real_mkdtemp(*args, **kwargs)
        created_dirs.append(Path(d))
        return d

    monkeypatch.setattr(tempfile_module, "mkdtemp", spying_mkdtemp)

    runner = _make_fake_runner(_fullkv_payload(), _rkv_payload())
    run_both_workers_via_subprocess("cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner)

    assert len(created_dirs) == 1
    assert not created_dirs[0].exists()


def test_validate_shared_identity_checks_all_four_fields():
    fullkv = FullKVWorkerResult.model_validate(_fullkv_payload())
    rkv = RKVWorkerResult.model_validate(_rkv_payload())
    ok, mismatches = validate_shared_identity(fullkv, rkv)
    assert ok is True
    assert mismatches == []

    rkv_bad = RKVWorkerResult.model_validate(_rkv_payload(dataset_revision="e" * 40))
    ok2, mismatches2 = validate_shared_identity(fullkv, rkv_bad)
    assert ok2 is False
    assert len(mismatches2) == 1
