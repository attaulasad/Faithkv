"""CPU-only coordinator tests (B1B-R4 §11/§16/§18/§19) -- `subprocess_runner`
is injected so the entire coordination flow (temp dirs, launching both
workers, reading back JSON, schema validation, shared-identity checking,
cleanup, timeout handling) is exercised WITHOUT ever invoking a real Python
subprocess, torch, or CUDA."""
from __future__ import annotations

import json
import subprocess
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

_DETERMINISM_POLICY = dict(
    framework_seed=13, python_random_seeded=True, torch_cpu_seeded=True, torch_cuda_seeded=True,
    cudnn_deterministic_requested=True, attention_backend="flash_attention_2",
    bitwise_determinism_guaranteed=False, tolerance_note="note",
)
_RUNTIME_GENERATION = dict(
    generation_mode="greedy", do_sample=False, temperature=None, top_p=None, batch_size=1, max_new_tokens=48,
    eos_token_id=99, eos_append_feed_policy="p", one_prefill_policy="p", single_token_decode_policy="p",
    attention_backend="flash_attention_2", cache_implementation="DynamicCache", framework_seed=13,
    prompt_token_count=200,
)
_PARAMETER_PLACEMENT = dict(
    unique_device_types=["cuda"], every_parameter_on_cuda=True, hf_device_map=None, no_offload_verified=True,
    parameter_count=100,
)
_RUNTIME_IDENTITY = dict(
    requested_model_revision="modelrev", resolved_model_revision="modelrev", model_revision_match=True,
    requested_tokenizer_revision="tokrev", resolved_tokenizer_revision="tokrev", tokenizer_revision_match=True,
)
_MEMORY = dict(
    allocated_before_reset_bytes=100, reserved_before_reset_bytes=200, peak_allocated_bytes=1000,
    peak_reserved_bytes=2000, reset_point="after_model_and_tokenizer_load_before_measured_inference",
)


def _fullkv_payload(**overrides) -> dict:
    payload = dict(
        role="fullkv", model_revision="modelrev", tokenizer_revision="tokrev",
        dataset_repo="HuggingFaceH4/MATH-500", dataset_revision="d" * 40,
        manifest_hash=MANIFEST_HASH, prompt_token_ids_sha256=PROMPT_HASH, prompt_token_count=200,
        natural_generated_token_ids=[1, 2, 3], natural_answer="42", natural_answer_status="correct",
        cap_hit=False, prefill_call_count=1, decode_call_count=3, call_boundary_trace_hash="t" * 64,
        wall_seconds=1.5,
        determinism_policy=_DETERMINISM_POLICY, runtime_generation=_RUNTIME_GENERATION,
        runtime_generation_config_hash="g" * 64, parameter_placement=_PARAMETER_PLACEMENT,
        runtime_identity=_RUNTIME_IDENTITY, memory=_MEMORY,
        peak_cuda_allocated_bytes=1000, peak_cuda_reserved_bytes=2000,
        every_parameter_on_cuda=True, batch_size=1, software_versions={"torch": "2.0"},
    )
    payload.update(overrides)
    return payload


def _rkv_payload(**overrides) -> dict:
    payload = dict(
        role="rkv", model_revision="modelrev", tokenizer_revision="tokrev",
        dataset_repo="HuggingFaceH4/MATH-500", dataset_revision="d" * 40,
        manifest_hash=MANIFEST_HASH, prompt_token_ids_sha256=PROMPT_HASH, prompt_token_count=200,
        rkv_upstream_revision="r" * 40, runtime_rkv_config_hash="h" * 64, frozen_rkv_config_hash="h" * 64,
        rkv_config_hash_match=True,
        example_valid=True, natural_answer_status="correct",
        token_identical_replay=True, prefill_decode_boundary_parity=True, compaction_position_equality=True,
        capture_gather_parity=True, absolute_position_parity=True, no_op_numerical_parity=True,
        pass1_call_boundary={"prefill_call_count": 1, "prefill_token_count": 200, "decode_call_count": 300, "ordered_trace_hash": "a" * 64},
        pass2_call_boundary={"prefill_call_count": 1, "prefill_token_count": 200, "decode_call_count": 300, "ordered_trace_hash": "a" * 64},
        observed_total_compaction_events=5, eligible_compaction_events=3, selected_compaction_events=3,
        events_with_at_least_one_completed_real_pair=3,
        events_with_all_four_real_pairs_completed=3, attempted_real_pair_count=12, completed_real_pair_count=12,
        failed_real_pair_count=0, attempted_no_op_pair_count=1, completed_no_op_pair_count=1,
        pair_failure_details=[],
        selected_event_count_exact=True, real_pair_count_exact=True, no_op_count_exact=True,
        all_required_pair_evaluations_completed=True,
        observed_retention_ratio=0.5,
        wall_seconds_pass1=1.0, wall_seconds_pass2=1.0, wall_seconds_targeted_capture=0.1,
        real_pair_wall_seconds=[1.0] * 12, no_op_pair_wall_seconds=[0.5],
        determinism_policy=_DETERMINISM_POLICY, runtime_generation=_RUNTIME_GENERATION,
        runtime_generation_config_hash="g" * 64, parameter_placement=_PARAMETER_PLACEMENT,
        runtime_identity=_RUNTIME_IDENTITY, memory=_MEMORY, minimized_target_evidence=[],
        peak_cuda_allocated_bytes=1000, peak_cuda_reserved_bytes=2000, every_parameter_on_cuda=True,
        batch_size=1, software_versions={"torch": "2.0"},
    )
    payload.update(overrides)
    return payload


def _make_fake_runner(fullkv_payload: dict | None, rkv_payload: dict | None, *, fail_role: str | None = None):
    """Returns a fake `subprocess_runner(argv, **kwargs) -> CompletedProcess`
    -shaped callable that writes the requested payload to the `--output`
    path named in argv, simulating a successful worker subprocess -- never
    actually launching Python. Accepts (and ignores) the real
    `capture_output`/`text`/`timeout`/`check` kwargs the coordinator now
    always passes (B1B-R4 §16)."""

    def runner(argv, **kwargs):
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        assert kwargs.get("timeout") is not None
        assert kwargs.get("check") is False
        role = argv[argv.index("--role") + 1]
        output_path = Path(argv[argv.index("--output") + 1])
        if role == fail_role:
            return SimpleNamespace(returncode=1, stdout="", stderr="simulated worker failure")
        payload = fullkv_payload if role == "fullkv" else rkv_payload
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return runner


# --------------------------------------------------------------------------
# B1B-R4.1 §28: PYTHONHASHSEED set on the subprocess ENVIRONMENT before
# launch (random.seed() inside the already-running worker cannot do this)
# --------------------------------------------------------------------------


def test_framework_seed_for_env_reads_the_real_frozen_config():
    from kvcot.discovery.b2a_workers import _framework_seed_for_env

    assert _framework_seed_for_env("configs/discovery/llama8b_math500_b1024.yaml") == 13


def test_framework_seed_for_env_falls_back_to_schema_default_for_an_unloadable_path():
    from kvcot.discovery.b2a_workers import _framework_seed_for_env
    from kvcot.discovery.discovery_config import DiscoveryGenerationLock

    assert _framework_seed_for_env("this/path/does/not/exist.yaml") == (
        DiscoveryGenerationLock.model_fields["framework_seed"].default
    )


def test_launch_worker_sets_pythonhashseed_and_tokenizers_parallelism_on_the_subprocess_env(tmp_path):
    from kvcot.discovery.b2a_workers import _launch_worker

    captured_env: dict[str, object] = {}

    def _runner(argv, **kwargs):
        captured_env.update(kwargs.get("env") or {})
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    _launch_worker(
        "fullkv", "configs/discovery/llama8b_math500_b1024.yaml", "manifest.json", tmp_path / "out.json",
        "fake-python", _runner, 60,
    )
    assert captured_env["PYTHONHASHSEED"] == "13"
    assert captured_env["TOKENIZERS_PARALLELISM"] == "false"
    # The subprocess environment must still inherit the parent's own
    # environment (PATH etc.), never a stripped-down replacement.
    import os

    assert captured_env.get("PATH") == os.environ.get("PATH")


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


def test_fullkv_failure_error_carries_no_partial_result():
    runner = _make_fake_runner(_fullkv_payload(), _rkv_payload(), fail_role="fullkv")
    try:
        run_both_workers_via_subprocess("cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner)
        assert False, "expected WorkerFailedError"
    except WorkerFailedError as exc:
        assert exc.partial_fullkv_result is None


def test_coordinator_raises_on_rkv_worker_failure():
    runner = _make_fake_runner(_fullkv_payload(), _rkv_payload(), fail_role="rkv")
    with pytest.raises(WorkerFailedError, match="rkv worker exited"):
        run_both_workers_via_subprocess(
            "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
        )


def test_rkv_failure_preserves_partial_fullkv_result():
    """B1B-R4 §16: FullKV evidence must survive an R-KV failure -- never
    discarded."""
    runner = _make_fake_runner(_fullkv_payload(), _rkv_payload(), fail_role="rkv")
    try:
        run_both_workers_via_subprocess("cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner)
        assert False, "expected WorkerFailedError"
    except WorkerFailedError as exc:
        assert isinstance(exc.partial_fullkv_result, FullKVWorkerResult)
        assert exc.partial_fullkv_result.role == "fullkv"


def test_coordinator_raises_on_worker_timeout():
    def runner(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 0))

    with pytest.raises(WorkerFailedError, match="timed out"):
        run_both_workers_via_subprocess("cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner)


def test_coordinator_raises_if_worker_reports_success_but_writes_no_file():
    def runner(argv, **kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")  # never writes the output file

    with pytest.raises(WorkerFailedError, match="wrote no output file"):
        run_both_workers_via_subprocess(
            "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
        )


def test_coordinator_rejects_malformed_worker_output():
    def runner(argv, **kwargs):
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
