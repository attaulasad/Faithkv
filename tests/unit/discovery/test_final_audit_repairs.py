"""Focused regression tests for the final bounded B1 independent-audit
repairs (F1-F9). Every test is CPU-only: real worker bodies run against the
same injected deterministic fakes as `test_b2a_workers_real_bodies`, with
failures injected at specific execution stages."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from test_b2a_workers_real_bodies import (
    CONTROLLED_NUM_LAYERS,
    NUM_LAYERS,
    _build_controlled_discovery_config,
    _build_fake_discovery_config,
    _ControlledModel,
    _ControlledTokenizer,
    _FakeCache,
    _FakeCudaFacade,
    _FakeManifest,
    _FakeModel,
    _FakeTokenizer,
)

from kvcot.discovery import b2a_workers
from kvcot.discovery.b2a_workers import run_fullkv_worker, run_rkv_worker
from kvcot.discovery.execution_measurement import SynchronizedTimer
from kvcot.discovery.worker_partial_evidence import WorkerBodyFailure, WorkerExecutionState


def _install_controlled_environment(monkeypatch):
    from _fake_rkv_fixtures import install_fake_rkv_compression_module

    install_fake_rkv_compression_module(monkeypatch)
    fake_transformers = types.ModuleType("transformers")
    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.DynamicCache = lambda: _FakeCache(CONTROLLED_NUM_LAYERS)
    fake_transformers.cache_utils = fake_cache_utils
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)


def _inject_phase_failure(monkeypatch, matcher, fail_at: int = 1):
    original = SynchronizedTimer.measure
    state = {"count": 0}

    def failing(self, phase, operation):
        if matcher(phase):
            state["count"] += 1
            if state["count"] == fail_at:
                raise RuntimeError(f"injected failure during {phase}")
        return original(self, phase, operation)

    monkeypatch.setattr(SynchronizedTimer, "measure", failing)


def _run_controlled_rkv_expecting_failure(monkeypatch):
    _install_controlled_environment(monkeypatch)
    config = _build_controlled_discovery_config()
    manifest = _FakeManifest()
    model = _ControlledModel(config.rkv)
    with pytest.raises(WorkerBodyFailure) as info:
        run_rkv_worker(
            config, manifest,
            _load_model=lambda: model, _load_tokenizer=lambda: _ControlledTokenizer(),
            _fresh_cache_factory=lambda: _FakeCache(CONTROLLED_NUM_LAYERS),
            _cuda=_FakeCudaFacade(), _device="cpu",
        )
    return info.value.evidence


# ---------------------------------------------------------------------------
# F1: true failing-stage vs last-completed-stage distinction
# ---------------------------------------------------------------------------


def test_worker_execution_state_tracks_current_and_completed_separately():
    state = WorkerExecutionState(attempt_id="a1")
    assert state.current_stage == "worker startup"
    assert state.last_completed_stage is None
    state.enter("model_load")
    assert state.current_stage == "model_load"
    assert state.last_completed_stage is None
    state.complete("model_load")
    state.enter("Pass 1")
    assert state.current_stage == "Pass 1"
    assert state.last_completed_stage == "model_load"


def test_rkv_failure_during_model_load_reports_model_load_not_tokenizer(monkeypatch):
    monkeypatch.setenv("KVCOT_B2A_ATTEMPT_ID", "attempt-f1")
    monkeypatch.delenv("KVCOT_B2A_PROGRESS_PATH", raising=False)
    _install_controlled_environment(monkeypatch)
    config = _build_controlled_discovery_config()

    def exploding_model():
        raise RuntimeError("CUDA out of memory during model load")

    with pytest.raises(WorkerBodyFailure) as info:
        run_rkv_worker(
            config, _FakeManifest(),
            _load_model=exploding_model, _load_tokenizer=lambda: _ControlledTokenizer(),
            _fresh_cache_factory=lambda: _FakeCache(CONTROLLED_NUM_LAYERS),
            _cuda=_FakeCudaFacade(), _device="cpu",
        )
    evidence = info.value.evidence
    assert evidence.failing_stage == "model_load"
    assert evidence.last_completed_stage == "tokenizer_load"
    assert evidence.failing_stage != evidence.last_completed_stage
    assert evidence.attempt_id == "attempt-f1"
    assert evidence.is_oom is True


def test_fullkv_failure_during_model_load_reports_model_load(monkeypatch):
    monkeypatch.setenv("KVCOT_B2A_ATTEMPT_ID", "attempt-f1-full")
    monkeypatch.delenv("KVCOT_B2A_PROGRESS_PATH", raising=False)
    config = _build_fake_discovery_config()

    def exploding_model():
        raise RuntimeError("boom in load")

    with pytest.raises(WorkerBodyFailure) as info:
        run_fullkv_worker(
            config, _FakeManifest(),
            _load_model=exploding_model, _load_tokenizer=lambda: _FakeTokenizer(),
            _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS), _cuda=_FakeCudaFacade(), _device="cpu",
        )
    evidence = info.value.evidence
    assert evidence.failing_stage == "model_load"
    assert evidence.last_completed_stage == "tokenizer_load"
    assert evidence.attempt_id == "attempt-f1-full"


def test_failure_during_pass1_reports_pass1_decode(monkeypatch):
    _inject_phase_failure(monkeypatch, lambda phase: phase == "rkv_pass1_decode", fail_at=1)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert evidence.failing_stage == "rkv_pass1_decode"
    assert evidence.last_completed_stage == "rkv_pass1_prefill"
    assert evidence.failing_stage != evidence.last_completed_stage


def test_failure_during_pass2_reports_pass2_prefill(monkeypatch):
    _inject_phase_failure(monkeypatch, lambda phase: phase == "rkv_pass2_prefill", fail_at=1)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert evidence.failing_stage == "rkv_pass2_prefill"
    assert evidence.last_completed_stage == "rkv_complete_pass1"
    assert evidence.example_aborted is True
    assert evidence.pass1_token_ids  # Pass 1 evidence survives the Pass 2 failure


def test_failure_inside_capture_parity_reports_capture_stage(monkeypatch):
    _inject_phase_failure(monkeypatch, lambda phase: phase == "capture_gather_and_parity", fail_at=1)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert evidence.failing_stage == "capture_gather_and_parity"
    assert evidence.last_completed_stage is not None
    assert evidence.failing_stage != evidence.last_completed_stage


def test_failure_during_pre_branch_admission_reports_admission_stage(monkeypatch):
    def exploding_guard(**kwargs):
        raise RuntimeError("injected pre-branch admission failure")

    monkeypatch.setattr("kvcot.discovery.execution_measurement.check_pre_branch_memory", exploding_guard)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert evidence.failing_stage.startswith("pre-branch admission:")
    assert evidence.last_completed_stage is not None
    assert not str(evidence.last_completed_stage).startswith("pre-branch admission:")


def _run_controlled_rkv_to_completion(monkeypatch) -> dict:
    _install_controlled_environment(monkeypatch)
    config = _build_controlled_discovery_config()
    model = _ControlledModel(config.rkv)
    return run_rkv_worker(
        config, _FakeManifest(),
        _load_model=lambda: model, _load_tokenizer=lambda: _ControlledTokenizer(),
        _fresh_cache_factory=lambda: _FakeCache(CONTROLLED_NUM_LAYERS),
        _cuda=_FakeCudaFacade(), _device="cpu",
    )


def test_failure_during_baseline_evaluation_is_recorded_against_that_pair(monkeypatch):
    """A failure inside a branch evaluation is contained at the PAIR
    boundary by design (pair attrition, never a silent abort) -- the
    recorded failure detail must name the exact injected subphase, and the
    exact-count gate conditions must fail closed."""
    _inject_phase_failure(monkeypatch, lambda phase: phase.endswith(":baseline_bridge_plus_48_scored_tokens"), fail_at=1)
    result = _run_controlled_rkv_to_completion(monkeypatch)
    assert result["completed_real_pair_count"] == 11
    assert result["failed_real_pair_count"] == 1
    assert result["all_required_pair_evaluations_completed"] is False
    assert len(result["failed_pair_identities"]) == 1
    failed = result["failed_pair_identities"][0]
    assert failed["failure_stage"] == "branch_evaluation_failure"
    assert ":baseline_bridge_plus_48_scored_tokens" in failed["failure_detail"]
    details = result["pair_failure_details"]
    assert len(details) == 1
    assert "injected failure during" in details[0]["detail"]


def test_failure_during_semantic_mutation_reports_mutation_subphase(monkeypatch):
    _inject_phase_failure(monkeypatch, lambda phase: phase.endswith(":semantic_mutation"), fail_at=1)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert evidence.failing_stage.endswith(":semantic_mutation")
    assert evidence.failing_stage != evidence.last_completed_stage
    assert evidence.attempted_pair_identities
    assert evidence.completed_pair_identities == []


def test_failure_during_swapped_evaluation_is_recorded_against_that_pair(monkeypatch):
    _inject_phase_failure(monkeypatch, lambda phase: phase.endswith(":swapped_bridge_plus_48_scored_tokens"), fail_at=1)
    result = _run_controlled_rkv_to_completion(monkeypatch)
    assert result["completed_real_pair_count"] == 11
    assert result["failed_real_pair_count"] == 1
    assert result["all_required_pair_evaluations_completed"] is False
    failed = result["failed_pair_identities"][0]
    assert failed["failure_stage"] == "branch_evaluation_failure"
    assert ":swapped_bridge_plus_48_scored_tokens" in failed["failure_detail"]


def test_failure_during_result_construction_reports_result_construction(monkeypatch):
    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("injected result construction failure")

    monkeypatch.setattr(b2a_workers, "RKVWorkerResult", _Boom)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert evidence.failing_stage == "result construction"
    assert evidence.last_completed_stage == "call_trace_comparison"
    # F2: a result-serialization failure preserves the COMPLETE run's
    # already-produced evidence.
    assert len([i for i in evidence.completed_pair_identities if i["pair_kind"] == "real"]) == 12
    assert evidence.no_op_identity is not None
    assert evidence.no_op_evidence
    assert evidence.replay_evidence
    assert evidence.replay_evidence["pass1_token_sha256"] == evidence.replay_evidence["pass2_token_sha256"]
    assert len(evidence.minimized_target_evidence) == 3
    assert len(evidence.semantic_mutation_reports) == 13


def test_fullkv_failure_during_result_construction(monkeypatch):
    class _Boom:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("injected fullkv result construction failure")

    monkeypatch.setattr(b2a_workers, "FullKVWorkerResult", _Boom)
    config = _build_fake_discovery_config()
    with pytest.raises(WorkerBodyFailure) as info:
        run_fullkv_worker(
            config, _FakeManifest(),
            _load_model=lambda: _FakeModel(), _load_tokenizer=lambda: _FakeTokenizer(),
            _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS), _cuda=_FakeCudaFacade(), _device="cpu",
        )
    assert info.value.evidence.failing_stage == "result construction"
    assert info.value.evidence.last_completed_stage == "fullkv_complete_natural_generation"


# ---------------------------------------------------------------------------
# F2: exact partial-evidence preservation at each failure point
# ---------------------------------------------------------------------------


def test_failure_before_first_pair_preserves_zero_pair_evidence(monkeypatch):
    calls = {"n": 0}
    from kvcot.discovery.execution_measurement import check_pre_branch_memory as real_guard

    def guard(**kwargs):
        calls["n"] += 1
        raise RuntimeError("injected before first pair")

    monkeypatch.setattr("kvcot.discovery.execution_measurement.check_pre_branch_memory", guard)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert calls["n"] == 1
    assert len(evidence.attempted_pair_identities) == 1
    assert evidence.completed_pair_identities == []
    assert len(evidence.failed_pair_identities) == 1
    assert evidence.failed_pair_identities[0]["pair_kind"] == "real"
    assert evidence.example_aborted is True
    assert evidence.abort_failure_type == "RuntimeError"
    assert len(evidence.minimized_target_evidence) == 3
    assert evidence.replay_evidence is not None


def test_failure_after_two_completed_pairs_preserves_exact_identities(monkeypatch):
    _inject_phase_failure(monkeypatch, lambda phase: phase.endswith(":baseline_clone"), fail_at=3)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    completed = [i for i in evidence.completed_pair_identities if i["pair_kind"] == "real"]
    attempted = [i for i in evidence.attempted_pair_identities if i["pair_kind"] == "real"]
    assert len(completed) == 2
    assert len(attempted) == 3
    assert len(evidence.failed_pair_identities) == 1
    failed = evidence.failed_pair_identities[0]
    completed_keys = {tuple(sorted(i.items())) for i in completed}
    assert tuple(sorted({k: failed[k] for k in attempted[0]}.items())) not in completed_keys
    assert evidence.example_aborted is True
    assert evidence.no_op_evidence is None  # never reached -- not fabricated
    assert len(evidence.pre_branch_memory_evidence) == 3


def test_failure_after_no_op_completed_preserves_no_op_evidence(monkeypatch):
    """The single B2A no-op is the fifth pair of the FIRST selected event
    (orchestrator NoOpMode.B2A_SINGLE_CALIBRATION) -- failing on the LAST
    real pair therefore exercises 'no_op_evidence when already
    constructed': the completed no-op's evidence must survive into the
    partial-evidence snapshot, never be dropped with the abort."""
    _inject_phase_failure(monkeypatch, lambda phase: phase.startswith("real_pair:") and phase.endswith(":baseline_clone"), fail_at=12)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    completed_real = [i for i in evidence.completed_pair_identities if i["pair_kind"] == "real"]
    completed_noop = [i for i in evidence.completed_pair_identities if i["pair_kind"] == "no_op"]
    assert len(completed_real) == 11
    assert len(completed_noop) == 1
    assert evidence.no_op_identity is not None
    assert evidence.no_op_evidence is not None
    assert evidence.no_op_evidence["baseline_nll"] == evidence.no_op_evidence["no_op_nll"]
    assert len(evidence.failed_pair_identities) == 1
    assert evidence.failed_pair_identities[0]["pair_kind"] == "real"
    assert evidence.example_aborted is True


def test_failure_during_no_op_preserves_prior_real_pairs(monkeypatch):
    _inject_phase_failure(monkeypatch, lambda phase: phase.startswith("no_op_pair:") and phase.endswith(":baseline_clone"), fail_at=1)
    evidence = _run_controlled_rkv_expecting_failure(monkeypatch)
    assert evidence.failing_stage.startswith("no_op_pair:")
    assert evidence.failing_stage.endswith(":baseline_clone")
    completed_real = [i for i in evidence.completed_pair_identities if i["pair_kind"] == "real"]
    assert len(completed_real) == 4  # the first event's four real pairs precede the no-op
    assert evidence.no_op_identity is not None
    assert evidence.no_op_evidence is None  # never fabricated for work not done
    assert len(evidence.failed_pair_identities) == 1
    assert evidence.failed_pair_identities[0]["pair_kind"] == "no_op"
    assert evidence.pass1_compaction_positions == evidence.pass2_compaction_positions


def test_partial_evidence_uses_same_derivations_as_success_path():
    """The failure path and the success path must call the SAME helpers --
    verified structurally so a future edit cannot silently fork them."""
    import inspect

    from kvcot.discovery import worker_partial_evidence

    workers_source = inspect.getsource(b2a_workers)
    partial_source = inspect.getsource(worker_partial_evidence)
    for helper in ("derive_failed_pair_identities", "build_no_op_evidence", "build_replay_evidence"):
        assert helper in workers_source
        assert helper in partial_source


# ---------------------------------------------------------------------------
# F3: memory failure evidence carries the failure message
# ---------------------------------------------------------------------------


def test_memory_phase_failure_preserves_message_and_real_values():
    from kvcot.discovery.execution_measurement import CudaMemoryMeasurer, MemoryMeasuredOperationError

    meter = CudaMemoryMeasurer(_FakeCudaFacade())
    with pytest.raises(MemoryMeasuredOperationError) as info:
        meter.observe("model_load", lambda: (_ for _ in ()).throw(MemoryError("CUDA out of memory: 24GiB")))
    record = info.value.evidence
    assert record.completed is False
    assert record.failure_type == "MemoryError"
    assert record.failure_message == "CUDA out of memory: 24GiB"
    assert record.phase == "model_load"
    assert record.allocated_before == 1000
    assert record.reserved_before == 2000
    assert record.peak_allocated == 12345
    assert record.peak_reserved == 23456
    assert record.synchronized_before is True
    assert record.synchronized_after is True
    assert record.reset_point == "immediately_before:model_load"
    exported = meter.export()[0]
    assert exported["failure_message"] == "CUDA out of memory: 24GiB"


def test_memory_phase_success_has_no_failure_fields():
    from kvcot.discovery.execution_measurement import CudaMemoryMeasurer

    meter = CudaMemoryMeasurer(_FakeCudaFacade())
    meter.observe("model_load", lambda: None)
    record = meter.records[0]
    assert record.completed is True
    assert record.failure_type is None
    assert record.failure_message is None


# ---------------------------------------------------------------------------
# F6: complete provenance
# ---------------------------------------------------------------------------


def test_provenance_reports_ancestry_system_and_software(tmp_path):
    from kvcot.discovery.attempt_artifacts import (
        B1_REPAIR_ROUND4_STARTING_COMMIT,
        B1_REQUIRED_ANCESTOR_SHAS,
        collect_execution_provenance,
        total_physical_ram_bytes,
    )

    provenance = collect_execution_provenance(
        repository=Path.cwd(), expected_rkv_sha="not-the-real-sha", artifact_root=tmp_path,
    )
    git = provenance["git"]
    assert git["starting_commit"] == B1_REPAIR_ROUND4_STARTING_COMMIT
    assert set(git["required_ancestry"]) == set(B1_REQUIRED_ANCESTOR_SHAS)
    assert all(isinstance(v, bool) for v in git["required_ancestry"].values())
    assert isinstance(git["head"], str) and len(git["head"]) == 40
    assert "origin_branch_sha" in git
    assert git["rkv_submodule_match"] is False  # expected sha deliberately wrong
    assert isinstance(git["rkv_submodule_sha"], str)

    system = provenance["system"]
    for key in (
        "os", "platform", "kernel_release", "architecture", "cpu",
        "logical_cpu_count", "total_physical_ram_bytes",
        "artifact_free_disk_bytes", "model_cache_free_disk_bytes",
    ):
        assert key in system
    ram = total_physical_ram_bytes()
    assert ram is None or (isinstance(ram, int) and ram > 0)
    assert system["total_physical_ram_bytes"] == ram or system["total_physical_ram_bytes"] > 0

    software = provenance["software"]
    for package in ("python", "torch", "transformers", "accelerate", "flash-attn",
                    "datasets", "huggingface-hub", "pydantic", "numpy"):
        assert package in software
    assert provenance["gpu_evidence_cross_references"]


def test_provenance_submodule_match_true_for_observed_sha(tmp_path):
    from kvcot.discovery.attempt_artifacts import collect_execution_provenance

    first = collect_execution_provenance(
        repository=Path.cwd(), expected_rkv_sha="x", artifact_root=tmp_path,
    )
    observed = first["git"]["rkv_submodule_sha"]
    second = collect_execution_provenance(
        repository=Path.cwd(), expected_rkv_sha=observed, artifact_root=tmp_path,
    )
    assert second["git"]["rkv_submodule_match"] is True


# ---------------------------------------------------------------------------
# F7: placement gate negatives
# ---------------------------------------------------------------------------


def _valid_placement(**overrides) -> dict:
    base = dict(
        unique_device_types=["cuda"], unique_devices=["cuda:0"], requested_device="cuda:0",
        every_parameter_on_cuda=True, hf_device_map={"": "cuda:0"}, no_offload_verified=True,
        parameter_count=42,
    )
    base.update(overrides)
    return base


def test_placement_gate_accepts_fully_on_device_workers():
    from kvcot.discovery.strict_device import verify_placement_from_raw_evidence

    assert verify_placement_from_raw_evidence(_valid_placement(), _valid_placement()) is True


@pytest.mark.parametrize("mutation", [
    {"requested_device": None},
    {"requested_device": "cuda:1"},
    {"unique_devices": ["cuda:1"]},
    {"unique_devices": ["cpu", "cuda:0"], "unique_device_types": ["cpu", "cuda"]},
    {"hf_device_map": {"lm_head": "disk"}},
    {"hf_device_map": {"model.layers.0": "meta"}},
    {"hf_device_map": {"model.layers.0": "cpu"}},
    {"hf_device_map": {"": "cuda:1"}},
    {"every_parameter_on_cuda": False},
    {"no_offload_verified": False},
    {"parameter_count": 0},
])
def test_placement_gate_rejects_offload_meta_cpu_and_wrong_device(mutation):
    from kvcot.discovery.strict_device import verify_placement_from_raw_evidence

    assert verify_placement_from_raw_evidence(_valid_placement(**mutation), _valid_placement()) is False
    assert verify_placement_from_raw_evidence(_valid_placement(), _valid_placement(**mutation)) is False


def test_device_gate_rejects_missing_or_wrong_requested_device():
    from test_strict_device import _valid_device_evidence

    from kvcot.discovery.strict_device import verify_device_gate_from_raw_evidence

    missing = _valid_device_evidence()
    del missing["requested_device"]
    assert verify_device_gate_from_raw_evidence(missing, _valid_device_evidence()) is False
    wrong = _valid_device_evidence(requested_device="cuda:1")
    assert verify_device_gate_from_raw_evidence(wrong, _valid_device_evidence()) is False


def test_device_gate_rejects_cli_worker_hardware_mismatch():
    from test_strict_device import _valid_device_evidence

    from kvcot.discovery.strict_device import verify_device_gate_from_raw_evidence

    cli = _valid_device_evidence(driver_version="999.99")
    assert verify_device_gate_from_raw_evidence(
        _valid_device_evidence(), _valid_device_evidence(), cli
    ) is False


def test_parameter_placement_derives_device_identity_and_requested_device():
    from types import SimpleNamespace

    from kvcot.discovery.runtime_evidence import derive_parameter_placement

    class _Model:
        def named_parameters(self):
            return iter([
                ("a", SimpleNamespace(device=SimpleNamespace(type="cuda", index=0))),
                ("b", SimpleNamespace(device=SimpleNamespace(type="cuda", index=0))),
            ])

    placement = derive_parameter_placement(_Model(), requested_device="cuda:0")
    assert placement.unique_devices == ("cuda:0",)
    assert placement.requested_device == "cuda:0"
    assert placement.every_parameter_on_cuda is True


# ---------------------------------------------------------------------------
# F9: exact timing/memory multiplicities
# ---------------------------------------------------------------------------


def _timing(phase: str, duration: float = 0.5, completed: bool = True) -> dict:
    return {"phase": phase, "duration_seconds": duration, "completed": completed}


def _pair_names() -> list[str]:
    return [f"real_pair:{i}:1:2" for i in range(12)] + ["no_op_pair:0:3:3"]


def _valid_fullkv_timing() -> list[dict]:
    from kvcot.discovery.final_contract import FULLKV_TIMING_EXACT_MULTIPLICITY

    records = []
    for phase, count in FULLKV_TIMING_EXACT_MULTIPLICITY.items():
        records.extend(_timing(phase) for _ in range(count))
    records.extend(_timing("fullkv_decode") for _ in range(4))
    records.append(_timing("answer_verification"))
    return records


def _valid_rkv_timing() -> list[dict]:
    from kvcot.discovery.final_contract import (
        PAIR_REQUIRED_TIMING_SUBPHASES,
        RKV_TIMING_EXACT_MULTIPLICITY,
    )

    records = []
    for phase, count in RKV_TIMING_EXACT_MULTIPLICITY.items():
        records.extend(_timing(phase) for _ in range(count))
    records.extend(_timing("rkv_pass1_decode") for _ in range(5))
    records.extend(_timing("rkv_pass2_decode") for _ in range(5))
    records.append(_timing("answer_verification"))
    for name in _pair_names():
        records.append(_timing(name))
        records.extend(_timing(f"{name}:{subphase}") for subphase in PAIR_REQUIRED_TIMING_SUBPHASES)
    return records


def _fullkv_actual_calls() -> list[dict]:
    return [{"call_kind": "prefill"}] + [{"call_kind": "decode"} for _ in range(4)]


def _rkv_actual_calls() -> list[dict]:
    return [{"call_kind": "prefill"}, {"call_kind": "prefill"}] + [{"call_kind": "decode"} for _ in range(30)]


def test_timing_contract_accepts_exact_multiplicities():
    from kvcot.discovery.final_contract import timing_contract_satisfied

    assert timing_contract_satisfied(
        _valid_fullkv_timing(), _valid_rkv_timing(),
        fullkv_actual_calls=_fullkv_actual_calls(), rkv_actual_calls=_rkv_actual_calls(),
    ) is True


def test_timing_contract_rejects_duplicate_singleton():
    from kvcot.discovery.final_contract import timing_contract_satisfied

    assert timing_contract_satisfied(
        _valid_fullkv_timing() + [_timing("model_load")], _valid_rkv_timing(),
    ) is False
    assert timing_contract_satisfied(
        _valid_fullkv_timing(), _valid_rkv_timing() + [_timing("rkv_complete_pass1")],
    ) is False


def test_timing_contract_rejects_wrong_capture_count():
    from kvcot.discovery.final_contract import timing_contract_satisfied

    assert timing_contract_satisfied(
        _valid_fullkv_timing(), _valid_rkv_timing() + [_timing("capture_gather_and_parity")],
    ) is False
    reduced = [r for r in _valid_rkv_timing() if r["phase"] != "capture_gather_and_parity"]
    reduced.extend(_timing("capture_gather_and_parity") for _ in range(2))
    assert timing_contract_satisfied(_valid_fullkv_timing(), reduced) is False


def test_timing_contract_rejects_duplicate_pair_identity():
    from kvcot.discovery.final_contract import timing_contract_satisfied

    assert timing_contract_satisfied(
        _valid_fullkv_timing(), _valid_rkv_timing() + [_timing("real_pair:0:1:2")],
    ) is False


def test_timing_contract_rejects_invalid_durations_and_failed_records():
    from kvcot.discovery.final_contract import timing_contract_satisfied

    assert timing_contract_satisfied(
        _valid_fullkv_timing() + [_timing("extra", duration=float("nan"))], _valid_rkv_timing(),
    ) is False
    assert timing_contract_satisfied(
        _valid_fullkv_timing() + [_timing("extra", duration=0.0)], _valid_rkv_timing(),
    ) is False
    assert timing_contract_satisfied(
        _valid_fullkv_timing() + [_timing("extra", completed=False)], _valid_rkv_timing(),
    ) is False


def test_timing_contract_rejects_call_count_disagreement_with_actual_evidence():
    from kvcot.discovery.final_contract import timing_contract_satisfied

    short_actual = [{"call_kind": "prefill"}] + [{"call_kind": "decode"} for _ in range(3)]
    assert timing_contract_satisfied(
        _valid_fullkv_timing(), _valid_rkv_timing(),
        fullkv_actual_calls=short_actual, rkv_actual_calls=_rkv_actual_calls(),
    ) is False
    one_prefill = [{"call_kind": "prefill"}] + [{"call_kind": "decode"} for _ in range(30)]
    assert timing_contract_satisfied(
        _valid_fullkv_timing(), _valid_rkv_timing(),
        fullkv_actual_calls=_fullkv_actual_calls(), rkv_actual_calls=one_prefill,
    ) is False


def _memory(phase: str, completed: bool = True, synchronized: bool = True) -> dict:
    return {
        "phase": phase, "allocated_before": 1, "reserved_before": 2, "peak_allocated": 3,
        "peak_reserved": 4, "allocated_after": 1, "reserved_after": 2,
        "reset_point": f"immediately_before:{phase}", "synchronized_before": synchronized,
        "synchronized_after": synchronized, "completed": completed,
    }


def _valid_memory(role: str) -> list[dict]:
    from kvcot.discovery.final_contract import (
        FULLKV_MEMORY_EXACT_MULTIPLICITY,
        RKV_MEMORY_EXACT_MULTIPLICITY,
    )

    spec = RKV_MEMORY_EXACT_MULTIPLICITY if role == "rkv" else FULLKV_MEMORY_EXACT_MULTIPLICITY
    records = [_memory(phase) for phase in spec]
    if role == "rkv":
        records.extend(_memory(name) for name in _pair_names())
    return records


def test_memory_contract_accepts_exact_multiplicities():
    from kvcot.discovery.final_contract import memory_contract_satisfied

    assert memory_contract_satisfied(_valid_memory("fullkv"), _valid_memory("rkv")) is True


def test_memory_contract_rejects_duplicated_singleton_and_pair_records():
    from kvcot.discovery.final_contract import memory_contract_satisfied

    assert memory_contract_satisfied(
        _valid_memory("fullkv") + [_memory("model_load")], _valid_memory("rkv"),
    ) is False
    assert memory_contract_satisfied(
        _valid_memory("fullkv"), _valid_memory("rkv") + [_memory("real_pair:0:1:2")],
    ) is False


def test_memory_contract_rejects_unsynchronized_or_failed_records():
    from kvcot.discovery.final_contract import memory_contract_satisfied

    assert memory_contract_satisfied(
        _valid_memory("fullkv") + [_memory("extra", synchronized=False)], _valid_memory("rkv"),
    ) is False
    assert memory_contract_satisfied(
        _valid_memory("fullkv"), _valid_memory("rkv") + [_memory("extra", completed=False)],
    ) is False


# ---------------------------------------------------------------------------
# F8: snapshot index/shard revalidation against a real directory
# ---------------------------------------------------------------------------


def _write_snapshot_dir(tmp_path: Path) -> Path:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "config.json").write_text('{"model_type": "llama"}', encoding="utf-8")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"a" * 10)
    (root / "model-00002-of-00002.safetensors").write_bytes(b"b" * 10)
    index = {"weight_map": {"w1": "model-00001-of-00002.safetensors", "w2": "model-00002-of-00002.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def _directory_evidence(root: Path) -> dict:
    from kvcot.discovery.snapshot_boundary import (
        _inventory,
        _parse_weight_indexes,
        _recognized_weight_files,
        compute_inventory_sha256,
    )

    files = _inventory(root)
    inventory = [[name, (root / name).stat().st_size] for name in files]
    index_files, index_hashes, referenced = _parse_weight_indexes(root, files)
    return {
        "repository_id": "org/model", "requested_revision": "a" * 40, "resolved_revision": "a" * 40,
        "asset_type": "model", "local_path": str(root), "files": list(files),
        "total_bytes": sum(size for _, size in inventory), "required_free_bytes": 0,
        "free_bytes": 10**9, "local_files_only": True, "file_count": len(files),
        "file_inventory": inventory, "inventory_sha256": compute_inventory_sha256(inventory),
        "weight_index_files": list(index_files), "weight_index_sha256": [list(pair) for pair in index_hashes],
        "referenced_shards": list(referenced), "missing_referenced_shards": [],
        "recognized_weight_files": list(_recognized_weight_files(files)),
    }


def test_snapshot_disk_revalidation_accepts_exact_valid_snapshot(tmp_path):
    from kvcot.discovery.snapshot_boundary import (
        revalidate_snapshot_evidence_against_directory,
        verify_snapshot_evidence_raw,
    )

    root = _write_snapshot_dir(tmp_path)
    evidence = _directory_evidence(root)
    assert verify_snapshot_evidence_raw(
        evidence, expected_repository_id="org/model", expected_revision="a" * 40, asset_type="model",
    ) is True
    assert revalidate_snapshot_evidence_against_directory(evidence) is True


def test_snapshot_disk_revalidation_rejects_malformed_index(tmp_path):
    from kvcot.discovery.snapshot_boundary import revalidate_snapshot_evidence_against_directory

    root = _write_snapshot_dir(tmp_path)
    evidence = _directory_evidence(root)
    (root / "model.safetensors.index.json").write_text("{malformed", encoding="utf-8")
    assert revalidate_snapshot_evidence_against_directory(evidence) is False


def test_snapshot_disk_revalidation_rejects_missing_referenced_shard(tmp_path):
    from kvcot.discovery.snapshot_boundary import revalidate_snapshot_evidence_against_directory

    root = _write_snapshot_dir(tmp_path)
    evidence = _directory_evidence(root)
    (root / "model-00002-of-00002.safetensors").unlink()
    assert revalidate_snapshot_evidence_against_directory(evidence) is False


def test_snapshot_raw_verifier_rejects_inventory_tampering(tmp_path):
    from kvcot.discovery.snapshot_boundary import verify_snapshot_evidence_raw

    root = _write_snapshot_dir(tmp_path)
    valid = _directory_evidence(root)

    fake_entry = dict(valid)
    fake_entry["file_inventory"] = valid["file_inventory"] + [["injected.bin", 5]]
    assert verify_snapshot_evidence_raw(
        fake_entry, expected_repository_id="org/model", expected_revision="a" * 40, asset_type="model",
    ) is False

    wrong_hash = dict(valid)
    wrong_hash["inventory_sha256"] = "0" * 64
    assert verify_snapshot_evidence_raw(
        wrong_hash, expected_repository_id="org/model", expected_revision="a" * 40, asset_type="model",
    ) is False

    wrong_total = dict(valid)
    wrong_total["total_bytes"] = valid["total_bytes"] + 1
    assert verify_snapshot_evidence_raw(
        wrong_total, expected_repository_id="org/model", expected_revision="a" * 40, asset_type="model",
    ) is False

    wrong_path = dict(valid)
    wrong_path["local_path"] = ""
    assert verify_snapshot_evidence_raw(
        wrong_path, expected_repository_id="org/model", expected_revision="a" * 40, asset_type="model",
    ) is False

    missing_shard = dict(valid)
    missing_shard["missing_referenced_shards"] = ["model-00002-of-00002.safetensors"]
    assert verify_snapshot_evidence_raw(
        missing_shard, expected_repository_id="org/model", expected_revision="a" * 40, asset_type="model",
    ) is False
