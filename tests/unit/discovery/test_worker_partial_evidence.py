"""Independent-audit Gate H1 unit tests for
`kvcot.discovery.worker_partial_evidence`."""
from __future__ import annotations

from kvcot.discovery.worker_partial_evidence import (
    PartialWorkerEvidence,
    WorkerBodyFailure,
    capture_partial_evidence,
    classify_failure,
)


def test_classify_failure_detects_cuda_oom_by_message():
    is_oom, is_timeout = classify_failure(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom is True
    assert is_timeout is False


def test_classify_failure_detects_out_of_memory_error_by_type_name():
    class OutOfMemoryError(RuntimeError):
        pass

    is_oom, _ = classify_failure(OutOfMemoryError("allocator failure"))
    assert is_oom is True


def test_classify_failure_generic_exception_is_neither():
    is_oom, is_timeout = classify_failure(ValueError("bad config"))
    assert is_oom is False
    assert is_timeout is False


def test_classify_failure_detects_timeout():
    class TimeoutExpired(RuntimeError):
        pass

    _, is_timeout = classify_failure(TimeoutExpired("worker took too long"))
    assert is_timeout is True


def test_capture_partial_evidence_with_empty_scope_has_no_fabricated_fields():
    """A genuinely early failure (e.g. before any local variable of
    interest is bound) must report empty/`None` evidence, never a
    fabricated value standing in for a real observation."""
    exc = RuntimeError("config load failed")
    evidence = capture_partial_evidence(role="fullkv", failing_stage="config validation", exc=exc, scope={})
    assert evidence.role == "fullkv"
    assert evidence.failing_stage == "config validation"
    assert evidence.determinism_policy is None
    assert evidence.timing_evidence == []
    assert evidence.memory_phase_evidence == []
    assert evidence.device_evidence is None
    assert evidence.snapshot_evidence is None
    assert evidence.actual_call_evidence == []
    assert evidence.peak_cuda_allocated_bytes is None
    assert evidence.failure_type == "RuntimeError"
    assert evidence.failure_message == "config load failed"
    assert evidence.is_oom is False


class _FakeTimer:
    def __init__(self, records):
        self._records = records

    def export(self):
        return self._records


class _FakeMemoryMeter:
    def __init__(self, records, peak_allocated, peak_reserved):
        self._records = records
        self.maximum_peak_allocated = peak_allocated
        self.maximum_peak_reserved = peak_reserved

    def export(self):
        return self._records


class _FakeActualCalls:
    def __init__(self, events):
        self._events = events

    def export(self):
        return self._events


class _FakeAttritionCounters:
    def __init__(self, total_entered, dropped_at, passed_all):
        self.total_entered = total_entered
        self.dropped_at = dropped_at
        self.passed_all = passed_all


class _FakeDeterminismPolicy:
    def __init__(self):
        self.framework_seed = 13
        self.attention_backend = "sdpa"


def test_capture_partial_evidence_pulls_real_evidence_from_scope():
    """Whatever local variables a worker body had already bound before its
    failure -- `timer`, `memory_meter`, `determinism_policy`, `actual_calls`,
    `device_evidence`, attrition counters -- must be threaded into the
    typed `PartialWorkerEvidence`, never dropped just because the body
    never got a chance to return normally."""
    scope = {
        "timer": _FakeTimer([{"phase": "model_load", "duration_seconds": 1.5, "completed": True}]),
        "memory_meter": _FakeMemoryMeter(
            [{"phase": "model_load", "peak_allocated": 100, "completed": True}], 100, 200
        ),
        "determinism_policy": _FakeDeterminismPolicy(),
        "device_evidence": {"verified": True, "gpu_name": "RTX 3090"},
        "actual_calls": _FakeActualCalls([{"call_kind": "prefill", "batch_size": 1}]),
        "example_attrition": _FakeAttritionCounters(1, {"cap_hit": 0}, 1),
        "pair_attrition": _FakeAttritionCounters(3, {"branch_evaluation_failure": 1}, 2),
    }
    exc = RuntimeError("CUDA out of memory during pair evaluation")
    evidence = capture_partial_evidence(
        role="rkv", failing_stage="real_pair:0:5:9", exc=exc, scope=scope,
        attempt_id="attempt-123", last_completed_stage="model-load completion",
    )

    assert evidence.attempt_id == "attempt-123"
    assert evidence.last_completed_stage == "model-load completion"
    assert evidence.failing_stage == "real_pair:0:5:9"
    assert evidence.timing_evidence == [{"phase": "model_load", "duration_seconds": 1.5, "completed": True}]
    assert evidence.memory_phase_evidence == [{"phase": "model_load", "peak_allocated": 100, "completed": True}]
    assert evidence.peak_cuda_allocated_bytes == 100
    assert evidence.peak_cuda_reserved_bytes == 200
    assert evidence.determinism_policy == {"framework_seed": 13, "attention_backend": "sdpa"}
    assert evidence.device_evidence == {"verified": True, "gpu_name": "RTX 3090"}
    assert evidence.actual_call_evidence == [{"call_kind": "prefill", "batch_size": 1}]
    assert evidence.example_attrition == {"total_entered": 1, "dropped_at": {"cap_hit": 0}, "passed_all": 1}
    assert evidence.pair_attrition == {
        "total_entered": 3, "dropped_at": {"branch_evaluation_failure": 1}, "passed_all": 2,
    }
    assert evidence.is_oom is True
    assert evidence.failure_type == "RuntimeError"


def test_capture_partial_evidence_reads_example_result_pair_evidence():
    """When `run_example` returned an `aborted=True` `ExampleResult` (Gate
    H1's `orchestrator.py` repair) before some LATER unrelated exception
    occurred, its pair-level evidence must still be threaded through."""

    class _FakeTrace:
        full_token_ids = (1, 2, 3, 4)

    class _FakeExampleResult:
        selected_event_evidence = ({"compaction_event_id": 0},)
        attempted_pair_identities = ({"pair_kind": "real"},)
        completed_pair_identities = ()
        semantic_mutation_reports = ({"pair_identity": {"pair_kind": "real"}, "attempted": True},)
        trace = _FakeTrace()
        pass2_replayed_token_ids = (1, 2, 3, 4)

    exc = ValueError("pydantic validation failed while building RKVWorkerResult")
    evidence = capture_partial_evidence(
        role="rkv", failing_stage="result construction", exc=exc,
        scope={"example_result": _FakeExampleResult()},
    )
    assert evidence.selected_event_evidence == [{"compaction_event_id": 0}]
    assert evidence.attempted_pair_identities == [{"pair_kind": "real"}]
    assert evidence.completed_pair_identities == []
    assert evidence.semantic_mutation_reports == [
        {"pair_identity": {"pair_kind": "real"}, "attempted": True}
    ]
    assert evidence.pass1_token_ids == [1, 2, 3, 4]
    assert evidence.pass2_fed_token_ids == [1, 2, 3, 4]
    assert evidence.is_oom is False


def test_worker_body_failure_chains_original_exception_as_cause():
    evidence = PartialWorkerEvidence(role="fullkv", failing_stage="model_load")
    cause = RuntimeError("original failure")
    failure = WorkerBodyFailure(evidence, cause)
    assert failure.__cause__ is cause
    assert failure.evidence is evidence
    assert "model_load" in str(failure)
    assert "original failure" in str(failure)


def test_worker_body_failure_exposes_oom_and_timeout_flags():
    evidence = PartialWorkerEvidence(role="rkv", failing_stage="pair", is_oom=True, is_timeout=False)
    failure = WorkerBodyFailure(evidence, RuntimeError("oom"))
    assert failure.is_oom is True
    assert failure.is_timeout is False
