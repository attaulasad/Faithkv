"""Durable partial worker evidence (B1B-R4-independent-audit Gate H1).

`run_fullkv_worker`/`run_rkv_worker` (`kvcot.discovery.b2a_workers`)
accumulate real, serializable evidence -- timing records, memory records,
the applied determinism policy, actual model-call evidence, device/snapshot
evidence -- incrementally, in local variables, over their (long) bodies. If
any statement raises partway through (a real CUDA OOM during model load or
mid-inference is the production-relevant case; a CPU test injects a fake
exception at any of these points), the ORIGINAL code let every one of those
local variables simply go out of scope with the propagating exception --
`kvcot.discovery.b2a_worker_entry.main()`'s except-block had access only to
the raw exception, and constructed a failure envelope with
`partial_measurements=None, determinism_policy=None` regardless of how much
real evidence existed a moment before the failure.

This module is the fix: `capture_partial_evidence` reads whatever
evidence-bearing local variables happen to already be bound in the calling
function's scope (via `locals()`, passed in explicitly -- never inspected by
frame-walking, which would be fragile and implicit) and assembles a typed,
schema-validated `PartialWorkerEvidence` snapshot. `WorkerBodyFailure` is
the exception `run_fullkv_worker`/`run_rkv_worker` raise (chaining the
original exception as `__cause__`, never swallowing it) so
`b2a_worker_entry.main()` can thread this real partial evidence into the
failure envelope instead of `None`.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, NoReturn

from pydantic import BaseModel, Field

# Exact substrings PyTorch's own CUDA allocator uses across the versions
# this repository has observed (`RuntimeError: CUDA out of memory. ...`,
# and modern `torch.cuda.OutOfMemoryError`, itself a `RuntimeError`
# subclass whose class name is `OutOfMemoryError`) -- checked by message
# content, never by `isinstance` against `torch.cuda.OutOfMemoryError`,
# so this module (like the rest of `kvcot.discovery`'s CPU-testable core)
# never imports torch itself.
_OOM_MARKERS = ("out of memory", "cuda oom", "cublas_status_alloc_failed")


def classify_failure(exc: BaseException) -> tuple[bool, bool]:
    """Returns `(is_oom, is_timeout)`. Never both -- a worker-body exception
    is classified as a timeout only when it is genuinely a timeout error
    type (this module's callers only ever see in-process exceptions; the
    subprocess-level `subprocess.TimeoutExpired` case is handled separately
    by the coordinator, `kvcot.discovery.b2a_workers.run_both_workers_via_subprocess`,
    which never reaches this function)."""
    type_name = type(exc).__name__
    message = str(exc).lower()
    is_oom = type_name == "OutOfMemoryError" or any(marker in message for marker in _OOM_MARKERS)
    is_timeout = type_name in ("TimeoutError", "TimeoutExpired")
    return is_oom, is_timeout


@dataclass
class WorkerExecutionState:
    """F1 (final independent-audit repair): explicit, typed execution-state
    tracking for a worker body. `current_stage` is updated immediately
    BEFORE each material operation begins; `last_completed_stage` is
    updated only AFTER an operation completes successfully. On failure,
    `failing_stage = current_stage` -- never the last COMPLETED stage,
    which the prior implementation wrongly reported as the failing one."""

    attempt_id: str | None = None
    current_stage: str = "worker startup"
    last_completed_stage: str | None = None
    entered_stages: list[str] = field(default_factory=list)

    def enter(self, stage: str) -> None:
        self.current_stage = stage
        self.entered_stages.append(stage)

    def complete(self, stage: str | None = None) -> None:
        self.last_completed_stage = stage if stage is not None else self.current_stage

    @contextmanager
    def track(self, stage: str) -> Iterator[None]:
        """R1 (residual independent-audit repair): the preferred call shape
        for a material operation that is not already routed through
        `SynchronizedTimer.measure` -- `enter(stage)` before the body runs,
        `complete(stage)` only if the body returns normally. On an
        exception, `complete` is never reached, so `current_stage` stays at
        `stage` and `last_completed_stage` stays at whatever finished
        before it -- `failing_stage` and `last_completed_stage` can never
        collide for an operation wrapped this way."""
        self.enter(stage)
        yield
        self.complete(stage)


class PartialWorkerEvidence(BaseModel):
    """Whatever real evidence a worker body accumulated before failing.

    Every field is optional/defaulted -- a genuinely early failure (e.g.
    config load, before any local variable of interest is even assigned)
    legitimately has almost nothing to report, and that must be visible as
    `None`/empty, never backfilled with a fabricated value."""

    role: str
    attempt_id: str | None = None
    last_completed_stage: str | None = None
    failing_stage: str | None = None

    determinism_policy: dict[str, Any] | None = None
    timing_evidence: list[dict[str, Any]] = Field(default_factory=list)
    memory_phase_evidence: list[dict[str, Any]] = Field(default_factory=list)
    device_evidence: dict[str, Any] | None = None
    snapshot_evidence: dict[str, Any] | None = None
    runtime_identity: dict[str, Any] | None = None
    parameter_placement: dict[str, Any] | None = None

    actual_call_evidence: list[dict[str, Any]] = Field(default_factory=list)
    pass1_token_ids: list[int] | None = None
    pass2_fed_token_ids: list[int] | None = None
    pass1_compaction_positions: list[int] | None = None
    pass2_compaction_positions: list[int] | None = None
    selected_event_evidence: list[dict[str, Any]] = Field(default_factory=list)
    minimized_target_evidence: list[dict[str, Any]] = Field(default_factory=list)
    attempted_pair_identities: list[dict[str, Any]] = Field(default_factory=list)
    completed_pair_identities: list[dict[str, Any]] = Field(default_factory=list)
    failed_pair_identities: list[dict[str, Any]] = Field(default_factory=list)
    # B2A-R2 forensic repair
    # (docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md): exactly
    # the SwapPairRecord objects `example_result.pair_records` already held
    # at the moment of failure -- never padded to a target count, never
    # fabricated for a pair that never completed. Empty for a failure before
    # any pair evaluation, by construction.
    pair_records: list[dict[str, Any]] = Field(default_factory=list)
    pair_failure_details: list[dict[str, Any]] = Field(default_factory=list)
    semantic_mutation_reports: list[dict[str, Any]] = Field(default_factory=list)
    pre_branch_memory_evidence: list[dict[str, Any]] = Field(default_factory=list)
    no_op_identity: dict[str, Any] | None = None
    no_op_evidence: dict[str, Any] | None = None
    replay_evidence: dict[str, Any] | None = None

    example_attrition: dict[str, Any] | None = None
    pair_attrition: dict[str, Any] | None = None
    example_aborted: bool = False
    abort_failure_type: str | None = None
    abort_failure_message: str | None = None
    abort_is_oom: bool = False

    peak_cuda_allocated_bytes: int | None = None
    peak_cuda_reserved_bytes: int | None = None

    is_oom: bool = False
    is_timeout: bool = False
    failure_type: str | None = None
    failure_message: str | None = None


def _export_timer(scope: dict[str, Any]) -> list[dict[str, Any]]:
    timer = scope.get("timer")
    if timer is None or not hasattr(timer, "export"):
        return []
    return timer.export()


def _export_memory(scope: dict[str, Any]) -> tuple[list[dict[str, Any]], int | None, int | None]:
    memory_meter = scope.get("memory_meter")
    if memory_meter is None or not hasattr(memory_meter, "export"):
        return [], None, None
    return (
        memory_meter.export(),
        int(memory_meter.maximum_peak_allocated),
        int(memory_meter.maximum_peak_reserved),
    )


def _attrition_snapshot(scope: dict[str, Any], name: str) -> dict[str, Any] | None:
    counters = scope.get(name)
    if counters is None:
        return None
    return {
        "total_entered": counters.total_entered,
        "dropped_at": dict(counters.dropped_at),
        "passed_all": counters.passed_all,
    }


def _dataclass_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return dict(value.__dict__)


def capture_partial_evidence(
    *,
    role: str,
    failing_stage: str,
    exc: BaseException,
    scope: dict[str, Any],
    attempt_id: str | None = None,
    last_completed_stage: str | None = None,
) -> PartialWorkerEvidence:
    """Builds a `PartialWorkerEvidence` snapshot from whatever local
    variables are already bound in `scope` (the failing function's own
    `locals()`, captured by the caller's `except` block -- so this reads
    exactly what that specific stack frame actually had, never a guess).
    """
    is_oom, is_timeout = classify_failure(exc)
    timing_evidence = _export_timer(scope)
    memory_phase_evidence, peak_allocated, peak_reserved = _export_memory(scope)

    device_evidence = scope.get("device_evidence")
    model_snapshot = scope.get("model_snapshot")
    tokenizer_snapshot = scope.get("tokenizer_snapshot")
    snapshot_evidence = None
    if model_snapshot is not None or tokenizer_snapshot is not None:
        snapshot_evidence = {
            "model": _dataclass_dict(model_snapshot),
            "tokenizer": _dataclass_dict(tokenizer_snapshot),
        }

    actual_calls = scope.get("actual_calls")
    actual_call_evidence = actual_calls.export() if actual_calls is not None else []

    example_result = scope.get("example_result")
    selected_event_evidence: list[dict[str, Any]] = []
    minimized_target_evidence: list[dict[str, Any]] = []
    attempted_pair_identities: list[dict[str, Any]] = []
    completed_pair_identities: list[dict[str, Any]] = []
    failed_pair_identities: list[dict[str, Any]] = []
    pair_failure_details: list[dict[str, Any]] = []
    semantic_mutation_reports: list[dict[str, Any]] = []
    pre_branch_memory_evidence: list[dict[str, Any]] = []
    no_op_identity: dict[str, Any] | None = None
    no_op_evidence: dict[str, Any] | None = None
    replay_evidence: dict[str, Any] | None = None
    pair_records: list[dict[str, Any]] = []
    pass1_token_ids: list[int] | None = None
    pass2_fed_token_ids: list[int] | None = None
    pass1_compaction_positions: list[int] | None = None
    pass2_compaction_positions: list[int] | None = None
    example_aborted = False
    abort_failure_type: str | None = None
    abort_failure_message: str | None = None
    abort_is_oom = False
    if example_result is not None:
        selected_event_evidence = list(example_result.selected_event_evidence)
        attempted_pair_identities = list(example_result.attempted_pair_identities)
        completed_pair_identities = list(example_result.completed_pair_identities)
        semantic_mutation_reports = list(example_result.semantic_mutation_reports)
        # B2A-R2 forensic repair: exactly the records that genuinely
        # completed before the failure -- never padded, never fabricated.
        # `getattr` (matching this function's existing defensive pattern for
        # every other optional attribute) so a test double or a genuinely
        # early failure that never reached pair construction contributes an
        # honestly empty list rather than raising.
        pair_records = [
            record.model_dump(mode="json") for record in getattr(example_result, "pair_records", ())
        ]
        minimized_target_evidence = [
            _dataclass_dict(item) for item in getattr(example_result, "minimized_target_evidence", ())
        ]
        pre_branch_memory_evidence = [
            _dataclass_dict(item) for item in getattr(example_result, "pre_branch_memory_evidence", ())
        ]
        raw_failure_details = list(getattr(example_result, "pair_failure_details", ()))
        pair_failure_details = [_dataclass_dict(item) for item in raw_failure_details]
        no_op_identity = next(
            (identity for identity in attempted_pair_identities if identity.get("pair_kind") == "no_op"), None
        )
        example_aborted = bool(getattr(example_result, "aborted", False))
        abort_failure_type = getattr(example_result, "abort_failure_type", None)
        abort_failure_message = getattr(example_result, "abort_failure_message", None)
        abort_is_oom = bool(getattr(example_result, "abort_is_oom", False))
        if example_result.trace is not None:
            pass1_token_ids = list(example_result.trace.full_token_ids)
        pass2_fed_token_ids = list(example_result.pass2_replayed_token_ids) or None

        # The scientific derivations below are the SAME single helpers the
        # successful worker path uses (`kvcot.discovery.b2a_evidence`) --
        # never a second, divergent reimplementation. Each is guarded so a
        # partially-constructed (or test-fake) example result contributes
        # whatever it genuinely has, and nothing more is fabricated.
        from kvcot.discovery.b2a_evidence import (
            build_no_op_evidence,
            build_replay_evidence,
            derive_compaction_positions,
            derive_failed_pair_identities,
        )

        failed_pair_identities = derive_failed_pair_identities(
            attempted_pair_identities, completed_pair_identities, raw_failure_details
        )
        if hasattr(example_result, "pair_records"):
            no_op_evidence = build_no_op_evidence(example_result) or None
        if example_result.trace is not None and hasattr(example_result.trace, "compaction_events"):
            pass1_compaction_positions, pass2_compaction_positions = derive_compaction_positions(example_result)
        instrumented = scope.get("instrumented")
        if instrumented is not None and example_result.trace is not None:
            replay_evidence = build_replay_evidence(
                example_result,
                pass1_events=instrumented.pass1_trace.events,
                pass2_events=instrumented.pass2_trace.events,
                actual_call_export=actual_call_evidence,
            )

    return PartialWorkerEvidence(
        role=role,
        attempt_id=attempt_id,
        last_completed_stage=last_completed_stage,
        failing_stage=failing_stage,
        determinism_policy=_dataclass_dict(scope.get("determinism_policy")),
        timing_evidence=timing_evidence,
        memory_phase_evidence=memory_phase_evidence,
        device_evidence=device_evidence,
        snapshot_evidence=snapshot_evidence,
        runtime_identity=_dataclass_dict(scope.get("runtime_identity")),
        parameter_placement=_dataclass_dict(scope.get("parameter_placement")),
        actual_call_evidence=actual_call_evidence,
        pass1_token_ids=pass1_token_ids,
        pass2_fed_token_ids=pass2_fed_token_ids,
        pass1_compaction_positions=pass1_compaction_positions,
        pass2_compaction_positions=pass2_compaction_positions,
        selected_event_evidence=selected_event_evidence,
        minimized_target_evidence=minimized_target_evidence,
        attempted_pair_identities=attempted_pair_identities,
        completed_pair_identities=completed_pair_identities,
        failed_pair_identities=failed_pair_identities,
        pair_failure_details=pair_failure_details,
        semantic_mutation_reports=semantic_mutation_reports,
        pre_branch_memory_evidence=pre_branch_memory_evidence,
        no_op_identity=no_op_identity,
        no_op_evidence=no_op_evidence,
        replay_evidence=replay_evidence,
        pair_records=pair_records,
        example_attrition=_attrition_snapshot(scope, "example_attrition"),
        pair_attrition=_attrition_snapshot(scope, "pair_attrition"),
        example_aborted=example_aborted,
        abort_failure_type=abort_failure_type,
        abort_failure_message=abort_failure_message,
        abort_is_oom=abort_is_oom,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        is_oom=is_oom,
        is_timeout=is_timeout,
        failure_type=type(exc).__name__,
        failure_message=str(exc),
    )


class WorkerBodyFailure(RuntimeError):
    """Raised by `run_fullkv_worker`/`run_rkv_worker` in place of letting the
    original exception propagate bare -- carries `.evidence` (a
    `PartialWorkerEvidence`) and chains the original exception as
    `__cause__`, so nothing about the original failure is lost, only
    enriched with whatever partial evidence existed at the point of
    failure."""

    def __init__(self, evidence: PartialWorkerEvidence, cause: BaseException):
        super().__init__(
            f"{evidence.role} worker body failed at stage={evidence.failing_stage!r}: "
            f"{type(cause).__name__}: {cause}"
        )
        self.evidence = evidence
        self.__cause__ = cause
        self.is_oom = evidence.is_oom
        self.is_timeout = evidence.is_timeout


def raise_worker_body_failure(
    *, role: str, execution_state: WorkerExecutionState, exc: BaseException, scope: dict[str, Any]
) -> NoReturn:
    """R1: the one call site `run_fullkv_worker`/`run_rkv_worker` use to
    convert a caught exception into a partial-evidence-bearing
    `WorkerBodyFailure` -- used both by their outer `except` blocks and by
    any narrower guard (e.g. the CUDA-availability preflight) that needs
    the identical treatment, so the capture-and-wrap logic exists in
    exactly one place."""
    evidence = capture_partial_evidence(
        role=role,
        failing_stage=execution_state.current_stage,
        exc=exc,
        scope=scope,
        attempt_id=execution_state.attempt_id,
        last_completed_stage=execution_state.last_completed_stage,
    )
    raise WorkerBodyFailure(evidence, exc) from exc
