"""Step 3R4 Finding 6: qualification artifact builder, atomic writer, and
sequential qualification coordinator tests. Every worker call is an
injected fake -- no torch, no CUDA, no R-KV import, no real FullKV
inference, no production filesystem path is ever touched."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from kvcot.discovery.b2a_r3_artifacts import (
    SELECTION_STATUS_NONE_QUALIFIED,
    SELECTION_STATUS_SELECTED,
    STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED,
    STOPPED_REASON_CANDIDATE_WORKER_TIMEOUT,
    STOPPED_REASON_FIRST_PASS,
    STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED,
    QualificationArtifactBuildRefused,
    QualificationArtifactWriteRefused,
    build_qualification_artifact as _build_qualification_artifact,
    verify_qualification_artifact as _verify_qualification_artifact,
    write_qualification_artifact_atomic as _write_qualification_artifact_atomic,
)
from kvcot.discovery.b2a_r3_contract import (
    CANDIDATE_MANIFEST_PATH,
    PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    QUALIFICATION_CANDIDATE_LIMIT,
)
from kvcot.discovery.b2a_r3_qualification import build_qualification_outcome
from kvcot.discovery.b2a_r3_qualification_coordinator import (
    CandidateWorkerTimeout,
    QualificationCoordinatorRefused,
    run_b2a_r3_qualification_coordinator,
)
from kvcot.discovery.b2a_r3_worker_adapter import FullKVWorkerResultR3, adapt_fullkv_worker_result_to_r3_evidence
from kvcot.discovery.b2a_workers import FullKVWorkerResult

from kvcot.discovery.b2a_r3_authorization import claim_authorization
from tests.unit.discovery.test_b2a_r3_authorization import _verified_stage_b as _build_verified_stage_b
from tests.unit.discovery.test_b2a_r3_worker_adapter import CONFIG_SHA, _candidate_manifest, _valid_worker_result


def _verified_stage_b(*args, **kwargs):
    payload, verified, document, candidate_manifest, git_state = _build_verified_stage_b(*args, **kwargs)
    consumed = claim_authorization(
        payload,
        repository_root=Path(document).parents[1],
        verified_context=verified,
        git_state=git_state,
    )
    return payload, consumed, document, candidate_manifest, git_state


def _context_for_limit(maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT, phase_seconds=3600):
    import tempfile

    _payload, context, _document, _candidate_manifest, _git_state = _verified_stage_b(
        Path(tempfile.mkdtemp()),
        document_overrides={
            "maximum_candidates": maximum_candidates,
            "phase_wall_time_limit_seconds": phase_seconds,
        },
    )
    return context


def build_qualification_artifact(*args, **kwargs):
    kwargs.setdefault("authorized_phase_wall_time_limit_seconds", 3600)
    requested_limit = kwargs.get("authorized_maximum_candidates", QUALIFICATION_CANDIDATE_LIMIT)
    context_limit = requested_limit if isinstance(requested_limit, int) and not isinstance(requested_limit, bool) and 1 <= requested_limit <= QUALIFICATION_CANDIDATE_LIMIT else QUALIFICATION_CANDIDATE_LIMIT
    kwargs.setdefault(
        "stage_b_authorization_context",
        _context_for_limit(context_limit),
    )
    return _build_qualification_artifact(*args, **kwargs)


def verify_qualification_artifact(artifact, **kwargs):
    kwargs.setdefault(
        "stage_b_authorization_context",
        _context_for_limit(
            artifact.get("authorized_maximum_candidates", QUALIFICATION_CANDIDATE_LIMIT),
            artifact.get("authorized_phase_wall_time_limit_seconds", 3600),
        ),
    )
    return _verify_qualification_artifact(artifact, **kwargs)


def write_qualification_artifact_atomic(artifact, **kwargs):
    kwargs.setdefault(
        "stage_b_authorization_context",
        _context_for_limit(
            artifact.get("authorized_maximum_candidates", QUALIFICATION_CANDIDATE_LIMIT),
            artifact.get("authorized_phase_wall_time_limit_seconds", 3600),
        ),
    )
    return _write_qualification_artifact_atomic(artifact, **kwargs)


class _FakeClock:
    """Returns a strictly increasing sequence of Unix timestamps, one call
    per `.tick()`-recorded advance -- deterministic, no real wall time."""

    def __init__(self, step: float = 1.0, start: float = 1_800_000_000.0):
        self._now = start
        self._step = step

    def __call__(self) -> float:
        value = self._now
        self._now += self._step
        return value

    def jump(self, seconds: float) -> None:
        self._now += seconds


def _runner(qualifies: dict[int, bool] | None = None, calls: list[int] | None = None,
            timeout_at: set[int] | None = None):
    """Builds a fake `fullkv_worker_runner(ordinal, timeout_seconds)`.
    `qualifies` maps ordinal -> whether that candidate's worker result
    should qualify (default True); `timeout_at` raises
    CandidateWorkerTimeout for those ordinals instead."""
    qualifies = qualifies or {}
    timeout_at = timeout_at or set()

    def runner(ordinal: int, timeout_seconds) -> FullKVWorkerResultR3:
        if calls is not None:
            calls.append(ordinal)
        if ordinal in timeout_at:
            raise CandidateWorkerTimeout(f"candidate {ordinal} timed out")
        # Step 3R4-Repair-2 Finding 4: the coordinator now caps the timeout
        # it hands the worker at whatever authorized phase-wide wall time
        # actually remains -- it may be LESS than the frozen per-candidate
        # constant, but must never exceed it.
        assert timeout_seconds <= PER_CANDIDATE_WORKER_TIMEOUT_SECONDS
        overrides = {} if qualifies.get(ordinal, True) else {"natural_answer_status": "incorrect"}
        return _valid_worker_result(ordinal, **overrides)

    return runner


def test_candidate_zero_passes_exactly_one_worker_call(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    calls: list[int] = []
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context, fullkv_worker_runner=_runner(calls=calls),
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0]
    assert artifact["selection_status"] == SELECTION_STATUS_SELECTED
    assert artifact["first_passing_candidate_ordinal"] == 0
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_FIRST_PASS
    verify_qualification_artifact(
        artifact, candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"]
    )


def test_candidate_zero_fails_candidate_one_passes_exactly_two_calls(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    calls: list[int] = []
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(qualifies={0: False}, calls=calls),
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0, 1]
    assert artifact["first_passing_candidate_ordinal"] == 1
    assert artifact["attempted_candidate_count"] == 2


def test_all_eight_fail_exactly_eight_calls(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    calls: list[int] = []
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(qualifies={i: False for i in range(8)}, calls=calls),
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == list(range(8))
    assert artifact["selection_status"] == SELECTION_STATUS_NONE_QUALIFIED
    assert artifact["first_passing_candidate_ordinal"] is None
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED


def test_authorization_maximum_candidates_three_never_calls_a_fourth(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(
        tmp_path, document_overrides={"maximum_candidates": 3}
    )
    calls: list[int] = []
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(qualifies={0: False, 1: False, 2: False}, calls=calls),
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0, 1, 2]
    assert artifact["selection_status"] == SELECTION_STATUS_NONE_QUALIFIED


def test_pass_at_ordinal_two_never_evaluates_three_through_seven(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    calls: list[int] = []
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(qualifies={0: False, 1: False, 2: True}, calls=calls),
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0, 1, 2]
    assert artifact["first_passing_candidate_ordinal"] == 2


def test_worker_returns_wrong_candidate_evidence_hard_refusal(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)

    def bad_runner(ordinal, timeout_seconds):
        # Always returns evidence bound to ordinal 5, regardless of which
        # ordinal the coordinator actually requested.
        return _valid_worker_result(5)

    with pytest.raises(QualificationCoordinatorRefused):
        run_b2a_r3_qualification_coordinator(
            candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
            consumed_authorization_context=context, fullkv_worker_runner=bad_runner,
            clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
        )


def test_worker_returns_legacy_result_hard_refusal(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)

    def legacy_runner(ordinal, timeout_seconds):
        return FullKVWorkerResult(
            role="fullkv", model_revision="x", tokenizer_revision="x", dataset_repo="x", dataset_revision="x",
            manifest_hash="a" * 64, prompt_token_ids_sha256="1" * 64, prompt_token_count=1,
            natural_generated_token_ids=[1], natural_answer="1", natural_answer_status="correct", cap_hit=False,
            prefill_call_count=1, decode_call_count=1, call_boundary_trace_hash="b" * 64, wall_seconds=1.0,
            determinism_policy={}, runtime_generation={}, runtime_generation_config_hash="c" * 64,
            parameter_placement={}, runtime_identity={}, memory={}, peak_cuda_allocated_bytes=1,
            peak_cuda_reserved_bytes=1, every_parameter_on_cuda=True, batch_size=1, software_versions={},
        )

    with pytest.raises(QualificationCoordinatorRefused):
        run_b2a_r3_qualification_coordinator(
            candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
            consumed_authorization_context=context, fullkv_worker_runner=legacy_runner,
            clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
        )


def test_candidate_timeout_exact_stopping_behavior(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    calls: list[int] = []
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(qualifies={0: False}, calls=calls, timeout_at={1}),
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0, 1]
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED
    assert artifact["attempted_candidate_count"] == 1
    assert artifact["selection_status"] == SELECTION_STATUS_NONE_QUALIFIED


def test_stdlib_timeout_error_also_stops_cleanly(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)

    def runner(ordinal, timeout_seconds):
        raise TimeoutError("stdlib timeout")

    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context, fullkv_worker_runner=runner,
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED
    assert artifact["attempted_candidate_count"] == 0


def _sequence_clock(values: list[float]):
    iterator = iter(values)

    def clock() -> float:
        return next(iterator)

    return clock


def test_phase_wide_time_exhausted_before_next_candidate_no_additional_call(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(
        tmp_path, document_overrides={"phase_wall_time_limit_seconds": 5}
    )
    calls: list[int] = []
    # Call sequence: started_at=0; ordinal-0 elapsed check=1 (within the
    # 5s limit, candidate 0 runs); ordinal-1 elapsed check=100 (exceeds
    # the limit -- stop before calling the runner a second time);
    # completed_at=101.
    clock = _sequence_clock([0.0, 1.0, 100.0, 101.0])

    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(qualifies={0: False}, calls=calls),
        clock=clock, per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0]
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED


def test_effective_worker_timeout_capped_at_remaining_phase_time(tmp_path):
    """Step 3R4-Repair-2 Finding 4's own example: authorized phase limit
    600s, 590s already elapsed before this candidate launches -- the
    worker must be capped at the 10s actually remaining, never the full
    7200s frozen per-candidate timeout."""
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(
        tmp_path, document_overrides={"phase_wall_time_limit_seconds": 600}
    )
    timeouts: list[float] = []

    def runner(ordinal, timeout_seconds):
        timeouts.append(timeout_seconds)
        return _valid_worker_result(ordinal, natural_answer_status="incorrect")

    clock = _sequence_clock([0.0, 590.0, 601.0, 602.0])
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context, fullkv_worker_runner=runner,
        clock=clock, per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert timeouts == [10.0]
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED


def test_effective_worker_timeout_never_exceeds_the_frozen_per_candidate_cap(tmp_path):
    """The other direction: when the remaining phase time is LARGER than
    the frozen per-candidate timeout, the worker still gets only the
    frozen cap, never the (larger) remaining phase time."""
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(
        tmp_path, document_overrides={"phase_wall_time_limit_seconds": 36000}
    )
    timeouts: list[float] = []

    def runner(ordinal, timeout_seconds):
        timeouts.append(timeout_seconds)
        return _valid_worker_result(ordinal)  # qualifies immediately -- stop after one call

    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context, fullkv_worker_runner=runner,
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert timeouts == [PER_CANDIDATE_WORKER_TIMEOUT_SECONDS]
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_FIRST_PASS


def test_post_worker_check_relabels_stop_reason_on_last_candidate(tmp_path):
    """Finding 4: even when the single authorized candidate is the LAST
    one the loop will ever run, a phase deadline blown during/after that
    candidate must be reported as phase_wall_time_exhausted -- never
    mislabeled as all_authorized_candidates_exhausted just because the
    `for` loop also happened to run out of candidates at the same time."""
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(
        tmp_path, document_overrides={"maximum_candidates": 1, "phase_wall_time_limit_seconds": 600}
    )
    calls: list[int] = []
    clock = _sequence_clock([0.0, 100.0, 601.0, 602.0])
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(qualifies={0: False}, calls=calls),
        clock=clock, per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0]
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED


def test_no_production_files_written(tmp_path):
    """The coordinator is a pure in-memory computation -- it must never
    write to the real production qualification-artifact path, whether or
    not that path is already populated by separately-committed, accepted
    evidence (e.g. after a Stage-B evidence-acceptance commit). Compare
    before/after content rather than asserting non-existence, since the
    real path may legitimately hold committed evidence outside this test."""
    from kvcot.discovery.b2a_r3_contract import QUALIFICATION_ARTIFACT_PATH

    path = Path(QUALIFICATION_ARTIFACT_PATH)
    before_exists = path.exists()
    before_bytes = path.read_bytes() if before_exists else None

    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context, fullkv_worker_runner=_runner(),
        clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )

    after_exists = path.exists()
    after_bytes = path.read_bytes() if after_exists else None
    assert after_exists == before_exists, "coordinator must not create or delete the production qualification artifact"
    assert after_bytes == before_bytes, "coordinator must not modify a pre-existing production qualification artifact"


def test_wrong_per_candidate_timeout_refused(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    with pytest.raises(QualificationCoordinatorRefused):
        run_b2a_r3_qualification_coordinator(
            candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
            consumed_authorization_context=context, fullkv_worker_runner=_runner(),
            clock=_FakeClock(), per_candidate_timeout_seconds=1,
        )


def test_unverified_context_object_refused(tmp_path):
    _payload, _context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)

    class FakeContext:
        maximum_candidates = 8
        phase_wall_time_limit_seconds = 3600
        _verification_token = object()

    with pytest.raises(QualificationCoordinatorRefused):
        run_b2a_r3_qualification_coordinator(
            candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
            consumed_authorization_context=FakeContext(), fullkv_worker_runner=_runner(),
            clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
        )


def test_mismatched_candidate_manifest_hash_refused(tmp_path):
    """The verified context authorized the REAL committed candidate
    manifest; supplying a different (but structurally valid) manifest at
    coordinator-call time -- e.g. one swapped in between verification and
    the coordinator run -- must be refused, never silently accepted."""
    from tests.unit.discovery.test_b2a_r3_freeze import _candidate_manifest as _synthetic_candidate_manifest

    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    swapped_manifest = _synthetic_candidate_manifest()
    assert swapped_manifest["canonical_sha256"] != candidate_manifest["canonical_sha256"]

    with pytest.raises(QualificationCoordinatorRefused):
        run_b2a_r3_qualification_coordinator(
            candidate_manifest=swapped_manifest, expected_config_sha256=swapped_manifest["config_sha256"],
            consumed_authorization_context=context, fullkv_worker_runner=_runner(),
            clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
        )


def test_final_artifact_passes_full_semantic_verification_every_scenario(tmp_path):
    scenarios = [
        {},
        {"qualifies": {0: False}},
        {"qualifies": {i: False for i in range(8)}},
    ]
    for index, scenario in enumerate(scenarios):
        sub_tmp = tmp_path / f"scenario-{index}"
        sub_tmp.mkdir()
        _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(sub_tmp)
        artifact = run_b2a_r3_qualification_coordinator(
            candidate_manifest=candidate_manifest, expected_config_sha256=candidate_manifest["config_sha256"],
            consumed_authorization_context=context, fullkv_worker_runner=_runner(**scenario),
            clock=_FakeClock(), per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
        )
        verify_qualification_artifact(
            artifact,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=candidate_manifest["config_sha256"],
            stage_b_authorization_context=context,
        )


def test_late_qualified_worker_result_is_not_accepted(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(
        tmp_path, document_overrides={"phase_wall_time_limit_seconds": 5}
    )
    calls: list[int] = []

    def runner(ordinal, timeout_seconds):
        calls.append(ordinal)
        return _valid_worker_result(ordinal)

    clock = _sequence_clock([0.0, 1.0, 6.0, 7.0])
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest,
        expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=runner,
        clock=clock,
        per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert calls == [0]
    assert artifact["attempted_candidate_count"] == 0
    assert artifact["qualification_stopped_reason"] == STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED


def test_artifact_is_bound_to_consumed_stage_b_authorization(tmp_path):
    _payload, context, _document, candidate_manifest, _git_state = _verified_stage_b(
        tmp_path, document_overrides={"maximum_candidates": 3, "phase_wall_time_limit_seconds": 600}
    )
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest,
        expected_config_sha256=candidate_manifest["config_sha256"],
        consumed_authorization_context=context,
        fullkv_worker_runner=_runner(),
        clock=_FakeClock(),
        per_candidate_timeout_seconds=PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    assert artifact["stage_b_authorization_id"] == context.claim.authorization_id
    assert artifact["authorization_document_sha256"] == context.claim.authorization_document_sha256
    assert artifact["authorization_claim_canonical_sha256"] == context.claim.canonical_sha256
    assert artifact["authorized_maximum_candidates"] == 3
    assert artifact["authorized_phase_wall_time_limit_seconds"] == 600
    verify_qualification_artifact(
        artifact,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=candidate_manifest["config_sha256"],
        stage_b_authorization_context=context,
    )


# --------------------------------------------------------------------- builder


def _outcome(ordinal: int, *, qualified: bool) -> dict:
    manifest = _candidate_manifest()
    worker_result = _valid_worker_result(ordinal, **({} if qualified else {"natural_answer_status": "incorrect"}))
    evidence = adapt_fullkv_worker_result_to_r3_evidence(
        worker_result=worker_result, candidate_manifest=manifest, candidate_ordinal=ordinal,
        expected_config_sha256=CONFIG_SHA,
    )
    return build_qualification_outcome(evidence, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)


def test_builder_produces_verifiable_artifact_on_first_pass():
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=False), _outcome(1, qualified=True)]
    artifact = build_qualification_artifact(
        attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        stopped_reason=STOPPED_REASON_FIRST_PASS, authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
        attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
    )
    verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)
    assert artifact["selection_status"] == SELECTION_STATUS_SELECTED
    assert artifact["first_passing_candidate_ordinal"] == 1


def test_builder_rejects_non_contiguous_ordinals():
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=False), _outcome(2, qualified=False)]
    with pytest.raises(QualificationArtifactBuildRefused):
        build_qualification_artifact(
            attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
            stopped_reason=STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED, authorized_maximum_candidates=len(attempted),
            attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
        )


def test_builder_rejects_over_authorized_candidate_count():
    manifest = _candidate_manifest()
    attempted = [_outcome(i, qualified=False) for i in range(9)]
    with pytest.raises(QualificationArtifactBuildRefused):
        build_qualification_artifact(
            attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
            stopped_reason=STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED,
            authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
            attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
        )


def test_builder_rejects_stopped_reason_selection_disagreement():
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=True)]
    with pytest.raises(QualificationArtifactBuildRefused):
        build_qualification_artifact(
            attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
            stopped_reason=STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED,  # WRONG -- ordinal 0 qualified
            authorized_maximum_candidates=len(attempted),
            attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
        )


def test_builder_rejects_unknown_stopped_reason():
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=True)]
    with pytest.raises(QualificationArtifactBuildRefused):
        build_qualification_artifact(
            attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
            stopped_reason="some_made_up_reason", authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
            attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
        )


def test_builder_rejects_unknown_authorized_maximum_candidates():
    """Step 3R4-Repair-2 Finding 6: `authorized_maximum_candidates` itself
    must be a valid int in 1..QUALIFICATION_CANDIDATE_LIMIT."""
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=True)]
    for bad_value in (0, 9, -1, True):
        with pytest.raises(QualificationArtifactBuildRefused):
            build_qualification_artifact(
                attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
                stopped_reason=STOPPED_REASON_FIRST_PASS, authorized_maximum_candidates=bad_value,
                attempt_started_at_utc="2026-07-23T00:00:00+00:00",
                attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
            )


def test_builder_rejects_exhausted_reason_when_count_disagrees_with_authorization():
    """Finding 6: `stopped_reason='all_authorized_candidates_exhausted'` is
    refused unless the attempted count equals the authorized limit exactly
    -- neither fewer nor more."""
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=False)]
    with pytest.raises(QualificationArtifactBuildRefused):
        build_qualification_artifact(
            attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
            stopped_reason=STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED, authorized_maximum_candidates=3,
            attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
        )


def test_builder_rejects_tampered_outcome_via_semantic_rederivation():
    manifest = _candidate_manifest()
    outcome = _outcome(0, qualified=True)
    tampered = dict(outcome)
    tampered["cap_hit"] = True  # contradicts stored qualified=True / trace_complete=True
    with pytest.raises(QualificationArtifactBuildRefused):
        build_qualification_artifact(
            attempted_outcomes=[tampered], candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
            stopped_reason=STOPPED_REASON_FIRST_PASS, authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
            attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
        )


# --------------------------------------------------------------------- atomic writer


def test_atomic_writer_writes_and_round_trips(tmp_path):
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=True)]
    artifact = build_qualification_artifact(
        attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        stopped_reason=STOPPED_REASON_FIRST_PASS, authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
        attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
    )
    output_path = tmp_path / "qualification.json"
    write_qualification_artifact_atomic(
        artifact, output_path=output_path, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
    )
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert written == artifact


def test_atomic_writer_refuses_overwrite(tmp_path):
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=True)]
    artifact = build_qualification_artifact(
        attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        stopped_reason=STOPPED_REASON_FIRST_PASS, authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
        attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
    )
    output_path = tmp_path / "qualification.json"
    write_qualification_artifact_atomic(
        artifact, output_path=output_path, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
    )
    with pytest.raises(QualificationArtifactWriteRefused):
        write_qualification_artifact_atomic(
            artifact, output_path=output_path, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        )


def test_atomic_writer_refuses_raced_existing_file_even_if_precheck_misses(tmp_path, monkeypatch):
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=True)]
    artifact = build_qualification_artifact(
        attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        stopped_reason=STOPPED_REASON_FIRST_PASS, authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
        attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
    )
    output_path = tmp_path / "qualification.json"
    output_path.write_text('{"already": true}\n', encoding="utf-8")

    real_exists = Path.exists

    def false_for_target(path):
        if path == output_path:
            return False
        return real_exists(path)

    monkeypatch.setattr(Path, "exists", false_for_target)
    with pytest.raises(QualificationArtifactWriteRefused):
        write_qualification_artifact_atomic(
            artifact, output_path=output_path, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        )
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"already": True}


def test_atomic_writer_refuses_invalid_artifact(tmp_path):
    manifest = _candidate_manifest()
    with pytest.raises(Exception):
        write_qualification_artifact_atomic(
            {"not": "a valid artifact"}, output_path=tmp_path / "x.json",
            candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        )
    assert not (tmp_path / "x.json").exists()


def test_atomic_writer_refuses_semantically_fabricated_but_canonically_rehashed_artifact(tmp_path):
    """Step 3R4-Repair-2 Finding 5: a canonically-rehashed artifact whose
    outcome fields are internally self-consistent but semantically
    fabricated (never actually reproducible from raw evidence) must be
    rejected by the writer -- not just a shallow schema/self-hash check."""
    manifest = _candidate_manifest()
    attempted = [_outcome(0, qualified=True)]
    artifact = build_qualification_artifact(
        attempted_outcomes=attempted, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        stopped_reason=STOPPED_REASON_FIRST_PASS, authorized_maximum_candidates=QUALIFICATION_CANDIDATE_LIMIT,
        attempt_started_at_utc="2026-07-23T00:00:00+00:00", attempt_completed_at_utc="2026-07-23T00:05:00+00:00",
    )
    tampered = dict(artifact)
    tampered["attempted"] = [dict(tampered["attempted"][0])]
    tampered["attempted"][0]["answer_verification_status"] = "incorrect"
    tampered.pop("canonical_sha256")
    from kvcot.utils.hashing import sha256_json

    tampered["canonical_sha256"] = sha256_json(tampered)

    with pytest.raises(Exception):
        write_qualification_artifact_atomic(
            tampered, output_path=tmp_path / "tampered.json",
            candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA,
        )
    assert not (tmp_path / "tampered.json").exists()


def test_atomic_writer_never_defaults_to_production_path():
    import inspect

    signature = inspect.signature(_write_qualification_artifact_atomic)
    assert signature.parameters["output_path"].default is inspect.Parameter.empty
    assert signature.parameters["candidate_manifest"].default is inspect.Parameter.empty
    assert signature.parameters["expected_config_sha256"].default is inspect.Parameter.empty
    assert signature.parameters["stage_b_authorization_context"].default is inspect.Parameter.empty


def test_coordinator_module_import_never_touches_rkv_torch_or_transformers():
    """Mirrors the established guard pattern in tests/unit/test_cli_b2a_r3.py:
    importing the coordinator module must never import torch/transformers/
    R-KV, even when those modules would raise immediately if imported."""
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    script = """
import sys

class _ForbiddenImportError(ImportError):
    pass

_FORBIDDEN_TOP = {"torch", "transformers", "flash_attn"}
_FORBIDDEN_SUBMODULES = (
    "kvcot.discovery.b2a_workers",
    "kvcot.discovery.schemas",
    "kvcot.discovery.scientific_summary",
    "kvcot.generation.policies",
)

class _Guard:
    def find_spec(self, name, path, target=None):
        top = name.split(".")[0]
        if top in _FORBIDDEN_TOP:
            raise _ForbiddenImportError("FORBIDDEN_IMPORT:" + name)
        if any(name == m or name.startswith(m + ".") for m in _FORBIDDEN_SUBMODULES):
            raise _ForbiddenImportError("FORBIDDEN_IMPORT:" + name)
        return None

sys.meta_path.insert(0, _Guard())
sys.path.insert(0, "src")

import kvcot.discovery.b2a_r3_qualification_coordinator as m

print("import-ok")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(repo_root), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout
