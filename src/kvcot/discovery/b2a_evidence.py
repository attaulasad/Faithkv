"""Real B2A evidence derivation (B1B-R4 §8/§22, superseding B1B-R3's
version of this module). Every function here computes gate-evidence fields
from ACTUAL observations already collected by the harness
(`kvcot.discovery.orchestrator.ExampleResult` /
`kvcot.discovery.pass1.NaturalRunTrace`) -- nothing here runs a GPU or
collects a new measurement, so it stays fully CPU-testable against
synthetic fixtures.

## B1B-R3 defect repaired

The prior version of this module derived FIVE independent trajectory/parity
conditions (`token_identical_replay`, `prefill_decode_boundary_parity`,
`compaction_position_equality`, `capture_gather_parity`,
`absolute_position_parity`) all from the single umbrella `example_result
.valid` boolean -- a worker could satisfy all five while never having
demonstrated any of them separately. This version requires the caller
(`kvcot.discovery.b2a_workers.run_rkv_worker`) to supply each one from an
INDEPENDENT raw observation: Pass 1 vs Pass 2 call-boundary comparison
(`kvcot.discovery.call_trace`), the token-identity check `run_pass2_capture`
already performs (`Pass2Result.valid`/`invalid_reason`), and the per-target
capture parity flags (`kvcot.discovery.capture.UpdateKvCaptureRecord
.gather_parity_passed` / `.observed_kept_indices_parity_passed`) read off
each selected target's capture record -- `example_result.valid` is no
longer read by this module at all for those five conditions.
"""
from __future__ import annotations

from dataclasses import dataclass

from kvcot.discovery.constants import (
    B2A_REAL_PAIR_EVALUATIONS_TOTAL,
    B2B_PILOT_EXAMPLE_COUNT,
    B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS,
    REAL_PAIR_EVALUATIONS_PER_EVENT,
)


@dataclass(frozen=True)
class PairCompletionEvidence:
    """B1B-R4 §8/§22: exact, independently-countable selection and
    pair-completion accounting -- never derived from `len(pair_records)`
    alone (which conflates real and no-op pairs) and never a single
    umbrella boolean.

    B1B-R4.1 §14 repair: `selected_compaction_events` (and the
    `selected_event_count_exact` gate condition it feeds) is now read from
    `example_result.selected_event_ids` -- the FROZEN Pass-1 plan's own
    selection, populated once by `kvcot.discovery.orchestrator.run_example`
    right after `build_pass1_plan` succeeds -- never re-derived by counting
    distinct event IDs across `pair_records`. That prior derivation
    conflated "selected by Pass 1" with "at least one pair survived
    attrition": an event every one of whose pairs failed would silently
    vanish from the count instead of being reported as a selected-but-failed
    event. `events_with_at_least_one_completed_real_pair` is the NEW,
    separately-named field for that weaker, completion-based quantity, so
    the two are never conflated under one name again."""

    observed_total_compaction_events: int
    eligible_compaction_events: int
    selected_compaction_events: int
    events_with_at_least_one_completed_real_pair: int
    events_with_all_four_real_pairs_completed: int
    attempted_real_pair_count: int
    completed_real_pair_count: int
    failed_real_pair_count: int
    attempted_no_op_pair_count: int
    completed_no_op_pair_count: int
    pair_failure_details: tuple["PairFailureDetail", ...]


def derive_pair_completion_evidence(*, trace, example_result) -> PairCompletionEvidence:
    """Derive every count from `example_result`/`trace` directly -- never
    from a re-run or a second independently-maintained counter.

    B1B-R4.1 §15 repair: `pair_failure_details` is read directly off
    `example_result.pair_failure_details` -- the structured records
    `kvcot.discovery.orchestrator.run_example` builds live, one per failed
    pair attempt. The prior `pair_attrition_dropped_stages` parameter was
    never actually threaded through by `kvcot.discovery.b2a_workers
    .run_rkv_worker`, so this field was always an empty tuple in the
    production R-KV worker path regardless of how many pairs actually
    failed -- removed rather than left as a silently-unused parameter that
    a future caller could pass without effect."""
    from kvcot.discovery.pass1 import eligible_event_ids

    observed_total = len(trace.compaction_events) if trace is not None else 0
    eligible = len(eligible_event_ids(trace)) if trace is not None else 0

    real_records = [r for r in example_result.pair_records if not r.is_noop_control]
    by_event: dict[int, int] = {}
    for r in real_records:
        by_event[r.compaction_event_id] = by_event.get(r.compaction_event_id, 0) + 1
    selected_events = len(example_result.selected_event_ids)
    events_with_at_least_one = len(by_event)
    events_with_all_four = sum(1 for count in by_event.values() if count == 4)

    attempted_real = example_result.attempted_real_pair_count
    completed_real = example_result.completed_real_pair_count
    attempted_no_op = example_result.attempted_no_op_pair_count
    completed_no_op = example_result.completed_no_op_pair_count

    return PairCompletionEvidence(
        observed_total_compaction_events=observed_total,
        eligible_compaction_events=eligible,
        selected_compaction_events=selected_events,
        events_with_at_least_one_completed_real_pair=events_with_at_least_one,
        events_with_all_four_real_pairs_completed=events_with_all_four,
        attempted_real_pair_count=attempted_real,
        completed_real_pair_count=completed_real,
        failed_real_pair_count=attempted_real - completed_real,
        attempted_no_op_pair_count=attempted_no_op,
        completed_no_op_pair_count=completed_no_op,
        pair_failure_details=example_result.pair_failure_details,
    )


@dataclass(frozen=True)
class TrajectoryParityEvidence:
    """The five previously-conflated conditions, each with its own
    independent raw source (B1B-R4 §8)."""

    token_identical_replay: bool
    prefill_decode_boundary_parity: bool
    compaction_position_equality: bool
    capture_gather_parity: bool
    absolute_position_parity: bool


def derive_trajectory_parity_evidence(
    *,
    pass2_result_valid: bool,
    pass2_invalid_reason: str | None,
    call_boundary_all_match: bool,
    target_capture_gather_parities: tuple[bool | None, ...],
    target_capture_absolute_parities: tuple[bool | None, ...],
    compaction_lists_match: bool | None = None,
) -> TrajectoryParityEvidence:
    """`token_identical_replay` comes from `Pass2Result`'s own token-by-
    token comparison against Pass 1's frozen trace
    (`kvcot.discovery.pass2.run_pass2_capture`, `INVALID_TOKEN_MISMATCH`) --
    specifically whether that WAS the failure reason (or no failure at all);
    a Pass-2 failure for an unrelated reason (e.g. a missing snapshot)
    reports this `True` (tokens genuinely matched) while the OTHER four
    conditions report `False` (never demonstrated for an incomplete run).

    `prefill_decode_boundary_parity` comes from an INDEPENDENT comparison of
    two separately-recorded `kvcot.discovery.call_trace.CallTraceRecorder`
    traces (Pass 1's vs Pass 2's) -- never inferred from `pass2_result.valid`.

    `capture_gather_parity`/`absolute_position_parity` are each `True` only
    when EVERY required selected target reported an explicit successful
    observation (`None` -- not evaluable -- counts as failure here, per
    B1B-R4 §8: "Aggregate conditions are true only when every required
    selected target has an explicit successful observation").

    `pass2_attempted` (derived, not a separate caller-supplied flag) is
    `True` iff Pass 2 was ever reached at all -- either it succeeded
    (`pass2_result_valid=True`) or it was reached and failed for a specific
    reason (`pass2_invalid_reason is not None`). An example that never got
    past Pass 1 (natural run invalid, wrong answer, cap hit, too few
    eligible events -- `pass2_result_valid=False` AND
    `pass2_invalid_reason=None`) reports `token_identical_replay=False`
    too: nothing was ever replayed, so no token-identity claim can be made,
    never vacuously `True` because "the check that would have caught a
    mismatch never ran"."""
    pass2_attempted = pass2_result_valid or pass2_invalid_reason is not None
    token_identical = pass2_attempted and pass2_invalid_reason != "pass2_token_mismatch"
    ran_to_completion = pass2_result_valid

    def _all_true(values: tuple[bool | None, ...]) -> bool:
        return len(values) > 0 and all(v is True for v in values)

    return TrajectoryParityEvidence(
        token_identical_replay=token_identical,
        prefill_decode_boundary_parity=ran_to_completion and call_boundary_all_match,
        compaction_position_equality=(
            ran_to_completion
            and pass2_invalid_reason is None
            and (compaction_lists_match is True if compaction_lists_match is not None else True)
        ),
        capture_gather_parity=ran_to_completion and _all_true(target_capture_gather_parities),
        absolute_position_parity=ran_to_completion and _all_true(target_capture_absolute_parities),
    )


def derive_observed_retention_ratio(example_result) -> float:
    trace = example_result.trace
    if trace is not None and trace.cache_length_final_per_layer:
        total_tokens = len(trace.full_token_ids)
        mean_final_len = sum(trace.cache_length_final_per_layer.values()) / len(trace.cache_length_final_per_layer)
        return mean_final_len / total_tokens if total_tokens > 0 else 0.0
    return 0.0


PairIdentity = tuple[int, int, int, int, int, str]


@dataclass(frozen=True)
class PairIdentityEvidence:
    """B1 execution-boundary closure §13: exact, DUPLICATE-DETECTING
    identity accounting -- a stable identity tuple `(compaction_event_id,
    layer_index, kv_head_index, evicted_absolute_position,
    donor_absolute_position, pair_kind)` per completed pair, never a bare
    prior at-least-four per-event count that cannot tell
    four genuinely distinct pairs apart from the same pair counted (or
    somehow recorded) four times. Computed entirely from
    `example_result.pair_records` -- no new per-pair state needed beyond
    what `kvcot.discovery.schemas.SwapPairRecord` already carries."""

    unique_completed_real_pair_count: int
    events_with_exactly_four_unique_real_pairs: int
    has_duplicate_real_pair_identity: bool
    completed_no_op_pair_count: int
    has_duplicate_no_op_pair_identity: bool


def _pair_identity(record) -> PairIdentity:
    kind = "no_op" if record.is_noop_control else "real"
    return (
        record.compaction_event_id, record.layer_index, record.kv_head_index,
        record.evicted_absolute_token_position, record.retained_absolute_token_position, kind,
    )


def derive_pair_identity_evidence(example_result) -> PairIdentityEvidence:
    real_identities = [_pair_identity(r) for r in example_result.pair_records if not r.is_noop_control]
    no_op_identities = [_pair_identity(r) for r in example_result.pair_records if r.is_noop_control]

    by_event: dict[int, set[PairIdentity]] = {}
    for identity in real_identities:
        by_event.setdefault(identity[0], set()).add(identity)
    events_with_exactly_four = sum(
        1 for identities in by_event.values() if len(identities) == REAL_PAIR_EVALUATIONS_PER_EVENT
    )

    return PairIdentityEvidence(
        unique_completed_real_pair_count=len(set(real_identities)),
        events_with_exactly_four_unique_real_pairs=events_with_exactly_four,
        has_duplicate_real_pair_identity=len(real_identities) != len(set(real_identities)),
        completed_no_op_pair_count=len(no_op_identities),
        has_duplicate_no_op_pair_identity=len(no_op_identities) != len(set(no_op_identities)),
    )


@dataclass(frozen=True)
class SemanticSwapCheckEvidence:
    """B1 execution-boundary closure §12: POSITIVE semantic-swap-check
    evidence -- `checks_required` is the frozen B2A constant (12 real
    pairs); `checks_attempted`/`checks_passed` are summed directly off
    `example_result.semantic_swap_checks_attempted`/`.semantic_swap_checks_passed`
    (`kvcot.discovery.orchestrator.run_example`, itself reading
    `PairBuildResult.semantic_swap_check_attempted`/`.semantic_swap_check_passed`
    at every return point in `kvcot.discovery.pipeline.build_swap_pair_record`)
    -- never derived from "no semantic_swap_parity_failure record exists in
    pair_failure_details", which is vacuously true for a pair whose check
    was never reached. The gate condition requires
    `checks_attempted == checks_required == 12` AND `checks_failed == 0`."""

    checks_required: int
    checks_attempted: int
    checks_passed: int
    checks_failed: int


def derive_semantic_swap_check_evidence(example_result) -> SemanticSwapCheckEvidence:
    attempted = example_result.semantic_swap_checks_attempted
    passed = example_result.semantic_swap_checks_passed
    return SemanticSwapCheckEvidence(
        checks_required=B2A_REAL_PAIR_EVALUATIONS_TOTAL,
        checks_attempted=attempted,
        checks_passed=passed,
        checks_failed=attempted - passed,
    )


def derive_no_op_numerical_parity(example_result) -> bool:
    """`True` only when an ACTUAL `is_noop_control=True` record exists among
    `example_result.pair_records` -- `kvcot.discovery.schemas.SwapPairRecord`'s
    own validators already require bit-exact
    `baseline_per_token_nll == swapped_per_token_nll` for any such record, so
    a present, schema-valid no-op record IS the calibration."""
    return any(pr.is_noop_control for pr in example_result.pair_records)


def derive_meaningful_compression_observed(*, selected_event_count: int, observed_retention_ratio: float) -> bool:
    return selected_event_count >= 1 and observed_retention_ratio < 1.0


def per_real_pair_projection_seconds(real_pair_wall_seconds: tuple[float, ...]) -> float:
    """B1B-R4 §12's frozen conservative per-pair statistic: the MAXIMUM
    total time among the completed real pair evaluations -- never the mean
    or an aggregate bucket. `0.0` if no real pair completed (never divides
    by zero, never fabricates a number)."""
    return max(real_pair_wall_seconds) if real_pair_wall_seconds else 0.0


def project_complete_pilot_gpu_hours(
    *,
    per_example_total_seconds: float,
    per_real_pair_seconds: float,
) -> float:
    """B1B-R4 §12's frozen projection formula:

    ```
    projected_seconds = B2B_PILOT_EXAMPLE_COUNT * per_example_total_seconds
                       + B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS * per_real_pair_seconds
    projected_gpu_hours = projected_seconds / 3600
    ```

    `per_example_total_seconds` must already be `fullkv_natural_generation +
    rkv_pass1 + rkv_pass2` (once-per-example components; the caller is
    responsible for not double-counting score/capture submeasurements that
    are already contained inside the Pass 2 total). `per_real_pair_seconds`
    is `per_real_pair_projection_seconds`'s output -- the MAXIMUM of the 12
    individually-measured real pair evaluations, never an aggregate bucket
    multiplied by 144 (the B1B-R3/B1B-R2 defect this repairs). The single
    B2A no-op calibration is deliberately excluded from this projection
    (B2B runs zero no-op evaluations, `kvcot.discovery.constants.NoOpMode
    .DISABLED`)."""
    projected_seconds = (
        B2B_PILOT_EXAMPLE_COUNT * per_example_total_seconds
        + B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS * per_real_pair_seconds
    )
    return projected_seconds / 3600.0


# --------------------------------------------------------------------------
# F2 (final independent-audit repair): the one canonical implementation of
# the failed-pair-identity, no-op, compaction-position, and replay evidence
# derivations. Both the successful worker path
# (`kvcot.discovery.b2a_workers.run_rkv_worker`) and the failure path
# (`kvcot.discovery.worker_partial_evidence.capture_partial_evidence`) call
# these -- never two divergent copies of the same scientific derivation.
# --------------------------------------------------------------------------


def derive_failed_pair_identities(attempted, completed, pair_failure_details) -> list:
    """attempted minus completed, enriched with the matching
    `PairFailureDetail`'s stage/detail where one exists."""
    completed_keys = {tuple(sorted(identity.items())) for identity in completed}
    failed = []
    for identity in attempted:
        if tuple(sorted(identity.items())) in completed_keys:
            continue
        detail = next(
            (
                item for item in pair_failure_details
                if item.compaction_event_id == identity.get("compaction_event_id")
                and item.layer_index == identity.get("layer_index")
                and item.kv_head_index == identity.get("kv_head_index")
                and item.evicted_absolute_position == identity.get("candidate_absolute_position")
                and item.donor_absolute_position == identity.get("donor_absolute_position")
                and item.pair_kind == identity.get("pair_kind")
            ),
            None,
        )
        failed.append({
            **identity,
            "failure_stage": None if detail is None else detail.stage,
            "failure_detail": None if detail is None else detail.detail,
        })
    return failed


def build_no_op_evidence(example_result) -> dict:
    """The numeric no-op parity evidence block. `{}` unless exactly one
    no-op record AND exactly one no-op mutation report exist -- evidence is
    only reported for work that genuinely happened."""
    from kvcot.utils.hashing import sha256_json

    no_op_records = [record for record in example_result.pair_records if record.is_noop_control]
    no_op_reports = [
        report for report in example_result.semantic_mutation_reports
        if report.get("pair_identity", {}).get("pair_kind") == "no_op"
    ]
    if len(no_op_records) != 1 or len(no_op_reports) != 1:
        return {}
    record = no_op_records[0]
    report = no_op_reports[0]
    differences = [abs(a - b) for a, b in zip(record.baseline_per_token_nll, record.swapped_per_token_nll)]
    return {
        "baseline_nll": list(record.baseline_per_token_nll),
        "no_op_nll": list(record.swapped_per_token_nll),
        "baseline_nll_sha256": sha256_json(list(record.baseline_per_token_nll)),
        "no_op_nll_sha256": sha256_json(list(record.swapped_per_token_nll)),
        "baseline_mean_nll": sum(record.baseline_per_token_nll) / len(record.baseline_per_token_nll),
        "no_op_mean_nll": sum(record.swapped_per_token_nll) / len(record.swapped_per_token_nll),
        "mean_difference": record.swap_gain,
        "maximum_absolute_per_token_difference": max(differences),
        "starting_snapshot_sha256": report.get("starting_snapshot_sha256"),
        "semantic_mutation_report": report,
        "physical_byte_delta": record.net_physical_bytes_changed,
        "provenance_before_sha256": report.get("provenance_before_sha256"),
        "provenance_after_sha256": report.get("provenance_after_sha256"),
        "kept_index_before_sha256": report.get("kept_index_before_sha256"),
        "kept_index_after_sha256": report.get("kept_index_after_sha256"),
    }


def derive_compaction_positions(example_result) -> tuple[list, list]:
    """(pass1_positions, pass2_positions) -- Pass 1 from the trace's own
    compaction events, Pass 2 from the replay's observed positions."""
    pass1 = [event.absolute_event_position for event in example_result.trace.compaction_events]
    pass2 = list(example_result.pass2_compaction_event_positions)
    return pass1, pass2


def build_replay_evidence(example_result, *, pass1_events, pass2_events, actual_call_export) -> dict:
    """The complete token/call/compaction replay-parity evidence block --
    moved verbatim out of `run_rkv_worker`'s success path so a failing
    worker can still report whatever replay evidence already existed."""
    from kvcot.discovery.mismatch import build_mismatch_record
    from kvcot.utils.hashing import sha256_int_ids, sha256_json

    def first_mismatch(left, right):
        return build_mismatch_record(left, right).first_mismatch_index

    pass1_calls = [
        {"call_kind": event.kind, "token_ids": list(event.token_ids), "token_count": len(event.token_ids)}
        for event in pass1_events
    ]
    pass2_calls = [
        {"call_kind": event.kind, "token_ids": list(event.token_ids), "token_count": len(event.token_ids)}
        for event in pass2_events
    ]
    pass1_actual_calls = actual_call_export[: len(pass1_calls)]
    pass2_actual_calls = actual_call_export[len(pass1_calls) : len(pass1_calls) + len(pass2_calls)]
    pass1_token_ids = list(example_result.trace.full_token_ids)
    pass2_fed_token_ids = list(example_result.pass2_replayed_token_ids)
    pass1_compaction_positions, pass2_compaction_positions = derive_compaction_positions(example_result)
    return {
        "pass1_token_ids": pass1_token_ids,
        "pass1_token_sha256": sha256_int_ids(pass1_token_ids),
        "pass2_fed_token_ids": pass2_fed_token_ids,
        "pass2_token_sha256": sha256_int_ids(pass2_fed_token_ids),
        "token_first_mismatch": first_mismatch(pass1_token_ids, pass2_fed_token_ids),
        "token_mismatch": build_mismatch_record(pass1_token_ids, pass2_fed_token_ids).export(),
        "pass1_calls": pass1_calls,
        "pass2_calls": pass2_calls,
        "pass1_call_sha256": sha256_json(pass1_calls),
        "pass2_call_sha256": sha256_json(pass2_calls),
        "call_first_mismatch": first_mismatch(pass1_calls, pass2_calls),
        "call_mismatch": build_mismatch_record(pass1_calls, pass2_calls).export(),
        "pass1_actual_calls": pass1_actual_calls,
        "pass2_actual_calls": pass2_actual_calls,
        "pass1_actual_call_sha256": sha256_json(pass1_actual_calls),
        "pass2_actual_call_sha256": sha256_json(pass2_actual_calls),
        "actual_call_first_mismatch": first_mismatch(pass1_actual_calls, pass2_actual_calls),
        "actual_call_mismatch": build_mismatch_record(pass1_actual_calls, pass2_actual_calls).export(),
        "pass1_compaction_positions": pass1_compaction_positions,
        "pass2_compaction_positions": pass2_compaction_positions,
        "pass1_compaction_sha256": sha256_json(pass1_compaction_positions),
        "pass2_compaction_sha256": sha256_json(pass2_compaction_positions),
        "compaction_first_mismatch": first_mismatch(pass1_compaction_positions, pass2_compaction_positions),
        "compaction_mismatch": build_mismatch_record(
            pass1_compaction_positions, pass2_compaction_positions
        ).export(),
        "complete_compaction_trace_match": pass1_compaction_positions == pass2_compaction_positions,
    }
