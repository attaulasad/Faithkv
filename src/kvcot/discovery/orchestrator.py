"""Whole-example orchestration — wires Pass 1, Pass 2, and per-pair branch
construction/evaluation together with structured attrition accounting
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9/§ Attrition, authorized by
CLAUDE.md §1b/§4b). This is the one place all of the harness's pieces are
connected end-to-end; every other module in `kvcot.discovery` stays usable
in isolation.

Two independent attrition populations, tracked with two separate
`AttritionCounters` (never conflated into one shared denominator):
`example_attrition` (one entry per example attempted; a natural-run/Pass-1/
Pass-2-level failure invalidates the WHOLE example, no pairs are ever
built) and `pair_attrition` (one entry per (event, candidate, donor) pair
attempted, `3 events x (4 cross-product pairs + 1 mandatory no-op) = 15`
per valid example; a pair-level failure never invalidates sibling pairs).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from kvcot.discovery.attrition import (
    STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE,
    STAGE_BRANCH_EVALUATION_FAILURE,
    STAGE_CAP_HIT,
    STAGE_CAPTURE_GATHER_PARITY_FAILURE,
    STAGE_COMPACTION_EVENT_MISMATCH,
    STAGE_FEWER_THAN_THREE_ELIGIBLE_EVENTS,
    STAGE_INVALID_CANDIDATE_DONOR_POOL,
    STAGE_MISSING_TARGET_SNAPSHOT,
    STAGE_NATURAL_RUN_INVALID,
    STAGE_OBSERVED_SURVIVOR_MISMATCH,
    STAGE_PASS2_EXECUTION_EXCEPTION,
    STAGE_PASS2_TOKEN_MISMATCH,
    STAGE_PREFILL_CONTRACT_VIOLATION,
    STAGE_SCHEMA_VALIDATION_FAILURE,
    STAGE_SEMANTIC_SWAP_PARITY_FAILURE,
    STAGE_UNCERTAINTY_MISSING,
    STAGE_UNEXPECTED_PAIR_EXCEPTION,
    AttritionCounters,
    PairFailureDetail,
)
from kvcot.discovery.constants import NoOpMode
from kvcot.discovery.compact_target import build_compact_branch_target
from kvcot.discovery.harness_types import DecodeOneFn, PrefillFn, SnapshotFn
from kvcot.discovery.pass1 import (
    PLAN_FAILURE_TOO_FEW_ELIGIBLE_EVENTS,
    AnswerFn,
    NaturalRunProvenance,
    NaturalRunTrace,
    build_pass1_plan,
    run_natural_pass1,
)
from kvcot.discovery.pass2 import (
    INVALID_COMPACTION_POSITION_MISMATCH,
    INVALID_MISSING_TARGET_CAPTURE,
    INVALID_MISSING_TARGET_SNAPSHOT,
    INVALID_PREFILL_SHAPE_MISMATCH,
    INVALID_SURVIVOR_MISMATCH_ACROSS_PASSES,
    INVALID_SURVIVOR_MISMATCH_WITHIN_PASS2,
    INVALID_TOKEN_MISMATCH,
    run_pass2_capture,
)
from kvcot.discovery.pipeline import (
    STAGE_BRANCH_EVALUATION_FAILURE as PAIR_STAGE_BRANCH_EVALUATION_FAILURE,
    STAGE_INVALID_CANDIDATE_DONOR_POOL as PAIR_STAGE_INVALID_CANDIDATE_DONOR_POOL,
    STAGE_SCHEMA_VALIDATION_FAILURE as PAIR_STAGE_SCHEMA_VALIDATION_FAILURE,
    UNCERTAINTY_POSITION_UNAVAILABLE_REASON,
    build_swap_pair_record,
)
from kvcot.discovery.sampling import IdentitySeedParts
from kvcot.discovery.schemas import SwapPairRecord

_PASS2_REASON_TO_STAGE = {
    INVALID_TOKEN_MISMATCH: STAGE_PASS2_TOKEN_MISMATCH,
    INVALID_PREFILL_SHAPE_MISMATCH: STAGE_PREFILL_CONTRACT_VIOLATION,
    INVALID_COMPACTION_POSITION_MISMATCH: STAGE_COMPACTION_EVENT_MISMATCH,
    INVALID_MISSING_TARGET_CAPTURE: STAGE_COMPACTION_EVENT_MISMATCH,
    INVALID_MISSING_TARGET_SNAPSHOT: STAGE_MISSING_TARGET_SNAPSHOT,
    INVALID_SURVIVOR_MISMATCH_WITHIN_PASS2: STAGE_CAPTURE_GATHER_PARITY_FAILURE,
    INVALID_SURVIVOR_MISMATCH_ACROSS_PASSES: STAGE_OBSERVED_SURVIVOR_MISMATCH,
}

_PAIR_STAGE_MAP = {
    PAIR_STAGE_INVALID_CANDIDATE_DONOR_POOL: STAGE_INVALID_CANDIDATE_DONOR_POOL,
    PAIR_STAGE_BRANCH_EVALUATION_FAILURE: STAGE_BRANCH_EVALUATION_FAILURE,
    PAIR_STAGE_SCHEMA_VALIDATION_FAILURE: STAGE_SCHEMA_VALIDATION_FAILURE,
}


@dataclass(frozen=True)
class PairExecutionPolicy:
    """B1B-R4 §7 repair: an execution policy that ACTUALLY CONTROLS pair
    construction in `run_example` below, not merely a documentation label
    layered on top of a fixed orchestrator behavior. `no_op_mode` is the
    only knob (`kvcot.discovery.constants.NoOpMode`); the default
    (`CPU_REQUIRED`) preserves this module's pre-existing, already-tested
    behavior exactly (one no-op pair per selected event) so every caller
    that does not pass a policy explicitly is unaffected by this repair."""

    no_op_mode: NoOpMode = NoOpMode.CPU_REQUIRED


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    valid: bool
    invalid_stage: str | None
    trace: NaturalRunTrace | None
    pair_records: tuple[SwapPairRecord, ...]
    # B1B-R4 §22: exact execution accounting, independent of pair_records
    # (which only contains SUCCESSFULLY built records -- a pair-level
    # attrition drop must not silently shrink this count). Always populated,
    # even for an invalid example (all zero in that case, since no pairs are
    # ever attempted once an example is dropped before pair construction).
    attempted_real_pair_count: int = 0
    attempted_no_op_pair_count: int = 0
    completed_real_pair_count: int = 0
    completed_no_op_pair_count: int = 0
    # B1B-R4 §12: one wall-clock duration PER COMPLETED pair evaluation
    # (real, then no-op), measured around the ENTIRE `build_swap_pair_record`
    # call for that pair -- never an aggregate bucket later multiplied by a
    # pair count. Sequential (never overlapping) since pairs are built one
    # at a time in the loop below; length == completed_real_pair_count /
    # completed_no_op_pair_count respectively (a failed pair contributes no
    # entry -- there is no meaningful "whole pair" duration for one that
    # never produced a record).
    real_pair_wall_seconds: tuple[float, ...] = ()
    no_op_pair_wall_seconds: tuple[float, ...] = ()
    # B1B-R4 §8: the RAW Pass-2 failure reason (e.g.
    # "pass2_token_mismatch"), independent of `invalid_stage`'s coarser
    # attrition-funnel bucket -- lets a caller derive `token_identical_replay`
    # specifically, never conflated with the other four trajectory/parity
    # conditions (`kvcot.discovery.b2a_evidence.derive_trajectory_parity_evidence`).
    pass2_invalid_reason: str | None = None
    # B1B-R4 §18: MINIMIZED per-target capture evidence only -- built here,
    # immediately after a successful Pass 2, so the object this function
    # RETURNS never carries a full-layer/full-cache tensor anywhere,
    # regardless of what a caller does with it afterward.
    minimized_target_evidence: tuple[Any, ...] = ()  # kvcot.discovery.capture_minimize.MinimizedTargetEvidence
    # B1B-R4.1 §15: one structured `PairFailureDetail` per FAILED pair
    # attempt (never populated from an aggregate counter after the fact) --
    # empty for a fully-successful example, by construction.
    pair_failure_details: tuple[PairFailureDetail, ...] = ()
    # B1B-R4.1 §14: the FROZEN Pass-1 plan's own selected compaction-event
    # IDs -- always exactly `len(plan.events)` (3) for any example that
    # reached this far, populated once, right after `build_pass1_plan`
    # succeeds, and never re-derived from which events happen to still have
    # a surviving pair record later. This is the authoritative "planned"
    # count `kvcot.discovery.b2a_evidence.derive_pair_completion_evidence`
    # must read -- counting distinct event IDs across `pair_records` instead
    # silently UNDER-counts whenever every pair for a selected event fails
    # (that event then has zero surviving records, even though it WAS
    # selected), conflating "selected by Pass 1" with "at least one pair
    # survived attrition."
    selected_event_ids: tuple[int, ...] = ()
    # B1 execution-boundary closure §12: POSITIVE semantic-swap-check
    # evidence, summed across every REAL pair attempt from
    # `PairBuildResult.semantic_swap_check_attempted`/`.semantic_swap_check_passed`
    # -- never derived from "no semantic_swap_parity_failure record exists"
    # (vacuously true for a pair whose check was never reached at all).
    semantic_swap_checks_attempted: int = 0
    semantic_swap_checks_passed: int = 0
    pre_branch_memory_evidence: tuple[Any, ...] = ()
    attempted_pair_identities: tuple[dict[str, Any], ...] = ()
    completed_pair_identities: tuple[dict[str, Any], ...] = ()
    semantic_mutation_reports: tuple[dict[str, Any], ...] = ()
    selected_event_evidence: tuple[dict[str, Any], ...] = ()
    pass2_replayed_token_ids: tuple[int, ...] = ()
    pass2_compaction_event_positions: tuple[int, ...] = ()
    # Independent-audit Gate H1: `True` only when the per-pair evaluation
    # loop below was cut short by an unexpected exception (e.g. a real CUDA
    # OOM mid-pair) rather than completing every planned pair for every
    # selected event. Every list/count field above is still whatever was
    # genuinely completed before the abort -- never discarded, never
    # padded out to look complete. A caller (`kvcot.discovery.b2a_workers`)
    # must never treat an aborted `ExampleResult` as if it were a normal,
    # merely-attrition-affected completion.
    aborted: bool = False
    abort_failure_type: str | None = None
    abort_failure_message: str | None = None
    abort_is_oom: bool = False


def run_example(
    *,
    example_id: str,
    model_revision: str,
    rkv_revision: str,
    provenance: NaturalRunProvenance,
    prompt_token_ids: list[int],
    pass1_initial_state: Any,
    pass2_initial_state_factory: Callable[[], Any],
    prefill_fn: PrefillFn,
    decode_one_fn: DecodeOneFn,
    snapshot_fn: SnapshotFn,
    max_new_tokens: int,
    eos_token_id: int | None,
    answer_fn: AnswerFn,
    num_hidden_layers: int,
    num_key_value_heads: int,
    identity: IdentitySeedParts,
    branch_step_fn,
    example_attrition: AttritionCounters,
    pair_attrition: AttritionCounters,
    pair_execution_policy: "PairExecutionPolicy | None" = None,
    clock_fn: Callable[[], float] | None = None,
    pre_branch_guard: Callable[[Any, str], Any] | None = None,
    operation_runner: Callable[[str, Callable[[], Any]], Any] | None = None,
    pair_phase_runner: Callable[[str, Callable[[], Any]], Any] | None = None,
    capture_timer_fn: Callable[[str, Callable[[], Any]], Any] | None = None,
) -> ExampleResult:
    """`capture_timer_fn` (independent-audit Gate H2.2 repair) is passed
    straight through to `kvcot.discovery.pass2.run_pass2_capture`, which
    passes it to `kvcot.discovery.capture.capture_update_kv` -- it times
    the REAL target capture gather/gather-parity/absolute-position-parity
    computation, never a later trace comparison. Defaults to `None`
    (every pre-existing caller/test is unaffected).

    `pair_execution_policy` defaults to `PairExecutionPolicy()`
    (`NoOpMode.CPU_REQUIRED`) -- every pre-existing caller that does not
    pass one explicitly gets the exact same pair-construction behavior this
    function always had (B1B-R4 §7). `clock_fn` (B1B-R4 §12) defaults to
    `time.monotonic` -- CPU tests inject a deterministic fake clock instead,
    so per-pair timing is exercised without depending on real wall-clock
    variance."""
    import time

    clock_fn = clock_fn or time.monotonic
    operation_runner = operation_runner or (lambda _phase, operation: operation())
    pair_phase_runner = pair_phase_runner or (lambda _phase, operation: operation())
    pair_execution_policy = pair_execution_policy or PairExecutionPolicy()
    example_attrition.record_entered()

    try:
        trace = operation_runner(
            "rkv_complete_pass1",
            lambda: run_natural_pass1(
                provenance, prompt_token_ids, pass1_initial_state, prefill_fn, decode_one_fn, max_new_tokens,
                eos_token_id, answer_fn,
            ),
        )
    except Exception as exc:
        # Independent-audit Gate H1 hostile-audit follow-up: `run_natural_pass1`
        # only ever raises (never returns an "invalid" sentinel -- see its
        # own contract) or returns a valid trace, so ANY exception here is a
        # genuine, unexpected failure (e.g. a real CUDA OOM during natural
        # generation), not a normal "answer didn't validate" case -- yet the
        # exception's type/message used to be discarded entirely, mapped to
        # the bare `STAGE_NATURAL_RUN_INVALID` funnel stage with no other
        # evidence. The stage NAME is unchanged (existing funnel consumers
        # are unaffected), but `aborted`/`abort_failure_type`/
        # `abort_failure_message`/`abort_is_oom` now preserve what actually
        # happened, exactly like the Pass-2 and per-pair equivalents below.
        example_attrition.record_dropped(STAGE_NATURAL_RUN_INVALID)
        return ExampleResult(
            example_id, False, STAGE_NATURAL_RUN_INVALID, None, (),
            aborted=True, abort_failure_type=type(exc).__name__, abort_failure_message=str(exc),
            abort_is_oom=("out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"),
        )

    if trace.natural_answer_status != "correct":
        example_attrition.record_dropped(STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE)
        return ExampleResult(example_id, False, STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE, trace, ())

    if trace.cap_hit:
        example_attrition.record_dropped(STAGE_CAP_HIT)
        return ExampleResult(example_id, False, STAGE_CAP_HIT, trace, ())

    # R1: routed through `operation_runner` -- in `run_rkv_worker` this is
    # the same tracked `measured` callable Pass 1/Pass 2 use, so an
    # unexpected exception here reports its own stage instead of leaving
    # `current_stage` stale at "rkv_complete_pass1" (already completed).
    plan, plan_failure = operation_runner(
        "pass1_plan_construction",
        lambda: build_pass1_plan(trace, num_hidden_layers, num_key_value_heads, identity),
    )
    if plan is None:
        stage = (
            STAGE_FEWER_THAN_THREE_ELIGIBLE_EVENTS
            if plan_failure == PLAN_FAILURE_TOO_FEW_ELIGIBLE_EVENTS
            else STAGE_INVALID_CANDIDATE_DONOR_POOL
        )
        example_attrition.record_dropped(stage)
        return ExampleResult(example_id, False, stage, trace, ())

    try:
        pass2_result = operation_runner(
            "rkv_complete_pass2",
            lambda: run_pass2_capture(
                plan, trace.full_token_ids, pass2_initial_state_factory(), prefill_fn, decode_one_fn, snapshot_fn,
                capture_timer_fn=capture_timer_fn,
            ),
        )
    except Exception as exc:
        # Independent-audit Gate H1: distinct from the `pass2_result.valid
        # is False` branch below (a normal return reporting a DETECTED
        # trajectory mismatch) -- this is Pass 2 execution itself raising
        # (e.g. a real CUDA OOM mid-capture). Pass 1's already-valid
        # `trace` and both attrition counters' state so far are preserved
        # in the returned `ExampleResult` rather than lost to a bare
        # propagating exception.
        example_attrition.record_dropped(STAGE_PASS2_EXECUTION_EXCEPTION)
        return ExampleResult(
            example_id, False, STAGE_PASS2_EXECUTION_EXCEPTION, trace, (),
            aborted=True, abort_failure_type=type(exc).__name__, abort_failure_message=str(exc),
            abort_is_oom=("out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"),
        )
    if not pass2_result.valid:
        stage = _PASS2_REASON_TO_STAGE[pass2_result.invalid_reason]
        example_attrition.record_dropped(stage)
        # Independent-audit Gate H3.7 repair: `pass2_result.replayed_token_ids`
        # is real diagnostic evidence -- Pass 2 replays every fed token
        # before detecting a mismatch, so this is exactly the sequence a
        # human (or `kvcot.discovery.mismatch.build_mismatch_record`) needs
        # to see WHAT was actually fed, compared against `trace
        # .full_token_ids`, to diagnose a `pass2_token_mismatch` without
        # re-running the model. Previously discarded here (defaulting to
        # the empty-tuple `ExampleResult.pass2_replayed_token_ids` field),
        # leaving only the bare stage name.
        return ExampleResult(
            example_id, False, stage, trace, (), pass2_invalid_reason=pass2_result.invalid_reason,
            pass2_replayed_token_ids=tuple(pass2_result.replayed_token_ids),
        )

    example_attrition.record_passed()
    pass2_replayed_token_ids = pass2_result.replayed_token_ids
    pass2_compaction_event_positions = pass2_result.compaction_event_positions
    selected_event_evidence = tuple(
        {
            "compaction_event_id": event.compaction_event_id,
            "absolute_event_position": event.absolute_event_position,
            "layer_index": event.layer_index,
            "kv_head_index": event.kv_head_index,
        }
        for event in plan.events
    )

    # B1B-R4 §18: minimize every selected target's capture evidence RIGHT
    # HERE, while `pass2_result.target_captures` (which holds full-layer/
    # full-cache tensors) is still in scope -- the object this function
    # returns never needs a full tensor again after this point.
    #
    # B1B-R4.1 §16 repair: `assert_minimized_bound` used to be exercised
    # only by `tests/unit/discovery/test_capture_minimize.py` -- never
    # called from any production code path, so a future regression growing
    # persistent per-target storage with model size would have gone
    # undetected outside that one test file. Called here, unconditionally,
    # for every target this function ever builds.
    from kvcot.discovery.capture_minimize import assert_minimized_bound, build_minimized_target_evidence

    def _build_minimized_target_evidence() -> tuple[Any, ...]:
        evidence_tuple = tuple(
            build_minimized_target_evidence(tc.event_plan, tc.capture_record) for tc in pass2_result.target_captures
        )
        for evidence in evidence_tuple:
            assert_minimized_bound(evidence)
        return evidence_tuple

    # R1: same rationale as `pass1_plan_construction` above -- otherwise an
    # unexpected exception here reports `current_stage` as
    # "rkv_complete_pass2" (already completed), not this derivation step.
    minimized_target_evidence = operation_runner(
        "minimized_target_evidence_construction", _build_minimized_target_evidence
    )

    # Convert transient full captures before the first branch starts.  Once
    # Pass2Result is deleted, pair evaluation can reach only selected vectors
    # and scalars plus the one required pristine snapshot per event.
    compact_targets = operation_runner(
        "compact_target_conversion",
        lambda: tuple(build_compact_branch_target(tc) for tc in pass2_result.target_captures),
    )
    del pass2_result

    pair_records: list[SwapPairRecord] = []
    pair_failure_details: list[PairFailureDetail] = []
    attempted_real = 0
    attempted_no_op = 0
    completed_real = 0
    completed_no_op = 0
    real_pair_wall_seconds: list[float] = []
    no_op_pair_wall_seconds: list[float] = []
    no_op_mode = pair_execution_policy.no_op_mode
    # B1 execution-boundary closure §12: POSITIVE semantic-swap-check
    # counts across every REAL pair attempted (no-op pairs excluded --
    # the no-op control has its own dedicated numerical-parity check,
    # never folded into this count) -- read directly off
    # `PairBuildResult.semantic_swap_check_attempted`/`.semantic_swap_check_passed`,
    # never derived after the fact from "no failure record exists".
    semantic_swap_checks_attempted = 0
    semantic_swap_checks_passed = 0
    pre_branch_memory_evidence: list[Any] = []
    attempted_pair_identities: list[dict[str, Any]] = []
    completed_pair_identities: list[dict[str, Any]] = []
    semantic_mutation_reports: list[dict[str, Any]] = []

    for event_index, target_capture in enumerate(compact_targets):
        cd = target_capture.event_plan.candidate_donor_selection
        noop_position = cd.donor_selected[0]
        real_pairs = list(cd.cross_product)

        # B1B-R4 §7: the no-op mode ACTUALLY CONTROLS whether a no-op pair
        # is even attempted for this event -- never unconditionally built
        # and then merely relabeled downstream.
        if no_op_mode == NoOpMode.CPU_REQUIRED:
            include_noop = True
        elif no_op_mode == NoOpMode.B2A_SINGLE_CALIBRATION:
            # Exactly ONE no-op pair for the whole example, drawn from the
            # FIRST selected event in the frozen plan -- deterministic,
            # never re-selected per run, never one per event.
            include_noop = event_index == 0
        elif no_op_mode == NoOpMode.DISABLED:
            include_noop = False
        else:
            raise ValueError(f"unrecognized NoOpMode: {no_op_mode!r}")

        pairs_to_build = [(pos, "real") for pos in real_pairs]
        if include_noop:
            pairs_to_build.append(((noop_position, noop_position), "no_op"))

        for (evicted_pos, donor_pos), kind in pairs_to_build:
            identity_record = {
                "compaction_event_id": target_capture.event_plan.compaction_event_id,
                "layer_index": target_capture.event_plan.layer_index,
                "kv_head_index": target_capture.event_plan.kv_head_index,
                "candidate_absolute_position": evicted_pos,
                "donor_absolute_position": donor_pos,
                "pair_kind": kind,
            }
            attempted_pair_identities.append(identity_record)
            if kind == "real":
                attempted_real += 1
            else:
                attempted_no_op += 1
            pair_attrition.record_entered()
            try:
                if pre_branch_guard is not None:
                    guard_evidence = pre_branch_guard(target_capture, kind)
                    pre_branch_memory_evidence.append(guard_evidence)
                    if not guard_evidence.accepted:
                        semantic_mutation_reports.append({
                            "pair_identity": identity_record,
                            "attempted": False,
                            "passed": False,
                            "k_slot_change_count": 0,
                            "v_slot_change_count": 0,
                            "provenance_update_count": 0,
                            "kept_index_update_count": 0,
                            "physical_byte_delta": 0,
                            "failure_reason": guard_evidence.rejection_reason,
                        })
                        pair_attrition.record_dropped(STAGE_BRANCH_EVALUATION_FAILURE)
                        pair_failure_details.append(
                            PairFailureDetail(
                                compaction_event_id=target_capture.event_plan.compaction_event_id,
                                layer_index=target_capture.event_plan.layer_index,
                                kv_head_index=target_capture.event_plan.kv_head_index,
                                evicted_absolute_position=evicted_pos,
                                donor_absolute_position=donor_pos,
                                pair_kind=kind,
                                stage=STAGE_BRANCH_EVALUATION_FAILURE,
                                detail=guard_evidence.rejection_reason,
                                elapsed_seconds=0.0,
                            )
                        )
                        continue
                # B1B-R4 §12: one non-overlapping wall-clock measurement
                # around the ENTIRE pair-construction call (clone/restore,
                # semantic swap, bridge-plus-scored evaluation for BOTH
                # baseline and swapped branches) -- never an aggregate
                # bucket shared across every pair in this loop.
                pair_start = clock_fn()
                result = operation_runner(
                    f"{kind}_pair:{target_capture.event_plan.compaction_event_id}:{evicted_pos}:{donor_pos}",
                    lambda: build_swap_pair_record(
                        example_id=example_id,
                        model_revision=model_revision,
                        rkv_revision=rkv_revision,
                        target_capture=target_capture,
                        evicted_absolute_position=evicted_pos,
                        donor_absolute_position=donor_pos,
                        trace=trace,
                        branch_step_fn=branch_step_fn,
                        phase_runner=pair_phase_runner,
                    ),
                )
                pair_elapsed = clock_fn() - pair_start
                ev = target_capture.event_plan

                mutation = result.semantic_mutation_report or {
                    "attempted": result.semantic_swap_check_attempted,
                    "passed": result.semantic_swap_check_passed,
                    "k_slot_change_count": 0,
                    "v_slot_change_count": 0,
                    "provenance_update_count": 0,
                    "kept_index_update_count": 0,
                    "physical_byte_delta": 0,
                    "failure_reason": result.failure_detail,
                }
                semantic_mutation_reports.append({"pair_identity": identity_record, **mutation})

                if kind == "real" and result.semantic_swap_check_attempted:
                    semantic_swap_checks_attempted += 1
                    if result.semantic_swap_check_passed:
                        semantic_swap_checks_passed += 1

                def _record_pair_failure(stage: str, detail: str | None) -> None:
                    pair_attrition.record_dropped(stage)
                    pair_failure_details.append(
                        PairFailureDetail(
                            compaction_event_id=ev.compaction_event_id,
                            layer_index=ev.layer_index,
                            kv_head_index=ev.kv_head_index,
                            evicted_absolute_position=evicted_pos,
                            donor_absolute_position=donor_pos,
                            pair_kind=kind,
                            stage=stage,
                            detail=detail,
                            elapsed_seconds=pair_elapsed,
                        )
                    )

                if result.record is None:
                    _record_pair_failure(_PAIR_STAGE_MAP[result.failure_stage], result.failure_detail)
                    continue
                if not result.record.valid_flag:
                    # B1B-R4.1 §18: a pair whose record WAS constructed but
                    # whose derived parity check failed (e.g. a real
                    # snapshot's semantic swap did not update provenance/
                    # kept-index bookkeeping) -- schema-valid, but not a
                    # success.
                    _record_pair_failure(STAGE_SEMANTIC_SWAP_PARITY_FAILURE, result.record.invalid_reason)
                    continue
                if _has_no_recorded_uncertainty_anywhere(result.record):
                    _record_pair_failure(STAGE_UNCERTAINTY_MISSING, "uncertainty_missing")
                    continue
                pair_attrition.record_passed()
                pair_records.append(result.record)
                completed_pair_identities.append(identity_record)
                if kind == "real":
                    completed_real += 1
                    real_pair_wall_seconds.append(pair_elapsed)
                else:
                    completed_no_op += 1
                    no_op_pair_wall_seconds.append(pair_elapsed)
            except Exception as exc:
                # Independent-audit Gate H1: every branch above that
                # detects an EXPECTED failure mode reports it as data (a
                # `PairFailureDetail`) and `continue`s to the next pair --
                # this catches only a genuinely UNEXPECTED exception (e.g.
                # a real CUDA OOM inside `pre_branch_guard`/
                # `build_swap_pair_record`). Every pair completed in
                # earlier loop iterations (`pair_records`,
                # `attempted_pair_identities`, `completed_pair_identities`,
                # `semantic_mutation_reports`, `pre_branch_memory_evidence`,
                # both attrition counters) is preserved in the returned
                # `ExampleResult` rather than lost to a bare propagating
                # exception.
                pair_attrition.record_dropped(STAGE_UNEXPECTED_PAIR_EXCEPTION)
                pair_failure_details.append(
                    PairFailureDetail(
                        compaction_event_id=target_capture.event_plan.compaction_event_id,
                        layer_index=target_capture.event_plan.layer_index,
                        kv_head_index=target_capture.event_plan.kv_head_index,
                        evicted_absolute_position=evicted_pos,
                        donor_absolute_position=donor_pos,
                        pair_kind=kind,
                        stage=STAGE_UNEXPECTED_PAIR_EXCEPTION,
                        detail=f"{type(exc).__name__}: {exc}",
                        elapsed_seconds=0.0,
                    )
                )
                return ExampleResult(
                    example_id, True, None, trace, tuple(pair_records),
                    attempted_real_pair_count=attempted_real,
                    attempted_no_op_pair_count=attempted_no_op,
                    completed_real_pair_count=completed_real,
                    completed_no_op_pair_count=completed_no_op,
                    real_pair_wall_seconds=tuple(real_pair_wall_seconds),
                    no_op_pair_wall_seconds=tuple(no_op_pair_wall_seconds),
                    pass2_invalid_reason=None,
                    minimized_target_evidence=minimized_target_evidence,
                    selected_event_ids=tuple(e.compaction_event_id for e in plan.events),
                    pair_failure_details=tuple(pair_failure_details),
                    semantic_swap_checks_attempted=semantic_swap_checks_attempted,
                    semantic_swap_checks_passed=semantic_swap_checks_passed,
                    pre_branch_memory_evidence=tuple(pre_branch_memory_evidence),
                    attempted_pair_identities=tuple(attempted_pair_identities),
                    completed_pair_identities=tuple(completed_pair_identities),
                    semantic_mutation_reports=tuple(semantic_mutation_reports),
                    selected_event_evidence=selected_event_evidence,
                    pass2_replayed_token_ids=pass2_replayed_token_ids,
                    pass2_compaction_event_positions=pass2_compaction_event_positions,
                    aborted=True,
                    abort_failure_type=type(exc).__name__,
                    abort_failure_message=str(exc),
                    abort_is_oom=("out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"),
                )

    return ExampleResult(
        example_id, True, None, trace, tuple(pair_records),
        attempted_real_pair_count=attempted_real,
        attempted_no_op_pair_count=attempted_no_op,
        completed_real_pair_count=completed_real,
        completed_no_op_pair_count=completed_no_op,
        real_pair_wall_seconds=tuple(real_pair_wall_seconds),
        no_op_pair_wall_seconds=tuple(no_op_pair_wall_seconds),
        pass2_invalid_reason=None,
        minimized_target_evidence=minimized_target_evidence,
        selected_event_ids=tuple(ev.compaction_event_id for ev in plan.events),
        pair_failure_details=tuple(pair_failure_details),
        semantic_swap_checks_attempted=semantic_swap_checks_attempted,
        semantic_swap_checks_passed=semantic_swap_checks_passed,
        pre_branch_memory_evidence=tuple(pre_branch_memory_evidence),
        attempted_pair_identities=tuple(attempted_pair_identities),
        completed_pair_identities=tuple(completed_pair_identities),
        semantic_mutation_reports=tuple(semantic_mutation_reports),
        selected_event_evidence=selected_event_evidence,
        pass2_replayed_token_ids=pass2_replayed_token_ids,
        pass2_compaction_event_positions=pass2_compaction_event_positions,
    )


def _has_no_recorded_uncertainty_anywhere(record: SwapPairRecord) -> bool:
    """A pair-level attrition drop (`uncertainty_missing`) fires ONLY when
    every one of the four uncertainty source values is missing for the
    structural reason "no logits were ever recorded at that position" --
    never for a legitimate, schema-expressible missingness (e.g. a
    non-finite computed value, or position 0). Those legitimate cases stay
    schema-valid and are NOT an attrition drop; the schema's own
    `*_missing_reason` fields already make them auditable."""
    reasons = (
        record.entropy_e_missing_reason,
        record.entropy_r_missing_reason,
        record.logit_margin_e_missing_reason,
        record.logit_margin_r_missing_reason,
    )
    values_present = (record.entropy_e, record.entropy_r, record.logit_margin_e, record.logit_margin_r)
    if any(v is not None for v in values_present):
        return False
    return all(reason == UNCERTAINTY_POSITION_UNAVAILABLE_REASON for reason in reasons)
