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
    STAGE_PASS2_TOKEN_MISMATCH,
    STAGE_PREFILL_CONTRACT_VIOLATION,
    STAGE_SCHEMA_VALIDATION_FAILURE,
    STAGE_SEMANTIC_SWAP_PARITY_FAILURE,
    STAGE_UNCERTAINTY_MISSING,
    AttritionCounters,
    PairFailureDetail,
)
from kvcot.discovery.constants import NoOpMode
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
) -> ExampleResult:
    """`pair_execution_policy` defaults to `PairExecutionPolicy()`
    (`NoOpMode.CPU_REQUIRED`) -- every pre-existing caller that does not
    pass one explicitly gets the exact same pair-construction behavior this
    function always had (B1B-R4 §7). `clock_fn` (B1B-R4 §12) defaults to
    `time.monotonic` -- CPU tests inject a deterministic fake clock instead,
    so per-pair timing is exercised without depending on real wall-clock
    variance."""
    import time

    clock_fn = clock_fn or time.monotonic
    pair_execution_policy = pair_execution_policy or PairExecutionPolicy()
    example_attrition.record_entered()

    try:
        trace = run_natural_pass1(
            provenance, prompt_token_ids, pass1_initial_state, prefill_fn, decode_one_fn, max_new_tokens,
            eos_token_id, answer_fn,
        )
    except Exception:
        example_attrition.record_dropped(STAGE_NATURAL_RUN_INVALID)
        return ExampleResult(example_id, False, STAGE_NATURAL_RUN_INVALID, None, ())

    if trace.natural_answer_status != "correct":
        example_attrition.record_dropped(STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE)
        return ExampleResult(example_id, False, STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE, trace, ())

    if trace.cap_hit:
        example_attrition.record_dropped(STAGE_CAP_HIT)
        return ExampleResult(example_id, False, STAGE_CAP_HIT, trace, ())

    plan, plan_failure = build_pass1_plan(trace, num_hidden_layers, num_key_value_heads, identity)
    if plan is None:
        stage = (
            STAGE_FEWER_THAN_THREE_ELIGIBLE_EVENTS
            if plan_failure == PLAN_FAILURE_TOO_FEW_ELIGIBLE_EVENTS
            else STAGE_INVALID_CANDIDATE_DONOR_POOL
        )
        example_attrition.record_dropped(stage)
        return ExampleResult(example_id, False, stage, trace, ())

    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, pass2_initial_state_factory(), prefill_fn, decode_one_fn, snapshot_fn
    )
    if not pass2_result.valid:
        stage = _PASS2_REASON_TO_STAGE[pass2_result.invalid_reason]
        example_attrition.record_dropped(stage)
        return ExampleResult(
            example_id, False, stage, trace, (), pass2_invalid_reason=pass2_result.invalid_reason
        )

    example_attrition.record_passed()

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

    minimized_target_evidence = tuple(
        build_minimized_target_evidence(tc.event_plan, tc.capture_record) for tc in pass2_result.target_captures
    )
    for evidence in minimized_target_evidence:
        assert_minimized_bound(evidence)

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

    for event_index, target_capture in enumerate(pass2_result.target_captures):
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
            if kind == "real":
                attempted_real += 1
            else:
                attempted_no_op += 1
            pair_attrition.record_entered()
            # B1B-R4 §12: one non-overlapping wall-clock measurement around
            # the ENTIRE pair-construction call (clone/restore, semantic
            # swap, bridge-plus-scored evaluation for BOTH baseline and
            # swapped branches) -- never an aggregate bucket shared across
            # every pair in this loop.
            pair_start = clock_fn()
            result = build_swap_pair_record(
                example_id=example_id,
                model_revision=model_revision,
                rkv_revision=rkv_revision,
                target_capture=target_capture,
                evicted_absolute_position=evicted_pos,
                donor_absolute_position=donor_pos,
                trace=trace,
                branch_step_fn=branch_step_fn,
            )
            pair_elapsed = clock_fn() - pair_start
            ev = target_capture.event_plan

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
                # whose derived parity check failed (e.g. a real snapshot's
                # semantic swap did not update provenance/kept-index
                # bookkeeping) -- schema-valid, but not a success.
                _record_pair_failure(STAGE_SEMANTIC_SWAP_PARITY_FAILURE, result.record.invalid_reason)
                continue
            if _has_no_recorded_uncertainty_anywhere(result.record):
                _record_pair_failure(STAGE_UNCERTAINTY_MISSING, "uncertainty_missing")
                continue
            pair_attrition.record_passed()
            pair_records.append(result.record)
            if kind == "real":
                completed_real += 1
                real_pair_wall_seconds.append(pair_elapsed)
            else:
                completed_no_op += 1
                no_op_pair_wall_seconds.append(pair_elapsed)

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
