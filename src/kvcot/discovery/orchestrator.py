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
    STAGE_NATURAL_RUN_INVALID,
    STAGE_OBSERVED_SURVIVOR_MISMATCH,
    STAGE_PASS2_TOKEN_MISMATCH,
    STAGE_SCHEMA_VALIDATION_FAILURE,
    STAGE_UNCERTAINTY_MISSING,
    AttritionCounters,
)
from kvcot.discovery.harness_types import NaturalStepFn
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
    INVALID_COMPACTION_POSITION_MISMATCH: STAGE_COMPACTION_EVENT_MISMATCH,
    INVALID_MISSING_TARGET_CAPTURE: STAGE_COMPACTION_EVENT_MISMATCH,
    INVALID_SURVIVOR_MISMATCH_WITHIN_PASS2: STAGE_CAPTURE_GATHER_PARITY_FAILURE,
    INVALID_SURVIVOR_MISMATCH_ACROSS_PASSES: STAGE_OBSERVED_SURVIVOR_MISMATCH,
}

_PAIR_STAGE_MAP = {
    PAIR_STAGE_INVALID_CANDIDATE_DONOR_POOL: STAGE_INVALID_CANDIDATE_DONOR_POOL,
    PAIR_STAGE_BRANCH_EVALUATION_FAILURE: STAGE_BRANCH_EVALUATION_FAILURE,
    PAIR_STAGE_SCHEMA_VALIDATION_FAILURE: STAGE_SCHEMA_VALIDATION_FAILURE,
}


@dataclass(frozen=True)
class ExampleResult:
    example_id: str
    valid: bool
    invalid_stage: str | None
    trace: NaturalRunTrace | None
    pair_records: tuple[SwapPairRecord, ...]


def run_example(
    *,
    example_id: str,
    model_revision: str,
    rkv_revision: str,
    provenance: NaturalRunProvenance,
    prompt_token_ids: list[int],
    pass1_initial_state: Any,
    pass2_initial_state_factory: Callable[[], Any],
    step_fn: NaturalStepFn,
    max_new_tokens: int,
    eos_token_id: int | None,
    answer_fn: AnswerFn,
    num_hidden_layers: int,
    num_key_value_heads: int,
    identity: IdentitySeedParts,
    branch_step_fn,
    example_attrition: AttritionCounters,
    pair_attrition: AttritionCounters,
) -> ExampleResult:
    example_attrition.record_entered()

    try:
        trace = run_natural_pass1(
            provenance, prompt_token_ids, pass1_initial_state, step_fn, max_new_tokens, eos_token_id, answer_fn
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

    pass2_result = run_pass2_capture(plan, trace.full_token_ids, pass2_initial_state_factory(), step_fn)
    if not pass2_result.valid:
        stage = _PASS2_REASON_TO_STAGE[pass2_result.invalid_reason]
        example_attrition.record_dropped(stage)
        return ExampleResult(example_id, False, stage, trace, ())

    example_attrition.record_passed()

    pair_records: list[SwapPairRecord] = []
    for target_capture in pass2_result.target_captures:
        cd = target_capture.event_plan.candidate_donor_selection
        noop_position = cd.donor_selected[0]
        pairs_to_build = list(cd.cross_product) + [(noop_position, noop_position)]

        for evicted_pos, donor_pos in pairs_to_build:
            pair_attrition.record_entered()
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
            if result.record is None:
                pair_attrition.record_dropped(_PAIR_STAGE_MAP[result.failure_stage])
                continue
            if _has_no_recorded_uncertainty_anywhere(result.record):
                pair_attrition.record_dropped(STAGE_UNCERTAINTY_MISSING)
                continue
            pair_attrition.record_passed()
            pair_records.append(result.record)

    return ExampleResult(example_id, True, None, trace, tuple(pair_records))


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
