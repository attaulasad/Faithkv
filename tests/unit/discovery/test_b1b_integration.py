"""B1B CPU harness integration tests
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §12). Synthetic Pass 1 ->
deterministic event/depth/head/pair plan -> token-identical Pass 2 ->
second-or-later compaction absolute parity -> candidate/donor capture ->
fixed-shape swap -> bridge call -> 48-token teacher-forced evaluation ->
uncertainty lookup -> `SwapPairRecord` validation -> attrition output, all
against injected synthetic/deterministic components. No real model is
loaded anywhere in this file.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch
from pydantic import ValidationError

from _synthetic_harness import (
    BUDGET,
    EOS_TOKEN_ID,
    NUM_HEADS,
    NUM_LAYERS,
    WINDOW,
    HarnessState,
    branch_step_fn,
    fresh_state_factory,
    install_fake_rkv_compression_module,
    make_natural_step_fn,
)
from _synthetic_harness_variants import make_query_salt_step_fn, make_schedule_shifted_step_fn

from kvcot.discovery.attrition import STAGE_UNCERTAINTY_MISSING, AttritionCounters
from kvcot.discovery.orchestrator import ExampleResult, _has_no_recorded_uncertainty_anywhere, run_example
from kvcot.discovery.pass1 import NaturalRunProvenance, build_pass1_plan, run_natural_pass1
from kvcot.discovery.pass2 import (
    INVALID_COMPACTION_POSITION_MISMATCH,
    INVALID_MISSING_TARGET_CAPTURE,
    INVALID_TOKEN_MISMATCH,
    run_pass2_capture,
)
from kvcot.discovery.pipeline import build_swap_pair_record
from kvcot.discovery.sampling import IdentitySeedParts
from kvcot.discovery.swap import SwapIndexError, apply_within_head_swap

PROMPT_LENGTH = 10
DESIRED_GENERATED_LENGTH = 290
STOP_AT = PROMPT_LENGTH + DESIRED_GENERATED_LENGTH
MAX_NEW_TOKENS = 295
PROMPT_TOKEN_IDS = list(range(1, PROMPT_LENGTH + 1))
IDENTITY = IdentitySeedParts(
    global_seed=13, dataset_name="synthetic", problem_index=0, model_revision="rev-a", rkv_revision="rkv-rev"
)
PROVENANCE = NaturalRunProvenance(
    model_name="synthetic-model",
    model_revision="rev-a",
    tokenizer_name="synthetic-tokenizer",
    tokenizer_revision="rev-a",
    rkv_revision="rkv-rev",
    config_sha256="deadbeef",
    dataset_name="synthetic",
    example_id="ex-1",
)


def _always_correct(generated_ids: list[int]) -> tuple[str, str]:
    return "42", "correct"


def _new_attrition_pair() -> tuple[AttritionCounters, AttritionCounters]:
    return AttritionCounters(), AttritionCounters()


def _run_example(monkeypatch, step_fn=None, example_id="ex-1"):
    install_fake_rkv_compression_module(monkeypatch)
    if step_fn is None:
        step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)
    example_attrition, pair_attrition = _new_attrition_pair()
    result = run_example(
        example_id=example_id,
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        provenance=PROVENANCE,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        pass1_initial_state=HarnessState(),
        pass2_initial_state_factory=fresh_state_factory(),
        step_fn=step_fn,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=EOS_TOKEN_ID,
        answer_fn=_always_correct,
        num_hidden_layers=NUM_LAYERS,
        num_key_value_heads=NUM_HEADS,
        identity=IDENTITY,
        branch_step_fn=branch_step_fn,
        example_attrition=example_attrition,
        pair_attrition=pair_attrition,
    )
    return result, example_attrition, pair_attrition


# --------------------------------------------------------------------------
# 1. Complete valid example (full injected orchestration, end to end)
# --------------------------------------------------------------------------


def test_complete_valid_example_end_to_end(monkeypatch):
    result, example_attrition, pair_attrition = _run_example(monkeypatch)

    assert result.valid is True
    assert result.invalid_stage is None
    assert result.trace.cap_hit is False
    assert result.trace.natural_answer_status == "correct"
    assert len(result.pair_records) == 3 * 5  # 3 events x (4 cross-product + 1 mandatory no-op)

    example_attrition.assert_consistent()
    pair_attrition.assert_consistent()
    assert example_attrition.passed_all == 1
    assert example_attrition.total_entered == 1
    assert pair_attrition.total_entered == 15

    for record in result.pair_records:
        assert record.valid_flag is True
        assert record.parity_check_passed is True
        assert len(record.baseline_per_token_nll) == 48
        assert len(record.swapped_per_token_nll) == 48


# --------------------------------------------------------------------------
# 2. Multi-event valid example with non-identity absolute map
# --------------------------------------------------------------------------


def test_multi_event_plan_has_non_identity_pre_event_map_on_a_later_event(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)

    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), step_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID, _always_correct
    )
    assert len(trace.compaction_events) >= 5  # plenty of events at DIVIDE_LENGTH spacing

    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert failure is None
    assert plan is not None
    assert len(plan.events) == 3

    pass2_result = run_pass2_capture(plan, trace.full_token_ids, HarnessState(), step_fn)
    assert pass2_result.valid is True

    non_identity_found = False
    for target_capture in pass2_result.target_captures:
        pre_map = target_capture.capture_record.pre_event_absolute_position_map
        num_heads = pre_map.shape[0]
        identity_map = torch.arange(pre_map.shape[1]).unsqueeze(0).expand(num_heads, -1)
        # An event's map is non-identity whenever it is not the FIRST
        # compaction event of the run (its predecessor's shuffled survivor
        # selection feeds into it) -- assert this holds for at least one
        # selected event, and verify ordered (not set) comparison is what's
        # being used.
        if not torch.equal(pre_map, identity_map):
            non_identity_found = True
    assert non_identity_found, "expected at least one selected event's pre-event map to be non-identity"


# --------------------------------------------------------------------------
# 3. Pass-2 token mismatch invalidates example
# --------------------------------------------------------------------------


def test_pass2_token_mismatch_invalidates_example(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), step_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID, _always_correct
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert plan is not None

    corrupted_tokens = list(trace.full_token_ids)
    corrupted_tokens[50] = (corrupted_tokens[50] + 1) % 64

    result = run_pass2_capture(plan, corrupted_tokens, HarnessState(), step_fn)
    assert result.valid is False
    assert result.invalid_reason == INVALID_TOKEN_MISMATCH
    assert result.target_captures == ()


# --------------------------------------------------------------------------
# 4. Compaction-position mismatch invalidates example
# --------------------------------------------------------------------------


def test_compaction_position_mismatch_invalidates_example(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    natural_step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), natural_step_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID, _always_correct
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert plan is not None

    shifted_step_fn = make_schedule_shifted_step_fn(schedule_offset=3, stop_at_predicted_position=STOP_AT)
    result = run_pass2_capture(plan, trace.full_token_ids, HarnessState(), shifted_step_fn)
    assert result.valid is False
    # Under the shifted schedule, the selected event's absolute position
    # either has no capture record at all (no update_kv call happened
    # there) or has one that did not compact -- both are the same
    # underlying failure (a compaction-event-position mismatch) and both
    # map to the same orchestrator-level attrition stage.
    assert result.invalid_reason in (INVALID_COMPACTION_POSITION_MISMATCH, INVALID_MISSING_TARGET_CAPTURE)


# --------------------------------------------------------------------------
# 5. Survivor order mismatch invalidates example
# --------------------------------------------------------------------------


def test_survivor_mismatch_invalidates_example(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    natural_step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), natural_step_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID, _always_correct
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert plan is not None

    diverged_step_fn = make_query_salt_step_fn(query_salt="DIFFERENT", stop_at_predicted_position=STOP_AT)
    result = run_pass2_capture(plan, trace.full_token_ids, HarnessState(), diverged_step_fn)
    assert result.valid is False
    assert result.invalid_reason in (
        "pass2_observed_survivor_parity_failed",
        "pass2_survivor_mismatch_vs_pass1",
    )


# --------------------------------------------------------------------------
# 6. Missing mandatory uncertainty -> explicit invalid/adjudicability state
# --------------------------------------------------------------------------


def test_missing_uncertainty_produces_explicit_missing_reason_and_attrition_signal(monkeypatch):
    result, _, _ = _run_example(monkeypatch)
    assert result.valid is True
    target_capture = None  # rebuild one directly to control the trace

    install_fake_rkv_compression_module(monkeypatch)
    step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), step_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID, _always_correct
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(plan, trace.full_token_ids, HarnessState(), step_fn)
    assert pass2_result.valid is True
    target_capture = pass2_result.target_captures[0]
    cd = target_capture.event_plan.candidate_donor_selection
    evicted_pos, donor_pos = cd.cross_product[0]

    stripped_trace = dataclasses.replace(trace, uncertainty_by_position={})
    pair_result = build_swap_pair_record(
        example_id="ex-1",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        target_capture=target_capture,
        evicted_absolute_position=evicted_pos,
        donor_absolute_position=donor_pos,
        trace=stripped_trace,
        branch_step_fn=branch_step_fn,
    )
    assert pair_result.record is not None  # schema-valid: missing_reason fields are populated, not fabricated
    record = pair_result.record
    assert record.entropy_e is None and record.entropy_e_missing_reason is not None
    assert record.entropy_r is None and record.entropy_r_missing_reason is not None
    assert record.logit_margin_e is None and record.logit_margin_e_missing_reason is not None
    assert record.logit_margin_r is None and record.logit_margin_r_missing_reason is not None
    assert _has_no_recorded_uncertainty_anywhere(record) is True


# --------------------------------------------------------------------------
# 7. No-op produces identical logits, identical NLL arrays, zero gain
# --------------------------------------------------------------------------


def test_noop_produces_identical_nll_and_zero_gain(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), step_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID, _always_correct
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(plan, trace.full_token_ids, HarnessState(), step_fn)
    target_capture = pass2_result.target_captures[0]
    donor_pos = target_capture.event_plan.candidate_donor_selection.donor_selected[0]

    pair_result = build_swap_pair_record(
        example_id="ex-1",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        target_capture=target_capture,
        evicted_absolute_position=donor_pos,
        donor_absolute_position=donor_pos,
        trace=trace,
        branch_step_fn=branch_step_fn,
    )
    assert pair_result.record is not None
    record = pair_result.record
    assert record.is_noop_control is True
    assert record.baseline_per_token_nll == record.swapped_per_token_nll
    assert record.swap_gain == 0.0
    assert record.net_physical_bytes_changed == 0


# --------------------------------------------------------------------------
# 8. Candidate dtype mismatch is rejected
# --------------------------------------------------------------------------


def test_candidate_dtype_mismatch_rejected_using_real_captured_tensors(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    step_fn = make_natural_step_fn(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), step_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID, _always_correct
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(plan, trace.full_token_ids, HarnessState(), step_fn)
    target_capture = pass2_result.target_captures[0]
    record = target_capture.capture_record
    head = target_capture.event_plan.kv_head_index

    real_candidate_key = record.pre_call_key_states[0, head, 0, :].clone()
    bad_candidate_key = real_candidate_key.to(torch.float64)  # dtype mismatch vs the target cache
    real_candidate_value = record.pre_call_value_states[0, head, 0, :].clone()

    with pytest.raises(SwapIndexError):
        apply_within_head_swap(
            key_cache=[record.returned_key_states],
            value_cache=[record.returned_value_states],
            layer_index=0,
            kv_head_index=head,
            retained_post_storage_position=0,
            candidate_key=bad_candidate_key,
            candidate_value=real_candidate_value,
        )


# --------------------------------------------------------------------------
# 9. Derived schema inconsistency is rejected
# --------------------------------------------------------------------------


def test_derived_schema_inconsistency_rejected_on_real_pipeline_output(monkeypatch):
    result, _, _ = _run_example(monkeypatch)
    assert result.valid is True
    real_record = result.pair_records[0]

    corrupted = real_record.model_dump()
    corrupted["score_margin_e_minus_r"] = corrupted["score_e"] - corrupted["score_r"] + 5.0
    with pytest.raises(ValidationError):
        type(real_record)(**corrupted)


# --------------------------------------------------------------------------
# 10. Repeated run with the same seeds produces byte-identical records
# --------------------------------------------------------------------------


def test_repeated_run_same_seeds_byte_identical_planning_records(monkeypatch):
    result_a, _, _ = _run_example(monkeypatch, example_id="ex-repeat")
    result_b, _, _ = _run_example(monkeypatch, example_id="ex-repeat")

    assert result_a.trace.reference_trace_sha256 == result_b.trace.reference_trace_sha256
    assert result_a.trace.full_token_ids == result_b.trace.full_token_ids
    assert len(result_a.pair_records) == len(result_b.pair_records)

    dumps_a = [r.model_dump_json() for r in result_a.pair_records]
    dumps_b = [r.model_dump_json() for r in result_b.pair_records]
    assert dumps_a == dumps_b
