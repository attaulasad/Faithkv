"""B1B/B1B-R2 CPU harness integration tests
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §12,
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §12). Synthetic
Pass 1 (one-shot prefill + one-token decode) -> deterministic event/depth/
head/pair plan -> token-identical Pass 2 (same call-boundary split) ->
second-or-later compaction absolute parity -> targeted, memory-bounded
capture -> complete multi-layer `ModelStateSnapshot` per target -> candidate/
donor capture -> fixed-shape swap on an independent snapshot clone ->
bridge call -> 48-token teacher-forced evaluation -> uncertainty lookup ->
`SwapPairRecord` validation -> attrition output, all against injected
synthetic/deterministic components. No real model is loaded anywhere in
this file.
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
    build_snapshot_from_state,
    fresh_state_factory,
    install_fake_rkv_compression_module,
    make_step_fns,
)
from _synthetic_harness_variants import make_query_salt_step_fns, make_schedule_shifted_step_fns

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
from kvcot.generation.state import ModelStateSnapshot

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


def _run_example(monkeypatch, step_fns=None, example_id="ex-1"):
    install_fake_rkv_compression_module(monkeypatch)
    if step_fns is None:
        step_fns = make_step_fns(stop_at_predicted_position=STOP_AT)
    prefill_fn, decode_one_fn = step_fns
    example_attrition, pair_attrition = _new_attrition_pair()
    result = run_example(
        example_id=example_id,
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        provenance=PROVENANCE,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        pass1_initial_state=HarnessState(),
        pass2_initial_state_factory=fresh_state_factory(),
        prefill_fn=prefill_fn,
        decode_one_fn=decode_one_fn,
        snapshot_fn=build_snapshot_from_state,
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
    assert len(result.pair_records) == 3 * 5  # 3 events x (4 real cross-product swaps + 1 mandatory no-op)

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
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    assert len(trace.compaction_events) >= 5  # plenty of events at DIVIDE_LENGTH spacing

    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert failure is None
    assert plan is not None
    assert len(plan.events) == 3

    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
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
        assert isinstance(target_capture.pristine_snapshot, ModelStateSnapshot)
        assert len(target_capture.pristine_snapshot.key_cache) == NUM_LAYERS
    assert non_identity_found, "expected at least one selected event's pre-event map to be non-identity"


# --------------------------------------------------------------------------
# 1a. B1B-R4.1 §16: the capture-minimization bound is enforced in production
# --------------------------------------------------------------------------


def test_run_example_enforces_the_minimized_capture_bound_not_just_the_test_suite(monkeypatch):
    """`kvcot.discovery.capture_minimize.assert_minimized_bound` used to be
    exercised only by `test_capture_minimize.py` -- never called from
    `run_example` itself, so a regression that grew persistent per-target
    storage would have gone undetected outside that one test file. Proven
    by monkeypatching `build_minimized_target_evidence` to return an
    evidence object that reports MORE persistent elements than its own
    `head_dim` bound allows, and confirming `run_example` now raises
    rather than silently returning an over-bound result."""
    import kvcot.discovery.capture_minimize as capture_minimize_mod
    from kvcot.discovery.capture_minimize import CaptureMinimizationError

    real_build = capture_minimize_mod.build_minimized_target_evidence

    def _oversized_build(event_plan, capture_record):
        import dataclasses

        evidence = real_build(event_plan, capture_record)
        return dataclasses.replace(evidence, persistent_tensor_numel=evidence.persistent_tensor_numel + 1_000_000)

    # `run_example` does `from kvcot.discovery.capture_minimize import
    # build_minimized_target_evidence` freshly on every call (a local, not
    # module-level, import) -- patching the source module's attribute is
    # what that fresh lookup actually resolves at call time.
    monkeypatch.setattr(capture_minimize_mod, "build_minimized_target_evidence", _oversized_build)

    with pytest.raises(CaptureMinimizationError):
        _run_example(monkeypatch)


# --------------------------------------------------------------------------
# 1b. B1B-R4.1 §15: structured per-pair failure evidence is actually populated
# --------------------------------------------------------------------------


def test_pair_failure_details_records_exactly_the_failed_pairs_not_an_empty_placeholder(monkeypatch):
    """`ExampleResult.pair_failure_details` used to be sourced from a
    parameter (`pair_attrition_dropped_stages`) the production R-KV worker
    path never actually populated, so it was always empty regardless of how
    many pairs failed. This proves the real, live-built path: force exactly
    ONE of the 15 pair attempts to fail (a `STAGE_BRANCH_EVALUATION_FAILURE`,
    via a patched `pipeline.build_swap_pair_record` that fails only for one
    specific (evicted, donor) pair) and confirms the resulting
    `pair_failure_details` names exactly that pair -- event, layer, head,
    positions, kind, stage, and a non-trivial elapsed time -- while every
    other pair still succeeds normally."""
    import kvcot.discovery.orchestrator as orchestrator_mod

    real_build = orchestrator_mod.build_swap_pair_record
    forced_failure: dict[str, object] = {}

    def _sometimes_failing_build(**kwargs):
        if not forced_failure and kwargs["evicted_absolute_position"] != kwargs["donor_absolute_position"]:
            forced_failure["event"] = kwargs["target_capture"].event_plan.compaction_event_id
            forced_failure["evicted"] = kwargs["evicted_absolute_position"]
            forced_failure["donor"] = kwargs["donor_absolute_position"]
            from kvcot.discovery.pipeline import STAGE_BRANCH_EVALUATION_FAILURE, PairBuildResult

            return PairBuildResult(None, STAGE_BRANCH_EVALUATION_FAILURE, "forced_failure_for_test")
        return real_build(**kwargs)

    monkeypatch.setattr(orchestrator_mod, "build_swap_pair_record", _sometimes_failing_build)

    result, example_attrition, pair_attrition = _run_example(monkeypatch)

    assert result.valid is True
    assert forced_failure, "the patched build function never saw a real pair -- test setup is broken"
    assert len(result.pair_failure_details) == 1
    failure = result.pair_failure_details[0]
    assert failure.compaction_event_id == forced_failure["event"]
    assert failure.evicted_absolute_position == forced_failure["evicted"]
    assert failure.donor_absolute_position == forced_failure["donor"]
    assert failure.pair_kind == "real"
    assert failure.stage == "branch_evaluation_failure"
    assert failure.detail == "forced_failure_for_test"
    assert failure.elapsed_seconds >= 0.0

    # Exactly one of the 15 total pair attempts failed -- every other pair
    # still succeeded normally (never dropped as collateral damage).
    assert len(result.pair_records) == 15 - 1
    pair_attrition.assert_consistent()
    assert pair_attrition.dropped_at["branch_evaluation_failure"] == 1


# --------------------------------------------------------------------------
# 2b. B1B-R4.1 §18: semantic-swap parity/byte evidence is derived, not hard-coded
# --------------------------------------------------------------------------


def test_semantic_swap_parity_is_derived_and_catches_a_missing_provenance_update(monkeypatch):
    """`build_swap_pair_record` must derive `parity_check_passed` from the
    real `SemanticSwapResult` mutation report, not report a hard-coded
    `True` regardless -- proven by monkeypatching
    `apply_semantic_within_head_swap` to report a missing provenance update
    on a snapshot that DOES carry provenance (the synthetic harness's own
    snapshots deliberately carry `provenance=None`, so a dummy provenance
    object is attached here to exercise the mandatory-update branch that
    only real-model-adapter snapshots would otherwise reach) and confirming
    the resulting build fails with a reason naming the exact defect, rather
    than silently succeeding."""
    import kvcot.discovery.pipeline as pipeline_mod
    from kvcot.generation.provenance import LayerProvenance, ModelProvenance

    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert failure is None and plan is not None

    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
    assert pass2_result.valid is True
    target_capture = pass2_result.target_captures[0]
    cd = target_capture.event_plan.candidate_donor_selection

    # A properly-SHAPED dummy provenance (one column per live cache slot at
    # each layer, matching the pristine snapshot's own key_cache length) --
    # `LayerProvenance.empty` alone has zero columns and cannot be indexed
    # at the swap's target slot.
    fake_provenance = ModelProvenance(
        layers={
            i: LayerProvenance(
                positions=torch.arange(target_capture.pristine_snapshot.key_cache[i].shape[-2])
                .unsqueeze(0)
                .expand(NUM_HEADS, -1)
                .clone()
            )
            for i in range(NUM_LAYERS)
        }
    )
    pristine_with_provenance = dataclasses.replace(target_capture.pristine_snapshot, provenance=fake_provenance)
    target_capture_with_provenance = dataclasses.replace(target_capture, pristine_snapshot=pristine_with_provenance)

    real_apply = pipeline_mod.apply_semantic_within_head_swap

    def _reporting_no_provenance_update(*args, **kwargs):
        result = real_apply(*args, **kwargs)
        return dataclasses.replace(result, provenance_updated=False)

    monkeypatch.setattr(pipeline_mod, "apply_semantic_within_head_swap", _reporting_no_provenance_update)

    build_result = build_swap_pair_record(
        example_id="ex-1",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        target_capture=target_capture_with_provenance,
        evicted_absolute_position=cd.evicted_selected[0],
        donor_absolute_position=cd.donor_selected[0],
        trace=trace,
        branch_step_fn=branch_step_fn,
    )

    # The record is still schema-valid (a well-formed report of a failed
    # pair, not dropped) -- `build_swap_pair_record`'s own docstring commits
    # to "the SAME code path, never a special case" for exactly this reason.
    assert build_result.record is not None, build_result.failure_detail
    assert build_result.record.parity_check_passed is False
    assert build_result.record.valid_flag is False
    assert "semantic_swap_parity_provenance_not_updated" in build_result.record.parity_failure_reason
    assert "semantic_swap_parity_provenance_not_updated" in build_result.record.invalid_reason


def test_baseline_snapshot_clone_is_released_before_swapped_clone_is_created(monkeypatch):
    """B1B-R4.1 §17: baseline and swapped snapshot clones must never be
    live at the same time -- proven via a weakref to each
    `ModelStateSnapshot.clone()` result: the baseline clone must already be
    unreachable (garbage-collected) once its evaluation has finished and
    `build_swap_pair_record` has moved on, not merely 'eventually' released
    after the whole pair completes. A custom step function that never hands
    back the SAME snapshot object it was given (`state.clone()` each call,
    exactly the shape the real-model adapter's restore-once branch stepping
    also has: `_LiveBranchState` is a different object than the
    `ModelStateSnapshot` it was restored from) is required -- the default
    synthetic `branch_step_fn` returns the identical snapshot object
    unchanged, which would keep it alive regardless of this repair and
    prove nothing."""
    import gc
    import weakref

    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
    target_capture = pass2_result.target_captures[0]
    cd = target_capture.event_plan.candidate_donor_selection

    clone_refs: list[weakref.ReferenceType] = []
    real_clone = ModelStateSnapshot.clone

    def _spying_clone(self):
        result = real_clone(self)
        clone_refs.append(weakref.ref(result))
        return result

    monkeypatch.setattr(ModelStateSnapshot, "clone", _spying_clone)

    def _non_retaining_step_fn(state, token_id):
        logits, _ = branch_step_fn(state, token_id)
        return logits, state.clone()  # never hands back the caller's own object

    build_result = build_swap_pair_record(
        example_id="ex-1", model_revision="rev-a", rkv_revision="rkv-rev",
        target_capture=target_capture,
        evicted_absolute_position=cd.evicted_selected[0], donor_absolute_position=cd.donor_selected[0],
        trace=trace, branch_step_fn=_non_retaining_step_fn,
    )
    assert build_result.record is not None, build_result.failure_detail

    # `clone_refs[0]`/`[1]` are the two top-level clones `build_swap_pair_record`
    # itself makes (baseline, then swapped) -- every later entry comes from
    # `_non_retaining_step_fn`'s own per-token re-clones inside evaluation.
    assert len(clone_refs) >= 2
    gc.collect()
    baseline_ref = clone_refs[0]
    assert baseline_ref() is None, (
        "the baseline snapshot clone was still reachable after its own evaluation finished -- "
        "it must be released before the swapped clone is created, never held simultaneously"
    )


def test_semantic_swap_parity_passes_and_reports_zero_bytes_changed_on_the_real_happy_path(monkeypatch):
    """The companion positive case: with the real (unpatched)
    `apply_semantic_within_head_swap`, `net_physical_bytes_changed` must be
    an actually-computed 0 (fixed-shape swap invariant), not merely a
    literal that happens to also be 0."""
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert failure is None and plan is not None
    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
    target_capture = pass2_result.target_captures[0]
    cd = target_capture.event_plan.candidate_donor_selection

    build_result = build_swap_pair_record(
        example_id="ex-1",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        target_capture=target_capture,
        evicted_absolute_position=cd.evicted_selected[0],
        donor_absolute_position=cd.donor_selected[0],
        trace=trace,
        branch_step_fn=branch_step_fn,
    )

    assert build_result.record is not None, build_result.failure_detail
    assert build_result.record.parity_check_passed is True
    assert build_result.record.net_physical_bytes_changed == 0


# --------------------------------------------------------------------------
# 3. Pass-2 token mismatch invalidates example
# --------------------------------------------------------------------------


def test_pass2_token_mismatch_invalidates_example(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert plan is not None

    corrupted_tokens = list(trace.full_token_ids)
    corrupted_tokens[50] = (corrupted_tokens[50] + 1) % 64

    result = run_pass2_capture(
        plan, corrupted_tokens, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
    assert result.valid is False
    assert result.invalid_reason == INVALID_TOKEN_MISMATCH
    assert result.target_captures == ()


# --------------------------------------------------------------------------
# 4. Compaction-position mismatch invalidates example
# --------------------------------------------------------------------------


def test_compaction_position_mismatch_invalidates_example(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    natural_prefill_fn, natural_decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), natural_prefill_fn, natural_decode_one_fn, MAX_NEW_TOKENS,
        EOS_TOKEN_ID, _always_correct,
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert plan is not None

    shifted_prefill_fn, shifted_decode_one_fn = make_schedule_shifted_step_fns(
        schedule_offset=3, stop_at_predicted_position=STOP_AT
    )
    result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), shifted_prefill_fn, shifted_decode_one_fn,
        build_snapshot_from_state,
    )
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
    natural_prefill_fn, natural_decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), natural_prefill_fn, natural_decode_one_fn, MAX_NEW_TOKENS,
        EOS_TOKEN_ID, _always_correct,
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert plan is not None

    diverged_prefill_fn, diverged_decode_one_fn = make_query_salt_step_fns(
        query_salt="DIFFERENT", stop_at_predicted_position=STOP_AT
    )
    result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), diverged_prefill_fn, diverged_decode_one_fn,
        build_snapshot_from_state,
    )
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

    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
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
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
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
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
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


# --------------------------------------------------------------------------
# 11. Prefill/decode call-boundary contract (B1B-R2 §6)
# --------------------------------------------------------------------------


def test_prefill_called_exactly_once_with_complete_prompt(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    natural_prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    prefill_calls: list[list[int]] = []

    def counting_prefill_fn(state, prompt_token_ids):
        prefill_calls.append(list(prompt_token_ids))
        return natural_prefill_fn(state, prompt_token_ids)

    run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), counting_prefill_fn, decode_one_fn, MAX_NEW_TOKENS,
        EOS_TOKEN_ID, _always_correct,
    )

    assert len(prefill_calls) == 1
    assert prefill_calls[0] == PROMPT_TOKEN_IDS


def test_every_continuation_token_uses_exactly_one_decode_one_call(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, natural_decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    decode_calls: list[int] = []

    def counting_decode_one_fn(state, token_id):
        decode_calls.append(token_id)
        return natural_decode_one_fn(state, token_id)

    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, counting_decode_one_fn, MAX_NEW_TOKENS,
        EOS_TOKEN_ID, _always_correct,
    )

    assert len(decode_calls) == len(trace.generated_token_ids)
    assert decode_calls == list(trace.generated_token_ids)


def test_pass1_and_pass2_have_identical_call_boundary_traces(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    pass1_prefill_calls: list[list[int]] = []
    pass1_decode_calls: list[int] = []

    def pass1_prefill(state, prompt_token_ids):
        pass1_prefill_calls.append(list(prompt_token_ids))
        return prefill_fn(state, prompt_token_ids)

    def pass1_decode(state, token_id):
        pass1_decode_calls.append(token_id)
        return decode_one_fn(state, token_id)

    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), pass1_prefill, pass1_decode, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)

    pass2_prefill_calls: list[list[int]] = []
    pass2_decode_calls: list[int] = []

    def pass2_prefill(state, prompt_token_ids):
        pass2_prefill_calls.append(list(prompt_token_ids))
        return prefill_fn(state, prompt_token_ids)

    def pass2_decode(state, token_id):
        pass2_decode_calls.append(token_id)
        return decode_one_fn(state, token_id)

    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), pass2_prefill, pass2_decode, build_snapshot_from_state
    )
    assert pass2_result.valid is True

    assert pass1_prefill_calls == pass2_prefill_calls  # identical single-call boundary, identical content
    assert pass1_decode_calls == pass2_decode_calls  # identical per-token call sequence


def test_uncertainty_derived_from_prefill_logits_not_repeated_decode_calls(monkeypatch):
    """Prompt-position uncertainty (`predicted_position` in
    `[1, prompt_length]`) must come from `prefill_fn`'s own
    `per_position_logits`, never from any `decode_one_fn` call -- there is
    no `decode_one_fn` call at all for those positions."""
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, natural_decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    decode_calls: list[int] = []

    def counting_decode_one_fn(state, token_id):
        decode_calls.append(token_id)
        return natural_decode_one_fn(state, token_id)

    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, counting_decode_one_fn, MAX_NEW_TOKENS,
        EOS_TOKEN_ID, _always_correct,
    )
    for predicted_position in range(1, PROMPT_LENGTH + 1):
        assert predicted_position in trace.uncertainty_by_position
    # None of those prompt-position entries required any decode call --
    # the first decode call only happens for predicted_position ==
    # prompt_length + 1 onward.
    assert len(decode_calls) == len(trace.generated_token_ids)


def test_replay_token_mismatch_causes_hard_failure_not_silent_repair(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)

    wrong_length_tokens = list(trace.full_token_ids) + [0]
    result = run_pass2_capture(
        plan, wrong_length_tokens, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
    assert result.valid is False
    assert result.invalid_reason == INVALID_TOKEN_MISMATCH


# --------------------------------------------------------------------------
# 12. Prefill-phase events are never eligible targets (B1B-R2 §5/§6)
# --------------------------------------------------------------------------


def test_prefill_phase_compaction_events_are_never_eligible(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    from kvcot.discovery.pass1 import eligible_event_ids

    eligible = eligible_event_ids(trace)
    event_by_id = {ev.compaction_event_id: ev for ev in trace.compaction_events}
    for event_id in eligible:
        assert event_by_id[event_id].absolute_event_position >= trace.prompt_length


# --------------------------------------------------------------------------
# 13. Baseline/swapped branch evaluation order cannot contaminate results
#     (B1B-R2 §5, requirement 7)
# --------------------------------------------------------------------------


def test_branch_evaluation_order_does_not_change_results(monkeypatch):
    from kvcot.discovery.branch_eval import evaluate_branch

    result, _, _ = _run_example(monkeypatch)
    assert result.valid is True

    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
    target_capture = pass2_result.target_captures[0]
    pristine = target_capture.pristine_snapshot
    baseline_snapshot = pristine.clone()
    swapped_snapshot = pristine.clone()
    swapped_snapshot.key_cache[target_capture.event_plan.layer_index][0, 0, 0, :] = -1.0

    bridge_token_id = 7
    reference_token_ids = list(range(48))

    baseline_first = evaluate_branch(branch_step_fn, baseline_snapshot.clone(), bridge_token_id, reference_token_ids)
    swapped_first_a = evaluate_branch(branch_step_fn, swapped_snapshot.clone(), bridge_token_id, reference_token_ids)
    # Reversed call order:
    swapped_first_b = evaluate_branch(branch_step_fn, swapped_snapshot.clone(), bridge_token_id, reference_token_ids)
    baseline_second = evaluate_branch(branch_step_fn, baseline_snapshot.clone(), bridge_token_id, reference_token_ids)

    assert baseline_first.per_token_nll == baseline_second.per_token_nll
    assert swapped_first_a.per_token_nll == swapped_first_b.per_token_nll


# --------------------------------------------------------------------------
# 14. Incomplete (one-layer) branch state is rejected (B1B-R2 §5)
# --------------------------------------------------------------------------


def test_incomplete_one_layer_snapshot_is_rejected(monkeypatch):
    """A `pristine_snapshot` truncated to one layer (mimicking the
    pre-repair bug: one layer's returned K/V standing in for the whole
    model's state) must be rejected when the target event's layer_index
    requires a different layer -- `apply_within_head_swap`'s own
    out-of-range check catches this, surfaced as a branch-evaluation
    failure rather than silently swapping the wrong layer."""
    import dataclasses as dc

    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT_TOKEN_IDS, HarnessState(), prefill_fn, decode_one_fn, MAX_NEW_TOKENS, EOS_TOKEN_ID,
        _always_correct,
    )
    plan, _ = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    pass2_result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill_fn, decode_one_fn, build_snapshot_from_state
    )
    target_capture = next(tc for tc in pass2_result.target_captures if tc.event_plan.layer_index != 0)

    truncated_snapshot = dc.replace(
        target_capture.pristine_snapshot,
        key_cache=[target_capture.pristine_snapshot.key_cache[0]],
        value_cache=[target_capture.pristine_snapshot.value_cache[0]],
    )
    truncated_capture = dc.replace(target_capture, pristine_snapshot=truncated_snapshot)

    cd = truncated_capture.event_plan.candidate_donor_selection
    evicted_pos, donor_pos = cd.cross_product[0]

    pair_result = build_swap_pair_record(
        example_id="ex-1",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        target_capture=truncated_capture,
        evicted_absolute_position=evicted_pos,
        donor_absolute_position=donor_pos,
        trace=trace,
        branch_step_fn=branch_step_fn,
    )
    assert pair_result.record is None
    assert pair_result.failure_stage == "branch_evaluation_failure"
    assert "swap_failed" in pair_result.failure_detail
