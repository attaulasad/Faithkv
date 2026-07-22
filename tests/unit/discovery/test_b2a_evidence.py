"""B1B-R4 §8/§12/§22 tests for `kvcot.discovery.b2a_evidence` -- the
repaired version that derives each trajectory/parity condition from its
own independent raw observation, never all five from one umbrella
`example_result.valid` boolean.
"""
from __future__ import annotations

from types import SimpleNamespace

from kvcot.discovery.b2a_evidence import (
    derive_meaningful_compression_observed,
    derive_no_op_numerical_parity,
    derive_observed_retention_ratio,
    derive_pair_completion_evidence,
    derive_pair_identity_evidence,
    derive_semantic_swap_check_evidence,
    derive_trajectory_parity_evidence,
    per_real_pair_projection_seconds,
    project_complete_pilot_gpu_hours,
)


def _pair(event_id: int, is_noop: bool = False):
    return SimpleNamespace(compaction_event_id=event_id, is_noop_control=is_noop)


def _identity_pair(event_id, layer, head, evicted, donor, is_noop=False):
    return SimpleNamespace(
        compaction_event_id=event_id, layer_index=layer, kv_head_index=head,
        evicted_absolute_token_position=evicted, retained_absolute_token_position=donor,
        is_noop_control=is_noop,
    )


def _trace(full_len: int, final_lens: dict[int, int], compaction_events=(), prompt_length=10):
    return SimpleNamespace(
        full_token_ids=tuple(range(full_len)), cache_length_final_per_layer=final_lens,
        compaction_events=tuple(compaction_events), prompt_length=prompt_length,
    )


def _example_result(
    pair_records, *, attempted_real=0, completed_real=0, attempted_no_op=0, completed_no_op=0, selected_event_ids=(),
    pair_failure_details=(),
):
    return SimpleNamespace(
        pair_records=tuple(pair_records), attempted_real_pair_count=attempted_real,
        completed_real_pair_count=completed_real, attempted_no_op_pair_count=attempted_no_op,
        completed_no_op_pair_count=completed_no_op, selected_event_ids=tuple(selected_event_ids),
        pair_failure_details=tuple(pair_failure_details),
    )


# --------------------------------------------------------------------------
# derive_trajectory_parity_evidence -- five INDEPENDENT conditions
# --------------------------------------------------------------------------


def test_all_five_conditions_true_only_when_every_independent_check_passes():
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=True, pass2_invalid_reason=None, call_boundary_all_match=True,
        target_capture_gather_parities=(True, True, True), target_capture_absolute_parities=(True, True, True),
    )
    assert evidence.token_identical_replay is True
    assert evidence.prefill_decode_boundary_parity is True
    assert evidence.compaction_position_equality is True
    assert evidence.capture_gather_parity is True
    assert evidence.absolute_position_parity is True


def test_token_mismatch_fails_only_token_identical_never_the_others_as_true():
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=False, pass2_invalid_reason="pass2_token_mismatch", call_boundary_all_match=True,
        target_capture_gather_parities=(), target_capture_absolute_parities=(),
    )
    assert evidence.token_identical_replay is False
    assert evidence.prefill_decode_boundary_parity is False  # ran_to_completion is False -- never vacuously True
    assert evidence.capture_gather_parity is False


def test_call_boundary_mismatch_is_independent_of_token_identity():
    """A run where tokens matched perfectly but the call-boundary trace
    diverged (e.g. Pass 2 issued an extra decode call) must fail
    `prefill_decode_boundary_parity` specifically -- proving the two
    conditions are not both silently derived from the same source."""
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=True, pass2_invalid_reason=None, call_boundary_all_match=False,
        target_capture_gather_parities=(True,), target_capture_absolute_parities=(True,),
    )
    assert evidence.token_identical_replay is True
    assert evidence.prefill_decode_boundary_parity is False
    assert evidence.capture_gather_parity is True  # unaffected -- independent condition


def test_capture_gather_parity_requires_every_target_true_not_just_one():
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=True, pass2_invalid_reason=None, call_boundary_all_match=True,
        target_capture_gather_parities=(True, True, False), target_capture_absolute_parities=(True, True, True),
    )
    assert evidence.capture_gather_parity is False
    assert evidence.absolute_position_parity is True


def test_none_gather_parity_counts_as_failure_never_vacuous_true():
    """`None` means "not evaluable" (kvcot.discovery.capture's own
    documented tri-state) -- an aggregate condition must never treat a
    missing observation as a pass."""
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=True, pass2_invalid_reason=None, call_boundary_all_match=True,
        target_capture_gather_parities=(True, None, True), target_capture_absolute_parities=(True, True, True),
    )
    assert evidence.capture_gather_parity is False


def test_empty_target_list_is_never_a_vacuous_pass():
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=True, pass2_invalid_reason=None, call_boundary_all_match=True,
        target_capture_gather_parities=(), target_capture_absolute_parities=(),
    )
    assert evidence.capture_gather_parity is False
    assert evidence.absolute_position_parity is False


def test_pass2_never_attempted_reports_token_identical_false_not_vacuous_true():
    """An example that fails BEFORE Pass 2 ever runs (natural run invalid,
    wrong answer, cap hit, too few eligible events) must never claim
    `token_identical_replay=True` -- nothing was ever replayed."""
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=False, pass2_invalid_reason=None, call_boundary_all_match=False,
        target_capture_gather_parities=(), target_capture_absolute_parities=(),
    )
    assert evidence.token_identical_replay is False


def test_compaction_position_equality_fails_on_a_pass2_failure_unrelated_to_tokens():
    evidence = derive_trajectory_parity_evidence(
        pass2_result_valid=False, pass2_invalid_reason="pass2_compaction_event_position_mismatch",
        call_boundary_all_match=True, target_capture_gather_parities=(), target_capture_absolute_parities=(),
    )
    assert evidence.token_identical_replay is True  # tokens matched; the OTHER check failed
    assert evidence.compaction_position_equality is False


# --------------------------------------------------------------------------
# derive_pair_completion_evidence
# --------------------------------------------------------------------------


def test_pair_completion_counts_real_and_no_op_independently():
    events = [
        SimpleNamespace(absolute_event_position=p, compaction_event_id=i)
        for i, p in enumerate((20, 500, 900))
    ]
    trace = _trace(1000, {0: 800}, compaction_events=events)
    pairs = [_pair(0), _pair(0), _pair(0), _pair(0), _pair(0, is_noop=True)]
    result = _example_result(
        pairs, attempted_real=4, completed_real=4, attempted_no_op=1, completed_no_op=1, selected_event_ids=(0,)
    )
    evidence = derive_pair_completion_evidence(trace=trace, example_result=result)
    assert evidence.attempted_real_pair_count == 4
    assert evidence.completed_real_pair_count == 4
    assert evidence.failed_real_pair_count == 0
    assert evidence.attempted_no_op_pair_count == 1
    assert evidence.completed_no_op_pair_count == 1
    assert evidence.selected_compaction_events == 1
    assert evidence.events_with_at_least_one_completed_real_pair == 1
    assert evidence.events_with_all_four_real_pairs_completed == 1


def test_failed_real_pair_count_reflects_attempted_minus_completed():
    trace = _trace(1000, {0: 800})
    pairs = [_pair(0), _pair(0)]  # only 2 of 4 attempted real pairs succeeded
    result = _example_result(pairs, attempted_real=4, completed_real=2, selected_event_ids=(0,))
    evidence = derive_pair_completion_evidence(trace=trace, example_result=result)
    assert evidence.failed_real_pair_count == 2
    assert evidence.events_with_all_four_real_pairs_completed == 0  # only 2/4 for this event


def test_event_count_is_distinct_events_not_pair_count():
    trace = _trace(1000, {0: 800})
    pairs = [_pair(0), _pair(0), _pair(0), _pair(0), _pair(0)]  # 5 pair attempts, all for the SAME event
    result = _example_result(pairs, attempted_real=5, completed_real=5, selected_event_ids=(0,))
    evidence = derive_pair_completion_evidence(trace=trace, example_result=result)
    assert evidence.selected_compaction_events == 1


def test_selected_event_count_comes_from_the_frozen_plan_not_surviving_pair_records():
    """B1B-R4.1 §14 regression: an event selected by Pass 1's frozen plan
    whose EVERY pair failed (zero surviving records) must still count as
    SELECTED -- the prior derivation (`len({distinct event ids in
    pair_records})`) would have silently reported 2 here instead of the
    true planned count of 3, conflating "selected" with "has a surviving
    pair"."""
    trace = _trace(1000, {0: 800})
    # Only events 0 and 1 have any surviving pair record; event 2 was
    # SELECTED by the plan but every one of its pairs failed attrition.
    pairs = [_pair(0), _pair(0), _pair(0), _pair(0), _pair(1)]
    result = _example_result(pairs, attempted_real=9, completed_real=5, selected_event_ids=(0, 1, 2))
    evidence = derive_pair_completion_evidence(trace=trace, example_result=result)
    assert evidence.selected_compaction_events == 3  # from the plan, not from pair_records
    assert evidence.events_with_at_least_one_completed_real_pair == 2  # the weaker, completion-based count
    assert evidence.events_with_all_four_real_pairs_completed == 1  # only event 0 reached 4


def test_missing_trace_reports_zero_observed_and_eligible():
    result = _example_result([], attempted_real=0, completed_real=0)
    evidence = derive_pair_completion_evidence(trace=None, example_result=result)
    assert evidence.observed_total_compaction_events == 0
    assert evidence.eligible_compaction_events == 0


# --------------------------------------------------------------------------
# B1 execution-boundary closure §12: derive_semantic_swap_check_evidence --
# POSITIVE checks_attempted/checks_passed, never absence-of-failure
# --------------------------------------------------------------------------


def test_semantic_swap_check_evidence_all_passed():
    result = SimpleNamespace(semantic_swap_checks_attempted=12, semantic_swap_checks_passed=12)
    evidence = derive_semantic_swap_check_evidence(result)
    assert evidence.checks_required == 12
    assert evidence.checks_attempted == 12
    assert evidence.checks_passed == 12
    assert evidence.checks_failed == 0


def test_semantic_swap_check_evidence_one_failure():
    result = SimpleNamespace(semantic_swap_checks_attempted=12, semantic_swap_checks_passed=11)
    evidence = derive_semantic_swap_check_evidence(result)
    assert evidence.checks_failed == 1


def test_semantic_swap_check_evidence_none_attempted_is_not_vacuously_passing():
    """Zero attempted must never present as zero failed with an implied
    pass -- `checks_attempted` staying below `checks_required` is itself
    what the gate condition checks for, precisely to catch this case."""
    result = SimpleNamespace(semantic_swap_checks_attempted=0, semantic_swap_checks_passed=0)
    evidence = derive_semantic_swap_check_evidence(result)
    assert evidence.checks_attempted == 0
    assert evidence.checks_failed == 0
    assert evidence.checks_attempted != evidence.checks_required


# --------------------------------------------------------------------------
# retention / no-op / meaningful compression
# --------------------------------------------------------------------------


def test_no_op_parity_requires_an_actual_noop_record():
    result_with = _example_result([_pair(0, is_noop=True)])
    result_without = _example_result([_pair(0, is_noop=False)])
    assert derive_no_op_numerical_parity(result_with) is True
    assert derive_no_op_numerical_parity(result_without) is False


def test_observed_retention_ratio_computed_from_real_cache_lengths():
    trace = _trace(full_len=1000, final_lens={0: 500, 1: 500})
    result = SimpleNamespace(trace=trace, full_token_ids=trace.full_token_ids)
    assert derive_observed_retention_ratio(result) == 0.5


def test_missing_trace_reports_zero_retention_never_a_crash():
    result = SimpleNamespace(trace=None)
    assert derive_observed_retention_ratio(result) == 0.0


def test_meaningful_compression_requires_event_and_strict_retention_below_one():
    assert derive_meaningful_compression_observed(selected_event_count=0, observed_retention_ratio=1.0) is False
    assert derive_meaningful_compression_observed(selected_event_count=1, observed_retention_ratio=1.0) is False
    assert derive_meaningful_compression_observed(selected_event_count=1, observed_retention_ratio=0.9) is True


# --------------------------------------------------------------------------
# timing / projection (B1B-R4 §12 -- never aggregate-time x 144)
# --------------------------------------------------------------------------


def test_per_real_pair_projection_uses_the_maximum_not_the_mean():
    assert per_real_pair_projection_seconds((1.0, 5.0, 2.0)) == 5.0


def test_per_real_pair_projection_is_zero_with_no_completed_pairs():
    assert per_real_pair_projection_seconds(()) == 0.0


def test_projection_formula_scales_per_example_and_per_real_pair():
    projected = project_complete_pilot_gpu_hours(per_example_total_seconds=25.0, per_real_pair_seconds=3.0)
    expected_seconds = 12 * 25.0 + 144 * 3.0
    assert projected == expected_seconds / 3600.0
    assert projected > 0.0


def test_projection_never_multiplies_an_aggregate_bucket_by_144():
    """Regression for the specific B1B-R2/B1B-R3 defect: if the caller
    passes the MAXIMUM of 12 individually-measured pairs (never their sum),
    the projection must be far smaller than naively summing all 12 pair
    times and then multiplying by 144."""
    twelve_pair_times = [1.0] * 12
    aggregate_sum = sum(twelve_pair_times)  # 12.0 -- the OLD (buggy) bucket
    correct_max = per_real_pair_projection_seconds(tuple(twelve_pair_times))  # 1.0

    buggy_projection = project_complete_pilot_gpu_hours(per_example_total_seconds=0.0, per_real_pair_seconds=aggregate_sum)
    correct_projection = project_complete_pilot_gpu_hours(per_example_total_seconds=0.0, per_real_pair_seconds=correct_max)

    assert correct_projection < buggy_projection
    assert correct_projection == (144 * 1.0) / 3600.0


# --------------------------------------------------------------------------
# B1 execution-boundary closure §13: derive_pair_identity_evidence --
# exact, DUPLICATE-DETECTING identity accounting, never a bare count
# --------------------------------------------------------------------------


def test_pair_identity_evidence_twelve_distinct_real_pairs_four_per_event():
    records = []
    for event_id in (0, 1, 2):
        for evicted, donor in [(10, 20), (10, 21), (11, 20), (11, 21)]:
            records.append(_identity_pair(event_id, layer=1, head=0, evicted=evicted, donor=donor))
    result = SimpleNamespace(pair_records=tuple(records))

    evidence = derive_pair_identity_evidence(result)
    assert evidence.unique_completed_real_pair_count == 12
    assert evidence.events_with_exactly_four_unique_real_pairs == 3
    assert evidence.has_duplicate_real_pair_identity is False


def test_pair_identity_evidence_detects_a_duplicate_pair_recorded_twice():
    """The exact scenario `count >= 4` could not distinguish: an event with
    FOUR pair records, but only THREE of them are actually distinct
    (event, layer, head, candidate, donor) identities -- one is a
    duplicate. A bare count would report `4 >= 4` (pass); the identity-based
    check must catch it."""
    records = [
        _identity_pair(0, layer=1, head=0, evicted=10, donor=20),
        _identity_pair(0, layer=1, head=0, evicted=10, donor=21),
        _identity_pair(0, layer=1, head=0, evicted=11, donor=20),
        _identity_pair(0, layer=1, head=0, evicted=10, donor=20),  # duplicate of the first
    ]
    result = SimpleNamespace(pair_records=tuple(records))

    evidence = derive_pair_identity_evidence(result)
    assert evidence.has_duplicate_real_pair_identity is True
    assert evidence.unique_completed_real_pair_count == 3  # only 3 DISTINCT identities among the 4 records
    assert evidence.events_with_exactly_four_unique_real_pairs == 0  # this event has only 3 unique, not exactly 4


def test_pair_identity_evidence_event_with_five_pairs_is_not_exactly_four():
    records = [
        _identity_pair(0, layer=1, head=0, evicted=e, donor=d)
        for e, d in [(10, 20), (10, 21), (11, 20), (11, 21), (12, 22)]
    ]
    result = SimpleNamespace(pair_records=tuple(records))

    evidence = derive_pair_identity_evidence(result)
    assert evidence.events_with_exactly_four_unique_real_pairs == 0


def test_pair_identity_evidence_no_op_duplicate_detected_separately_from_real():
    records = [
        _identity_pair(0, layer=1, head=0, evicted=20, donor=20, is_noop=True),
        _identity_pair(0, layer=1, head=0, evicted=20, donor=20, is_noop=True),  # duplicate no-op
    ]
    result = SimpleNamespace(pair_records=tuple(records))

    evidence = derive_pair_identity_evidence(result)
    assert evidence.completed_no_op_pair_count == 2
    assert evidence.has_duplicate_no_op_pair_identity is True
    assert evidence.has_duplicate_real_pair_identity is False  # independent from the no-op check
