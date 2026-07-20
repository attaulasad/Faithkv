from types import SimpleNamespace

from kvcot.discovery.b2a_evidence import derive_trajectory_evidence, project_complete_pilot_gpu_hours


def _pair(event_id: int, is_noop: bool = False):
    return SimpleNamespace(compaction_event_id=event_id, is_noop_control=is_noop)


def _trace(full_len: int, final_lens: dict[int, int]):
    return SimpleNamespace(full_token_ids=tuple(range(full_len)), cache_length_final_per_layer=final_lens)


def _example_result(valid: bool, pair_records, trace):
    return SimpleNamespace(valid=valid, pair_records=tuple(pair_records), trace=trace)


def test_invalid_example_reports_all_trajectory_fields_false():
    result = _example_result(False, [], None)
    evidence = derive_trajectory_evidence(result)
    assert evidence.token_identical_replay is False
    assert evidence.prefill_decode_boundary_parity is False
    assert evidence.compaction_position_equality is False
    assert evidence.capture_gather_parity is False
    assert evidence.absolute_position_parity is False
    assert evidence.sufficient_eligible_events is False


def test_valid_example_reports_trajectory_fields_true():
    trace = _trace(1000, {0: 800, 1: 800})
    pairs = [_pair(0), _pair(0), _pair(1), _pair(2, is_noop=True)]
    result = _example_result(True, pairs, trace)
    evidence = derive_trajectory_evidence(result)
    assert evidence.token_identical_replay is True
    assert evidence.sufficient_eligible_events is True
    assert evidence.event_count == 3  # distinct event ids: 0, 1, 2


def test_no_op_parity_requires_an_actual_noop_record():
    trace = _trace(1000, {0: 800})
    result_with = _example_result(True, [_pair(0, is_noop=True)], trace)
    result_without = _example_result(True, [_pair(0, is_noop=False)], trace)
    assert derive_trajectory_evidence(result_with).no_op_numerical_parity is True
    assert derive_trajectory_evidence(result_without).no_op_numerical_parity is False


def test_event_count_is_distinct_events_not_pair_count():
    trace = _trace(1000, {0: 800})
    pairs = [_pair(0), _pair(0), _pair(0), _pair(0), _pair(0)]  # 5 pair attempts, all for the SAME event
    result = _example_result(True, pairs, trace)
    evidence = derive_trajectory_evidence(result)
    assert evidence.event_count == 1
    assert len(pairs) == 5


def test_observed_retention_ratio_computed_from_real_cache_lengths():
    trace = _trace(full_len=1000, final_lens={0: 500, 1: 500})
    result = _example_result(True, [_pair(0)], trace)
    evidence = derive_trajectory_evidence(result)
    assert evidence.observed_retention_ratio == 0.5


def test_meaningful_compression_requires_event_and_strict_retention_below_one():
    trace_no_compaction = _trace(full_len=1000, final_lens={0: 1000})
    result = _example_result(True, [], trace_no_compaction)
    evidence = derive_trajectory_evidence(result)
    assert evidence.event_count == 0
    assert evidence.meaningful_compression_observed is False

    trace_compacted = _trace(full_len=1000, final_lens={0: 900})
    result2 = _example_result(True, [_pair(0)], trace_compacted)
    evidence2 = derive_trajectory_evidence(result2)
    assert evidence2.meaningful_compression_observed is True


def test_missing_trace_reports_zero_retention_never_a_crash():
    result = _example_result(False, [], None)
    evidence = derive_trajectory_evidence(result)
    assert evidence.observed_retention_ratio == 0.0


def test_projection_formula_scales_per_example_and_per_branch_components():
    projected = project_complete_pilot_gpu_hours(
        fullkv_natural_generation_wall_seconds=10.0,
        rkv_pass1_wall_seconds=10.0,
        token_identical_pass2_wall_seconds=10.0,
        score_recomputation_wall_seconds=5.0,
        targeted_capture_wall_seconds=5.0,
        cache_clone_restore_wall_seconds=1.0,
        one_fixed_shape_swap_wall_seconds=0.5,
        bridge_plus_48_scored_wall_seconds=2.0,
    )
    per_example = 10.0 + 10.0 + 10.0 + 5.0 + 5.0
    per_branch = 1.0 + 0.5 + 2.0
    expected_seconds = 12 * per_example + 144 * per_branch
    assert projected == expected_seconds / 3600.0
    assert projected > 0.0


def test_projection_is_nonzero_whenever_any_component_is_nonzero():
    projected = project_complete_pilot_gpu_hours(
        fullkv_natural_generation_wall_seconds=0.0,
        rkv_pass1_wall_seconds=0.0,
        token_identical_pass2_wall_seconds=0.0,
        score_recomputation_wall_seconds=0.0,
        targeted_capture_wall_seconds=0.0,
        cache_clone_restore_wall_seconds=0.0,
        one_fixed_shape_swap_wall_seconds=0.0,
        bridge_plus_48_scored_wall_seconds=1.0,
    )
    assert projected > 0.0
