import pytest
from pydantic import ValidationError

from kvcot.discovery.b2a_contract import (
    B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS,
    MANDATORY_GATE_CONDITIONS,
    MAX_PEAK_ALLOCATED_MEMORY_GIB,
    MAX_PROJECTED_PILOT_GPU_HOURS,
    B2AGateEvidence,
    B2AGateResult,
    B2AOneExampleMeasurement,
    build_gate_evidence_from_measurement,
    evaluate_b2a_gate,
)


def _good_measurement(**overrides) -> B2AOneExampleMeasurement:
    defaults = dict(
        fullkv_natural_generation_wall_seconds=10.0,
        rkv_pass1_wall_seconds=10.0,
        rkv_pass2_wall_seconds=10.0,
        targeted_capture_wall_seconds=1.0,
        per_example_total_wall_seconds=30.0,
        real_pair_wall_seconds=[1.0, 1.2, 0.9, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        no_op_pair_wall_seconds=[0.5],
        per_real_pair_seconds=1.2,
        peak_cuda_allocated_bytes=10 * 1024**3,
        peak_cuda_reserved_bytes=12 * 1024**3,
        every_parameter_on_cuda=True,
        observed_retention_ratio=0.2,
        event_count=9,
        projected_complete_pilot_gpu_hours=1.5,
    )
    defaults.update(overrides)
    return B2AOneExampleMeasurement(**defaults)


def _good_evidence_kwargs(**overrides) -> dict:
    defaults = dict(
        token_identical_replay=True,
        prefill_decode_boundary_parity=True,
        compaction_position_equality=True,
        capture_gather_parity=True,
        absolute_position_parity=True,
        no_op_numerical_parity=True,
        semantic_swap_parity=True,
        unique_real_pair_count_exact=True,
        events_with_four_unique_pairs_exact=True,
        no_duplicate_pair_identity=True,
        dataset_revision_match=True,
        dataset_row_identity_match=True,
        manifest_hash_match=True,
        prompt_token_hash_match=True,
        model_revision_match=True,
        tokenizer_revision_match=True,
        generation_config_hash_match=True,
        rkv_config_hash_match=True,
        batch_size_verified=True,
        one_example_only=True,
        meaningful_compression_observed=True,
        sufficient_eligible_events=True,
        selected_event_count_exact=True,
        real_pair_count_exact=True,
        no_op_count_exact=True,
        all_required_pair_evaluations_completed=True,
    )
    defaults.update(overrides)
    return defaults


def _good_evidence(measurement=None, **overrides) -> B2AGateEvidence:
    measurement = measurement or _good_measurement()
    return build_gate_evidence_from_measurement(measurement, **_good_evidence_kwargs(**overrides))


def test_required_measurements_checklist_is_complete_and_documented():
    assert len(B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS) == 14
    assert "peak_cuda_allocated_memory" in B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS
    assert "parameter_placement_assertion" in B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS


def test_measurement_rejects_negative_wall_time():
    with pytest.raises(ValidationError):
        _good_measurement(fullkv_natural_generation_wall_seconds=-1.0)


def test_mandatory_gate_conditions_checklist_has_every_task_brief_field():
    required_names = {
        "token_identical_replay",
        "prefill_decode_boundary_parity",
        "compaction_position_equality",
        "capture_gather_parity",
        "absolute_position_parity",
        "no_op_numerical_parity",
        "dataset_revision_match",
        "dataset_row_identity_match",
        "manifest_hash_match",
        "prompt_token_hash_match",
        "model_revision_match",
        "tokenizer_revision_match",
        "generation_config_hash_match",
        "rkv_config_hash_match",
        "no_offload_verified",
        "batch_size_verified",
        "runtime_within_limit",
        "peak_vram_within_limit",
        "one_example_only",
    }
    assert required_names.issubset(set(MANDATORY_GATE_CONDITIONS))


def test_evidence_cannot_be_constructed_with_a_missing_field():
    # B2AGateEvidence is the pydantic model that actually enforces "every
    # field required" -- omitting one is a ValidationError, never a silent
    # default True.
    kwargs = _good_evidence(measurement=_good_measurement()).model_dump()
    del kwargs["token_identical_replay"]
    with pytest.raises(ValidationError):
        B2AGateEvidence(**kwargs)


def test_gate_passes_when_everything_within_bounds():
    result = evaluate_b2a_gate(_good_evidence())
    assert result.passed is True
    assert result.failed_conditions == ()
    for name in MANDATORY_GATE_CONDITIONS:
        assert getattr(result, name) is True


@pytest.mark.parametrize(
    "field_name",
    [
        "token_identical_replay",
        "prefill_decode_boundary_parity",
        "compaction_position_equality",
        "capture_gather_parity",
        "absolute_position_parity",
        "no_op_numerical_parity",
        "dataset_revision_match",
        "dataset_row_identity_match",
        "manifest_hash_match",
        "prompt_token_hash_match",
        "model_revision_match",
        "tokenizer_revision_match",
        "generation_config_hash_match",
        "rkv_config_hash_match",
        "batch_size_verified",
        "one_example_only",
        "meaningful_compression_observed",
        "sufficient_eligible_events",
        "selected_event_count_exact",
        "real_pair_count_exact",
        "no_op_count_exact",
        "all_required_pair_evaluations_completed",
    ],
)
def test_gate_fails_hard_when_any_single_mandatory_condition_is_false(field_name):
    evidence = _good_evidence(**{field_name: False})
    result = evaluate_b2a_gate(evidence)
    assert result.passed is False
    assert field_name in result.failed_conditions


def test_gate_fails_when_no_offload_verified_derives_false_from_measurement():
    # `no_offload_verified` is derived from the measurement's own
    # `every_parameter_on_cuda`, never an independently-supplied claim.
    evidence = _good_evidence(measurement=_good_measurement(every_parameter_on_cuda=False))
    result = evaluate_b2a_gate(evidence)
    assert result.passed is False
    assert "no_offload_verified" in result.failed_conditions


def test_gate_fails_on_projected_pilot_runtime_over_threshold():
    measurement = _good_measurement(projected_complete_pilot_gpu_hours=MAX_PROJECTED_PILOT_GPU_HOURS + 0.01)
    result = evaluate_b2a_gate(_good_evidence(measurement=measurement))
    assert result.passed is False
    assert "runtime_within_limit" in result.failed_conditions


def test_gate_fails_closed_on_unavailable_runtime_projection_never_fabricated_as_within_limit():
    """B2A-R1 zero-event coordinator repair (2026-07-22): an insufficient
    real-pair count (e.g. zero compaction events) makes the runtime
    projection genuinely unavailable -- `None`, never `0.0`/`inf`/the
    4.00-hour limit itself. `evaluate_b2a_gate` must fail `
    runtime_within_limit` closed from that `None`, never compare it against
    the threshold as if it were a real (and coincidentally passing)
    number."""
    measurement = _good_measurement(per_real_pair_seconds=None, projected_complete_pilot_gpu_hours=None)
    result = evaluate_b2a_gate(_good_evidence(measurement=measurement))
    assert result.passed is False
    assert "runtime_within_limit" in result.failed_conditions
    assert measurement.per_real_pair_seconds is None
    assert measurement.projected_complete_pilot_gpu_hours is None


def test_gate_fails_on_peak_allocated_memory_over_threshold():
    over_bytes = int((MAX_PEAK_ALLOCATED_MEMORY_GIB + 0.5) * 1024**3)
    measurement = _good_measurement(peak_cuda_allocated_bytes=over_bytes)
    result = evaluate_b2a_gate(_good_evidence(measurement=measurement))
    assert result.passed is False
    assert "peak_vram_within_limit" in result.failed_conditions


def test_gate_fails_on_peak_reserved_memory_over_threshold_even_when_allocated_is_low():
    """B1B-R4 §14: the gate uses max(allocated, reserved) -- a reserved-heavy
    run over the limit must fail even with a small allocated value."""
    over_bytes = int((MAX_PEAK_ALLOCATED_MEMORY_GIB + 0.5) * 1024**3)
    measurement = _good_measurement(peak_cuda_allocated_bytes=1 * 1024**3, peak_cuda_reserved_bytes=over_bytes)
    assert measurement.peak_vram_gib > MAX_PEAK_ALLOCATED_MEMORY_GIB
    result = evaluate_b2a_gate(_good_evidence(measurement=measurement))
    assert result.passed is False
    assert "peak_vram_within_limit" in result.failed_conditions


def test_gate_passes_when_both_allocated_and_reserved_are_under_threshold():
    measurement = _good_measurement(
        peak_cuda_allocated_bytes=5 * 1024**3, peak_cuda_reserved_bytes=6 * 1024**3,
    )
    result = evaluate_b2a_gate(_good_evidence(measurement=measurement))
    assert "peak_vram_within_limit" not in result.failed_conditions


def test_gate_passes_at_exactly_the_inclusive_threshold():
    exact_bytes = int(MAX_PEAK_ALLOCATED_MEMORY_GIB * 1024**3)
    measurement = _good_measurement(peak_cuda_allocated_bytes=exact_bytes, peak_cuda_reserved_bytes=exact_bytes)
    result = evaluate_b2a_gate(_good_evidence(measurement=measurement))
    assert "peak_vram_within_limit" not in result.failed_conditions


def test_gate_reports_multiple_simultaneous_failures():
    measurement = _good_measurement(
        projected_complete_pilot_gpu_hours=999.0,
        every_parameter_on_cuda=False,
    )
    evidence = _good_evidence(
        measurement=measurement, meaningful_compression_observed=False, sufficient_eligible_events=False
    )
    result = evaluate_b2a_gate(evidence)
    assert result.passed is False
    assert len(result.failed_conditions) == 4  # runtime, no_offload, compression, eligible_events


def test_gate_result_cannot_be_hand_constructed_as_passing_with_a_false_field():
    good = evaluate_b2a_gate(_good_evidence())
    fields = {name: getattr(good, name) for name in MANDATORY_GATE_CONDITIONS}
    fields["token_identical_replay"] = False  # one condition now false
    with pytest.raises(ValueError):
        B2AGateResult(passed=True, failed_conditions=(), **fields)


def test_gate_result_cannot_be_hand_constructed_as_failing_with_all_true_fields():
    good = evaluate_b2a_gate(_good_evidence())
    fields = {name: getattr(good, name) for name in MANDATORY_GATE_CONDITIONS}
    with pytest.raises(ValueError):
        B2AGateResult(passed=False, failed_conditions=("token_identical_replay",), **fields)


def test_gate_never_imports_torch_or_touches_gpu():
    import kvcot.discovery.b2a_contract as mod

    assert "torch" not in mod.__dict__
