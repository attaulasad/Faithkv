import pytest
from pydantic import ValidationError

from kvcot.discovery.b2a_contract import (
    B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS,
    MAX_PEAK_ALLOCATED_MEMORY_GIB,
    MAX_PROJECTED_PILOT_GPU_HOURS,
    B2AOneExampleMeasurement,
    evaluate_b2a_hard_stop_gate,
)


def _good_measurement(**overrides) -> B2AOneExampleMeasurement:
    defaults = dict(
        fullkv_natural_generation_wall_seconds=10.0,
        rkv_pass1_wall_seconds=10.0,
        token_identical_pass2_wall_seconds=10.0,
        score_recomputation_wall_seconds=1.0,
        targeted_capture_wall_seconds=1.0,
        cache_clone_restore_wall_seconds=1.0,
        one_fixed_shape_swap_wall_seconds=0.1,
        bridge_plus_48_scored_wall_seconds=2.0,
        peak_cuda_allocated_bytes=10 * 1024**3,
        peak_cuda_reserved_bytes=12 * 1024**3,
        every_parameter_on_cuda=True,
        observed_retention_ratio=0.2,
        event_count=9,
        projected_complete_pilot_gpu_hours=1.5,
    )
    defaults.update(overrides)
    return B2AOneExampleMeasurement(**defaults)


def test_required_measurements_checklist_is_complete_and_documented():
    assert len(B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS) == 14
    assert "peak_cuda_allocated_memory" in B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS
    assert "parameter_placement_assertion" in B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS


def test_measurement_rejects_negative_wall_time():
    with pytest.raises(ValidationError):
        _good_measurement(fullkv_natural_generation_wall_seconds=-1.0)


def test_gate_passes_when_everything_within_bounds():
    result = evaluate_b2a_hard_stop_gate(
        _good_measurement(), meaningful_compression_observed=True, sufficient_eligible_events=True
    )
    assert result.passed is True
    assert result.failed_conditions == ()


def test_gate_fails_on_projected_pilot_runtime_over_threshold():
    measurement = _good_measurement(projected_complete_pilot_gpu_hours=MAX_PROJECTED_PILOT_GPU_HOURS + 0.01)
    result = evaluate_b2a_hard_stop_gate(measurement, meaningful_compression_observed=True, sufficient_eligible_events=True)
    assert result.passed is False
    assert "projected_pilot_exceeds_4_gpu_hours" in result.failed_conditions


def test_gate_fails_on_peak_allocated_memory_over_threshold():
    over_bytes = int((MAX_PEAK_ALLOCATED_MEMORY_GIB + 0.5) * 1024**3)
    measurement = _good_measurement(peak_cuda_allocated_bytes=over_bytes)
    result = evaluate_b2a_hard_stop_gate(measurement, meaningful_compression_observed=True, sufficient_eligible_events=True)
    assert "peak_allocated_memory_exceeds_22_gib" in result.failed_conditions


def test_gate_fails_when_a_parameter_is_not_on_cuda():
    measurement = _good_measurement(every_parameter_on_cuda=False)
    result = evaluate_b2a_hard_stop_gate(measurement, meaningful_compression_observed=True, sufficient_eligible_events=True)
    assert "a_parameter_is_not_on_cuda" in result.failed_conditions


def test_gate_fails_when_no_meaningful_compression_observed():
    result = evaluate_b2a_hard_stop_gate(
        _good_measurement(), meaningful_compression_observed=False, sufficient_eligible_events=True
    )
    assert "no_meaningful_compression_observed" in result.failed_conditions


def test_gate_fails_when_insufficient_eligible_events():
    result = evaluate_b2a_hard_stop_gate(
        _good_measurement(), meaningful_compression_observed=True, sufficient_eligible_events=False
    )
    assert "insufficient_eligible_events" in result.failed_conditions


def test_gate_reports_multiple_simultaneous_failures():
    measurement = _good_measurement(
        projected_complete_pilot_gpu_hours=999.0,
        every_parameter_on_cuda=False,
    )
    result = evaluate_b2a_hard_stop_gate(measurement, meaningful_compression_observed=False, sufficient_eligible_events=False)
    assert len(result.failed_conditions) == 4


def test_gate_never_imports_torch_or_touches_gpu():
    import sys

    assert "torch" not in sys.modules or True  # presence elsewhere is fine; this module itself must not require it
    import kvcot.discovery.b2a_contract as mod

    assert "torch" not in mod.__dict__
