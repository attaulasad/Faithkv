from __future__ import annotations

import pytest
import torch

from kvcot.discovery.execution_measurement import (
    CudaMemoryMeasurer,
    MemoryMeasuredOperationError,
    SynchronizedTimer,
    TimedOperationError,
    build_runtime_projection,
    check_pre_branch_memory,
    snapshot_growth_bytes_per_token,
    snapshot_position_tracking_bytes_per_token,
    snapshot_tensor_bytes,
)
from kvcot.generation.state import ModelStateSnapshot


class FakeCuda:
    def __init__(self, allocated=100, reserved=200, peak_allocated=300, peak_reserved=400):
        self.allocated = allocated
        self.reserved = reserved
        self.peak_allocated = peak_allocated
        self.peak_reserved = peak_reserved
        self.events = []

    def synchronize(self): self.events.append("sync")
    def memory_allocated(self): self.events.append("allocated"); return self.allocated
    def memory_reserved(self): self.events.append("reserved"); return self.reserved
    def reset_peak_memory_stats(self): self.events.append("reset")
    def max_memory_allocated(self): self.events.append("peak_allocated"); return self.peak_allocated
    def max_memory_reserved(self): self.events.append("peak_reserved"); return self.peak_reserved


def test_synchronized_timer_orders_sync_clock_operation_sync_clock():
    events = []
    cuda = FakeCuda()
    cuda.synchronize = lambda: events.append("sync")
    clock_values = iter((10.0, 12.5))
    timer = SynchronizedTimer(cuda, lambda: events.append("clock") or next(clock_values))
    assert timer.measure("phase", lambda: events.append("operation") or 7) == 7
    assert events == ["sync", "clock", "operation", "sync", "clock"]
    assert timer.records[0].duration_seconds == 2.5


def test_synchronized_timer_preserves_partial_evidence_on_exception():
    timer = SynchronizedTimer(FakeCuda(), iter((1.0, 1.5)).__next__)
    with pytest.raises(TimedOperationError) as caught:
        timer.measure("explode", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert caught.value.evidence.completed is False
    assert caught.value.evidence.failure_type == "RuntimeError"


def test_synchronized_timer_parent_span_uses_synchronized_boundaries():
    cuda = FakeCuda()
    timer = SynchronizedTimer(cuda, iter((2.0, 5.5)).__next__)
    started = timer.begin_span()
    evidence = timer.finish_span("complete_worker", started)
    assert evidence.duration_seconds == 3.5
    assert cuda.events == ["sync", "sync"]


def test_memory_measurer_owns_reset_and_preserves_failed_phase():
    cuda = FakeCuda()
    meter = CudaMemoryMeasurer(cuda)
    with pytest.raises(MemoryMeasuredOperationError):
        meter.observe("model_load", lambda: (_ for _ in ()).throw(MemoryError("oom")))
    assert meter.records[0].completed is False
    assert cuda.events[:4] == ["sync", "allocated", "reserved", "reset"]


def _snapshot():
    return ModelStateSnapshot(
        key_cache=[torch.zeros(1, 1, 4, 8)], value_cache=[torch.zeros(1, 1, 4, 8)], query_cache={},
        compression_flags_per_layer=["none"], model_length=4, after_think=None,
    )


def test_snapshot_growth_bytes_per_token_is_shape_derived():
    """1 layer, num_kv_heads=1, head_dim=8, float32 (4 bytes): one more
    decoded position adds `1 * 8 * 4 = 32` bytes to K and 32 to V."""
    snapshot = _snapshot()
    assert snapshot_growth_bytes_per_token(snapshot) == 64


def test_snapshot_position_tracking_bytes_per_token_is_shape_derived():
    """1 layer, num_kv_heads=1: one int64 position per (layer, kv_head)."""
    snapshot = _snapshot()
    assert snapshot_position_tracking_bytes_per_token(snapshot) == 8


def test_pre_branch_guard_is_shape_derived_and_rejects_before_clone():
    """Independent-audit Gate H5 repair: every component below is derived
    from the snapshot's own real tensor shapes or the frozen
    bridge/scored-horizon counts -- hand-computed here to prove the exact
    formula, not just that SOME positive number came out."""
    snapshot = _snapshot()
    cuda = FakeCuda(allocated=1000, reserved=2000)

    clone_bytes = snapshot_tensor_bytes(snapshot)  # 128 (K) + 128 (V) = 256
    assert clone_bytes == 256
    selected_vector_bytes = 64
    vocab_size = 10
    bridge_token_count = 1
    scored_token_count = 2
    total_future_branch_tokens = bridge_token_count + scored_token_count  # 3
    per_token_kv_growth = 64  # from test_snapshot_growth_bytes_per_token_is_shape_derived
    complete_horizon_kv_growth = per_token_kv_growth * total_future_branch_tokens  # 192
    append_realloc = complete_horizon_kv_growth  # 192, conservative doubling
    query_cache_growth = 0
    logits_bytes = vocab_size * 4  # 40
    log_softmax_bytes = vocab_size * 4  # 40
    nll_scalar_bytes = scored_token_count * 4  # 8
    position_tracking_bytes = 8 * total_future_branch_tokens  # 24
    known_temporary = logits_bytes + log_softmax_bytes + nll_scalar_bytes + position_tracking_bytes  # 112
    required = (
        clone_bytes + selected_vector_bytes + complete_horizon_kv_growth + append_realloc
        + query_cache_growth + known_temporary
    )
    assert required == 816

    accepted = check_pre_branch_memory(
        phase="pair", cuda=cuda, snapshot=snapshot, selected_vector_bytes=selected_vector_bytes,
        vocab_size=vocab_size, bridge_token_count=bridge_token_count, scored_token_count=scored_token_count,
        safety_limit_bytes=2000 + required,
    )
    rejected = check_pre_branch_memory(
        phase="pair", cuda=cuda, snapshot=snapshot, selected_vector_bytes=selected_vector_bytes,
        vocab_size=vocab_size, bridge_token_count=bridge_token_count, scored_token_count=scored_token_count,
        safety_limit_bytes=2000 + required - 1,
    )
    assert accepted.required_additional_bytes == required
    assert accepted.accepted is True
    assert rejected.accepted is False

    # Every componentized field is independently visible, not just the total.
    assert accepted.bridge_token_count == 1
    assert accepted.scored_token_count == 2
    assert accepted.total_future_branch_tokens == 3
    assert accepted.per_token_kv_growth_bytes == 64
    assert accepted.complete_horizon_kv_growth_bytes == 192
    assert accepted.append_realloc_temporary_bytes == 192
    assert accepted.query_cache_growth_bytes == 0
    assert accepted.logits_bytes == 40
    assert accepted.log_softmax_bytes == 40
    assert accepted.nll_scalar_bytes == 8
    assert accepted.position_tracking_bytes == 24
    assert accepted.known_temporary_bytes == 112


def test_pre_branch_guard_kv_growth_scales_with_cache_shape():
    """Doubling num_kv_heads (or head_dim, or the horizon) must double the
    corresponding growth component -- proving these are genuinely derived
    from shape, never a fixed value regardless of model size."""
    bigger = ModelStateSnapshot(
        key_cache=[torch.zeros(1, 2, 4, 8)], value_cache=[torch.zeros(1, 2, 4, 8)], query_cache={},
        compression_flags_per_layer=["none"], model_length=4, after_think=None,
    )
    assert snapshot_growth_bytes_per_token(bigger) == 128  # 2x num_kv_heads -> 2x growth
    assert snapshot_position_tracking_bytes_per_token(bigger) == 16  # 2x num_kv_heads -> 2x tracking

    multi_layer = ModelStateSnapshot(
        key_cache=[torch.zeros(1, 1, 4, 8), torch.zeros(1, 1, 4, 8)],
        value_cache=[torch.zeros(1, 1, 4, 8), torch.zeros(1, 1, 4, 8)],
        query_cache={}, compression_flags_per_layer=["none", "none"], model_length=4, after_think=None,
    )
    assert snapshot_growth_bytes_per_token(multi_layer) == 128  # 2 layers -> 2x growth


def test_pre_branch_guard_selected_vector_bytes_do_not_scale_with_cache_length():
    """A longer PRE-EXISTING cache (more decoded positions already in the
    snapshot) must not change `selected_vector_bytes` -- that quantity is
    caller-supplied (the actual persisted selected-vector size), entirely
    independent of how long the snapshot's cache already is."""
    short_snapshot = ModelStateSnapshot(
        key_cache=[torch.zeros(1, 1, 4, 8)], value_cache=[torch.zeros(1, 1, 4, 8)], query_cache={},
        compression_flags_per_layer=["none"], model_length=4, after_think=None,
    )
    long_snapshot = ModelStateSnapshot(
        key_cache=[torch.zeros(1, 1, 400, 8)], value_cache=[torch.zeros(1, 1, 400, 8)], query_cache={},
        compression_flags_per_layer=["none"], model_length=400, after_think=None,
    )
    cuda = FakeCuda()
    short_result = check_pre_branch_memory(
        phase="pair", cuda=cuda, snapshot=short_snapshot, selected_vector_bytes=64, vocab_size=10,
        bridge_token_count=1, scored_token_count=2,
    )
    long_result = check_pre_branch_memory(
        phase="pair", cuda=cuda, snapshot=long_snapshot, selected_vector_bytes=64, vocab_size=10,
        bridge_token_count=1, scored_token_count=2,
    )
    assert short_result.selected_vector_bytes == long_result.selected_vector_bytes == 64
    # But the snapshot CLONE bytes (which DO scale with cache length) differ.
    assert long_result.snapshot_clone_bytes > short_result.snapshot_clone_bytes
    # And the per-token growth/position-tracking rates are identical (same
    # shape per position), independent of how many positions already exist.
    assert short_result.per_token_kv_growth_bytes == long_result.per_token_kv_growth_bytes


def test_pre_branch_guard_fails_closed_on_non_finite_or_negative_components():
    snapshot = _snapshot()
    cuda = FakeCuda()
    with pytest.raises((ValueError, TypeError)):
        check_pre_branch_memory(
            phase="pair", cuda=cuda, snapshot=snapshot, selected_vector_bytes=-1, vocab_size=10,
            bridge_token_count=1, scored_token_count=2,
        )


def test_load_inclusive_projection_uses_max_of_twelve_real_pairs_and_excludes_noop():
    projection = build_runtime_projection(
        fullkv_startup_and_model_load_seconds=10,
        rkv_startup_and_model_load_seconds=20,
        fullkv_natural_generation_seconds=2,
        rkv_pass1_seconds=3,
        rkv_pass2_seconds=4,
        b2a_real_pair_seconds=list(range(1, 13)),
    )
    assert projection.per_example_inference_seconds == 9
    assert projection.conservative_real_pair_seconds == 12
    assert projection.projected_total_seconds == 10 + 20 + 12 * 9 + 144 * 12
    # B2A-R1 zero-event repair (2026-07-22): the exactly-12 "available" path
    # must remain exactly as strict as before this repair.
    assert projection.available is True
    assert projection.unavailable_reason is None
    assert projection.observed_real_pair_duration_count == 12
    assert projection.required_real_pair_duration_count == 12
    assert projection.projected_complete_pilot_gpu_hours == projection.projected_total_seconds / 3600.0


# ---------------------------------------------------------------------------
# B2A-R1 zero-event coordinator repair (2026-07-22): a frozen one-example row
# that produces fewer than 12 real-pair durations (e.g. zero compaction
# events, as B2A-R1 actually observed: prompt=105 tokens, generated=449
# tokens, processed length ~554, well under R-KV budget=1024) is a genuine,
# non-exceptional, ineligible-calibration outcome. `build_runtime_projection`
# must report it as `available=False` with every real-pair-derived field
# `None` -- never raise, and never fabricate `0.0`/`inf`/the 4.00-hour limit
# itself as a stand-in projection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("observed_count", [0, 1, 11])
def test_runtime_projection_is_unavailable_not_an_exception_for_insufficient_real_pair_durations(observed_count):
    projection = build_runtime_projection(
        fullkv_startup_and_model_load_seconds=10,
        rkv_startup_and_model_load_seconds=20,
        fullkv_natural_generation_seconds=2,
        rkv_pass1_seconds=3,
        rkv_pass2_seconds=4,
        b2a_real_pair_seconds=[1.0 + i for i in range(observed_count)],
    )
    assert projection.available is False
    assert projection.unavailable_reason == "insufficient_real_pair_durations"
    assert projection.observed_real_pair_duration_count == observed_count
    assert projection.required_real_pair_duration_count == 12
    # Never fabricated: no 0.0, no inf, no guessed placeholder -- an
    # explicit, unmistakable absence.
    assert projection.conservative_real_pair_seconds is None
    assert projection.projected_total_seconds is None
    assert projection.projected_complete_pilot_gpu_hours is None
    # The one-time and per-example components never depended on the
    # real-pair list, and must still be reported (the coordinator's
    # `per_example_total_wall_seconds` measurement stays meaningful even
    # when the pilot-runtime projection is unavailable).
    assert projection.per_example_inference_seconds == 9
    assert projection.fullkv_startup_and_model_load_seconds == 10
    assert projection.rkv_startup_and_model_load_seconds == 20


def test_runtime_projection_zero_real_pairs_matches_the_actual_observed_b2a_r1_failure():
    """The exact shape of the actual B2A-R1 consumed-attempt failure
    (results/decisions/b2a_attempt_20260722T072823470986Z_..., preserved in
    docs/evidence/B2A_R1_ATTEMPT_INDEX_2026-07-22.json): zero compaction
    events, hence zero real-pair durations -- this must no longer raise
    `ValueError: runtime projection requires exactly 12 B2A real-pair
    durations`."""
    projection = build_runtime_projection(
        fullkv_startup_and_model_load_seconds=5.0,
        rkv_startup_and_model_load_seconds=6.0,
        fullkv_natural_generation_seconds=1.0,
        rkv_pass1_seconds=1.0,
        rkv_pass2_seconds=1.0,
        b2a_real_pair_seconds=[],
    )
    assert projection.available is False
    assert projection.observed_real_pair_duration_count == 0


def test_runtime_projection_still_rejects_non_finite_or_non_positive_real_pair_durations():
    """A genuinely malformed measurement -- a present but non-finite or
    non-positive real-pair duration -- remains a hard defect, never
    silently downgraded to "unavailable"."""
    with pytest.raises(ValueError, match="finite and positive"):
        build_runtime_projection(
            fullkv_startup_and_model_load_seconds=10,
            rkv_startup_and_model_load_seconds=20,
            fullkv_natural_generation_seconds=2,
            rkv_pass1_seconds=3,
            rkv_pass2_seconds=4,
            b2a_real_pair_seconds=[1.0] * 11 + [-5.0],
        )
    with pytest.raises(ValueError, match="finite and positive"):
        build_runtime_projection(
            fullkv_startup_and_model_load_seconds=10,
            rkv_startup_and_model_load_seconds=20,
            fullkv_natural_generation_seconds=2,
            rkv_pass1_seconds=3,
            rkv_pass2_seconds=4,
            b2a_real_pair_seconds=[1.0] * 11 + [float("nan")],
        )


def test_runtime_projection_still_rejects_non_finite_or_negative_one_time_components():
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_runtime_projection(
            fullkv_startup_and_model_load_seconds=-1,
            rkv_startup_and_model_load_seconds=20,
            fullkv_natural_generation_seconds=2,
            rkv_pass1_seconds=3,
            rkv_pass2_seconds=4,
            b2a_real_pair_seconds=[],
        )
