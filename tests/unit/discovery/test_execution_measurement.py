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


def test_pre_branch_guard_is_shape_derived_and_rejects_before_clone():
    snapshot = _snapshot()
    cuda = FakeCuda(allocated=1000, reserved=2000)
    # One complete working clone: discovery restore transfers its tensor
    # ownership into the live cache instead of allocating a second clone.
    required = (2 * 1 * 1 * 4 * 8 * 4) + 64 + 128
    accepted = check_pre_branch_memory(
        phase="pair", cuda=cuda, snapshot=snapshot, selected_vector_bytes=64,
        known_temporary_bytes=128, safety_limit_bytes=2000 + required,
    )
    rejected = check_pre_branch_memory(
        phase="pair", cuda=cuda, snapshot=snapshot, selected_vector_bytes=64,
        known_temporary_bytes=128, safety_limit_bytes=2000 + required - 1,
    )
    assert accepted.required_additional_bytes == required
    assert accepted.accepted is True
    assert rejected.accepted is False


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
