"""Step 3R4-Repair-2: FullKV timing/memory contract separation tests.

Step 3R4's version of this module asserted that qualification's timing
vocabulary was exactly the historical `FULLKV_REQUIRED_TIMING_PHASES`
(10 phases) and that `before_model_load`/`post_load_baseline` could never
legitimately appear as TIMING phases. An independent re-audit found that
claim false against the REAL `run_fullkv_worker` body: its `measured()`
helper times AND memory-samples `before_model_load`/`post_load_baseline` in
one call, so both genuinely appear in `timer.export()`, and
`answer_verification` is genuinely nested BEFORE (never after)
`fullkv_complete_natural_generation`. This module now tests the corrected,
real-worker-shaped vocabulary (`_FULLKV_QUALIFICATION_PHASE_ORDER`,
`_FULLKV_QUALIFICATION_MEMORY_PHASES`) -- see
`docs/B2A_R3_STEP3R4_REPAIR2_2026-07-23.md` §2. No torch, no CUDA -- pure
dict fixtures.
"""
from __future__ import annotations

import copy
import math

import pytest

from kvcot.discovery.final_contract import (
    FULLKV_MEMORY_EXACT_MULTIPLICITY,
    FULLKV_REQUIRED_MEMORY_PHASES,
    FULLKV_REQUIRED_TIMING_PHASES,
    _FULLKV_QUALIFICATION_MEMORY_PHASES,
    _FULLKV_QUALIFICATION_PHASE_ORDER,
    fullkv_qualification_memory_complete,
    fullkv_qualification_timing_complete,
    peak_cuda_bytes_from_qualification_memory_evidence,
)
from tests.unit.discovery.test_b2a_r3_independent_audit_repair import _memory, _timing


def test_canonical_historical_fullkv_timing_passes():
    assert fullkv_qualification_timing_complete(_timing(), fullkv_wall_seconds=3.0) is True


def test_canonical_historical_fullkv_memory_passes():
    assert fullkv_qualification_memory_complete(_memory()) is True


@pytest.mark.parametrize("memory_phase", ["before_model_load", "post_load_baseline"])
def test_duplicate_before_model_load_or_post_load_baseline_fails(memory_phase):
    """`before_model_load`/`post_load_baseline` are now legitimate,
    required-once TIMING phases (the real worker genuinely times them) --
    a SECOND occurrence must still be rejected as a duplicate singleton,
    exactly like any other required-once phase."""
    timing = _timing()
    timing.append(
        {
            "phase": memory_phase,
            "started_at": 0.0,
            "ended_at": 1.0,
            "duration_seconds": 1.0,
            "synchronize_before_start": True,
            "synchronize_before_end": True,
            "completed": True,
            "failure_type": None,
            "failure_message": None,
        }
    )
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


def test_unknown_phase_name_in_timing_list_fails():
    timing = _timing()
    timing.append(
        {
            "phase": "not_a_real_phase",
            "started_at": 0.0,
            "ended_at": 1.0,
            "duration_seconds": 1.0,
            "synchronize_before_start": True,
            "synchronize_before_end": True,
            "completed": True,
            "failure_type": None,
            "failure_message": None,
        }
    )
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


@pytest.mark.parametrize("phase", ["before_model_load", "post_load_baseline"])
def test_missing_before_model_load_or_post_load_baseline_timing_fails(phase):
    timing = [row for row in _timing() if row["phase"] != phase]
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


@pytest.mark.parametrize("phase", ["tokenizer_load", "post_load_validation"])
def test_missing_extra_real_worker_memory_phase_fails(phase):
    memory = [row for row in _memory() if row["phase"] != phase]
    assert fullkv_qualification_memory_complete(memory) is False


def test_unknown_phase_name_in_memory_list_fails():
    memory = _memory()
    memory.append(
        {
            "phase": "not_a_real_phase",
            "allocated_before": 0, "reserved_before": 0, "peak_allocated": 1, "peak_reserved": 1,
            "allocated_after": 1, "reserved_after": 1,
            "reset_point": "after_model_and_tokenizer_load_before_measured_inference",
            "synchronized_before": True, "synchronized_after": True, "completed": True,
        }
    )
    assert fullkv_qualification_memory_complete(memory) is False


def test_qualification_timing_order_matches_frozen_vocabulary():
    assert list(_FULLKV_QUALIFICATION_PHASE_ORDER) == [
        "before_model_load",
        "fullkv_worker_startup",
        "snapshot_tokenizer_resolution",
        "tokenizer_load",
        "model_load",
        "post_load_validation",
        "post_load_baseline",
        "fullkv_prefill",
        "fullkv_decode",
        "answer_verification",
        "fullkv_complete_natural_generation",
        "fullkv_complete_worker",
    ]


def test_qualification_memory_phases_match_frozen_vocabulary():
    assert set(_FULLKV_QUALIFICATION_MEMORY_PHASES) == {
        "before_model_load",
        "tokenizer_load",
        "model_load",
        "post_load_validation",
        "post_load_baseline",
        "fullkv_complete_natural_generation",
        "fullkv_complete_worker",
    }


def test_timing_phase_inside_memory_evidence_fails():
    memory = _memory()
    memory.append(
        {
            "phase": "fullkv_prefill",
            "allocated_before": 0, "reserved_before": 0, "peak_allocated": 1, "peak_reserved": 1,
            "allocated_after": 1, "reserved_after": 1,
            "reset_point": "after_model_and_tokenizer_load_before_measured_inference",
            "synchronized_before": True, "synchronized_after": True, "completed": True,
        }
    )
    assert fullkv_qualification_memory_complete(memory) is False


@pytest.mark.parametrize("phase", FULLKV_REQUIRED_TIMING_PHASES)
def test_missing_timing_phase_fails(phase):
    timing = [row for row in _timing() if row["phase"] != phase]
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


@pytest.mark.parametrize("phase", [p for p in FULLKV_REQUIRED_MEMORY_PHASES])
def test_missing_memory_phase_fails(phase):
    memory = [row for row in _memory() if row["phase"] != phase]
    assert fullkv_qualification_memory_complete(memory) is False


def test_duplicate_singleton_timing_phase_fails():
    timing = _timing()
    timing.insert(1, copy.deepcopy(next(r for r in timing if r["phase"] == "model_load")))
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


def test_duplicate_singleton_memory_phase_fails():
    memory = _memory()
    memory.append(copy.deepcopy(next(r for r in memory if r["phase"] == "model_load")))
    assert fullkv_qualification_memory_complete(memory) is False


def test_wrong_decode_multiplicity_of_zero_fails():
    timing = [row for row in _timing() if row["phase"] != "fullkv_decode"]
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


def test_multiple_fullkv_decode_records_pass():
    timing = _timing()
    decode = next(r for r in timing if r["phase"] == "fullkv_decode")
    index = timing.index(decode)
    timing.insert(index, copy.deepcopy(decode))
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is True


def test_wrong_order_fails():
    timing = _timing()
    idx_a = next(i for i, r in enumerate(timing) if r["phase"] == "fullkv_complete_natural_generation")
    idx_b = next(i for i, r in enumerate(timing) if r["phase"] == "answer_verification")
    timing[idx_a], timing[idx_b] = timing[idx_b], timing[idx_a]
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


def test_negative_duration_fails():
    timing = _timing()
    timing[0]["duration_seconds"] = -1.0
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


def test_nan_duration_fails():
    timing = _timing()
    timing[0]["duration_seconds"] = float("nan")
    assert fullkv_qualification_timing_complete(timing, fullkv_wall_seconds=3.0) is False


def test_negative_memory_reading_fails():
    memory = _memory()
    memory[0]["peak_allocated"] = -1
    assert fullkv_qualification_memory_complete(memory) is False


def test_boolean_memory_reading_fails():
    memory = _memory()
    memory[0]["peak_allocated"] = True
    assert fullkv_qualification_memory_complete(memory) is False


def test_wrong_wall_time_binding_fails():
    assert fullkv_qualification_timing_complete(_timing(), fullkv_wall_seconds=4.0) is False


def test_unsynchronized_memory_record_fails():
    memory = _memory()
    memory[0]["synchronized_after"] = False
    assert fullkv_qualification_memory_complete(memory) is False


def test_incomplete_memory_record_fails():
    memory = _memory()
    memory[0]["completed"] = False
    assert fullkv_qualification_memory_complete(memory) is False


def test_peak_cuda_bytes_extracted_from_complete_worker_phase():
    memory = _memory()
    for record in memory:
        if record["phase"] == "fullkv_complete_worker":
            record["peak_allocated"] = 12345
            record["peak_reserved"] = 67890
    allocated, reserved = peak_cuda_bytes_from_qualification_memory_evidence(memory)
    assert (allocated, reserved) == (12345, 67890)


def test_empty_timing_list_fails():
    assert fullkv_qualification_timing_complete([], fullkv_wall_seconds=3.0) is False


def test_empty_memory_list_fails():
    assert fullkv_qualification_memory_complete([]) is False
