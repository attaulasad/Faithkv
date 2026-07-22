"""B1B-R4 §18 tests for `kvcot.discovery.capture_minimize` -- proves the
minimized target evidence retains no full-cache tensor and stays within a
fixed bound regardless of a (synthetically large) cache length.
"""
from __future__ import annotations

import torch

from kvcot.discovery.capture_minimize import (
    CaptureMinimizationError,
    assert_minimized_bound,
    build_minimized_target_evidence,
)
from kvcot.discovery.capture import UpdateKvCaptureRecord
from kvcot.discovery.pass1 import EventPlan
from kvcot.discovery.sampling import CandidateDonorSelection

HEAD_DIM = 8
NUM_HEADS = 2
WINDOW_SIZE = 2


def _make_record(cache_len: int, evicted_positions, donor_positions, head=0):
    """A real (small) `UpdateKvCaptureRecord`-shaped fixture: full-layer
    tensors sized by `cache_len`, so a bound violation would be visible if
    the minimization function accidentally retained them."""
    non_recent_len = cache_len - WINDOW_SIZE
    pre_key = torch.arange(NUM_HEADS * cache_len * HEAD_DIM, dtype=torch.float32).reshape(
        1, NUM_HEADS, cache_len, HEAD_DIM
    )
    pre_value = pre_key + 1000.0
    scores = torch.arange(NUM_HEADS * non_recent_len, dtype=torch.float32).reshape(1, NUM_HEADS, non_recent_len)

    # pre_event_absolute_position_map: identity mapping (physical slot i -> absolute position i)
    position_map = torch.arange(cache_len, dtype=torch.long).unsqueeze(0).expand(NUM_HEADS, -1).clone()

    return UpdateKvCaptureRecord(
        had_compaction=True,
        pre_call_key_states=pre_key,
        pre_call_value_states=pre_value,
        pre_call_key_shape=tuple(pre_key.shape),
        pre_call_value_shape=tuple(pre_value.shape),
        pre_call_dtype=str(pre_key.dtype),
        pre_call_device=str(pre_key.device),
        recomputed_final_score=scores,
        recomputed_attention_component=scores,
        recomputed_similarity_component=scores,
        recomputed_topk_indices=torch.zeros(1, NUM_HEADS, 1, dtype=torch.long),
        window_size=WINDOW_SIZE,
        returned_key_states=pre_key.clone(),
        returned_value_states=pre_value.clone(),
        gather_parity_passed=True,
        pre_event_absolute_position_map=position_map,
        recomputed_kept_absolute_positions=position_map,
        observed_kept_absolute_positions=position_map,
        observed_kept_indices_parity_passed=True,
        parity_check_passed=True,
        parity_failure_reason=None,
    )


def _make_event_plan(evicted_selected, donor_selected):
    cross_product = tuple((e, d) for e in evicted_selected for d in donor_selected)
    cd = CandidateDonorSelection(
        evicted_selected=tuple(evicted_selected), donor_selected=tuple(donor_selected), cross_product=cross_product
    )
    return EventPlan(
        compaction_event_id=0, absolute_event_position=100, chronological_event_ordinal=0, depth_stratum=0,
        layer_index=3, kv_head_index=0, candidate_donor_selection=cd,
    )


def test_extracts_exact_candidate_and_donor_vectors():
    record = _make_record(cache_len=20, evicted_positions=None, donor_positions=None)
    plan = _make_event_plan(evicted_selected=(1, 2), donor_selected=(10, 11))

    evidence = build_minimized_target_evidence(plan, record)

    assert evidence.candidate_absolute_positions == (1, 2)
    assert evidence.donor_absolute_positions == (10, 11)
    assert len(evidence.candidate_key_vectors) == 2
    assert len(evidence.donor_key_vectors) == 2
    expected_candidate_1_key = tuple(record.pre_call_key_states[0, 0, 1, :].tolist())
    assert evidence.candidate_key_vectors[0] == expected_candidate_1_key
    assert evidence.head_dim == HEAD_DIM


def test_gather_and_absolute_parity_flags_pass_through():
    record = _make_record(cache_len=20, evicted_positions=None, donor_positions=None)
    plan = _make_event_plan(evicted_selected=(1, 2), donor_selected=(10, 11))
    evidence = build_minimized_target_evidence(plan, record)
    assert evidence.gather_parity_passed is True
    assert evidence.absolute_position_parity_passed is True
    assert evidence.failure_reason is None


def test_bound_holds_regardless_of_synthetically_large_cache_length():
    """The core B1B-R4 §18 regression: persistent storage must not scale
    with cache length -- prove it by using a cache 1000x larger than the
    small fixture case and checking the SAME fixed bound holds."""
    small_record = _make_record(cache_len=20, evicted_positions=None, donor_positions=None)
    large_record = _make_record(cache_len=20_000, evicted_positions=None, donor_positions=None)
    plan = _make_event_plan(evicted_selected=(1, 2), donor_selected=(10, 11))

    small_evidence = build_minimized_target_evidence(plan, small_record)
    large_evidence = build_minimized_target_evidence(plan, large_record)

    assert small_evidence.persistent_tensor_numel == large_evidence.persistent_tensor_numel
    max_vectors = (2 + 2) * 2  # CANDIDATES_PER_EVENT + DONORS_PER_EVENT, times K and V
    assert large_evidence.persistent_tensor_numel == max_vectors * HEAD_DIM

    assert_minimized_bound(small_evidence)
    assert_minimized_bound(large_evidence)


def test_no_full_cache_tensor_reachable_from_the_returned_object():
    record = _make_record(cache_len=500, evicted_positions=None, donor_positions=None)
    plan = _make_event_plan(evicted_selected=(1, 2), donor_selected=(10, 11))
    evidence = build_minimized_target_evidence(plan, record)

    for field_value in evidence.__dict__.values():
        assert not isinstance(field_value, torch.Tensor)
        if isinstance(field_value, tuple):
            for item in field_value:
                assert not isinstance(item, torch.Tensor)


def test_missing_candidate_position_raises():
    record = _make_record(cache_len=20, evicted_positions=None, donor_positions=None)
    plan = _make_event_plan(evicted_selected=(999, 2), donor_selected=(10, 11))  # 999 not in the position map
    try:
        build_minimized_target_evidence(plan, record)
        assert False, "expected CaptureMinimizationError"
    except CaptureMinimizationError:
        pass


def test_no_compaction_record_returns_empty_evidence_not_a_crash():
    record = UpdateKvCaptureRecord(
        had_compaction=False, pre_call_key_states=torch.zeros(1, 1, 1, HEAD_DIM),
        pre_call_value_states=torch.zeros(1, 1, 1, HEAD_DIM), pre_call_key_shape=(1, 1, 1, HEAD_DIM),
        pre_call_value_shape=(1, 1, 1, HEAD_DIM), pre_call_dtype="torch.float32", pre_call_device="cpu",
        recomputed_final_score=None, recomputed_attention_component=None, recomputed_similarity_component=None,
        recomputed_topk_indices=None, window_size=None, returned_key_states=torch.zeros(1, 1, 1, HEAD_DIM),
        returned_value_states=torch.zeros(1, 1, 1, HEAD_DIM), gather_parity_passed=None,
        pre_event_absolute_position_map=None, recomputed_kept_absolute_positions=None,
        observed_kept_absolute_positions=None, observed_kept_indices_parity_passed=None,
        parity_check_passed=True, parity_failure_reason=None,
    )
    plan = _make_event_plan(evicted_selected=(1, 2), donor_selected=(10, 11))
    evidence = build_minimized_target_evidence(plan, record)
    assert evidence.persistent_tensor_numel == 0
    assert evidence.failure_reason == "no_compaction_or_missing_position_map"
