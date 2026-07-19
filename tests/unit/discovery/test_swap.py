import pytest
import torch

from kvcot.discovery.swap import SwapAliasingError, SwapIndexError, apply_within_head_swap

NUM_LAYERS = 3
NUM_HEADS = 4
SEQ_LEN = 10
HEAD_DIM = 6


def _make_cache(seed: int):
    g = torch.Generator().manual_seed(seed)
    key_cache = [torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=g) for _ in range(NUM_LAYERS)]
    value_cache = [torch.randn(1, NUM_HEADS, SEQ_LEN, HEAD_DIM, generator=g) for _ in range(NUM_LAYERS)]
    return key_cache, value_cache


def test_swap_changes_exactly_one_slot_shape_preserved():
    key_cache, value_cache = _make_cache(seed=1)
    layer_index, head_index, slot = 1, 2, 5
    candidate_key = torch.full((HEAD_DIM,), 99.0)
    candidate_value = torch.full((HEAD_DIM,), -99.0)

    result = apply_within_head_swap(
        key_cache, value_cache, layer_index, head_index, slot, candidate_key, candidate_value
    )

    assert torch.equal(result.key_cache[layer_index][0, head_index, slot, :], candidate_key)
    assert torch.equal(result.value_cache[layer_index][0, head_index, slot, :], candidate_value)

    for l in range(NUM_LAYERS):
        assert result.key_cache[l].shape == key_cache[l].shape
        assert result.value_cache[l].shape == value_cache[l].shape


def test_swap_touches_no_other_layer_head_or_slot():
    key_cache, value_cache = _make_cache(seed=2)
    layer_index, head_index, slot = 0, 1, 3
    candidate_key = torch.zeros(HEAD_DIM)
    candidate_value = torch.ones(HEAD_DIM)

    result = apply_within_head_swap(
        key_cache, value_cache, layer_index, head_index, slot, candidate_key, candidate_value
    )

    for l in range(NUM_LAYERS):
        expected_k = key_cache[l].clone()
        expected_v = value_cache[l].clone()
        if l == layer_index:
            expected_k[0, head_index, slot, :] = candidate_key
            expected_v[0, head_index, slot, :] = candidate_value
        assert torch.equal(result.key_cache[l], expected_k), f"layer {l} key diverged unexpectedly"
        assert torch.equal(result.value_cache[l], expected_v), f"layer {l} value diverged unexpectedly"


def test_swap_does_not_mutate_original_input_tensors():
    key_cache, value_cache = _make_cache(seed=3)
    key_before = [t.clone() for t in key_cache]
    value_before = [t.clone() for t in value_cache]

    apply_within_head_swap(
        key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), torch.ones(HEAD_DIM)
    )

    for l in range(NUM_LAYERS):
        assert torch.equal(key_cache[l], key_before[l])
        assert torch.equal(value_cache[l], value_before[l])


def test_net_physical_bytes_unchanged():
    key_cache, value_cache = _make_cache(seed=4)
    total_before = sum(t.numel() * t.element_size() for t in key_cache + value_cache)

    result = apply_within_head_swap(
        key_cache, value_cache, 2, 3, 9, torch.zeros(HEAD_DIM), torch.ones(HEAD_DIM)
    )

    total_after = sum(t.numel() * t.element_size() for t in result.key_cache + result.value_cache)
    assert total_after == total_before


def test_noop_swap_e_equals_r_is_detected_and_bit_exact():
    key_cache, value_cache = _make_cache(seed=5)
    layer_index, head_index, slot = 1, 1, 4
    existing_key = key_cache[layer_index][0, head_index, slot, :].clone()
    existing_value = value_cache[layer_index][0, head_index, slot, :].clone()

    result = apply_within_head_swap(
        key_cache, value_cache, layer_index, head_index, slot, existing_key, existing_value
    )

    assert result.is_noop is True
    for l in range(NUM_LAYERS):
        assert torch.equal(result.key_cache[l], key_cache[l])
        assert torch.equal(result.value_cache[l], value_cache[l])


def test_non_noop_swap_reports_is_noop_false():
    key_cache, value_cache = _make_cache(seed=6)
    result = apply_within_head_swap(
        key_cache, value_cache, 0, 0, 0, torch.full((HEAD_DIM,), 12345.0), torch.full((HEAD_DIM,), -12345.0)
    )
    assert result.is_noop is False


@pytest.mark.parametrize(
    "layer_index,head_index,slot",
    [(-1, 0, 0), (NUM_LAYERS, 0, 0), (0, -1, 0), (0, NUM_HEADS, 0), (0, 0, -1), (0, 0, SEQ_LEN)],
)
def test_invalid_indices_are_rejected(layer_index, head_index, slot):
    key_cache, value_cache = _make_cache(seed=7)
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(
            key_cache, value_cache, layer_index, head_index, slot, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM)
        )


def test_wrong_candidate_shape_is_rejected():
    key_cache, value_cache = _make_cache(seed=8)
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM + 1), torch.zeros(HEAD_DIM))
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM - 1))


def test_aliased_key_and_value_layer_tensor_is_rejected():
    key_cache, value_cache = _make_cache(seed=9)
    value_cache[0] = key_cache[0]  # simulate a caller bug: same tensor object for K and V
    with pytest.raises(SwapAliasingError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM))


def test_aliased_candidate_key_and_value_is_rejected():
    key_cache, value_cache = _make_cache(seed=10)
    candidate = torch.zeros(HEAD_DIM)
    with pytest.raises(SwapAliasingError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, candidate, candidate)


def test_mismatched_key_value_cache_lengths_rejected():
    key_cache, value_cache = _make_cache(seed=11)
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache[:-1], 0, 0, 0, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM))
