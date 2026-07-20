import pytest
import torch

from kvcot.discovery.swap import (
    SwapAliasingError,
    SwapIndexError,
    apply_semantic_within_head_swap,
    apply_within_head_swap,
    apply_within_head_swap_owned,
)
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.state import ModelStateSnapshot

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


# --------------------------------------------------------------------------
# B1 execution-boundary closure §10: owned (no redundant clone) variant
# --------------------------------------------------------------------------


def test_owned_swap_mutates_in_place_returns_same_objects_no_second_clone():
    key_cache, value_cache = _make_cache(seed=10)
    layer_index, head_index, slot = 1, 2, 5
    candidate_key = torch.full((HEAD_DIM,), 99.0)
    candidate_value = torch.full((HEAD_DIM,), -99.0)
    original_storage_ptrs = [t.untyped_storage().data_ptr() for t in key_cache + value_cache]

    result = apply_within_head_swap_owned(
        key_cache, value_cache, layer_index, head_index, slot, candidate_key, candidate_value
    )

    # The SAME list/tensor objects are returned -- never freshly cloned.
    assert result.key_cache is key_cache
    assert result.value_cache is value_cache
    result_storage_ptrs = [t.untyped_storage().data_ptr() for t in result.key_cache + result.value_cache]
    assert result_storage_ptrs == original_storage_ptrs, "owned mutation must never allocate a second full-cache clone"

    assert torch.equal(key_cache[layer_index][0, head_index, slot, :], candidate_key)
    assert torch.equal(value_cache[layer_index][0, head_index, slot, :], candidate_value)


def test_owned_swap_touches_no_other_layer_head_or_slot():
    key_cache, value_cache = _make_cache(seed=11)
    key_before = [t.clone() for t in key_cache]
    value_before = [t.clone() for t in value_cache]
    layer_index, head_index, slot = 0, 1, 3
    candidate_key = torch.zeros(HEAD_DIM)
    candidate_value = torch.ones(HEAD_DIM)

    apply_within_head_swap_owned(key_cache, value_cache, layer_index, head_index, slot, candidate_key, candidate_value)

    for l in range(NUM_LAYERS):
        expected_k = key_before[l].clone()
        expected_v = value_before[l].clone()
        if l == layer_index:
            expected_k[0, head_index, slot, :] = candidate_key
            expected_v[0, head_index, slot, :] = candidate_value
        assert torch.equal(key_cache[l], expected_k), f"layer {l} key diverged unexpectedly"
        assert torch.equal(value_cache[l], expected_v), f"layer {l} value diverged unexpectedly"


def test_owned_swap_noop_detection_matches_cloning_variant():
    key_cache, value_cache = _make_cache(seed=12)
    layer_index, head_index, slot = 0, 0, 0
    existing_key = key_cache[layer_index][0, head_index, slot, :].clone().contiguous()
    existing_value = value_cache[layer_index][0, head_index, slot, :].clone().contiguous()

    result = apply_within_head_swap_owned(
        key_cache, value_cache, layer_index, head_index, slot, existing_key, existing_value
    )
    assert result.is_noop is True


def test_owned_swap_still_enforces_every_validation():
    key_cache, value_cache = _make_cache(seed=13)
    with pytest.raises(SwapIndexError, match="candidate_key must have shape"):
        apply_within_head_swap_owned(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM + 1), torch.ones(HEAD_DIM))
    with pytest.raises(SwapAliasingError):
        aliased = key_cache[0][0, 0, 0, :]
        apply_within_head_swap_owned(key_cache, value_cache, 0, 0, 1, aliased, torch.ones(HEAD_DIM))


def test_owned_swap_rejects_out_of_range_indices_same_as_cloning_variant():
    key_cache, value_cache = _make_cache(seed=14)
    with pytest.raises(SwapIndexError):
        apply_within_head_swap_owned(key_cache, value_cache, NUM_LAYERS, 0, 0, torch.zeros(HEAD_DIM), torch.ones(HEAD_DIM))


def test_semantic_swap_owned_true_mutates_in_place_no_second_clone(monkeypatch):
    """`apply_semantic_within_head_swap(..., owned=True)` must dispatch to
    the owned primitive -- proven by monkeypatching the CLONING primitive
    to raise if it is ever called, so any accidental fallback would fail
    loudly rather than silently double-cloning."""
    import kvcot.discovery.swap as swap_mod

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("apply_within_head_swap (the cloning primitive) must not be called when owned=True")

    monkeypatch.setattr(swap_mod, "apply_within_head_swap", _must_not_be_called)

    key_cache, value_cache = _make_cache(seed=15)
    snapshot = ModelStateSnapshot(
        key_cache=key_cache, value_cache=value_cache, query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS, model_length=SEQ_LEN, after_think=None,
        provenance=None, kv_cluster_bookkeeping_per_layer=None,
    )
    original_storage_ptr = snapshot.key_cache[0].untyped_storage().data_ptr()

    result = apply_semantic_within_head_swap(
        snapshot, layer_index=0, kv_head_index=0, retained_post_storage_position=0,
        candidate_key=torch.zeros(HEAD_DIM), candidate_value=torch.ones(HEAD_DIM),
        donor_absolute_position=5, candidate_absolute_position=7, owned=True,
    )

    assert snapshot.key_cache[0].untyped_storage().data_ptr() == original_storage_ptr
    assert result.swap_result.key_cache is key_cache


def test_semantic_swap_owned_false_default_still_clones_unchanged():
    """`owned` defaults to `False` -- every pre-existing caller (this
    module's own other tests) gets the exact prior cloning behavior."""
    key_cache, value_cache = _make_cache(seed=16)
    snapshot = ModelStateSnapshot(
        key_cache=key_cache, value_cache=value_cache, query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS, model_length=SEQ_LEN, after_think=None,
        provenance=None, kv_cluster_bookkeeping_per_layer=None,
    )
    original_storage_ptr = snapshot.key_cache[0].untyped_storage().data_ptr()

    apply_semantic_within_head_swap(
        snapshot, layer_index=0, kv_head_index=0, retained_post_storage_position=0,
        candidate_key=torch.zeros(HEAD_DIM), candidate_value=torch.ones(HEAD_DIM),
        donor_absolute_position=5, candidate_absolute_position=7,
    )

    assert snapshot.key_cache[0].untyped_storage().data_ptr() != original_storage_ptr


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


# --------------------------------------------------------------------------
# Blocker 4: dtype/device/batch-size guards and storage-overlap hardening
# --------------------------------------------------------------------------


def test_float64_candidate_into_float32_cache_rejected():
    key_cache, value_cache = _make_cache(seed=12)
    assert key_cache[0].dtype == torch.float32
    candidate_key = torch.zeros(HEAD_DIM, dtype=torch.float64)
    candidate_value = torch.zeros(HEAD_DIM, dtype=torch.float32)
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, candidate_key, candidate_value)


def test_cpu_candidate_into_another_device_cache_rejected():
    # This CPU-only test environment has no second real accelerator to
    # exercise a true cross-device mismatch against, so `meta` (torch's
    # always-available "no real storage" device) stands in as the "another
    # device" -- only ONE tensor in play is meta (candidate_key alone) so
    # this cannot collide with the meta-vs-meta storage-identity quirk
    # (two independent meta tensors both report untyped_storage().data_ptr()
    # == 0, which would otherwise look like a false storage overlap).
    key_cache, value_cache = _make_cache(seed=13)
    candidate_key = torch.zeros(HEAD_DIM).to("meta")
    candidate_value = torch.zeros(HEAD_DIM)  # cpu, matches the cache's own device
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, candidate_key, candidate_value)


def test_batch_size_two_rejected():
    key_cache, value_cache = _make_cache(seed=14)
    key_cache[0] = torch.randn(2, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    value_cache[0] = torch.randn(2, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM))


def test_key_value_cache_dtype_mismatch_rejected():
    key_cache, value_cache = _make_cache(seed=15)
    value_cache[0] = value_cache[0].to(torch.float64)
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM, dtype=torch.float64))


def test_key_value_cache_device_mismatch_rejected():
    key_cache, value_cache = _make_cache(seed=16)
    value_cache[0] = value_cache[0].to("meta")
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM))


def test_offset_views_sharing_the_same_storage_rejected():
    key_cache, value_cache = _make_cache(seed=17)
    # value_cache[0] is a genuine, in-bounds VIEW into key_cache[0]'s own
    # storage (one position later along the seq dimension) -- a different
    # starting offset, same underlying storage. A plain
    # `tensor.data_ptr() == tensor.data_ptr()` comparison on the two 4-D
    # tensors would already catch differing starting addresses as
    # "different", missing the shared storage entirely; untyped_storage()
    # identity must catch it regardless.
    base = key_cache[0]
    key_cache[0] = base
    value_cache[0] = base[:, :, 1:, :]
    with pytest.raises(SwapAliasingError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), torch.zeros(HEAD_DIM))


def test_candidate_view_into_cache_storage_rejected():
    key_cache, value_cache = _make_cache(seed=18)
    # candidate_key is a view (a single [layer,head,slot] row) into
    # key_cache[0]'s own storage at a DIFFERENT slot than the one being
    # written -- starting data_ptr() differs from the target slot's, so a
    # naive data_ptr()-only check on the write target would miss this, but
    # untyped_storage() identity catches it.
    candidate_key = key_cache[0][0, 0, 3, :]
    candidate_value = torch.zeros(HEAD_DIM)
    with pytest.raises(SwapAliasingError):
        apply_within_head_swap(key_cache, value_cache, 0, 0, 0, candidate_key, candidate_value)


def test_non_contiguous_candidate_rejected():
    key_cache, value_cache = _make_cache(seed=19)
    base = torch.randn(HEAD_DIM, 2)
    non_contiguous_candidate = base[:, 0]  # stride (2,) over a (HEAD_DIM, 2) tensor -- not contiguous
    assert not non_contiguous_candidate.is_contiguous()
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(
            key_cache, value_cache, 0, 0, 0, non_contiguous_candidate, torch.zeros(HEAD_DIM)
        )
    with pytest.raises(SwapIndexError):
        apply_within_head_swap(
            key_cache, value_cache, 0, 0, 0, torch.zeros(HEAD_DIM), non_contiguous_candidate
        )


def test_valid_noop_remains_exact_with_new_guards_in_place():
    key_cache, value_cache = _make_cache(seed=20)
    layer_index, head_index, slot = 1, 2, 5
    existing_key = key_cache[layer_index][0, head_index, slot, :].clone().contiguous()
    existing_value = value_cache[layer_index][0, head_index, slot, :].clone().contiguous()

    result = apply_within_head_swap(
        key_cache, value_cache, layer_index, head_index, slot, existing_key, existing_value
    )

    assert result.is_noop is True
    for l in range(NUM_LAYERS):
        assert torch.equal(result.key_cache[l], key_cache[l])
        assert torch.equal(result.value_cache[l], value_cache[l])


def test_every_untouched_cache_slice_remains_bit_exact():
    key_cache, value_cache = _make_cache(seed=21)
    layer_index, head_index, slot = 2, 1, 6
    candidate_key = torch.full((HEAD_DIM,), 7.0)
    candidate_value = torch.full((HEAD_DIM,), -7.0)

    result = apply_within_head_swap(
        key_cache, value_cache, layer_index, head_index, slot, candidate_key, candidate_value
    )

    for l in range(NUM_LAYERS):
        for h in range(NUM_HEADS):
            for s in range(SEQ_LEN):
                if (l, h, s) == (layer_index, head_index, slot):
                    continue
                assert torch.equal(result.key_cache[l][0, h, s, :], key_cache[l][0, h, s, :]), (l, h, s)
                assert torch.equal(result.value_cache[l][0, h, s, :], value_cache[l][0, h, s, :]), (l, h, s)


# --------------------------------------------------------------------------
# B1B-R3 §10: apply_semantic_within_head_swap -- provenance/kept-index
# bookkeeping consistency
# --------------------------------------------------------------------------


def _make_full_snapshot(seed: int) -> ModelStateSnapshot:
    key_cache, value_cache = _make_cache(seed)
    provenance = ModelProvenance(
        layers={
            l: LayerProvenance(positions=torch.arange(100 * (l + 1), 100 * (l + 1) + SEQ_LEN).unsqueeze(0).expand(NUM_HEADS, -1).clone())
            for l in range(NUM_LAYERS)
        }
    )
    kv_cluster_bookkeeping = [
        {"kept_token_indices": [provenance.layers[l].positions.clone()]} for l in range(NUM_LAYERS)
    ]
    return ModelStateSnapshot(
        key_cache=key_cache,
        value_cache=value_cache,
        query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS,
        model_length=SEQ_LEN,
        after_think=None,
        provenance=provenance,
        kv_cluster_bookkeeping_per_layer=kv_cluster_bookkeeping,
    )


def test_semantic_swap_updates_kv_content_provenance_and_bookkeeping_consistently():
    snapshot = _make_full_snapshot(seed=30)
    layer_index, head_index, slot = 1, 2, 5
    donor_absolute_position = int(snapshot.provenance.layers[layer_index].positions[head_index, slot].item())
    candidate_absolute_position = donor_absolute_position + 12345  # a distinguishable, different identity
    candidate_key = torch.full((HEAD_DIM,), 42.0)
    candidate_value = torch.full((HEAD_DIM,), -42.0)

    result = apply_semantic_within_head_swap(
        snapshot,
        layer_index=layer_index,
        kv_head_index=head_index,
        retained_post_storage_position=slot,
        candidate_key=candidate_key,
        candidate_value=candidate_value,
        donor_absolute_position=donor_absolute_position,
        candidate_absolute_position=candidate_absolute_position,
    )

    assert result.is_noop is False
    assert result.provenance_updated is True
    assert result.kept_index_bookkeeping_updated is True
    assert torch.equal(snapshot.key_cache[layer_index][0, head_index, slot, :], candidate_key)
    assert torch.equal(snapshot.value_cache[layer_index][0, head_index, slot, :], candidate_value)
    assert int(snapshot.provenance.layers[layer_index].positions[head_index, slot].item()) == candidate_absolute_position
    assert int(
        snapshot.kv_cluster_bookkeeping_per_layer[layer_index]["kept_token_indices"][-1][head_index, slot].item()
    ) == candidate_absolute_position


def test_semantic_swap_touches_only_the_one_targeted_identity():
    snapshot = _make_full_snapshot(seed=31)
    layer_index, head_index, slot = 0, 1, 3
    before_positions = {l: snapshot.provenance.layers[l].positions.clone() for l in range(NUM_LAYERS)}
    donor_absolute_position = int(before_positions[layer_index][head_index, slot].item())

    apply_semantic_within_head_swap(
        snapshot,
        layer_index=layer_index,
        kv_head_index=head_index,
        retained_post_storage_position=slot,
        candidate_key=torch.zeros(HEAD_DIM),
        candidate_value=torch.ones(HEAD_DIM),
        donor_absolute_position=donor_absolute_position,
        candidate_absolute_position=donor_absolute_position + 999,
    )

    for l in range(NUM_LAYERS):
        for h in range(NUM_HEADS):
            for s in range(SEQ_LEN):
                expected = before_positions[l][h, s].item()
                if (l, h, s) == (layer_index, head_index, slot):
                    assert int(snapshot.provenance.layers[l].positions[h, s].item()) != expected
                else:
                    assert int(snapshot.provenance.layers[l].positions[h, s].item()) == expected


def test_semantic_swap_noop_leaves_identity_unchanged():
    snapshot = _make_full_snapshot(seed=32)
    layer_index, head_index, slot = 1, 1, 4
    existing_key = snapshot.key_cache[layer_index][0, head_index, slot, :].clone().contiguous()
    existing_value = snapshot.value_cache[layer_index][0, head_index, slot, :].clone().contiguous()
    donor_absolute_position = int(snapshot.provenance.layers[layer_index].positions[head_index, slot].item())

    result = apply_semantic_within_head_swap(
        snapshot,
        layer_index=layer_index,
        kv_head_index=head_index,
        retained_post_storage_position=slot,
        candidate_key=existing_key,
        candidate_value=existing_value,
        donor_absolute_position=donor_absolute_position,
        candidate_absolute_position=donor_absolute_position,
    )

    assert result.is_noop is True
    assert int(snapshot.provenance.layers[layer_index].positions[head_index, slot].item()) == donor_absolute_position


def test_semantic_swap_does_not_alias_or_mutate_a_pristine_clone():
    pristine = _make_full_snapshot(seed=33)
    baseline = pristine.clone()
    swapped = pristine.clone()
    layer_index, head_index, slot = 2, 0, 7
    donor_absolute_position = int(swapped.provenance.layers[layer_index].positions[head_index, slot].item())

    apply_semantic_within_head_swap(
        swapped,
        layer_index=layer_index,
        kv_head_index=head_index,
        retained_post_storage_position=slot,
        candidate_key=torch.full((HEAD_DIM,), 5.0),
        candidate_value=torch.full((HEAD_DIM,), -5.0),
        donor_absolute_position=donor_absolute_position,
        candidate_absolute_position=donor_absolute_position + 1,
    )

    assert torch.equal(baseline.key_cache[layer_index], pristine.key_cache[layer_index])
    assert int(baseline.provenance.layers[layer_index].positions[head_index, slot].item()) == donor_absolute_position
    assert not torch.equal(swapped.key_cache[layer_index], pristine.key_cache[layer_index])


def test_semantic_swap_without_provenance_or_bookkeeping_is_best_effort_and_still_swaps_kv():
    key_cache, value_cache = _make_cache(seed=34)
    snapshot = ModelStateSnapshot(
        key_cache=key_cache, value_cache=value_cache, query_cache={},
        compression_flags_per_layer=["none"] * NUM_LAYERS, model_length=SEQ_LEN, after_think=None,
        provenance=None, kv_cluster_bookkeeping_per_layer=None,
    )
    result = apply_semantic_within_head_swap(
        snapshot, layer_index=0, kv_head_index=0, retained_post_storage_position=0,
        candidate_key=torch.zeros(HEAD_DIM), candidate_value=torch.ones(HEAD_DIM),
        donor_absolute_position=0, candidate_absolute_position=1,
    )
    assert result.provenance_updated is False
    assert result.kept_index_bookkeeping_updated is False
    assert torch.equal(snapshot.key_cache[0][0, 0, 0, :], torch.zeros(HEAD_DIM))
