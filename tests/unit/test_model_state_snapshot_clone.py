"""`ModelStateSnapshot.clone()` (B1B-R2 §5): every tensor field
independently cloned, every container deep-copied, never a view/alias onto
the original. Added specifically so `kvcot.discovery.pipeline
.build_swap_pair_record` can produce two independent (baseline, swapped)
working copies from one pristine post-event snapshot with no live model in
the loop -- this file proves the round-trip and aliasing guarantees the
rest of that machinery depends on.
"""
from __future__ import annotations

import torch

from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.state import ModelStateSnapshot


def _make_snapshot(num_layers=3, num_heads=2, seq_len=5, head_dim=4) -> ModelStateSnapshot:
    key_cache = [torch.randn(1, num_heads, seq_len, head_dim) for _ in range(num_layers)]
    value_cache = [torch.randn(1, num_heads, seq_len, head_dim) for _ in range(num_layers)]
    provenance = ModelProvenance(
        layers={i: LayerProvenance(positions=torch.arange(seq_len).unsqueeze(0).expand(num_heads, -1).clone())
                for i in range(num_layers)},
        prompt_length=2,
        think_start_absolute=0,
        think_end_absolute=1,
    )
    return ModelStateSnapshot(
        key_cache=key_cache,
        value_cache=value_cache,
        query_cache={0: torch.randn(num_heads, head_dim)},
        compression_flags_per_layer=["none"] * num_layers,
        model_length=seq_len,
        after_think=None,
        compaction_event_steps=[2, 4],
        tokens_since_last_compaction=1,
        absolute_position=seq_len,
        provenance=provenance,
        kv_cluster_bookkeeping_per_layer=[
            {"evicted_token_num": 0, "kept_token_indices": [], "kept_attention_scores": [],
             "kept_similarity_scores": [], "kept_final_scores": []}
            for _ in range(num_layers)
        ],
    )


def test_clone_round_trips_every_field():
    snap = _make_snapshot()
    clone = snap.clone()

    assert len(clone.key_cache) == len(snap.key_cache)
    for a, b in zip(snap.key_cache, clone.key_cache):
        assert torch.equal(a, b)
    for a, b in zip(snap.value_cache, clone.value_cache):
        assert torch.equal(a, b)
    assert set(clone.query_cache) == set(snap.query_cache)
    for k in snap.query_cache:
        assert torch.equal(snap.query_cache[k], clone.query_cache[k])
    assert clone.compression_flags_per_layer == snap.compression_flags_per_layer
    assert clone.model_length == snap.model_length
    assert clone.after_think == snap.after_think
    assert clone.compaction_event_steps == snap.compaction_event_steps
    assert clone.tokens_since_last_compaction == snap.tokens_since_last_compaction
    assert clone.absolute_position == snap.absolute_position
    assert clone.kv_cluster_bookkeeping_per_layer == snap.kv_cluster_bookkeeping_per_layer
    assert torch.equal(clone.provenance.layers[0].positions, snap.provenance.layers[0].positions)


def test_clone_shares_no_tensor_storage_with_original():
    snap = _make_snapshot()
    clone = snap.clone()

    for a, b in zip(snap.key_cache, clone.key_cache):
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    for a, b in zip(snap.value_cache, clone.value_cache):
        assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
    for k in snap.query_cache:
        assert snap.query_cache[k].untyped_storage().data_ptr() != clone.query_cache[k].untyped_storage().data_ptr()
    assert (
        snap.provenance.layers[0].positions.untyped_storage().data_ptr()
        != clone.provenance.layers[0].positions.untyped_storage().data_ptr()
    )


def test_mutating_clone_does_not_affect_original():
    snap = _make_snapshot()
    clone = snap.clone()

    clone.key_cache[0].fill_(999.0)
    clone.value_cache[1].fill_(-999.0)
    clone.kv_cluster_bookkeeping_per_layer[0]["evicted_token_num"] = 42
    clone.compaction_event_steps.append(999)

    assert not torch.equal(snap.key_cache[0], clone.key_cache[0])
    assert not torch.equal(snap.value_cache[1], clone.value_cache[1])
    assert snap.kv_cluster_bookkeeping_per_layer[0]["evicted_token_num"] == 0
    assert snap.compaction_event_steps == [2, 4]


def test_two_independent_clones_of_the_same_pristine_snapshot_never_alias():
    """Mirrors `kvcot.discovery.pipeline.build_swap_pair_record`'s own
    `baseline = pristine.clone(); swapped = pristine.clone()` pattern."""
    pristine = _make_snapshot()
    baseline = pristine.clone()
    swapped = pristine.clone()

    assert baseline.key_cache[0].untyped_storage().data_ptr() != swapped.key_cache[0].untyped_storage().data_ptr()

    swapped.key_cache[1][0, 0, 0, :] = -1.0
    assert not torch.equal(swapped.key_cache[1], baseline.key_cache[1])
    assert torch.equal(baseline.key_cache[1], pristine.key_cache[1])  # baseline is unaffected by swapped's mutation


def test_clone_is_independent_of_provenance_none():
    snap = _make_snapshot()
    snap.provenance = None
    clone = snap.clone()
    assert clone.provenance is None
