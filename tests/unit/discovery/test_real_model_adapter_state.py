"""CPU-only (no GPU, no real model download) regression tests for
`kvcot.discovery.real_model_adapter` (B1B-R3 §8/§9).

`FakeEvictingModel` below is a lightweight, deterministic stand-in for a
real R-KV-patched CausalLM forward -- real torch CPU tensors, real
`kv_cluster.kept_token_indices` growth, real cache-length shrinkage -- but
with a trivial "keep the most recent `budget` positions" eviction policy
(never R-KV's real topk/similarity formula, which is exercised elsewhere,
`tests/unit/discovery/test_capture.py`) chosen specifically so this
module's expected `model_provenance.layers[i].positions` after several
calls is trivial to state independently and assert against -- proving
`advance_after_forward` keeps `RealModelState.model_provenance` correctly
populated across prefill, decode, AND an eviction event, closing the gap
`docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md` §8 identified (B1B-R2's adapter
never called `append_new_tokens_prefill`/`append_new_token` at all, so
`model_provenance.layers[i].positions` stayed permanently empty)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from kvcot.discovery.real_model_adapter import (  # noqa: E402
    RealModelState,
    _LiveBranchState,
    advance_after_forward,
    build_real_branch_step_fn_restore_once,
    build_real_decode_one_fn,
    build_real_prefill_fn,
    build_real_snapshot_fn,
    compute_kept_indices_lengths,
    evaluate_branch_from_snapshot,
    restore_compaction_tracker_from_snapshot,
)
from kvcot.generation.provenance import LayerProvenance, ModelProvenance
from kvcot.generation.replay import CompactionTracker
from kvcot.generation.state import ModelStateSnapshot, reset_patched_state

NUM_LAYERS = 2
NUM_HEADS = 1
HEAD_DIM = 4
VOCAB_SIZE = 8
BUDGET = 5
WINDOW_SIZE = 2


class _FakeConfig:
    compression = None
    divide_method = "step_length"
    divide_length = 1_000_000  # never triggers the schedule flag itself -- eviction is driven by budget only
    compression_content = "all"
    update_kv = True


class _FakeKVCluster:
    def __init__(self, budget: int, window_size: int):
        self.budget = budget
        self.window_size = window_size
        self.record_kept_token_indices = True
        self.evicted_token_num = 0
        self.kept_token_indices: list = []
        self.kept_attention_scores: list = []
        self.kept_similarity_scores: list = []
        self.kept_final_scores: list = []


class _FakeAttn:
    def __init__(self, budget: int, window_size: int):
        self.config = _FakeConfig()
        self.kv_cluster = _FakeKVCluster(budget, window_size)


class _FakeLayer:
    def __init__(self, budget: int, window_size: int):
        self.self_attn = _FakeAttn(budget, window_size)


class _FakeCache:
    def __init__(self, num_layers: int):
        self.key_cache = [torch.empty(1, NUM_HEADS, 0, HEAD_DIM) for _ in range(num_layers)]
        self.value_cache = [torch.empty(1, NUM_HEADS, 0, HEAD_DIM) for _ in range(num_layers)]

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if self.key_cache[layer_idx].numel() == 0:
            self.key_cache[layer_idx] = key
            self.value_cache[layer_idx] = value
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value], dim=-2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class FakeEvictingModel:
    """Deterministic "keep most recent `budget` absolute positions" fake --
    real cache tensors and real `kept_token_indices` absolute-position
    bookkeeping (never physical-slot indices standing in for absolute
    positions), so this fixture's expected post-eviction survivor identity
    is externally computable without depending on the code under test."""

    def __init__(self, budget: int = BUDGET, window_size: int = WINDOW_SIZE, num_layers: int = NUM_LAYERS):
        self.model = SimpleNamespace(layers=[_FakeLayer(budget, window_size) for _ in range(num_layers)])
        self.config = SimpleNamespace(num_key_value_heads=NUM_HEADS)
        self._fed_absolute_positions: list[int] = []  # ground truth for the test to check against

    def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=None, cache_position=None):
        cache = past_key_values
        seq_len = input_ids.shape[1]
        new_absolute_positions = [int(p) for p in cache_position.tolist()]
        self._fed_absolute_positions.extend(new_absolute_positions)

        for i, layer in enumerate(self.model.layers):
            kv_cluster = layer.self_attn.kv_cluster
            new_k = torch.full((1, NUM_HEADS, seq_len, HEAD_DIM), float(cache_position[0].item()))
            new_v = new_k.clone()
            cache.key_cache[i] = torch.cat([cache.key_cache[i], new_k], dim=-2)
            cache.value_cache[i] = torch.cat([cache.value_cache[i], new_v], dim=-2)
            current_len = cache.key_cache[i].shape[-2]
            if current_len >= kv_cluster.budget:
                survivors_absolute = self._fed_absolute_positions[-kv_cluster.budget:]
                kept = torch.tensor(survivors_absolute, dtype=torch.long).unsqueeze(0).expand(NUM_HEADS, -1).clone()
                kv_cluster.kept_token_indices.append(kept)
                kv_cluster.evicted_token_num += current_len - kv_cluster.budget
                cache.key_cache[i] = cache.key_cache[i][:, :, -kv_cluster.budget:, :]
                cache.value_cache[i] = cache.value_cache[i][:, :, -kv_cluster.budget:, :]

        logits = torch.zeros(1, seq_len, VOCAB_SIZE)
        return SimpleNamespace(logits=logits)


def _fresh_state(model: FakeEvictingModel, num_layers: int = NUM_LAYERS) -> RealModelState:
    provenance = ModelProvenance(layers={i: LayerProvenance.empty(NUM_HEADS) for i in range(num_layers)})
    return RealModelState(
        model=model, cache=_FakeCache(num_layers), model_provenance=provenance,
        compaction=CompactionTracker(), absolute_position=0, device="cpu",
    )


# --------------------------------------------------------------------------
# advance_after_forward / compute_kept_indices_lengths
# --------------------------------------------------------------------------


def test_prefill_appends_prompt_positions_to_every_layer_provenance():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")

    result = prefill_fn(state, [10, 11, 12])  # 3 prompt tokens, absolute positions 0,1,2

    for layer_idx in range(NUM_LAYERS):
        positions = state.model_provenance.layers[layer_idx].positions
        assert positions.shape[-1] == 3
        assert positions[0].tolist() == [0, 1, 2]
    assert len(result.per_position_logits) == 3
    assert len(result.per_position_layer_observations) == 3


def test_decode_appends_exactly_one_position_per_call():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")

    prefill_fn(state, [10, 11, 12])
    decode_fn(state, 20)
    decode_fn(state, 21)

    for layer_idx in range(NUM_LAYERS):
        positions = state.model_provenance.layers[layer_idx].positions
        assert positions[0].tolist() == [0, 1, 2, 3, 4]


def test_provenance_reflects_real_eviction_not_left_empty():
    """The core B1B-R3 §8 regression: before the fix, `model_provenance
    .layers[i].positions` never grew via append at all (only wholesale-
    replaced on an eviction event), so `pre_event_absolute_position_map`
    read at the moment of an eviction would be wrong/empty. With
    budget=5, feeding 3 prompt + 3 decode tokens (6 total, positions
    0..5) crosses the budget at the 5th fed token (position 4) -- this
    test proves the pre-event map available to the caller at that exact
    call is the CORRECT dense 0..3 sequence, not empty."""
    model = FakeEvictingModel(budget=5)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")

    prefill_fn(state, [10, 11, 12])  # absolute positions 0,1,2 -- cache len 3, no eviction yet
    result_a = decode_fn(state, 20)  # absolute position 3 fed -- cache len 4, still no eviction
    assert not any(obs.had_compaction for obs in result_a.layer_observations.values())

    result_b = decode_fn(state, 21)  # absolute position 4 fed -- cache len 5 == budget: eviction fires
    assert all(obs.had_compaction for obs in result_b.layer_observations.values())
    for layer_idx in range(NUM_LAYERS):
        pre_map = result_b.layer_observations[layer_idx].pre_event_absolute_position_map
        assert pre_map is not None
        assert pre_map[0].tolist() == [0, 1, 2, 3, 4]  # dense, correct -- not empty

    for layer_idx in range(NUM_LAYERS):
        positions_after = state.model_provenance.layers[layer_idx].positions
        assert positions_after[0].tolist() == [0, 1, 2, 3, 4]


def test_event_fired_recorded_at_most_once_per_forward_call():
    model = FakeEvictingModel(budget=3)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")

    prefill_fn(state, [10, 11, 12])  # 3 tokens, hits budget=3 exactly within the one prefill call
    assert len(state.compaction.event_steps) == 1


def test_multiple_decode_evictions_recorded_once_each():
    model = FakeEvictingModel(budget=2)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")

    prefill_fn(state, [10, 11])  # positions 0,1 -- hits budget=2 already
    decode_fn(state, 20)  # position 2 -- evicts again
    decode_fn(state, 21)  # position 3 -- evicts again

    assert len(state.compaction.event_steps) == 3
    assert state.compaction.event_steps == sorted(state.compaction.event_steps)


def test_compute_kept_indices_lengths_matches_live_model():
    model = FakeEvictingModel(budget=3)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    prefill_fn(state, [10, 11, 12])

    lengths = compute_kept_indices_lengths(model)
    assert lengths == {i: 1 for i in range(NUM_LAYERS)}


def test_advance_after_forward_rejects_multi_position_decode():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    with pytest.raises(ValueError, match="exactly one fed position"):
        advance_after_forward(
            state.model, state.cache, state.model_provenance, state.compaction,
            fed_absolute_positions=[0, 1], expected_cache_lengths_if_no_eviction={i: 2 for i in range(NUM_LAYERS)},
            kept_indices_lengths_before_call=compute_kept_indices_lengths(model), call_kind="decode",
        )


# --------------------------------------------------------------------------
# restore-once branch evaluation (B1B-R3 §9)
# --------------------------------------------------------------------------


def _make_pristine_snapshot(model: FakeEvictingModel, prompt_len: int = 3) -> ModelStateSnapshot:
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    prefill_fn(state, list(range(100, 100 + prompt_len)))
    snapshot_fn = build_real_snapshot_fn()
    return snapshot_fn(state)


def test_branch_step_fn_restores_exactly_once_per_branch():
    """The cache object is constructed fresh (via `restore_snapshot`) only
    on the first call -- every subsequent call in the same branch reuses
    the identical cache object, never constructing (and re-restoring into)
    a new one."""
    model = FakeEvictingModel(budget=100)  # large budget -- no eviction noise during the branch itself
    snapshot = _make_pristine_snapshot(model)
    step_fn = build_real_branch_step_fn_restore_once(model, "cpu")

    state = snapshot
    cache_ids: list[int] = []
    for token_id in [200, 201, 202, 203]:
        _, state = step_fn(state, token_id)
        assert isinstance(state, _LiveBranchState)
        cache_ids.append(id(state.cache))

    assert len(set(cache_ids)) == 1


def test_branch_step_fn_second_call_reuses_live_state_without_restoring():
    model = FakeEvictingModel(budget=100)
    snapshot = _make_pristine_snapshot(model)
    step_fn = build_real_branch_step_fn_restore_once(model, "cpu")

    _, live_after_first = step_fn(snapshot, 200)
    assert isinstance(live_after_first, _LiveBranchState)
    cache_id_before = id(live_after_first.cache)

    _, live_after_second = step_fn(live_after_first, 201)
    assert live_after_second is live_after_first
    assert id(live_after_second.cache) == cache_id_before


def test_evaluate_branch_from_snapshot_scores_full_horizon():
    model = FakeEvictingModel(budget=100)
    snapshot = _make_pristine_snapshot(model)
    reference_tokens = [i % VOCAB_SIZE for i in range(48)]

    result = evaluate_branch_from_snapshot(model, "cpu", snapshot, bridge_token_id=250, reference_token_ids=reference_tokens)

    assert len(result.per_token_nll) == 48
    assert isinstance(result.final_cache_state, _LiveBranchState)


def test_two_independent_branches_from_the_same_snapshot_do_not_alias():
    model = FakeEvictingModel(budget=100)
    snapshot = _make_pristine_snapshot(model)
    step_fn = build_real_branch_step_fn_restore_once(model, "cpu")

    _, branch_a = step_fn(snapshot, 200)
    _, branch_b = step_fn(snapshot, 201)

    assert branch_a is not branch_b
    assert branch_a.cache is not branch_b.cache
    for layer_idx in range(NUM_LAYERS):
        assert branch_a.cache.key_cache[layer_idx].data_ptr() != branch_b.cache.key_cache[layer_idx].data_ptr()


# --------------------------------------------------------------------------
# restore_compaction_tracker_from_snapshot (B1B-R4 §13)
# --------------------------------------------------------------------------


def test_restore_compaction_tracker_pure_formula_against_a_hand_built_snapshot():
    """Isolated test of the derivation formula itself, independent of any
    real forward call: `last_event_absolute_position = absolute_position -
    tokens_since_last_compaction`, the exact inverse of
    `CompactionTracker.tokens_since_last`."""
    fake_snapshot = SimpleNamespace(
        compaction_event_steps=[5, 12], tokens_since_last_compaction=3, absolute_position=15,
    )
    tracker = restore_compaction_tracker_from_snapshot(fake_snapshot)
    assert tracker.event_steps == [5, 12]
    assert tracker.last_event_absolute_position == 12
    assert tracker.tokens_since_last(15) == 3


def test_restore_compaction_tracker_no_event_case_matches_fresh_tracker_default():
    fake_snapshot = SimpleNamespace(compaction_event_steps=[], tokens_since_last_compaction=7, absolute_position=7)
    tracker = restore_compaction_tracker_from_snapshot(fake_snapshot)
    assert tracker.event_steps == []
    assert tracker.last_event_absolute_position == 0  # matches CompactionTracker()'s own default


def test_snapshot_carries_complete_event_history_after_real_evictions():
    model = FakeEvictingModel(budget=2)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")

    prefill_fn(state, [10, 11])  # positions 0,1 -- hits budget=2 within prefill: event attributed to abs pos 2 (end of the one opaque prefill call)
    decode_fn(state, 20)  # position 2 fed -- evicts again: event at abs pos 3

    snapshot_fn = build_real_snapshot_fn()
    snapshot = snapshot_fn(state)

    assert snapshot.compaction_event_steps == [2, 3]
    assert snapshot.tokens_since_last_compaction == 0

    tracker = restore_compaction_tracker_from_snapshot(snapshot)
    assert tracker.event_steps == [2, 3]
    assert tracker.last_event_absolute_position == 3
    assert tracker.tokens_since_last(snapshot.absolute_position) == 0


def test_branch_step_fn_appends_to_restored_history_never_replaces_it():
    model = FakeEvictingModel(budget=2)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")

    prefill_fn(state, [10, 11])  # event attributed to abs pos 2
    decode_fn(state, 20)  # event at abs pos 3
    snapshot_fn = build_real_snapshot_fn()
    snapshot = snapshot_fn(state)
    assert snapshot.compaction_event_steps == [2, 3]

    step_fn = build_real_branch_step_fn_restore_once(model, "cpu")
    _, live = step_fn(snapshot, 200)  # abs pos 4 fed -- cache stays at budget, fires again

    assert live.compaction.event_steps == [2, 3, 4]  # APPENDED, not replaced/reset to [4]
    assert live.compaction.last_event_absolute_position == 4


def test_restoring_same_snapshot_twice_gives_independent_trackers():
    model = FakeEvictingModel(budget=2)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    prefill_fn(state, [10, 11])  # event attributed to abs pos 2
    snapshot_fn = build_real_snapshot_fn()
    snapshot = snapshot_fn(state)
    assert snapshot.compaction_event_steps == [2]

    step_fn = build_real_branch_step_fn_restore_once(model, "cpu")
    _, branch_a = step_fn(snapshot, 200)
    _, branch_b = step_fn(snapshot, 201)

    assert branch_a.compaction is not branch_b.compaction
    assert branch_a.compaction.event_steps == [2, 3] == branch_b.compaction.event_steps

    # Mutating one branch's tracker must never be visible on the other --
    # baseline and swapped branches cannot mutate each other's tracker.
    branch_a.compaction.event_steps.append(9999)
    assert 9999 not in branch_b.compaction.event_steps
    branch_a.compaction.note_event(9999)
    assert branch_b.compaction.last_event_absolute_position == 3
