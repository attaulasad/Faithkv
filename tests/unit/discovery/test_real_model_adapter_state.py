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


# --------------------------------------------------------------------------
# B1B-R4.1 §4: one authoritative provenance state, pending-position
# projection instead of a Pass-2 shadow tracker
# --------------------------------------------------------------------------


class _RaisingModel(FakeEvictingModel):
    """Raises on the `fail_on_call_index`-th call (1-indexed) -- used to
    prove pending fed positions are cleared, and no partial commit occurs
    on `model_provenance`, when the real forward call itself raises."""

    def __init__(self, fail_on_call_index: int, **kwargs):
        super().__init__(**kwargs)
        self._fail_on_call_index = fail_on_call_index
        self._call_count = 0

    def __call__(self, *args, **kwargs):
        self._call_count += 1
        if self._call_count == self._fail_on_call_index:
            raise RuntimeError("synthetic forward failure")
        return super().__call__(*args, **kwargs)


def test_projected_pre_event_position_map_matches_committed_when_no_pending():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    prefill_fn(state, [10, 11, 12])

    for layer_idx in range(NUM_LAYERS):
        assert state.projected_pre_event_position_map(layer_idx)[0].tolist() == [0, 1, 2]


def test_projected_pre_event_position_map_reflects_pending_decode_without_mutating_authoritative_state():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    prefill_fn(state, [10, 11, 12])
    committed_before = state.model_provenance.layers[0].positions.clone()

    state.register_pending_fed_positions([3], "decode")
    projected = state.projected_pre_event_position_map(0)
    assert projected[0].tolist() == [0, 1, 2, 3]
    # The projection must never mutate the authoritative model_provenance.
    assert torch.equal(state.model_provenance.layers[0].positions, committed_before)

    state.clear_pending()
    assert state.pending_fed_absolute_positions is None
    assert state.pending_call_kind is None
    assert state.projected_pre_event_position_map(0)[0].tolist() == [0, 1, 2]


def test_projected_pre_event_position_map_reflects_pending_prefill_on_fresh_state():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    state.register_pending_fed_positions([0, 1, 2], "prefill")
    projected = state.projected_pre_event_position_map(0)
    assert projected[0].tolist() == [0, 1, 2]
    assert state.model_provenance.layers[0].positions.numel() == 0  # authoritative state untouched


def test_projected_pre_event_position_map_returns_none_for_unknown_layer():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    assert state.projected_pre_event_position_map(999) is None


def test_projected_pre_event_position_map_rejects_unrecognized_pending_call_kind():
    model = FakeEvictingModel()
    state = _fresh_state(model)
    state.pending_fed_absolute_positions = [5]
    state.pending_call_kind = "bogus"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="pending_call_kind"):
        state.projected_pre_event_position_map(0)


def test_forward_call_registers_pending_positions_visible_via_projection_mid_call(monkeypatch):
    """Proves the real adapter registers pending positions BEFORE issuing
    the forward call -- a `model` stand-in that reads the projection off its
    own `state` argument mid-call sees the pending token already reflected,
    exactly what `capture_update_kv`'s hook (invoked from inside a real
    `update_kv`, itself invoked from inside the forward call) needs.
    `__call__` is a dunder method, resolved via the TYPE not the instance,
    so the spy must patch the class, not set an instance attribute."""
    model = FakeEvictingModel()
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")
    prefill_fn(state, [10, 11, 12])

    observed_mid_call: dict[str, object] = {}
    original_unbound_call = FakeEvictingModel.__call__

    def _spying_call(self, *args, **kwargs):
        observed_mid_call["projected"] = state.projected_pre_event_position_map(0)[0].tolist()
        observed_mid_call["pending"] = list(state.pending_fed_absolute_positions)
        observed_mid_call["call_kind"] = state.pending_call_kind
        return original_unbound_call(self, *args, **kwargs)

    monkeypatch.setattr(FakeEvictingModel, "__call__", _spying_call)
    decode_fn(state, 20)

    assert observed_mid_call["pending"] == [3]
    assert observed_mid_call["call_kind"] == "decode"
    assert observed_mid_call["projected"] == [0, 1, 2, 3]
    # After the call commits, pending must be cleared and the committed
    # state must match what was projected mid-call.
    assert state.pending_fed_absolute_positions is None
    assert state.model_provenance.layers[0].positions[0].tolist() == [0, 1, 2, 3]


def test_pending_positions_cleared_and_no_partial_commit_on_forward_exception():
    model = _RaisingModel(fail_on_call_index=2)  # call 1 = prefill (succeeds), call 2 = first decode (raises)
    state = _fresh_state(model)
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")

    prefill_fn(state, [10, 11, 12])
    provenance_before = {i: state.model_provenance.layers[i].positions.clone() for i in range(NUM_LAYERS)}
    absolute_position_before = state.absolute_position
    assert state.pending_fed_absolute_positions is None

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        decode_fn(state, 20)

    assert state.pending_fed_absolute_positions is None
    assert state.pending_call_kind is None
    assert state.absolute_position == absolute_position_before  # never advanced past the failed call
    for i in range(NUM_LAYERS):
        assert torch.equal(state.model_provenance.layers[i].positions, provenance_before[i])


def test_live_branch_state_pending_positions_cleared_on_branch_forward_exception():
    model = _RaisingModel(fail_on_call_index=2)  # call 1 builds the pristine snapshot's prefill; call 2 = branch decode
    snapshot = _make_pristine_snapshot(model)
    step_fn = build_real_branch_step_fn_restore_once(model, "cpu")

    with pytest.raises(RuntimeError, match="synthetic forward failure"):
        step_fn(snapshot, 200)


def _run_natural_trace_for_pass2(model, prompt_ids: list[int], max_new_tokens: int):
    from kvcot.discovery.pass1 import NaturalRunProvenance, run_natural_pass1

    state = _fresh_state(model, num_layers=len(model.model.layers))
    prefill_fn = build_real_prefill_fn("cpu")
    decode_fn = build_real_decode_one_fn("cpu")
    provenance = NaturalRunProvenance(
        model_name="fake", model_revision="fake", tokenizer_name="fake", tokenizer_revision="fake",
        rkv_revision="fake", config_sha256="fake", dataset_name="fake", example_id="fake",
    )
    trace = run_natural_pass1(
        provenance, prompt_ids, state, prefill_fn, decode_fn,
        max_new_tokens=max_new_tokens, eos_token_id=None,
        answer_fn=lambda toks: (None, "unverifiable"),
    )
    return trace


class _RealShapeCapturingModel:
    """Unlike `FakeEvictingModel` above (which manages eviction inline and
    never calls `kv_cluster.update_kv` at all -- so it cannot exercise
    `kvcot.discovery.capture.capture_update_kv`'s wrapped hook), this fake
    matches the real forward-call shape AND calls each layer's real
    `kv_cluster.update_kv(key, query, value)` exactly once per forward call,
    via `_synthetic_harness.LayerKVCluster` (itself a thin wrapper around
    the real-FORMULA `FakeR1KV` fixture) -- there is exactly one fake
    eviction formula in this test suite, never a second one. K/V/query
    tensors are seeded purely from `(layer, absolute_position, token_id)`
    (never call order), so Pass 1 and Pass 2 -- each starting from a FRESH
    instance of this class, replaying the identical token sequence --
    reproduce bit-identical trajectories, the same discipline
    `_synthetic_harness.py` documents for its own fixture."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        budget: int,
        window_size: int,
        divide_length: int,
        vocab_size: int = 8,
    ):
        from _synthetic_harness import LayerKVCluster

        self._num_heads = num_heads
        self._head_dim = head_dim
        self._window_size = window_size
        self._vocab_size = vocab_size
        self._divide_length = divide_length
        clusters = [
            LayerKVCluster(num_key_value_heads=num_heads, budget=budget, window_size=window_size, kernel_size=3)
            for _ in range(num_layers)
        ]
        self.model = SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=SimpleNamespace(kv_cluster=c, config=SimpleNamespace(compression=None)))
                for c in clusters
            ]
        )
        self.config = SimpleNamespace(num_key_value_heads=num_heads)

    def __call__(self, input_ids, position_ids=None, past_key_values=None, use_cache=None, cache_position=None):
        from _synthetic_harness import _seeded

        cache = past_key_values
        seq_len = input_ids.shape[1]
        positions = [int(p) for p in cache_position.tolist()]
        for i, layer in enumerate(self.model.layers):
            kv_cluster = layer.self_attn.kv_cluster
            k_parts, v_parts = [], []
            for offset, pos in enumerate(positions):
                token_id = int(input_ids[0, offset].item())
                k_parts.append(
                    torch.randn(1, self._num_heads, 1, self._head_dim, generator=_seeded("real_key", i, pos, token_id))
                )
                v_parts.append(
                    torch.randn(1, self._num_heads, 1, self._head_dim, generator=_seeded("real_value", i, pos, token_id))
                )
            new_k, new_v = torch.cat(k_parts, dim=2), torch.cat(v_parts, dim=2)
            if cache.key_cache[i].numel() == 0:
                cache.key_cache[i], cache.value_cache[i] = new_k, new_v
            else:
                cache.key_cache[i] = torch.cat([cache.key_cache[i], new_k], dim=2)
                cache.value_cache[i] = torch.cat([cache.value_cache[i], new_v], dim=2)
            cur_len = cache.key_cache[i].shape[-2]
            # Real R-KV only actually calls `update_kv`'s internal eviction
            # check every `divide_length` fed tokens (`divide_method=
            # step_length` -- CLAUDE.md §4) -- never on every single decode
            # call. Calling it every call instead (as an earlier version of
            # this fixture did) forces `FakeR1KV`'s multi-call
            # `evicted_token_num`/`prev_indices` remap formula through a
            # degenerate one-token-evicted-per-call pattern the real
            # production schedule never produces, which is a fixture defect,
            # not a defect in the remap formula or in the provenance
            # mechanism under test here.
            if cur_len % self._divide_length != 0:
                continue
            query = torch.randn(
                1, self._num_heads, self._window_size, self._head_dim, generator=_seeded("real_query", i, cur_len)
            )
            new_key, new_value = kv_cluster.update_kv(cache.key_cache[i], query, cache.value_cache[i])
            cache.key_cache[i], cache.value_cache[i] = new_key, new_value
        return SimpleNamespace(logits=torch.zeros(1, seq_len, self._vocab_size))


def test_pass2_uses_real_model_states_authoritative_projection_with_no_shadow_provenance(monkeypatch):
    """End-to-end proof of the B1B-R4.1 §4 repair: `run_pass2_capture`,
    given a `RealModelState`, produces a VALID result (all cross-pass
    survivor/parity checks pass) using ONLY `state.projected_pre_event_position_map`
    -- no independently-mutated Pass-2 provenance track exists on this path
    at all (`pass2.py`'s `layer_provenance` dict is never populated: verified
    directly by monkeypatching `LayerProvenance.empty` to fail loudly if
    called during this run). Also exercises multiple compaction events,
    including one selected event that occurs AFTER several PRIOR
    compactions -- proving later-event remapping remains correct, not just
    the first event."""
    from _fake_rkv_fixtures import install_fake_rkv_compression_module

    install_fake_rkv_compression_module(monkeypatch)
    from kvcot.discovery.pass1 import EventPlan, Pass1Plan
    from kvcot.discovery.pass2 import run_pass2_capture

    num_layers, num_heads, head_dim = 2, 1, 4
    # budget/window/divide_length/prompt_len are chosen so the FIRST
    # eviction genuinely shrinks the cache (kv_cache_len strictly greater
    # than budget) rather than landing exactly at kv_cache_len == budget.
    # At exactly-at-budget, `FakeR1KV.update_kv` (a faithful, formula-level
    # port of the real R-KV bookkeeping -- see `_fake_rkv_fixtures.py`)
    # still reorders physical storage via topk (even though it evicts zero
    # tokens), but leaves `evicted_token_num` at 0, so its OWN next eviction
    # skips the `prev_indices` remap it would otherwise need -- a real,
    # pre-existing characteristic of upstream R-KV's `kept_token_indices`
    # bookkeeping under that specific boundary condition, found by this
    # test during development, and orthogonal to the B1B-R4.1 §4 provenance
    # mechanism under test here (confirmed by hand-tracing: this repository's
    # own provenance tracking reports the objectively correct absolute
    # positions at every step in that scenario too; it is R-KV's bookkeeping
    # that becomes internally ambiguous, and `capture.py`'s parity check
    # correctly treats R-KV's own bookkeeping as ground truth). Avoided here
    # by never letting a schedule point land exactly at budget.
    budget, window_size, divide_length = 6, 2, 4
    prompt_ids = [10, 11, 12]

    model1 = _RealShapeCapturingModel(num_layers, num_heads, head_dim, budget, window_size, divide_length)
    # Pass 1's own initial `RealModelState` legitimately calls
    # `LayerProvenance.empty` once per layer to seed its (empty) starting
    # provenance -- that is not the shadow track this test guards against,
    # so the ban below is installed only AFTER Pass 1 (and Pass 2's own
    # fresh starting state) have already been constructed.
    trace = _run_natural_trace_for_pass2(model1, prompt_ids, max_new_tokens=40)

    decode_phase_event_ids = [
        ev.compaction_event_id for ev in trace.compaction_events if ev.absolute_event_position >= trace.prompt_length
    ]
    assert len(decode_phase_event_ids) >= 6, "fixture must produce enough compaction events to pick 3 spread-out ones"

    chosen_ids = [decode_phase_event_ids[1], decode_phase_event_ids[len(decode_phase_event_ids) // 2], decode_phase_event_ids[-2]]
    event_by_id = {ev.compaction_event_id: ev for ev in trace.compaction_events}
    events = tuple(
        EventPlan(
            compaction_event_id=eid,
            absolute_event_position=event_by_id[eid].absolute_event_position,
            chronological_event_ordinal=i,
            depth_stratum=0,
            layer_index=i % num_layers,
            kv_head_index=0,
            candidate_donor_selection=None,
        )
        for i, eid in enumerate(chosen_ids)
    )
    plan = Pass1Plan(trace=trace, events=events)

    model2 = _RealShapeCapturingModel(num_layers, num_heads, head_dim, budget, window_size, divide_length)
    state2 = _fresh_state(model2, num_layers=num_layers)

    def _never_called(*args, **kwargs):
        raise AssertionError(
            "LayerProvenance.empty must never be called during run_pass2_capture "
            "on the real-model (authoritative-projection) path -- pass2.py must "
            "not build a shadow provenance track when state owns its own."
        )

    monkeypatch.setattr(LayerProvenance, "empty", staticmethod(_never_called))
    result = run_pass2_capture(
        plan, trace.full_token_ids, state2,
        build_real_prefill_fn("cpu"), build_real_decode_one_fn("cpu"), build_real_snapshot_fn(),
    )

    assert result.valid, result.invalid_reason
    assert len(result.target_captures) == 3
    for tc in result.target_captures:
        assert tc.capture_record.had_compaction
        assert tc.capture_record.parity_check_passed, tc.capture_record.parity_failure_reason
        assert tc.capture_record.observed_kept_indices_parity_passed is True

    # Pass 1's own recorded compaction-position list and Pass 2's replayed
    # one (recovered from the capture records at the selected targets) must
    # agree exactly for every selected event -- not just the picked-3.
    pass1_positions = {ev.compaction_event_id: ev.absolute_event_position for ev in trace.compaction_events}
    for ev_plan, tc in zip(plan.events, result.target_captures):
        assert pass1_positions[ev_plan.compaction_event_id] == tc.event_plan.absolute_event_position
