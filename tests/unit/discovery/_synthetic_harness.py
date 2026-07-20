"""Shared synthetic multi-layer harness for the B1B/B1B-R2 integration tests
(`tests/unit/discovery/test_b1b_integration.py`). Deterministic, seeded
purely from `(layer, absolute_position, token_id)` tuples so Pass 1 and
Pass 2 (each starting from a FRESH state) reproduce bit-identical
trajectories when replaying the same token sequence -- never a real model,
never any network access.

Mirrors real R-KV's actual schedule shape: `update_kv` is invoked only
every `DIVIDE_LENGTH` tokens (matching `divide_method=step_length`), not on
every single forward call -- otherwise every call past the first eviction
would immediately re-evict exactly one token (post-eviction length is
already at budget), giving pools far too small for 2-candidate/2-donor
sampling to ever succeed.

## Prefill/decode split (B1B-R2 §6)

`make_step_fns` returns `(prefill_fn, decode_one_fn)` sharing one per-
position core (`_process_position`) -- `prefill_fn` is the ONE call Pass 1/
Pass 2 make for the complete prompt (looping over prompt positions
internally, but invoked exactly once by the caller); `decode_one_fn` is
called once per continuation token thereafter. Both variants below
(`_synthetic_harness_variants.py`) follow the identical split.

## Complete multi-layer branch state (B1B-R2 §5)

`build_snapshot_from_state` builds a complete
`kvcot.generation.state.ModelStateSnapshot` (every layer's K/V, plus each
layer's `kv_cluster` bookkeeping) from a `HarnessState` -- this is what
`kvcot.discovery.pass2.run_pass2_capture`'s injected `snapshot_fn` uses, and
what `branch_step_fn` below now scores from (never a bare single-layer K/V
tuple).
"""
from __future__ import annotations

import copy
import hashlib
import sys
import types
from typing import Sequence

import torch

from kvcot.discovery.harness_types import LayerStepObservation, NaturalStepResult, PrefillStepResult
from kvcot.generation.state import ModelStateSnapshot

NUM_LAYERS = 4
NUM_HEADS = 2
HEAD_DIM = 4
WINDOW = 3
BUDGET = 8
DIVIDE_LENGTH = 20
VOCAB_SIZE = 64
EOS_TOKEN_ID = 63


def install_fake_rkv_compression_module(monkeypatch) -> None:
    from _fake_rkv_fixtures import fake_cal_similarity, fake_compute_attention_scores

    fake_rkv = types.ModuleType("rkv")
    fake_compression = types.ModuleType("rkv.compression")
    fake_compression.cal_similarity = fake_cal_similarity
    fake_compression.compute_attention_scores = fake_compute_attention_scores
    fake_rkv.compression = fake_compression
    monkeypatch.setitem(sys.modules, "rkv", fake_rkv)
    monkeypatch.setitem(sys.modules, "rkv.compression", fake_compression)


def _seeded(*parts: object) -> torch.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big", signed=False) % (2**63)
    return torch.Generator().manual_seed(seed)


class LayerKVCluster:
    """A `FakeR1KV`-shaped per-layer cluster with `num_key_value_heads`
    added -- the one extra attribute `kvcot.discovery.pass2.run_pass2_capture`
    needs to build a fresh `LayerProvenance` for a targeted layer."""

    def __init__(self, num_key_value_heads: int, budget: int, window_size: int, kernel_size: int = 3):
        from _fake_rkv_fixtures import FakeR1KV

        self._inner = FakeR1KV(
            budget=budget, window_size=window_size, kernel_size=kernel_size, record_kept_token_indices=True
        )
        self.num_key_value_heads = num_key_value_heads

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def update_kv(self, key_states, query_states, value_states):
        return self._inner.update_kv(key_states, query_states, value_states)


class HarnessState:
    def __init__(self, num_layers: int = NUM_LAYERS):
        self.layer_kv_clusters = {
            i: LayerKVCluster(num_key_value_heads=NUM_HEADS, budget=BUDGET, window_size=WINDOW)
            for i in range(num_layers)
        }
        self.layer_full_kv: dict[int, tuple[torch.Tensor, torch.Tensor] | tuple[None, None]] = {
            i: (None, None) for i in range(num_layers)
        }
        self.layer_positions: dict[int, torch.Tensor | None] = {i: None for i in range(num_layers)}
        self.pos = 0

    def kv_cluster_for_layer(self, layer_index: int):
        return self.layer_kv_clusters[layer_index]


def fresh_state_factory(num_layers: int = NUM_LAYERS):
    return lambda: HarnessState(num_layers=num_layers)


def build_snapshot_from_state(state: HarnessState) -> ModelStateSnapshot:
    """`SnapshotFn` implementation: a complete, deep-cloned
    `ModelStateSnapshot` covering every layer -- never just the targeted
    layer's K/V. Mirrors `kvcot.generation.replay.capture_snapshot`'s own
    field-by-field construction, adapted to this harness's synthetic
    per-layer bookkeeping shape."""
    num_layers = len(state.layer_full_kv)
    key_cache = [state.layer_full_kv[i][0].clone() for i in range(num_layers)]
    value_cache = [state.layer_full_kv[i][1].clone() for i in range(num_layers)]

    kv_cluster_bookkeeping: list[dict] = []
    for i in range(num_layers):
        kv_cluster = state.layer_kv_clusters[i]
        kv_cluster_bookkeeping.append(
            {
                "evicted_token_num": kv_cluster.evicted_token_num,
                "kept_token_indices": copy.deepcopy(kv_cluster.kept_token_indices),
                "kept_attention_scores": copy.deepcopy(kv_cluster.kept_attention_scores),
                "kept_similarity_scores": copy.deepcopy(getattr(kv_cluster, "kept_similarity_scores", [])),
                "kept_final_scores": copy.deepcopy(getattr(kv_cluster, "kept_final_scores", [])),
            }
        )

    return ModelStateSnapshot(
        key_cache=key_cache,
        value_cache=value_cache,
        query_cache={},
        compression_flags_per_layer=["none"] * num_layers,
        model_length=state.pos,
        after_think=None,
        compaction_event_steps=[],
        tokens_since_last_compaction=0,
        absolute_position=state.pos,
        provenance=None,
        kv_cluster_bookkeeping_per_layer=kv_cluster_bookkeeping,
    )


def _process_position(state: HarnessState, pos: int, token_id: int) -> tuple[torch.Tensor, dict[int, LayerStepObservation]]:
    """One position's worth of per-layer K/V append + (maybe) compaction,
    shared identically by both the prefill loop and single-token decode --
    the ONLY difference between prefill and decode is how many times this
    is called and by which public function, never the per-position logic
    itself."""
    layer_observations: dict[int, LayerStepObservation] = {}
    is_schedule_point = (pos + 1) % DIVIDE_LENGTH == 0

    for layer_idx, kv_cluster in state.layer_kv_clusters.items():
        new_k = torch.randn(1, NUM_HEADS, 1, HEAD_DIM, generator=_seeded("key", layer_idx, pos, token_id))
        new_v = torch.randn(1, NUM_HEADS, 1, HEAD_DIM, generator=_seeded("value", layer_idx, pos, token_id))
        full_k, full_v = state.layer_full_kv[layer_idx]
        full_k = new_k if full_k is None else torch.cat([full_k, new_k], dim=2)
        full_v = new_v if full_v is None else torch.cat([full_v, new_v], dim=2)

        pos_col = torch.full((NUM_HEADS, 1), pos, dtype=torch.long)
        positions = state.layer_positions[layer_idx]
        positions = pos_col if positions is None else torch.cat([positions, pos_col], dim=1)
        pre_event_map = positions.clone()

        had_compaction = False
        observed_kept = None
        if is_schedule_point:
            query = torch.randn(1, NUM_HEADS, WINDOW, HEAD_DIM, generator=_seeded("query", layer_idx, pos))
            k_out, v_out = kv_cluster.update_kv(full_k, query, full_v)
            had_compaction = True  # is_schedule_point implies accumulated length >= BUDGET by construction
            observed_kept = kv_cluster.kept_token_indices[-1].clone()
            positions = observed_kept.clone()
            full_k, full_v = k_out, v_out

        state.layer_full_kv[layer_idx] = (full_k, full_v)
        state.layer_positions[layer_idx] = positions
        layer_observations[layer_idx] = LayerStepObservation(
            had_compaction=had_compaction,
            cache_length_after=full_k.shape[-2],
            pre_event_absolute_position_map=pre_event_map if had_compaction else None,
            observed_kept_absolute_positions=observed_kept,
            window_size=WINDOW,
        )

    logits = torch.randn(VOCAB_SIZE, generator=_seeded("logits", pos, token_id))
    return logits, layer_observations


def _apply_eos_forcing(logits: torch.Tensor, predicted_position: int, stop_at_predicted_position: int | None) -> torch.Tensor:
    if stop_at_predicted_position is not None and predicted_position == stop_at_predicted_position:
        logits[EOS_TOKEN_ID] = 1e9
    else:
        logits[EOS_TOKEN_ID] = -1e9
    return logits


def make_step_fns(stop_at_predicted_position: int | None = None):
    """Returns `(prefill_fn, decode_one_fn)` -- the exact `PrefillFn`/
    `DecodeOneFn` pair Pass 1/Pass 2 call. `stop_at_predicted_position`:
    when the logits about to be returned would be used to select the token
    at exactly this absolute position, force `EOS_TOKEN_ID` to be the
    argmax -- deterministic, controlled generation length for test
    scenarios that need a clean, non-cap-hit natural run. `None` disables
    forcing (EOS never fires; used by tests that supply their own fixed
    token sequence and never call argmax)."""

    def prefill_fn(state: HarnessState, prompt_token_ids: Sequence[int]) -> PrefillStepResult:
        per_position_logits: list[torch.Tensor] = []
        per_position_obs: list[dict[int, LayerStepObservation]] = []
        for token_id in prompt_token_ids:
            pos = state.pos
            logits, obs = _process_position(state, pos, token_id)
            logits = _apply_eos_forcing(logits, pos + 1, stop_at_predicted_position)
            per_position_logits.append(logits)
            per_position_obs.append(obs)
            state.pos = pos + 1
        return PrefillStepResult(
            new_state=state,
            per_position_logits=tuple(per_position_logits),
            per_position_layer_observations=tuple(per_position_obs),
        )

    def decode_one_fn(state: HarnessState, token_id: int) -> NaturalStepResult:
        pos = state.pos
        logits, obs = _process_position(state, pos, token_id)
        logits = _apply_eos_forcing(logits, pos + 1, stop_at_predicted_position)
        state.pos = pos + 1
        return NaturalStepResult(next_token_logits=logits, new_state=state, layer_observations=obs)

    return prefill_fn, decode_one_fn


def branch_step_fn(snapshot: ModelStateSnapshot, token_id):
    """Dependency-injected `kvcot.discovery.branch_eval.StepFn`: scores a
    branch from its COMPLETE multi-layer `ModelStateSnapshot` (every layer's
    K/V, never a single layer), deterministically seeded from `(cache
    content hash, token_id)` so a genuine no-op swap (bit-identical
    snapshot) reproduces bit-identical logits, and any real content change
    at ANY layer changes the seed."""
    content_fingerprint = int(
        hashlib.sha256(
            b"".join(
                t.detach().contiguous().numpy().tobytes() for t in list(snapshot.key_cache) + list(snapshot.value_cache)
            )
        ).hexdigest()[:8],
        16,
    )
    logits = torch.randn(VOCAB_SIZE, generator=_seeded("branch_logits", content_fingerprint, token_id))
    return logits, snapshot
