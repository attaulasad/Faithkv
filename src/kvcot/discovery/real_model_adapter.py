"""Real-model `PrefillFn`/`DecodeOneFn`/`SnapshotFn` adapters
(`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §11) — the seam a
future, separately-authorized B2A execution plugs into. This module is
CODE, not a stub: every function below is a real, reviewable
implementation built entirely from primitives the PRIMARY pipeline already
uses and has exercised (`kvcot.generation.decode`, `kvcot.generation.state`,
`kvcot.generation.replay`, `kvcot.generation.provenance`) — it never
reimplements the call shape or the eviction-detection logic independently.

**This module is never imported or executed by any test or code path in
this repository as of this pass** — `cmd_b2a_calibrate`'s `--execute` mode
is the only caller, and that mode always hard-stops on a CPU-checkable
precondition (CUDA required; the frozen one-example manifest's
prompt-token identity is unresolved) before it would ever reach here. No
GPU has run this code; no model has been loaded through it. Every
`import torch`/`transformers` reference below is deferred to inside a
function body, matching this repository's existing discipline for
GPU-only code (`kvcot.generation`, `kvcot.cli`).

## Prefill logits, all positions (documented simplification vs.
`kvcot.generation.decode.prefill`)

`kvcot.generation.decode.prefill` (the primary/frozen pipeline's own
prefill call) intentionally returns only the LAST position's logits — the
primary pipeline never needs the earlier ones. This module's own
`_prefill_forward_all_positions` issues the IDENTICAL forward call (same
`input_ids`/`position_ids`/`cache_position`/`use_cache` construction,
verified line-for-line against `kvcot.generation.decode.prefill`) but reads
`out.logits[0, :, :]` (every position) instead of `out.logits[0, -1, :]` —
required because `PrefillFn`'s contract needs one logits tensor per prompt
position (`kvcot.discovery.harness_types.PrefillStepResult`). This is
additive (a different slice of the SAME forward call's output), never a
second, differently-shaped forward call.

## Per-position compaction attribution during prefill (documented limit)

A real transformer prefill call is opaque: R-KV's `kept_token_indices`
bookkeeping only tells us whether at least one compaction fired somewhere
across the whole prompt, never at which exact intra-prompt position. This
adapter attributes any prefill-phase compaction to the LAST prompt
position — harmless for this harness's purposes, because
`kvcot.discovery.pass1.eligible_event_ids` already excludes every
prefill-phase event from target selection (B1B-R2 §6): no prefill-phase
observation this adapter produces is ever used to build a branch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcot.discovery.harness_types import (
    DecodeOneFn,
    LayerStepObservation,
    NaturalStepResult,
    PrefillFn,
    PrefillStepResult,
    SnapshotFn,
)


@dataclass
class RealModelState:
    """The `state` object threaded through Pass 1/Pass 2 for a REAL model —
    bundles exactly what `kvcot.generation.replay.replay_and_snapshot`
    already bundles as local variables, so the same reset/sync helpers
    apply unchanged. `kv_cluster_for_layer` is the one method
    `kvcot.discovery.pass2.run_pass2_capture` requires of any state object.
    """

    model: Any
    cache: Any
    model_provenance: Any  # kvcot.generation.provenance.ModelProvenance
    compaction: Any  # kvcot.generation.replay.CompactionTracker
    absolute_position: int
    device: str

    def kv_cluster_for_layer(self, layer_index: int):
        return self.model.model.layers[layer_index].self_attn.kv_cluster


def _prefill_forward_all_positions(model, cache, prompt_token_ids: list[int], device: str):
    """IDENTICAL call shape to `kvcot.generation.decode.prefill` (verified
    line-for-line against it), except this reads every position's logits,
    not only the last. Never called from the primary pipeline — only from
    this module's `PrefillFn`."""
    import torch

    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    n = len(prompt_token_ids)
    position_ids = torch.arange(0, n, device=device).unsqueeze(0)
    cache_position = torch.arange(0, n, device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
    return out.logits[0, :, :]  # (n, vocab) -- one row per prompt position


def _layer_observations_for_all_layers(
    model,
    cache,
    model_provenance,
    kept_indices_lengths: dict[int, int],
    expected_len_if_no_evict: dict[int, int],
    absolute_position_after: int,
) -> dict[int, LayerStepObservation]:
    """Builds one `LayerStepObservation` per layer for a single forward
    call, reusing `kvcot.generation.replay`'s own eviction-detection helper
    (`_sync_layer_after_call`) — the SAME cross-checked logic the primary
    pipeline already relies on, never reimplemented independently here.
    `kept_indices_lengths` is mutated in place (threaded across calls,
    exactly like `replay_and_snapshot`'s own local variable of the same
    name)."""
    from kvcot.generation.replay import _has_kv_cluster, _sync_layer_after_call

    num_layers = len(model.model.layers)
    observations: dict[int, LayerStepObservation] = {}
    for layer_idx in range(num_layers):
        pre_event_map = model_provenance.layers[layer_idx].positions.clone()
        event_fired = _sync_layer_after_call(
            model,
            cache,
            layer_idx,
            model_provenance,
            kept_indices_lengths,
            expected_len_if_no_evict=expected_len_if_no_evict[layer_idx],
            absolute_position_after=absolute_position_after,
        )
        kv_cluster = _has_kv_cluster(model, layer_idx)
        observed_kept = None
        window_size = None
        if event_fired and kv_cluster is not None:
            observed_kept = kv_cluster.kept_token_indices[-1].clone()
            window_size = kv_cluster.window_size
        observations[layer_idx] = LayerStepObservation(
            had_compaction=event_fired,
            cache_length_after=int(cache.key_cache[layer_idx].shape[-2]),
            pre_event_absolute_position_map=pre_event_map if event_fired else None,
            observed_kept_absolute_positions=observed_kept,
            window_size=window_size,
        )
    return observations


def build_real_prefill_fn(device: str) -> PrefillFn:
    """Real-model `PrefillFn` — one opaque forward call over the complete
    prompt (B1B-R2 §6), never repeated one-token calls. `state` must be a
    `RealModelState` whose `cache` is freshly constructed
    (`kvcot.generation.state.reset_patched_state`) and whose
    `model_provenance` already has every layer's positions appended via
    `LayerProvenance.append_new_tokens_prefill` — matching
    `kvcot.generation.replay.replay_and_snapshot`'s own prefill ordering."""

    def prefill_fn(state: RealModelState, prompt_token_ids) -> PrefillStepResult:
        prompt_token_ids = list(prompt_token_ids)
        num_layers = len(state.model.model.layers)
        expected_len_if_no_evict = {i: len(prompt_token_ids) for i in range(num_layers)}
        kept_indices_lengths: dict[int, int] = {i: 0 for i in range(num_layers)}

        all_position_logits = _prefill_forward_all_positions(
            state.model, state.cache, prompt_token_ids, device
        )

        # Per-position observations: intra-prompt attribution is not
        # available from one opaque call (see module docstring) -- every
        # position except the last reports had_compaction=False; the
        # single real eviction-detection pass runs once, after the whole
        # call, attributed to the LAST prompt position.
        per_position_observations = [
            {i: LayerStepObservation(had_compaction=False, cache_length_after=int(state.cache.key_cache[i].shape[-2]))
             for i in range(num_layers)}
            for _ in range(len(prompt_token_ids) - 1)
        ]
        last_position_observations = _layer_observations_for_all_layers(
            state.model, state.cache, state.model_provenance, kept_indices_lengths, expected_len_if_no_evict,
            absolute_position_after=len(prompt_token_ids),
        )
        per_position_observations.append(last_position_observations)

        state.absolute_position = len(prompt_token_ids)
        return PrefillStepResult(
            new_state=state,
            per_position_logits=tuple(all_position_logits[i] for i in range(len(prompt_token_ids))),
            per_position_layer_observations=tuple(per_position_observations),
        )

    return prefill_fn


def build_real_decode_one_fn(device: str) -> DecodeOneFn:
    """Real-model `DecodeOneFn` — one single-token forward call, matching
    `kvcot.generation.decode.decode_step`'s exact call shape (imported and
    reused directly, never reimplemented)."""

    def decode_one_fn(state: RealModelState, token_id: int) -> NaturalStepResult:
        from kvcot.generation.decode import decode_step

        num_layers = len(state.model.model.layers)
        expected_len_if_no_evict = {
            i: int(state.cache.key_cache[i].shape[-2]) + 1 for i in range(num_layers)
        }
        kept_indices_lengths = {i: len(state.model.model.layers[i].self_attn.kv_cluster.kept_token_indices)
                                 if getattr(state.model.model.layers[i].self_attn, "kv_cluster", None) is not None
                                 else 0
                                 for i in range(num_layers)}

        logits = decode_step(state.model, state.cache, token_id, state.absolute_position, device)
        state.absolute_position += 1
        # Per-token position bookkeeping (LayerProvenance.append_new_token)
        # is the CALLER's job (kvcot.discovery.pass1/pass2), matching
        # kvcot.generation.replay.replay_and_snapshot's own division of
        # responsibility -- this adapter only reports what happened on
        # THIS call, never mutates provenance position bookkeeping itself.

        observations = _layer_observations_for_all_layers(
            state.model, state.cache, state.model_provenance, kept_indices_lengths, expected_len_if_no_evict,
            absolute_position_after=state.absolute_position,
        )
        return NaturalStepResult(next_token_logits=logits, new_state=state, layer_observations=observations)

    return decode_one_fn


def build_real_snapshot_fn() -> SnapshotFn:
    """Real-model `SnapshotFn` — delegates entirely to
    `kvcot.generation.replay.capture_snapshot`, the SAME complete-state
    capture the primary pipeline already uses (never a second,
    independently-written snapshot constructor)."""

    def snapshot_fn(state: RealModelState):
        from kvcot.generation.replay import capture_snapshot

        return capture_snapshot(
            state.model, state.cache, state.model_provenance, state.compaction, state.absolute_position
        )

    return snapshot_fn
