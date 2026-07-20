"""Pass 2 — token-identical replay and targeted capture
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9,
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §4/§5/§6, authorized
by CLAUDE.md §1b/§4b: CPU-side harness architecture only, exercised against
dependency-injected `PrefillFn`/`DecodeOneFn`/`SnapshotFn`, never a real
model).

Pass 2 replays EXACTLY the token sequence Pass 1 froze (teacher-forced,
never re-decided by argmax/sampling here), from a fresh
(`state.kv_cluster_for_layer` must be a NEWLY constructed per-layer
bookkeeping object, never the Pass-1 instance reused) state, using the
IDENTICAL one-shot-prefill-then-one-token-decode call boundary Pass 1 uses
(B1B-R2 §6) — never a prefill simulated as repeated one-token calls. It
never reconstructs evicted states from FullKV, and it never modifies the
pinned R-KV submodule -- absolute survivor identity for the targeted
(layer, kv_head) pairs is obtained entirely through
`kvcot.discovery.capture.capture_update_kv`, wired to a FRESH
`kvcot.generation.provenance.LayerProvenance` built during this exact
replay, exactly the contract Blocker 2 repaired.

## Target-only capture (B1B-R2 §4)

`capture_update_kv` is wired with `should_capture` so that ONLY the exact
preselected (absolute_position, layer_index) pairs Pass 1 selected are ever
captured — every other `update_kv` call (there can be orders of magnitude
more of them over a full generation) passes straight through to the
original method: no clone, no stored tensor, no capture record. Retained
capture state is therefore bounded by the number of selected targets (3),
never by the total number of decode/compaction calls in the run.

## Complete post-event snapshot per target (B1B-R2 §5)

For each selected target event, `snapshot_fn(state)` is called immediately
after that event's `decode_one_fn` call returns, capturing one pristine,
complete `kvcot.generation.state.ModelStateSnapshot` (every transformer
layer's K/V, not just the targeted layer's) — this is the state branch
construction (`kvcot.discovery.pipeline.build_swap_pair_record`) actually
originates from, never a bare single-layer K/V tuple.

Any of the following invalidates the WHOLE example (never a partial/
degraded result): a token mismatch against Pass 1's frozen trace, a
compaction-event-position mismatch at a targeted layer, an observed-
survivor mismatch (either the within-Pass-2 recomputed-vs-observed parity
`capture_update_kv` itself performs, or a cross-pass mismatch against Pass
1's own recorded survivors for the same event/layer/head), or a missing
pristine snapshot for a selected target.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from kvcot.discovery.capture import UpdateKvCaptureRecord, capture_update_kv
from kvcot.discovery.harness_types import DecodeOneFn, PrefillFn, SnapshotFn
from kvcot.discovery.pass1 import EventPlan, Pass1Plan
from kvcot.generation.provenance import LayerProvenance
from kvcot.generation.state import ModelStateSnapshot

INVALID_TOKEN_MISMATCH = "pass2_token_mismatch"
INVALID_COMPACTION_POSITION_MISMATCH = "pass2_compaction_event_position_mismatch"
INVALID_SURVIVOR_MISMATCH_WITHIN_PASS2 = "pass2_observed_survivor_parity_failed"
INVALID_SURVIVOR_MISMATCH_ACROSS_PASSES = "pass2_survivor_mismatch_vs_pass1"
INVALID_MISSING_TARGET_CAPTURE = "pass2_missing_target_capture_record"
INVALID_MISSING_TARGET_SNAPSHOT = "pass2_missing_target_snapshot"
INVALID_PREFILL_SHAPE_MISMATCH = "pass2_prefill_result_shape_mismatch"


@dataclass(frozen=True)
class TargetCapture:
    event_plan: EventPlan
    capture_record: UpdateKvCaptureRecord
    pristine_snapshot: ModelStateSnapshot


@dataclass(frozen=True)
class Pass2Result:
    valid: bool
    invalid_reason: str | None
    replayed_token_ids: tuple[int, ...]
    target_captures: tuple[TargetCapture, ...]


def run_pass2_capture(
    pass1_plan: Pass1Plan,
    replay_token_ids: Sequence[int],
    initial_state: Any,
    prefill_fn: PrefillFn,
    decode_one_fn: DecodeOneFn,
    snapshot_fn: SnapshotFn,
) -> Pass2Result:
    trace = pass1_plan.trace
    expected_tokens = trace.full_token_ids
    replay_token_ids = tuple(replay_token_ids)

    # (2)/(3): require token identity at EVERY replayed position, before
    # any capture instrumentation is even attached.
    if len(replay_token_ids) != len(expected_tokens):
        return Pass2Result(False, INVALID_TOKEN_MISMATCH, replay_token_ids, ())
    for pos, (expected, actual) in enumerate(zip(expected_tokens, replay_token_ids)):
        if expected != actual:
            return Pass2Result(False, INVALID_TOKEN_MISMATCH, replay_token_ids, ())

    prompt_length = trace.prompt_length
    prompt_tokens = replay_token_ids[:prompt_length]
    continuation_tokens = replay_token_ids[prompt_length:]

    target_layers = sorted({ev.layer_index for ev in pass1_plan.events})
    # Every selected event is decode-phase-only by construction
    # (`kvcot.discovery.pass1.eligible_event_ids`), so a target's
    # (position, layer) pair can never coincide with any position visited
    # during the single opaque prefill call.
    target_position_layer_pairs = {(ev.absolute_event_position, ev.layer_index) for ev in pass1_plan.events}
    event_positions = {ev.absolute_event_position for ev in pass1_plan.events}

    layer_provenance: dict[int, LayerProvenance] = {}
    sinks: dict[int, list[UpdateKvCaptureRecord]] = {layer: [] for layer in target_layers}
    records_by_position: dict[int, dict[int, UpdateKvCaptureRecord]] = {layer: {} for layer in target_layers}
    pristine_snapshot_by_position: dict[int, ModelStateSnapshot] = {}

    current_position = {"pos": -1}

    def _should_capture(pos: int, layer: int) -> bool:
        return (pos, layer) in target_position_layer_pairs

    state = initial_state
    with contextlib.ExitStack() as stack:
        kv_clusters: dict[int, Any] = {}
        for layer_index in target_layers:
            kv_cluster = state.kv_cluster_for_layer(layer_index)
            kv_clusters[layer_index] = kv_cluster

            def _map_fn(_layer_index=layer_index):
                lp = layer_provenance.get(_layer_index)
                return lp.positions if lp is not None else None

            stack.enter_context(
                capture_update_kv(
                    kv_cluster,
                    sinks[layer_index],
                    pre_event_position_map_fn=_map_fn,
                    layer_idx=layer_index,
                    current_position_fn=lambda: current_position["pos"],
                    should_capture=_should_capture,
                )
            )

        for layer_index in target_layers:
            layer_provenance[layer_index] = LayerProvenance.empty(kv_clusters[layer_index].num_key_value_heads)

        # (1)/reset: a fresh LayerProvenance per targeted layer, built
        # entirely from THIS replay, never carried over from Pass 1.

        # ---- prefill: exactly one call for the complete prompt ----
        for layer_index in target_layers:
            layer_provenance[layer_index].append_new_tokens_prefill(list(range(prompt_length)))

        prefill_result = prefill_fn(state, prompt_tokens)
        if (
            len(prefill_result.per_position_logits) != prompt_length
            or len(prefill_result.per_position_layer_observations) != prompt_length
        ):
            return Pass2Result(False, INVALID_PREFILL_SHAPE_MISMATCH, replay_token_ids, ())
        state = prefill_result.new_state

        # Provenance bookkeeping only -- no target event can ever fall
        # inside the prefill (decode-phase-only eligibility), so no capture
        # record is expected here regardless of whether a compaction fired.
        for pos in range(prompt_length):
            obs_by_layer = prefill_result.per_position_layer_observations[pos]
            for layer_index in target_layers:
                obs = obs_by_layer.get(layer_index)
                if obs is not None and obs.had_compaction:
                    layer_provenance[layer_index].adopt_upstream_kept_indices(obs.observed_kept_absolute_positions)

        # ---- decode: one single-token call per continuation token ----
        for offset, token_id in enumerate(continuation_tokens):
            pos = prompt_length + offset
            current_position["pos"] = pos
            for layer_index in target_layers:
                layer_provenance[layer_index].append_new_token(pos)

            sink_lengths_before = {layer_index: len(sinks[layer_index]) for layer_index in target_layers}
            result = decode_one_fn(state, token_id)
            state = result.new_state

            for layer_index in target_layers:
                if len(sinks[layer_index]) > sink_lengths_before[layer_index]:
                    records_by_position[layer_index][pos] = sinks[layer_index][-1]
                obs = result.layer_observations.get(layer_index)
                if obs is not None and obs.had_compaction:
                    layer_provenance[layer_index].adopt_upstream_kept_indices(obs.observed_kept_absolute_positions)

            if pos in event_positions:
                pristine_snapshot_by_position[pos] = snapshot_fn(state)

    # (4)/(5)/(9)/(10): for each selected event, locate this replay's
    # capture record and pristine snapshot at that event's absolute
    # position and cross-check them.
    target_captures: list[TargetCapture] = []
    for ev in pass1_plan.events:
        record = records_by_position[ev.layer_index].get(ev.absolute_event_position)
        if record is None:
            return Pass2Result(False, INVALID_MISSING_TARGET_CAPTURE, replay_token_ids, ())

        if not record.had_compaction:
            return Pass2Result(False, INVALID_COMPACTION_POSITION_MISMATCH, replay_token_ids, ())
        # `parity_check_passed` covers BOTH gather parity (the returned
        # compressed cache actually matches the independently recomputed
        # gather) and absolute-survivor-identity parity together -- a
        # candidate/donor extracted from a record that failed either half
        # cannot be trusted, so the whole example must be invalidated.
        if not record.parity_check_passed:
            return Pass2Result(False, INVALID_SURVIVOR_MISMATCH_WITHIN_PASS2, replay_token_ids, ())

        pass1_layer_obs = trace.compaction_events[ev.compaction_event_id].layer_observations.get(ev.layer_index)
        if pass1_layer_obs is None or pass1_layer_obs.observed_kept_absolute_positions is None:
            return Pass2Result(False, INVALID_SURVIVOR_MISMATCH_ACROSS_PASSES, replay_token_ids, ())
        pass1_head_positions = pass1_layer_obs.observed_kept_absolute_positions[ev.kv_head_index]
        pass2_head_positions = record.observed_kept_absolute_positions[ev.kv_head_index]
        if pass1_head_positions.shape != pass2_head_positions.shape or not torch.equal(
            pass1_head_positions, pass2_head_positions
        ):
            return Pass2Result(False, INVALID_SURVIVOR_MISMATCH_ACROSS_PASSES, replay_token_ids, ())

        pristine_snapshot = pristine_snapshot_by_position.get(ev.absolute_event_position)
        if pristine_snapshot is None:
            return Pass2Result(False, INVALID_MISSING_TARGET_SNAPSHOT, replay_token_ids, ())

        target_captures.append(
            TargetCapture(event_plan=ev, capture_record=record, pristine_snapshot=pristine_snapshot)
        )

    return Pass2Result(True, None, replay_token_ids, tuple(target_captures))
