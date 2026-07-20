"""Pass 2 — token-identical replay and targeted capture
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9, authorized by CLAUDE.md
§1b/§4b: CPU-side harness architecture only, exercised against a
dependency-injected `NaturalStepFn`, never a real model).

Pass 2 replays EXACTLY the token sequence Pass 1 froze (teacher-forced,
never re-decided by argmax/sampling here), from a fresh
(`state.kv_cluster_for_layer` must be a NEWLY constructed per-layer
bookkeeping object, never the Pass-1 instance reused) state. It never
reconstructs evicted states from FullKV, and it never modifies the pinned
R-KV submodule -- absolute survivor identity for the targeted
(layer, kv_head) pairs is obtained entirely through
`kvcot.discovery.capture.capture_update_kv`, wired to a FRESH
`kvcot.generation.provenance.LayerProvenance` built during this exact
replay, exactly the contract Blocker 2 repaired.

Any of the following invalidates the WHOLE example (never a partial/
degraded result): a token mismatch against Pass 1's frozen trace, a
compaction-event-position mismatch at a targeted layer, an observed-
survivor mismatch (either the within-Pass-2 recomputed-vs-observed parity
`capture_update_kv` itself performs, or a cross-pass mismatch against Pass
1's own recorded survivors for the same event/layer/head).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from kvcot.discovery.capture import UpdateKvCaptureRecord, capture_update_kv
from kvcot.discovery.harness_types import NaturalStepFn
from kvcot.discovery.pass1 import EventPlan, Pass1Plan
from kvcot.generation.provenance import LayerProvenance

INVALID_TOKEN_MISMATCH = "pass2_token_mismatch"
INVALID_COMPACTION_POSITION_MISMATCH = "pass2_compaction_event_position_mismatch"
INVALID_SURVIVOR_MISMATCH_WITHIN_PASS2 = "pass2_observed_survivor_parity_failed"
INVALID_SURVIVOR_MISMATCH_ACROSS_PASSES = "pass2_survivor_mismatch_vs_pass1"
INVALID_MISSING_TARGET_CAPTURE = "pass2_missing_target_capture_record"


@dataclass(frozen=True)
class TargetCapture:
    event_plan: EventPlan
    capture_record: UpdateKvCaptureRecord


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
    step_fn: NaturalStepFn,
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

    target_layers = sorted({ev.layer_index for ev in pass1_plan.events})
    events_by_layer: dict[int, list[EventPlan]] = {}
    for ev in pass1_plan.events:
        events_by_layer.setdefault(ev.layer_index, []).append(ev)

    # (1): reset all mutable state -- a fresh LayerProvenance per targeted
    # layer, built entirely from THIS replay, never carried over from Pass 1.
    layer_provenance: dict[int, LayerProvenance] = {}
    sinks: dict[int, list[UpdateKvCaptureRecord]] = {layer: [] for layer in target_layers}
    # `capture_update_kv` appends to a sink only when the wrapped
    # `update_kv` is ACTUALLY called -- which, matching the real system's
    # `divide_method=step_length` schedule, is not every position. A sink
    # LIST INDEX is therefore never the same thing as an absolute position;
    # this dict is built by watching the sink grow position-by-position, so
    # a specific event's record can be looked up by the absolute position
    # it actually happened at, never by assuming index == position.
    records_by_position: dict[int, dict[int, UpdateKvCaptureRecord]] = {layer: {} for layer in target_layers}

    state = initial_state
    with contextlib.ExitStack() as stack:
        kv_clusters: dict[int, Any] = {}
        for layer_index in target_layers:
            kv_cluster = state.kv_cluster_for_layer(layer_index)
            kv_clusters[layer_index] = kv_cluster

            def _map_fn(_layer_index=layer_index):
                lp = layer_provenance.get(_layer_index)
                return lp.positions if lp is not None else None

            stack.enter_context(capture_update_kv(kv_cluster, sinks[layer_index], pre_event_position_map_fn=_map_fn))

        for pos, token_id in enumerate(replay_token_ids):
            for layer_index in target_layers:
                if layer_index not in layer_provenance:
                    num_heads = kv_clusters[layer_index].num_key_value_heads
                    layer_provenance[layer_index] = LayerProvenance.empty(num_heads)
                layer_provenance[layer_index].append_new_token(pos)

            sink_lengths_before = {layer_index: len(sinks[layer_index]) for layer_index in target_layers}
            result = step_fn(state, token_id)
            state = result.new_state

            for layer_index in target_layers:
                if len(sinks[layer_index]) > sink_lengths_before[layer_index]:
                    records_by_position[layer_index][pos] = sinks[layer_index][-1]
                obs = result.layer_observations.get(layer_index)
                if obs is not None and obs.had_compaction:
                    layer_provenance[layer_index].adopt_upstream_kept_indices(obs.observed_kept_absolute_positions)

    # (4)/(5)/(9)/(10): for each selected event, locate this replay's
    # capture record at that event's absolute position and cross-check it.
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

        target_captures.append(TargetCapture(event_plan=ev, capture_record=record))

    return Pass2Result(True, None, replay_token_ids, tuple(target_captures))
