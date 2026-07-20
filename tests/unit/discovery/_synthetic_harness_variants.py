"""Deliberately-divergent step-function variants used only by the B1B
integration tests that must prove trajectory/survivor mismatches are
actually detected (scenarios 4 and 5) -- kept in a separate file from
`_synthetic_harness.py` so the "normal" harness stays simple."""
from __future__ import annotations

import torch

from _synthetic_harness import BUDGET, DIVIDE_LENGTH, EOS_TOKEN_ID, HEAD_DIM, NUM_HEADS, VOCAB_SIZE, WINDOW, HarnessState, _seeded
from kvcot.discovery.harness_types import LayerStepObservation, NaturalStepResult


def make_schedule_shifted_step_fn(schedule_offset: int, stop_at_predicted_position: int | None = None):
    """IDENTICAL to `_synthetic_harness.make_natural_step_fn` except
    compaction fires on a schedule shifted by `schedule_offset` positions --
    produces the SAME token trajectory (token generation itself is
    unaffected) but compaction events land at DIFFERENT absolute positions,
    so a Pass-2 replay using this variant against a Pass-1 plan built from
    the unshifted schedule must find a compaction-position mismatch."""

    def step_fn(state: HarnessState, token_id: int) -> NaturalStepResult:
        pos = state.pos
        layer_observations: dict[int, LayerStepObservation] = {}
        is_schedule_point = (pos + 1 + schedule_offset) % DIVIDE_LENGTH == 0

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
            if is_schedule_point and full_k.shape[-2] >= BUDGET:
                query = torch.randn(1, NUM_HEADS, WINDOW, HEAD_DIM, generator=_seeded("query", layer_idx, pos))
                k_out, v_out = kv_cluster.update_kv(full_k, query, full_v)
                had_compaction = True
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
        predicted_position = pos + 1
        if stop_at_predicted_position is not None and predicted_position == stop_at_predicted_position:
            logits[EOS_TOKEN_ID] = 1e9
        else:
            logits[EOS_TOKEN_ID] = -1e9

        state.pos = pos + 1
        return NaturalStepResult(next_token_logits=logits, new_state=state, layer_observations=layer_observations)

    return step_fn


def make_query_salt_step_fn(query_salt: str, stop_at_predicted_position: int | None = None):
    """IDENTICAL to `_synthetic_harness.make_natural_step_fn` except the
    per-call query tensor is seeded with an extra salt -- compaction still
    fires at EXACTLY the same absolute positions (same schedule, same token
    trajectory), but the resulting top-k selection at each event genuinely
    differs, so a Pass-2 replay using this variant must find a survivor
    IDENTITY mismatch (never a position mismatch)."""

    def step_fn(state: HarnessState, token_id: int) -> NaturalStepResult:
        pos = state.pos
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
                query = torch.randn(
                    1, NUM_HEADS, WINDOW, HEAD_DIM, generator=_seeded("query", query_salt, layer_idx, pos)
                )
                k_out, v_out = kv_cluster.update_kv(full_k, query, full_v)
                had_compaction = True
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
        predicted_position = pos + 1
        if stop_at_predicted_position is not None and predicted_position == stop_at_predicted_position:
            logits[EOS_TOKEN_ID] = 1e9
        else:
            logits[EOS_TOKEN_ID] = -1e9

        state.pos = pos + 1
        return NaturalStepResult(next_token_logits=logits, new_state=state, layer_observations=layer_observations)

    return step_fn
