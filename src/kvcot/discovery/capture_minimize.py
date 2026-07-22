"""Selected-capture minimization (B1B-R4 §18). `kvcot.discovery.capture
.UpdateKvCaptureRecord` necessarily holds COMPLETE pre-call/returned K/V
tensors and full-layer score tensors transiently -- `kvcot.discovery
.pipeline.build_swap_pair_record` needs them to extract each of an event's
several (candidate, donor) pairs one at a time. But nothing downstream of
pair construction needs the full tensors: this module extracts exactly the
handful of per-selected-head row vectors and scalar scores a target event's
evidence actually requires, so a caller can drop its reference to the full
`UpdateKvCaptureRecord` afterward without losing anything the B2A evidence
schema (`kvcot.discovery.b2a_workers`) reports.

## Bound (never scales with model size)

For one selected (layer, kv_head) target, at most `CANDIDATES_PER_EVENT +
DONORS_PER_EVENT` (4) row vectors of shape `(head_dim,)` are retained, per
K and V (8 vectors total) -- this is a small, FIXED constant multiple of
`head_dim`, independent of `num_hidden_layers`, `num_key_value_heads`, or
cache length. `persistent_tensor_numel`/`persistent_tensor_bytes` make this
an assertable, auditable fact rather than a claim.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from kvcot.discovery.constants import CANDIDATES_PER_EVENT, DONORS_PER_EVENT
from kvcot.discovery.pass1 import EventPlan


class CaptureMinimizationError(RuntimeError):
    pass


def _find_physical_index(positions_1d: torch.Tensor, absolute_position: int) -> int | None:
    matches = (positions_1d == absolute_position).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


@dataclass(frozen=True)
class MinimizedTargetEvidence:
    """Everything B1B-R4 §8's "Capture parity" evidence needs for one
    selected target, and nothing else -- no full-layer/full-cache tensor is
    reachable from this object."""

    compaction_event_id: int
    layer_index: int
    kv_head_index: int
    candidate_absolute_positions: tuple[int, ...]
    donor_absolute_positions: tuple[int, ...]
    candidate_key_vectors: tuple[tuple[float, ...], ...]
    candidate_value_vectors: tuple[tuple[float, ...], ...]
    donor_key_vectors: tuple[tuple[float, ...], ...]
    donor_value_vectors: tuple[tuple[float, ...], ...]
    candidate_scores: tuple[float, ...]
    donor_scores: tuple[float, ...]
    gather_parity_passed: bool | None
    absolute_position_parity_passed: bool | None
    failure_reason: str | None
    head_dim: int
    persistent_tensor_numel: int
    persistent_tensor_bytes: int
    largest_persistent_tensor_shape: tuple[int, ...]


def build_minimized_target_evidence(event_plan: EventPlan, capture_record) -> MinimizedTargetEvidence:
    """Extract exactly the selected candidate/donor rows and scalar scores
    from a (still-full) `UpdateKvCaptureRecord`, and return an object that
    retains no reference to any full-layer/full-cache tensor -- every
    tensor this function reads is used only transiently, inside this call,
    to build small `float`/`tuple` values; nothing here stores a `torch
    .Tensor` on the returned object."""
    head = event_plan.kv_head_index
    record = capture_record

    if not record.had_compaction or record.pre_event_absolute_position_map is None:
        return MinimizedTargetEvidence(
            compaction_event_id=event_plan.compaction_event_id, layer_index=event_plan.layer_index,
            kv_head_index=head, candidate_absolute_positions=(), donor_absolute_positions=(),
            candidate_key_vectors=(), candidate_value_vectors=(), donor_key_vectors=(), donor_value_vectors=(),
            candidate_scores=(), donor_scores=(), gather_parity_passed=None, absolute_position_parity_passed=None,
            failure_reason="no_compaction_or_missing_position_map", head_dim=0,
            persistent_tensor_numel=0, persistent_tensor_bytes=0, largest_persistent_tensor_shape=(),
        )

    cd = event_plan.candidate_donor_selection
    candidate_positions = tuple(sorted({e for e, _ in cd.cross_product}))
    donor_positions = tuple(sorted({d for _, d in cd.cross_product}))

    pre_map_head = record.pre_event_absolute_position_map[head]
    window_size = record.window_size
    non_recent_map = pre_map_head[: pre_map_head.shape[0] - window_size] if window_size else pre_map_head

    pre_key = record.pre_call_key_states
    pre_value = record.pre_call_value_states
    scores = record.recomputed_final_score
    head_dim = int(pre_key.shape[-1])

    def _extract(positions: tuple[int, ...]) -> tuple[list[tuple[float, ...]], list[tuple[float, ...]], list[float]]:
        keys: list[tuple[float, ...]] = []
        values: list[tuple[float, ...]] = []
        score_values: list[float] = []
        for absolute_position in positions:
            phys = _find_physical_index(non_recent_map, absolute_position)
            if phys is None:
                raise CaptureMinimizationError(
                    f"absolute_position={absolute_position} not found in pre_event_absolute_position_map "
                    f"for layer={event_plan.layer_index} head={head}"
                )
            keys.append(tuple(pre_key[0, head, phys, :].tolist()))
            values.append(tuple(pre_value[0, head, phys, :].tolist()))
            score_values.append(float(scores[0, head, phys].item()))
        return keys, values, score_values

    candidate_keys, candidate_values, candidate_scores = _extract(candidate_positions)
    donor_keys, donor_values, donor_scores = _extract(donor_positions)

    n_vectors = len(candidate_keys) + len(candidate_values) + len(donor_keys) + len(donor_values)
    numel = n_vectors * head_dim
    bytes_ = numel * 4  # stored as Python float / float32-equivalent accounting

    return MinimizedTargetEvidence(
        compaction_event_id=event_plan.compaction_event_id,
        layer_index=event_plan.layer_index,
        kv_head_index=head,
        candidate_absolute_positions=candidate_positions,
        donor_absolute_positions=donor_positions,
        candidate_key_vectors=tuple(candidate_keys),
        candidate_value_vectors=tuple(candidate_values),
        donor_key_vectors=tuple(donor_keys),
        donor_value_vectors=tuple(donor_values),
        candidate_scores=tuple(candidate_scores),
        donor_scores=tuple(donor_scores),
        gather_parity_passed=record.gather_parity_passed,
        absolute_position_parity_passed=record.observed_kept_indices_parity_passed,
        failure_reason=record.parity_failure_reason,
        head_dim=head_dim,
        persistent_tensor_numel=numel,
        persistent_tensor_bytes=bytes_,
        largest_persistent_tensor_shape=(head_dim,),
    )


def assert_minimized_bound(evidence: MinimizedTargetEvidence) -> None:
    """Hard assertion (B1B-R4 §18): persistent storage for ONE selected
    target must never exceed `(CANDIDATES_PER_EVENT + DONORS_PER_EVENT) * 2
    (K and V) * head_dim` elements -- independent of how many layers, heads,
    or cache positions the real model actually has."""
    max_vectors = (CANDIDATES_PER_EVENT + DONORS_PER_EVENT) * 2
    max_numel = max_vectors * evidence.head_dim
    if evidence.persistent_tensor_numel > max_numel:
        raise CaptureMinimizationError(
            f"minimized target evidence retains {evidence.persistent_tensor_numel} scalar elements, "
            f"exceeding the fixed bound of {max_numel} ({max_vectors} vectors x head_dim={evidence.head_dim}) -- "
            "this must never scale with num_layers/num_kv_heads/cache_length."
        )
