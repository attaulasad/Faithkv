"""Memory-bounded branch inputs for B2A discovery pair evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import torch

from kvcot.discovery.pass1 import EventPlan
from kvcot.discovery.pass2 import TargetCapture
from kvcot.generation.state import ModelStateSnapshot


class CompactTargetBoundError(ValueError):
    """A compact target retained more tensor storage than its derived bound."""


@dataclass(frozen=True)
class SelectedPositionData:
    absolute_position: int
    pre_storage_position: int
    post_storage_position: int | None
    key_vector: torch.Tensor
    value_vector: torch.Tensor
    score: float
    attention: float
    similarity: float


@dataclass(frozen=True)
class CaptureParityEvidence:
    gather_parity_passed: bool
    absolute_position_parity_passed: bool
    parity_check_passed: bool
    failure_reason: str | None


@dataclass(frozen=True)
class CompactBranchTarget:
    event_plan: EventPlan
    pristine_snapshot: ModelStateSnapshot
    positions: Mapping[int, SelectedPositionData]
    capture_parity_evidence: CaptureParityEvidence
    persistent_tensor_numel: int
    persistent_tensor_bytes: int
    derived_tensor_numel_bound: int
    derived_tensor_byte_bound: int


def _find_position(values: torch.Tensor, absolute_position: int) -> int | None:
    matches = (values == absolute_position).nonzero(as_tuple=True)[0]
    return None if matches.numel() == 0 else int(matches[0].item())


def build_compact_branch_target(target: TargetCapture) -> CompactBranchTarget:
    """Gather only selected pair vectors/scalars from one transient capture."""
    event = target.event_plan
    record = target.capture_record
    head = event.kv_head_index
    if not record.parity_check_passed:
        raise CompactTargetBoundError("cannot compact a capture whose parity check failed")
    required = tuple(dict.fromkeys(
        position for pair in event.candidate_donor_selection.cross_product for position in pair
    ))
    pre_map = record.pre_event_absolute_position_map
    observed_map = record.observed_kept_absolute_positions
    if pre_map is None or observed_map is None or record.window_size is None:
        raise CompactTargetBoundError("capture is missing required absolute-position evidence")
    pre_head = pre_map[head]
    non_recent = pre_head[: pre_head.shape[0] - record.window_size]

    selected: dict[int, SelectedPositionData] = {}
    for absolute_position in required:
        pre_index = _find_position(non_recent, absolute_position)
        if pre_index is None:
            raise CompactTargetBoundError(
                f"selected absolute position {absolute_position} is absent from the non-recent pool"
            )
        key = record.pre_call_key_states[0, head, pre_index, :].detach().clone().contiguous()
        value = record.pre_call_value_states[0, head, pre_index, :].detach().clone().contiguous()
        selected[absolute_position] = SelectedPositionData(
            absolute_position=absolute_position,
            pre_storage_position=pre_index,
            post_storage_position=_find_position(observed_map[head], absolute_position),
            key_vector=key,
            value_vector=value,
            score=float(record.recomputed_final_score[0, head, pre_index].item()),
            attention=float(record.recomputed_attention_component[0, head, pre_index].item()),
            similarity=float(record.recomputed_similarity_component[0, head, pre_index].item()),
        )

    actual_numel = sum(item.key_vector.numel() + item.value_vector.numel() for item in selected.values())
    actual_bytes = sum(
        item.key_vector.numel() * item.key_vector.element_size()
        + item.value_vector.numel() * item.value_vector.element_size()
        for item in selected.values()
    )
    head_dimension = int(record.pre_call_key_states.shape[-1])
    derived_numel_bound = len(required) * 2 * head_dimension
    derived_byte_bound = len(required) * head_dimension * (
        record.pre_call_key_states.element_size() + record.pre_call_value_states.element_size()
    )
    compact = CompactBranchTarget(
        event_plan=event,
        pristine_snapshot=target.pristine_snapshot,
        positions=MappingProxyType(selected),
        capture_parity_evidence=CaptureParityEvidence(
            gather_parity_passed=record.gather_parity_passed is True,
            absolute_position_parity_passed=record.observed_kept_indices_parity_passed is True,
            parity_check_passed=record.parity_check_passed,
            failure_reason=record.parity_failure_reason,
        ),
        persistent_tensor_numel=actual_numel,
        persistent_tensor_bytes=actual_bytes,
        derived_tensor_numel_bound=derived_numel_bound,
        derived_tensor_byte_bound=derived_byte_bound,
    )
    assert_compact_target_bound(compact)
    return compact


def assert_compact_target_bound(target: CompactBranchTarget) -> None:
    """Enforce the selected-vector shape/dtype-derived storage ceiling."""
    if target.persistent_tensor_numel > target.derived_tensor_numel_bound:
        raise CompactTargetBoundError(
            f"compact target tensor elements {target.persistent_tensor_numel} exceed derived bound "
            f"{target.derived_tensor_numel_bound}"
        )
    if target.persistent_tensor_bytes > target.derived_tensor_byte_bound:
        raise CompactTargetBoundError(
            f"compact target bytes {target.persistent_tensor_bytes} exceed derived bound "
            f"{target.derived_tensor_byte_bound}"
        )
