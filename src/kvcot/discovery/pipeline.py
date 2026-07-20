"""Branch construction and evaluation — connects candidate/donor identities,
captured K/V and score components, the fixed-shape swap, baseline/swapped
branch evaluation, and final `SwapPairRecord` construction
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9,
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §5, authorized by
CLAUDE.md §1b/§4b). Dependency-injected (`BranchStepFn`, model-agnostic,
matching `kvcot.discovery.branch_eval.StepFn`'s existing shape) so synthetic
deterministic models can exercise the complete pipeline on CPU.

One (layer, kv_head) capture record's PRE-event non-recent score pool
(`recomputed_final_score`/`recomputed_attention_component`/
`recomputed_similarity_component`, all shape
`(1, num_kv_heads, pre_event_len - window_size)`) is the single source of
truth for `score_e`/`score_r` and their components — candidate and donor
are both drawn from that pool by `kvcot.discovery.pass1._pools_for_layer_head`
(the protected recent window is excluded there, never scored here).

## Branching from a complete `ModelStateSnapshot` (B1B-R2 §5, repaired)

Previously this module treated ONE (layer, kv_head)'s returned K/V tensors
as the complete branch continuation state — not a valid causal-LM
continuation state (every other layer's K/V, and every other piece of
mutable model state, was simply absent). Branch construction now starts
from `target_capture.pristine_snapshot` — a complete, independently-cloned
`kvcot.generation.state.ModelStateSnapshot` covering every layer — and
`kvcot.discovery.swap.apply_within_head_swap` mutates only the selected
layer/head/slot of an independent clone of it, leaving every other layer,
head, and slot byte-identical to the pristine snapshot. `BranchStepFn` now
receives a full `ModelStateSnapshot` as its "cache state" argument, never a
bare per-layer tensor tuple.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from kvcot.discovery.branch_eval import SCORED_HORIZON, StepFn as BranchStepFn, SwapBranchComparison, evaluate_branch
from kvcot.discovery.nll import mean_nll
from kvcot.discovery.pass1 import NaturalRunTrace
from kvcot.discovery.pass2 import TargetCapture
from kvcot.discovery.schemas import SwapPairRecord
from kvcot.discovery.swap import SwapAliasingError, SwapIndexError, apply_semantic_within_head_swap
from kvcot.discovery.uncertainty import UncertaintySignal, compute_pair_uncertainty_signals
from kvcot.generation.state import ModelStateSnapshot
from kvcot.utils.hashing import sha256_int_ids

STAGE_INVALID_CANDIDATE_DONOR_POOL = "invalid_candidate_donor_pool"
STAGE_BRANCH_EVALUATION_FAILURE = "branch_evaluation_failure"
STAGE_SCHEMA_VALIDATION_FAILURE = "schema_validation_failure"


def _find_physical_index(positions_1d: torch.Tensor, absolute_position: int) -> int | None:
    matches = (positions_1d == absolute_position).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


def _cache_total_bytes(key_cache: list[torch.Tensor], value_cache: list[torch.Tensor]) -> int:
    """Total physical bytes across every layer's K and V cache tensors --
    used to derive `net_physical_bytes_changed` from a real before/after
    comparison (B1B-R4.1 §18) instead of the literal `0` this module used to
    hard-code. `apply_within_head_swap` already raises on any shape
    mismatch, so this always computes to 0 for a successful swap -- the
    point is that it is COMPUTED, so a future regression that actually
    changed a shape would be caught here rather than silently reported as
    zero regardless."""
    return sum(t.numel() * t.element_size() for t in key_cache) + sum(t.numel() * t.element_size() for t in value_cache)


@dataclass(frozen=True)
class PairBuildResult:
    record: SwapPairRecord | None
    failure_stage: str | None
    failure_detail: str | None


UNCERTAINTY_POSITION_UNAVAILABLE_REASON = "uncertainty_lookup_position_not_available"


def _uncertainty_signal_at(trace: NaturalRunTrace, absolute_position: int, kind: str) -> UncertaintySignal:
    position_uncertainty = trace.uncertainty_by_position.get(absolute_position)
    if position_uncertainty is None:
        return UncertaintySignal(value=None, missing_reason=UNCERTAINTY_POSITION_UNAVAILABLE_REASON)
    return position_uncertainty.entropy if kind == "entropy" else position_uncertainty.logit_margin


def build_swap_pair_record(
    *,
    example_id: str,
    model_revision: str,
    rkv_revision: str,
    target_capture: TargetCapture,
    evicted_absolute_position: int,
    donor_absolute_position: int,
    trace: NaturalRunTrace,
    branch_step_fn: BranchStepFn,
    scored_horizon: int = SCORED_HORIZON,
) -> PairBuildResult:
    """Build and validate exactly one `SwapPairRecord` for one
    (event, candidate, donor) pair. `evicted_absolute_position ==
    donor_absolute_position` is the mandatory no-op control (Part IX.20) —
    handled by the SAME code path, never a special case: the candidate and
    donor resolve to the identical pre-event physical slot, so every
    derived quantity (score margin, component diffs, swap content) comes
    out exactly zero/identical by construction, not by a separate branch.
    """
    ev = target_capture.event_plan
    record = target_capture.capture_record
    head = ev.kv_head_index

    t = ev.absolute_event_position
    bridge_pos = t + 1
    first_scored = t + 2
    if first_scored + scored_horizon > len(trace.full_token_ids):
        return PairBuildResult(None, STAGE_BRANCH_EVALUATION_FAILURE, "insufficient_future_tokens_for_horizon")

    bridge_token_id = trace.full_token_ids[bridge_pos]
    reference_token_ids = list(trace.full_token_ids[first_scored : first_scored + scored_horizon])

    pre_map_head = record.pre_event_absolute_position_map[head]
    window_size = record.window_size
    non_recent_map = pre_map_head[: pre_map_head.shape[0] - window_size]

    evicted_phys = _find_physical_index(non_recent_map, evicted_absolute_position)
    donor_phys_pre = _find_physical_index(non_recent_map, donor_absolute_position)
    if evicted_phys is None or donor_phys_pre is None:
        return PairBuildResult(None, STAGE_INVALID_CANDIDATE_DONOR_POOL, "candidate_or_donor_not_in_pre_event_pool")

    observed_head = record.observed_kept_absolute_positions[head]
    donor_post_idx = _find_physical_index(observed_head, donor_absolute_position)
    if donor_post_idx is None:
        return PairBuildResult(None, STAGE_INVALID_CANDIDATE_DONOR_POOL, "donor_not_in_observed_kept_positions")

    pre_key = record.pre_call_key_states
    pre_value = record.pre_call_value_states
    candidate_key = pre_key[0, head, evicted_phys, :].clone().contiguous()
    candidate_value = pre_value[0, head, evicted_phys, :].clone().contiguous()
    donor_pre_key = pre_key[0, head, donor_phys_pre, :]
    donor_pre_value = pre_value[0, head, donor_phys_pre, :]

    score_e = record.recomputed_final_score[0, head, evicted_phys].item()
    score_r = record.recomputed_final_score[0, head, donor_phys_pre].item()
    attn_e = record.recomputed_attention_component[0, head, evicted_phys].item()
    attn_r = record.recomputed_attention_component[0, head, donor_phys_pre].item()
    sim_e = record.recomputed_similarity_component[0, head, evicted_phys].item()
    sim_r = record.recomputed_similarity_component[0, head, donor_phys_pre].item()

    # Branch from a COMPLETE, independently-cloned post-event
    # ModelStateSnapshot (B1B-R2 §5) -- never one layer's returned K/V
    # tensors standing in for the whole model's continuation state.
    #
    # B1B-R4.1 §17 repair: baseline and swapped are cloned and evaluated
    # SEQUENTIALLY, never both live at once -- baseline is cloned from
    # `pristine`, fully evaluated, and its clone reference dropped BEFORE
    # the swapped clone is even created. The prior version cloned both
    # up front and held both local variables for the rest of this function
    # (both still reachable through `evaluate_swap_branches`'s combined
    # call), doubling peak branch-construction memory for no reason -- on a
    # real model this is a full multi-layer K/V cache clone, not a toy
    # tensor. `pristine` itself is never mutated by either clone.
    pristine = target_capture.pristine_snapshot

    baseline_snapshot = pristine.clone()
    try:
        baseline_result = evaluate_branch(branch_step_fn, baseline_snapshot, bridge_token_id, reference_token_ids)
    except Exception as exc:
        return PairBuildResult(None, STAGE_BRANCH_EVALUATION_FAILURE, f"branch_eval_raised: {type(exc).__name__}: {exc}")
    finally:
        # Drop the local reference the moment baseline evaluation is done,
        # whether it succeeded or raised -- the swapped clone below must
        # never coexist with a still-reachable baseline clone.
        del baseline_snapshot

    swapped_snapshot = pristine.clone()
    try:
        semantic_swap = apply_semantic_within_head_swap(
            swapped_snapshot,
            layer_index=ev.layer_index,
            kv_head_index=head,
            retained_post_storage_position=donor_post_idx,
            candidate_key=candidate_key,
            candidate_value=candidate_value,
            donor_absolute_position=donor_absolute_position,
            candidate_absolute_position=evicted_absolute_position,
        )
    except (SwapIndexError, SwapAliasingError) as exc:
        del swapped_snapshot
        return PairBuildResult(None, STAGE_BRANCH_EVALUATION_FAILURE, f"swap_failed: {exc}")

    try:
        swapped_result = evaluate_branch(branch_step_fn, swapped_snapshot, bridge_token_id, reference_token_ids)
    except Exception as exc:
        return PairBuildResult(None, STAGE_BRANCH_EVALUATION_FAILURE, f"branch_eval_raised: {type(exc).__name__}: {exc}")
    finally:
        del swapped_snapshot

    comparison = SwapBranchComparison(
        baseline_per_token_nll=baseline_result.per_token_nll,
        swapped_per_token_nll=swapped_result.per_token_nll,
        baseline_mean_nll=baseline_result.mean_nll,
        swapped_mean_nll=swapped_result.mean_nll,
        swap_gain=baseline_result.mean_nll - swapped_result.mean_nll,
        baseline_final_cache_state=baseline_result.final_cache_state,
        swapped_final_cache_state=swapped_result.final_cache_state,
        baseline_per_token_logits=baseline_result.per_token_logits,
        swapped_per_token_logits=swapped_result.per_token_logits,
    )

    reference_horizon_sha256 = sha256_int_ids(reference_token_ids)
    is_noop_control = evicted_absolute_position == donor_absolute_position

    # B1B-R4.1 §18: derive parity_check_passed/net_physical_bytes_changed
    # from the real mutation report instead of hard-coding True/0.
    # `record.parity_check_passed` (this target's within-Pass-2 capture
    # parity) is already required True for ANY target capture to have
    # survived into `Pass2Result.target_captures` -- re-checked here
    # defensively, never assumed silently. For a snapshot that carries full
    # provenance (every real-model snapshot does, per
    # `kvcot.discovery.swap.apply_semantic_within_head_swap`'s own
    # docstring; a synthetic/CPU-test snapshot without provenance
    # legitimately skips those updates), both the provenance update and the
    # kept-index bookkeeping update are mandatory -- missing either is a
    # hard parity failure, never silently ignored.
    # Provenance and kept-index bookkeeping are independent pieces of
    # snapshot state (`apply_semantic_within_head_swap`'s own docstring
    # scopes each to its own presence check, never a single combined flag)
    # -- a real-model snapshot always carries both together in practice, but
    # the CPU synthetic harness snapshot deliberately carries bookkeeping
    # without provenance (`tests/unit/discovery/_synthetic_harness.py
    # .build_snapshot_from_state` sets `provenance=None`), so each mandatory
    # update is gated on its OWN presence signal, matching the swap
    # primitive's own logic exactly, never a shared proxy for both.
    provenance_present = pristine.provenance is not None
    layer_bookkeeping = (
        pristine.kv_cluster_bookkeeping_per_layer[ev.layer_index] if pristine.kv_cluster_bookkeeping_per_layer else None
    )
    kept_index_bookkeeping_present = bool(layer_bookkeeping and layer_bookkeeping.get("kept_token_indices"))

    swap_parity_failures: list[str] = []
    if not record.parity_check_passed:
        swap_parity_failures.append("capture_parity_check_failed")
    if provenance_present and not semantic_swap.provenance_updated:
        swap_parity_failures.append("semantic_swap_parity_provenance_not_updated")
    if kept_index_bookkeeping_present and not semantic_swap.kept_index_bookkeeping_updated:
        swap_parity_failures.append("semantic_swap_parity_kept_index_bookkeeping_not_updated")

    parity_check_passed = not swap_parity_failures
    parity_failure_reason = ",".join(swap_parity_failures) if swap_parity_failures else None

    net_physical_bytes_changed = _cache_total_bytes(
        semantic_swap.swap_result.key_cache, semantic_swap.swap_result.value_cache
    ) - _cache_total_bytes(pristine.key_cache, pristine.value_cache)

    uncertainty_signals = compute_pair_uncertainty_signals(
        entropy_e=_uncertainty_signal_at(trace, evicted_absolute_position, "entropy"),
        entropy_r=_uncertainty_signal_at(trace, donor_absolute_position, "entropy"),
        logit_margin_e=_uncertainty_signal_at(trace, evicted_absolute_position, "logit_margin"),
        logit_margin_r=_uncertainty_signal_at(trace, donor_absolute_position, "logit_margin"),
    )

    try:
        pair = SwapPairRecord(
            example_id=example_id,
            model_revision=model_revision,
            rkv_revision=rkv_revision,
            compaction_event_id=ev.compaction_event_id,
            chronological_event_ordinal=ev.chronological_event_ordinal,
            depth_stratum=ev.depth_stratum,
            layer_index=ev.layer_index,
            kv_head_index=head,
            event_token_absolute_position=t,
            bridge_token_absolute_position=bridge_pos,
            first_affected_forward_input_absolute_position=bridge_pos,
            first_affected_logit_target_absolute_position=first_scored,
            first_scored_absolute_position=first_scored,
            evicted_absolute_token_position=evicted_absolute_position,
            evicted_pre_storage_position=evicted_phys,
            retained_absolute_token_position=donor_absolute_position,
            retained_pre_storage_position=donor_phys_pre,
            retained_post_storage_position=donor_post_idx,
            score_e=score_e,
            score_r=score_r,
            score_margin_e_minus_r=score_e - score_r,
            attention_component_diff=attn_e - attn_r,
            similarity_component_diff=sim_e - sim_r,
            recency_diff=evicted_absolute_position - donor_absolute_position,
            key_norm_diff=(candidate_key.float().norm() - donor_pre_key.float().norm()).item(),
            value_norm_diff=(candidate_value.float().norm() - donor_pre_value.float().norm()).item(),
            entropy_e=uncertainty_signals.entropy_e.value,
            entropy_e_missing_reason=uncertainty_signals.entropy_e.missing_reason,
            entropy_r=uncertainty_signals.entropy_r.value,
            entropy_r_missing_reason=uncertainty_signals.entropy_r.missing_reason,
            entropy_diff=uncertainty_signals.entropy_diff,
            logit_margin_e=uncertainty_signals.logit_margin_e.value,
            logit_margin_e_missing_reason=uncertainty_signals.logit_margin_e.missing_reason,
            logit_margin_r=uncertainty_signals.logit_margin_r.value,
            logit_margin_r_missing_reason=uncertainty_signals.logit_margin_r.missing_reason,
            logit_margin_diff=uncertainty_signals.logit_margin_diff,
            parity_check_passed=parity_check_passed,
            parity_failure_reason=parity_failure_reason,
            is_noop_control=is_noop_control,
            net_physical_bytes_changed=net_physical_bytes_changed,
            cap_hit_flag=trace.cap_hit,
            valid_flag=parity_check_passed,
            invalid_reason=parity_failure_reason,
            reference_horizon_sha256=reference_horizon_sha256,
            swap_gain=comparison.swap_gain,
            baseline_per_token_nll=comparison.baseline_per_token_nll,
            swapped_per_token_nll=comparison.swapped_per_token_nll,
        )
    except Exception as exc:  # pydantic ValidationError or a constructed invariant violation
        return PairBuildResult(None, STAGE_SCHEMA_VALIDATION_FAILURE, str(exc))

    return PairBuildResult(pair, None, None)
