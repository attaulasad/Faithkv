"""Branch construction and evaluation — connects candidate/donor identities,
captured K/V and score components, the fixed-shape swap, baseline/swapped
branch evaluation, and final `SwapPairRecord` construction
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9, authorized by CLAUDE.md
§1b/§4b). Dependency-injected (`BranchStepFn`, model-agnostic, matching
`kvcot.discovery.branch_eval.StepFn`'s existing shape) so synthetic
deterministic models can exercise the complete pipeline on CPU.

One (layer, kv_head) capture record's PRE-event non-recent score pool
(`recomputed_final_score`/`recomputed_attention_component`/
`recomputed_similarity_component`, all shape
`(1, num_kv_heads, pre_event_len - window_size)`) is the single source of
truth for `score_e`/`score_r` and their components — candidate and donor
are both drawn from that pool by `kvcot.discovery.pass1._pools_for_layer_head`
(the protected recent window is excluded there, never scored here).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from kvcot.discovery.branch_eval import SCORED_HORIZON, StepFn as BranchStepFn, evaluate_swap_branches
from kvcot.discovery.nll import mean_nll
from kvcot.discovery.pass1 import NaturalRunTrace
from kvcot.discovery.pass2 import TargetCapture
from kvcot.discovery.schemas import SwapPairRecord
from kvcot.discovery.swap import SwapAliasingError, SwapIndexError, apply_within_head_swap
from kvcot.discovery.uncertainty import UncertaintySignal, compute_pair_uncertainty_signals
from kvcot.utils.hashing import sha256_int_ids

STAGE_INVALID_CANDIDATE_DONOR_POOL = "invalid_candidate_donor_pool"
STAGE_BRANCH_EVALUATION_FAILURE = "branch_evaluation_failure"
STAGE_SCHEMA_VALIDATION_FAILURE = "schema_validation_failure"


def _find_physical_index(positions_1d: torch.Tensor, absolute_position: int) -> int | None:
    matches = (positions_1d == absolute_position).nonzero(as_tuple=True)[0]
    if matches.numel() == 0:
        return None
    return int(matches[0].item())


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

    try:
        swap_result = apply_within_head_swap(
            key_cache=[record.returned_key_states],
            value_cache=[record.returned_value_states],
            layer_index=0,
            kv_head_index=head,
            retained_post_storage_position=donor_post_idx,
            candidate_key=candidate_key,
            candidate_value=candidate_value,
        )
    except (SwapIndexError, SwapAliasingError) as exc:
        return PairBuildResult(None, STAGE_BRANCH_EVALUATION_FAILURE, f"swap_failed: {exc}")

    baseline_state = (record.returned_key_states, record.returned_value_states)
    swapped_state = (swap_result.key_cache[0], swap_result.value_cache[0])

    try:
        comparison = evaluate_swap_branches(
            branch_step_fn, baseline_state, swapped_state, bridge_token_id, reference_token_ids
        )
    except Exception as exc:
        return PairBuildResult(None, STAGE_BRANCH_EVALUATION_FAILURE, f"branch_eval_raised: {type(exc).__name__}: {exc}")

    reference_horizon_sha256 = sha256_int_ids(reference_token_ids)
    is_noop_control = evicted_absolute_position == donor_absolute_position

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
            parity_check_passed=True,
            parity_failure_reason=None,
            is_noop_control=is_noop_control,
            net_physical_bytes_changed=0,
            cap_hit_flag=trace.cap_hit,
            valid_flag=True,
            invalid_reason=None,
            reference_horizon_sha256=reference_horizon_sha256,
            swap_gain=comparison.swap_gain,
            baseline_per_token_nll=comparison.baseline_per_token_nll,
            swapped_per_token_nll=comparison.swapped_per_token_nll,
        )
    except Exception as exc:  # pydantic ValidationError or a constructed invariant violation
        return PairBuildResult(None, STAGE_SCHEMA_VALIDATION_FAILURE, str(exc))

    return PairBuildResult(pair, None, None)
