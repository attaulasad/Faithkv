"""Pass 1 — natural-run bookkeeping contract and deterministic, outcome-blind
selection (`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9, authorized by
CLAUDE.md §1b/§4b: CPU-side harness architecture only, exercised against a
dependency-injected `NaturalStepFn`, never a real model).

Pass 1's job, in order:

1. Freeze the complete token trace (`run_natural_pass1`) by driving the
   injected `step_fn` one token at a time (this repository's frozen
   batch-1, token-by-token call shape, CLAUDE.md §4) and recording every
   piece of required bookkeeping as it happens.
2. Identify eligible compaction events ONLY AFTER the complete trace length
   is known (`eligible_event_ids`) — never provisionally, mid-generation.
3. Deterministically select exactly 3 eligible events, independently assign
   depth strata, and select layer/head/candidates/donors
   (`build_pass1_plan`) using ONLY `kvcot.discovery.sampling`'s existing,
   already-tested, outcome-blind draws — none of these functions accept a
   branch-gain or NLL argument, so no downstream outcome can influence
   selection, by construction of their signatures.

The resulting `Pass1Plan` is a complete, immutable plan for Pass 2 — Pass 2
never re-derives or second-guesses any selection decision made here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch

from kvcot.discovery.constants import MINIMUM_FUTURE_TOKENS_AFTER_EVENT
from kvcot.discovery.harness_types import LayerStepObservation, NaturalStepFn
from kvcot.discovery.sampling import (
    CandidateDonorSelection,
    IdentitySeedParts,
    assign_depth_strata,
    select_candidates_and_donors,
    select_events,
    select_kv_head,
    select_layer,
)
from kvcot.discovery.uncertainty import UncertaintySignal, compute_entropy_nats, compute_logit_margin
from kvcot.utils.hashing import sha256_int_ids

NaturalAnswerStatus = Literal["correct", "incorrect", "unverifiable"]


@dataclass(frozen=True)
class NaturalRunProvenance:
    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    rkv_revision: str
    config_sha256: str
    dataset_name: str
    example_id: str


@dataclass(frozen=True)
class PositionUncertainty:
    entropy: UncertaintySignal
    logit_margin: UncertaintySignal


@dataclass(frozen=True)
class CompactionEventObservation:
    """One compaction event observed during Pass 1, chronologically
    ordered. `layer_observations` covers every layer the harness tracks —
    Pass 1 records ALL of them (never just the eventually-selected layer),
    since which layer/head gets selected is not decided until after the
    complete trace is known."""

    compaction_event_id: int
    absolute_event_position: int  # t
    layer_observations: dict[int, LayerStepObservation]


@dataclass(frozen=True)
class NaturalRunTrace:
    provenance: NaturalRunProvenance
    prompt_length: int
    full_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    natural_answer: str | None
    natural_answer_status: NaturalAnswerStatus
    cap_hit: bool
    uncertainty_by_position: dict[int, PositionUncertainty]
    compaction_events: tuple[CompactionEventObservation, ...]
    cache_length_final_per_layer: dict[int, int]
    reference_trace_sha256: str


AnswerFn = Callable[[list[int]], tuple["str | None", NaturalAnswerStatus]]


def run_natural_pass1(
    provenance: NaturalRunProvenance,
    prompt_token_ids: Sequence[int],
    initial_state: Any,
    step_fn: NaturalStepFn,
    max_new_tokens: int,
    eos_token_id: int | None,
    answer_fn: AnswerFn,
) -> NaturalRunTrace:
    """Drive `step_fn` one token at a time over the prompt, then greedily
    (argmax, deterministic -- no sampling, since Pass 1 has no seed/
    temperature concept of its own) over up to `max_new_tokens` generated
    tokens, freezing the complete trace before any eligibility decision is
    made anywhere else in this module.
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    full_token_ids: list[int] = list(prompt_token_ids)
    prompt_length = len(full_token_ids)
    if prompt_length == 0:
        raise ValueError("prompt_token_ids must be non-empty")

    uncertainty_by_position: dict[int, PositionUncertainty] = {}
    compaction_events: list[CompactionEventObservation] = []
    cache_length_final_per_layer: dict[int, int] = {}
    state = initial_state
    cap_hit = False
    pos = 0

    while pos < len(full_token_ids):
        token_id = full_token_ids[pos]
        result = step_fn(state, token_id)
        state = result.new_state

        predicted_position = pos + 1
        entropy = compute_entropy_nats(result.next_token_logits)
        margin = compute_logit_margin(result.next_token_logits)
        uncertainty_by_position[predicted_position] = PositionUncertainty(entropy=entropy, logit_margin=margin)

        for layer_index, obs in result.layer_observations.items():
            cache_length_final_per_layer[layer_index] = obs.cache_length_after
        if any(obs.had_compaction for obs in result.layer_observations.values()):
            compaction_events.append(
                CompactionEventObservation(
                    compaction_event_id=len(compaction_events),
                    absolute_event_position=pos,
                    layer_observations=dict(result.layer_observations),
                )
            )

        is_last_known_position = predicted_position == len(full_token_ids)
        generated_so_far = len(full_token_ids) - prompt_length
        if is_last_known_position and pos >= prompt_length - 1:
            if generated_so_far >= max_new_tokens:
                cap_hit = True
            else:
                next_token_id = int(torch.argmax(result.next_token_logits).item())
                if eos_token_id is not None and next_token_id == eos_token_id:
                    pass  # natural stop -- do not append the EOS token itself
                else:
                    full_token_ids.append(next_token_id)
        pos += 1

    generated_token_ids = tuple(full_token_ids[prompt_length:])
    natural_answer, natural_answer_status = answer_fn(list(generated_token_ids))

    return NaturalRunTrace(
        provenance=provenance,
        prompt_length=prompt_length,
        full_token_ids=tuple(full_token_ids),
        generated_token_ids=generated_token_ids,
        natural_answer=natural_answer,
        natural_answer_status=natural_answer_status,
        cap_hit=cap_hit,
        uncertainty_by_position=uncertainty_by_position,
        compaction_events=tuple(compaction_events),
        cache_length_final_per_layer=cache_length_final_per_layer,
        reference_trace_sha256=sha256_int_ids(list(full_token_ids)),
    )


def eligible_event_ids(trace: NaturalRunTrace) -> list[int]:
    """Compaction events eligible for selection: NEVER the first or last
    compaction event of the run, and at least
    `MINIMUM_FUTURE_TOKENS_AFTER_EVENT` (49) real future tokens must exist
    after the event's absolute position. Evaluated only against the
    COMPLETE frozen trace (`len(trace.full_token_ids)`), never a
    provisional/in-progress length."""
    events = trace.compaction_events
    if len(events) < 3:
        return []
    total_len = len(trace.full_token_ids)
    eligible = []
    for i, ev in enumerate(events):
        if i == 0 or i == len(events) - 1:
            continue
        future_tokens = (total_len - 1) - ev.absolute_event_position
        if future_tokens >= MINIMUM_FUTURE_TOKENS_AFTER_EVENT:
            eligible.append(ev.compaction_event_id)
    return eligible


def _pools_for_layer_head(
    layer_obs: LayerStepObservation, kv_head_index: int
) -> tuple[list[int], list[int]] | None:
    """`(evicted_pool, retained_pool)` of ABSOLUTE token positions for one
    (layer, kv_head) at one event, derived from that layer's own
    bookkeeping -- never a shadow reconstruction. `None` if the layer had
    no compaction (nothing to sample from).

    The DONOR pool is restricted to the topk-SELECTED (non-recent-window)
    survivors only -- the protected recent window (the last `window_size`
    columns of `observed_kept_absolute_positions`, `kvcot.discovery.capture`'s
    own storage ordering) is always kept unconditionally and never competes
    on score, so it has no comparable `score_r` value and is excluded from
    donor sampling (documented simplification of this harness's
    architecture, not a change to R-KV's real eviction behavior)."""
    if not layer_obs.had_compaction:
        return None
    pre_map = layer_obs.pre_event_absolute_position_map
    observed = layer_obs.observed_kept_absolute_positions
    if pre_map is None or observed is None:
        return None
    window_size = layer_obs.window_size
    observed_head = observed[kv_head_index]
    retained_pool = (observed_head[:-window_size] if window_size else observed_head).tolist()
    retained_set = set(observed_head.tolist())
    evicted_pool = [p for p in pre_map[kv_head_index].tolist() if p not in retained_set]
    return evicted_pool, retained_pool


@dataclass(frozen=True)
class EventPlan:
    compaction_event_id: int
    absolute_event_position: int
    chronological_event_ordinal: int
    depth_stratum: int
    layer_index: int
    kv_head_index: int
    candidate_donor_selection: CandidateDonorSelection


@dataclass(frozen=True)
class Pass1Plan:
    trace: NaturalRunTrace
    events: tuple[EventPlan, EventPlan, EventPlan]


PLAN_FAILURE_TOO_FEW_ELIGIBLE_EVENTS = "fewer_than_three_eligible_events"
PLAN_FAILURE_INVALID_CANDIDATE_DONOR_POOL = "invalid_candidate_donor_pool"


def build_pass1_plan(
    trace: NaturalRunTrace,
    num_hidden_layers: int,
    num_key_value_heads: int,
    identity: IdentitySeedParts,
) -> tuple[Pass1Plan | None, str | None]:
    """Deterministic, outcome-blind event/depth/layer/head/candidate/donor
    selection. Returns `(plan, None)` on success or `(None, failure_reason)`
    -- NEVER a plan with fewer than 3 events, and never influenced by any
    branch-gain or NLL value (none of the `kvcot.discovery.sampling`
    functions this calls accept one)."""
    eligible = eligible_event_ids(trace)
    selection = select_events(eligible, identity)
    if selection is None:
        return None, PLAN_FAILURE_TOO_FEW_ELIGIBLE_EVENTS

    depth = assign_depth_strata(selection.selected_events_chronological, identity)
    event_by_id = {ev.compaction_event_id: ev for ev in trace.compaction_events}

    plans: list[EventPlan] = []
    for event_id in selection.selected_events_chronological:
        ev = event_by_id[event_id]
        stratum = depth.depth_stratum_by_event[event_id]
        layer_sel = select_layer(event_id, stratum, num_hidden_layers, identity)
        kv_head_index = select_kv_head(event_id, num_key_value_heads, identity)

        layer_obs = ev.layer_observations.get(layer_sel.layer_index)
        if layer_obs is None:
            return None, PLAN_FAILURE_INVALID_CANDIDATE_DONOR_POOL
        pools = _pools_for_layer_head(layer_obs, kv_head_index)
        if pools is None:
            return None, PLAN_FAILURE_INVALID_CANDIDATE_DONOR_POOL
        evicted_pool, retained_pool = pools

        cd = select_candidates_and_donors(
            evicted_pool, retained_pool, event_id, layer_sel.layer_index, kv_head_index, identity
        )
        if cd is None:
            return None, PLAN_FAILURE_INVALID_CANDIDATE_DONOR_POOL

        plans.append(
            EventPlan(
                compaction_event_id=event_id,
                absolute_event_position=ev.absolute_event_position,
                chronological_event_ordinal=selection.chronological_ordinal_by_event[event_id],
                depth_stratum=stratum,
                layer_index=layer_sel.layer_index,
                kv_head_index=kv_head_index,
                candidate_donor_selection=cd,
            )
        )

    assert len(plans) == 3
    return Pass1Plan(trace=trace, events=(plans[0], plans[1], plans[2])), None
