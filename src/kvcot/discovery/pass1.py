"""Pass 1 — natural-run bookkeeping contract and deterministic, outcome-blind
selection (`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9,
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §6, authorized by
CLAUDE.md §1b/§4b: CPU-side harness architecture only, exercised against
dependency-injected `PrefillFn`/`DecodeOneFn`, never a real model).

Pass 1's job, in order:

1. Freeze the complete token trace (`run_natural_pass1`) by feeding the
   COMPLETE prompt through exactly one `prefill_fn` call, then driving the
   injected `decode_one_fn` one continuation token at a time (this
   repository's frozen batch-1, one-shot-prefill-then-token-by-token-decode
   call shape, CLAUDE.md §4, B1B-R2 §6 — never a prefill simulated as
   repeated one-token calls), recording every piece of required bookkeeping
   as it happens.
2. Identify eligible compaction events ONLY AFTER the complete trace length
   is known (`eligible_event_ids`) — never provisionally, mid-generation.
   Only DECODE-phase events are eligible (`absolute_event_position >=
   prompt_length`): a real one-shot prefill call is architecturally opaque
   from the outside (see `docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md`
   §5/§6) — there is no valid "immediately after this specific mid-prefill
   position" model-state boundary to snapshot from, so an event occurring
   during prefill can never be a branch-construction target.
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
from kvcot.discovery.harness_types import DecodeOneFn, LayerStepObservation, PrefillFn
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
    prefill_fn: PrefillFn,
    decode_one_fn: DecodeOneFn,
    max_new_tokens: int,
    eos_token_id: int | None,
    answer_fn: AnswerFn,
) -> NaturalRunTrace:
    """Feed the COMPLETE prompt through exactly one `prefill_fn` call, then
    drive `decode_one_fn` one continuation token at a time, greedily
    (argmax, deterministic -- no sampling, since Pass 1 has no seed/
    temperature concept of its own) over up to `max_new_tokens` generated
    tokens, freezing the complete trace before any eligibility decision is
    made anywhere else in this module. `prefill_fn` is never called more
    than once (B1B-R2 §6: "Do not simulate a full prefill through repeated
    one-token calls").
    """
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    prompt_token_ids = list(prompt_token_ids)
    prompt_length = len(prompt_token_ids)
    if prompt_length == 0:
        raise ValueError("prompt_token_ids must be non-empty")

    # --- prefill: exactly one call for the complete prompt ---
    prefill_result = prefill_fn(initial_state, prompt_token_ids)
    if len(prefill_result.per_position_logits) != prompt_length:
        raise ValueError(
            f"prefill_fn must return exactly one logits entry per prompt position "
            f"({prompt_length}), got {len(prefill_result.per_position_logits)}"
        )
    if len(prefill_result.per_position_layer_observations) != prompt_length:
        raise ValueError(
            f"prefill_fn must return exactly one layer_observations entry per prompt position "
            f"({prompt_length}), got {len(prefill_result.per_position_layer_observations)}"
        )

    full_token_ids: list[int] = list(prompt_token_ids)
    uncertainty_by_position: dict[int, PositionUncertainty] = {}
    compaction_events: list[CompactionEventObservation] = []
    cache_length_final_per_layer: dict[int, int] = {}
    state = prefill_result.new_state
    cap_hit = False

    for pos in range(prompt_length):
        predicted_position = pos + 1
        logits = prefill_result.per_position_logits[pos]
        layer_observations = prefill_result.per_position_layer_observations[pos]
        uncertainty_by_position[predicted_position] = PositionUncertainty(
            entropy=compute_entropy_nats(logits), logit_margin=compute_logit_margin(logits)
        )
        for layer_index, obs in layer_observations.items():
            cache_length_final_per_layer[layer_index] = obs.cache_length_after
        if any(obs.had_compaction for obs in layer_observations.values()):
            compaction_events.append(
                CompactionEventObservation(
                    compaction_event_id=len(compaction_events),
                    absolute_event_position=pos,
                    layer_observations=dict(layer_observations),
                )
            )

    # --- decode: one single-token call per continuation token ---
    next_logits = prefill_result.per_position_logits[-1]
    pos = prompt_length  # absolute position of the next token to be fed/generated
    while True:
        generated_so_far = pos - prompt_length
        if generated_so_far >= max_new_tokens:
            cap_hit = True
            break
        next_token_id = int(torch.argmax(next_logits).item())
        if eos_token_id is not None and next_token_id == eos_token_id:
            break  # natural stop -- do not append or feed the EOS token itself
        full_token_ids.append(next_token_id)

        result = decode_one_fn(state, next_token_id)
        state = result.new_state

        predicted_position = pos + 1
        uncertainty_by_position[predicted_position] = PositionUncertainty(
            entropy=compute_entropy_nats(result.next_token_logits),
            logit_margin=compute_logit_margin(result.next_token_logits),
        )
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
        next_logits = result.next_token_logits
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


def eligible_event_positions(
    event_positions: Sequence[int], *, prompt_length: int, total_len: int
) -> list[int]:
    """The position-based eligibility RULE alone, independent of the
    CPU-harness's `CompactionEventObservation`/`NaturalRunTrace` types:
    NEVER the first or last event in `event_positions` (chronological
    order), NEVER a prefill-phase event (B1B-R2 §6: a one-shot prefill call
    has no valid mid-prefill snapshot boundary to branch from — see this
    module's docstring), and at least `MINIMUM_FUTURE_TOKENS_AFTER_EVENT`
    (49) real future tokens must exist after the event's absolute position,
    evaluated against the COMPLETE trace length. `eligible_event_ids` below
    and FullKV-only B2A-R2 row qualification
    (`kvcot.discovery.b2a_qualification`, which has only predicted event
    POSITIONS from `kvcot.analysis.rkv_schedule
    .predicted_compaction_event_positions` -- no per-layer
    `CompactionEventObservation` objects, since R-KV is never imported
    during qualification) both call this one rule, never two independently
    maintained copies of it."""
    if len(event_positions) < 3:
        return []
    eligible = []
    for i, position in enumerate(event_positions):
        if i == 0 or i == len(event_positions) - 1:
            continue
        if position < prompt_length:
            continue
        future_tokens = (total_len - 1) - position
        if future_tokens >= MINIMUM_FUTURE_TOKENS_AFTER_EVENT:
            eligible.append(i)
    return eligible


def eligible_event_ids(trace: NaturalRunTrace) -> list[int]:
    """Compaction events eligible for selection: NEVER the first or last
    compaction event of the run, NEVER a prefill-phase event (B1B-R2 §6:
    a one-shot prefill call has no valid mid-prefill snapshot boundary to
    branch from — see this module's docstring), and at least
    `MINIMUM_FUTURE_TOKENS_AFTER_EVENT` (49) real future tokens must exist
    after the event's absolute position. Evaluated only against the
    COMPLETE frozen trace (`len(trace.full_token_ids)`), never a
    provisional/in-progress length. Delegates the position-based rule
    itself to `eligible_event_positions` (shared with FullKV-only B2A-R2
    row qualification) -- this function's only remaining job is translating
    between `CompactionEventObservation`s and `compaction_event_id`s."""
    events = trace.compaction_events
    positions = [ev.absolute_event_position for ev in events]
    eligible_indices = eligible_event_positions(
        positions, prompt_length=trace.prompt_length, total_len=len(trace.full_token_ids)
    )
    return [events[i].compaction_event_id for i in eligible_indices]


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
