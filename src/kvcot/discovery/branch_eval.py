"""Teacher-forced branch NLL evaluator and the mandatory no-op control
(Part IX.20 of `docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`).

Deliberately model-agnostic: `step_fn` is dependency-injected
(`Callable[[cache_state, token_id], tuple[logits, new_cache_state]]`), so
this module never imports a real model and never downloads anything —
tests supply a small deterministic toy step function. As of B1B-R2 §5,
`cache_state` is always a complete
`kvcot.generation.state.ModelStateSnapshot` (every layer's K/V, not one
layer's tensors) — `kvcot.discovery.pipeline.build_swap_pair_record` is the
one caller that constructs `baseline_initial_cache_state`/
`swapped_initial_cache_state` this way, from two independent clones of one
pristine post-event snapshot. Evaluation rules, frozen: feed one unscored
bridge token; score exactly the 48 supplied reference tokens; compute NLL
via float32 `log_softmax` against the real target token; evolve each
branch's own cache independently (never share mutable state between
baseline and swapped); never sample or take argmax; never truncate the
horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from kvcot.discovery.constants import SCORED_HORIZON
from kvcot.discovery.nll import mean_nll
from kvcot.utils.hashing import sha256_json

StepFn = Callable[[Any, int], tuple[torch.Tensor, Any]]


@dataclass(frozen=True)
class BranchEvalResult:
    per_token_nll: list[float]
    per_token_logits: list[torch.Tensor]
    mean_nll: float
    final_cache_state: Any


@dataclass(frozen=True)
class CompactBranchScore:
    """B1 execution-boundary closure §8: exactly what discovery scoring
    scientifically needs from one branch's `BranchEvalResult`, and nothing
    else -- no full-vocabulary per-token logits, no live final cache state
    (a real-model `final_cache_state` is a `_LiveBranchState` holding a
    complete multi-layer `DynamicCache`; a synthetic-harness one is
    whatever the injected `step_fn` returns, potentially also large).
    `kvcot.discovery.pipeline.build_swap_pair_record` extracts this
    immediately after `evaluate_branch` returns and releases the full
    `BranchEvalResult` before evaluating the OTHER branch (baseline before
    swapped) -- the two branches' full live cache/logits must never be
    reachable at the same time."""

    per_token_nll: tuple[float, ...]
    mean_nll: float
    nll_sha256: str


def compact_branch_score(result: BranchEvalResult) -> CompactBranchScore:
    """Pure extraction -- reads `result.per_token_nll`/`.mean_nll` only;
    never touches `.per_token_logits`/`.final_cache_state`, so a caller
    that discards `result` immediately after calling this never actually
    dereferences the fields it's about to release. Hashed via the
    project's existing canonical-JSON hash (`kvcot.utils.hashing
    .sha256_json`) -- never a second, independently-invented hashing
    scheme."""
    per_token_nll = tuple(result.per_token_nll)
    return CompactBranchScore(
        per_token_nll=per_token_nll, mean_nll=result.mean_nll, nll_sha256=sha256_json(list(per_token_nll))
    )


def evaluate_branch_compact(
    step_fn: StepFn,
    initial_cache_state: Any,
    bridge_token_id: int,
    reference_token_ids: Sequence[int],
) -> CompactBranchScore:
    """Score one discovery branch without retaining logits or final cache."""
    if len(reference_token_ids) != SCORED_HORIZON:
        raise ValueError(
            f"reference_token_ids must have exactly {SCORED_HORIZON} entries, got {len(reference_token_ids)}"
        )
    cache_state = initial_cache_state
    logits, cache_state = step_fn(cache_state, bridge_token_id)
    per_token_nll: list[float] = []
    last_index = len(reference_token_ids) - 1
    for index, target_token_id in enumerate(reference_token_ids):
        nll = -torch.log_softmax(logits.float(), dim=-1)[target_token_id]
        per_token_nll.append(float(nll.item()))
        if index < last_index:
            logits, cache_state = step_fn(cache_state, target_token_id)
    values = tuple(per_token_nll)
    score = CompactBranchScore(
        per_token_nll=values,
        mean_nll=mean_nll(values),
        nll_sha256=sha256_json(list(values)),
    )
    del logits
    del cache_state
    return score


def evaluate_branch(step_fn: StepFn, initial_cache_state: Any, bridge_token_id: int, reference_token_ids: Sequence[int]) -> BranchEvalResult:
    """One branch's teacher-forced evaluation. `reference_token_ids` must
    have exactly `SCORED_HORIZON` entries — never padded, never truncated;
    an event that cannot supply the full horizon must be rejected by the
    caller before this function is ever invoked (eligibility, not scoring,
    per `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §3.2).
    """
    if len(reference_token_ids) != SCORED_HORIZON:
        raise ValueError(
            f"reference_token_ids must have exactly {SCORED_HORIZON} entries, got {len(reference_token_ids)}"
        )

    cache_state = initial_cache_state
    logits, cache_state = step_fn(cache_state, bridge_token_id)

    per_token_nll: list[float] = []
    per_token_logits: list[torch.Tensor] = []
    last_index = len(reference_token_ids) - 1
    for k, target_token_id in enumerate(reference_token_ids):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        nll = -log_probs[target_token_id]
        per_token_nll.append(nll.item())
        per_token_logits.append(logits)
        if k < last_index:
            # Feed the real reference token (teacher-forced, never the
            # model's own prediction) to advance to the next step's logits.
            logits, cache_state = step_fn(cache_state, target_token_id)

    return BranchEvalResult(
        per_token_nll=per_token_nll,
        per_token_logits=per_token_logits,
        mean_nll=mean_nll(per_token_nll),
        final_cache_state=cache_state,
    )


@dataclass(frozen=True)
class SwapBranchComparison:
    baseline_per_token_nll: list[float]
    swapped_per_token_nll: list[float]
    baseline_mean_nll: float
    swapped_mean_nll: float
    swap_gain: float
    baseline_final_cache_state: Any
    swapped_final_cache_state: Any
    baseline_per_token_logits: list[torch.Tensor]
    swapped_per_token_logits: list[torch.Tensor]


def evaluate_swap_branches(
    step_fn: StepFn,
    baseline_initial_cache_state: Any,
    swapped_initial_cache_state: Any,
    bridge_token_id: int,
    reference_token_ids: Sequence[int],
) -> SwapBranchComparison:
    """`swap_gain = baseline_mean_nll - swapped_mean_nll` (positive means
    replacing donor `r` with rejected candidate `e` REDUCES mean NLL over
    the scored window — `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §3.3's sign
    convention, restated here so it is never recomputed independently)."""
    baseline = evaluate_branch(step_fn, baseline_initial_cache_state, bridge_token_id, reference_token_ids)
    swapped = evaluate_branch(step_fn, swapped_initial_cache_state, bridge_token_id, reference_token_ids)
    return SwapBranchComparison(
        baseline_per_token_nll=baseline.per_token_nll,
        swapped_per_token_nll=swapped.per_token_nll,
        baseline_mean_nll=baseline.mean_nll,
        swapped_mean_nll=swapped.mean_nll,
        swap_gain=baseline.mean_nll - swapped.mean_nll,
        baseline_final_cache_state=baseline.final_cache_state,
        swapped_final_cache_state=swapped.final_cache_state,
        baseline_per_token_logits=baseline.per_token_logits,
        swapped_per_token_logits=swapped.per_token_logits,
    )


def assert_timing_invariants(
    event_token_absolute_position: int, bridge_token_absolute_position: int, first_scored_absolute_position: int
) -> None:
    """Standalone check of the frozen timing rule (Part II), usable at
    branch-evaluation time before a full `SwapPairRecord` is assembled —
    mirrors (never re-derives independently) the same invariant enforced
    by `kvcot.discovery.schemas.SwapPairRecord`."""
    if bridge_token_absolute_position != event_token_absolute_position + 1:
        raise ValueError("bridge_token_absolute_position must equal event_token_absolute_position + 1")
    if first_scored_absolute_position != bridge_token_absolute_position + 1:
        raise ValueError("first_scored_absolute_position must equal bridge_token_absolute_position + 1 (i.e. t + 2)")
