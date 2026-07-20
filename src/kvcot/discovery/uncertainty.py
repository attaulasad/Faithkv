"""Operational definitions of entropy and logit margin (Part IV of
`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`).

Both signals are token-specific, computed from the raw next-token-prediction
logits of the natural R-KV run at the moment the token was actually
predicted — never from the compaction-event logits (unless that event
happened to be the same forward call that predicted the token), never after
a synthetic swap is applied. Every candidate/donor position gets its own
value; they are never shared across all candidates at one eviction event.

Imports torch at module scope by design — this module is only ever used
from a real capture/replay code path (never from `kvcot.analysis` or
`kvcot.cli --dry-run`), matching the existing deferred-import discipline
documented in `kvcot.generation.state`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

UNCERTAINTY_SIGNAL_SOURCE = "raw_next_token_logits_at_token_prediction_time"

POSITION_ZERO_MISSING_REASON = "position_zero_has_no_preceding_prediction_distribution"
NON_FINITE_MISSING_REASON = "computed_value_is_not_finite"
VOCAB_TOO_SMALL_MISSING_REASON = "vocabulary_size_below_2_cannot_form_a_top2_margin"


@dataclass(frozen=True)
class UncertaintySignal:
    """A single scalar signal, or an explicit reason it is unavailable.
    Exactly one of `value`/`missing_reason` is non-None (never both, never
    neither) — enforced in `__post_init__` so a caller can never silently
    treat a missing signal as `0.0`.
    """

    value: float | None
    missing_reason: str | None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.missing_reason is None):
            raise ValueError(
                "UncertaintySignal requires exactly one of value/missing_reason to be set, "
                f"got value={self.value!r} missing_reason={self.missing_reason!r}"
            )

    @property
    def is_available(self) -> bool:
        return self.value is not None


def _missing(reason: str) -> UncertaintySignal:
    return UncertaintySignal(value=None, missing_reason=reason)


def compute_entropy_nats(raw_logits: torch.Tensor) -> UncertaintySignal:
    """Shannon entropy of softmax(raw_logits), in nats.

    `raw_logits` must be the raw vocabulary logits before temperature
    scaling, top-p filtering, sampling, argmax, or any other logits
    processor/warper — a 1-D tensor over the vocabulary dimension. Computed
    in float32, natural log, never normalized by vocabulary size, never
    coerced to zero on non-finite input (`store null plus an explicit
    missing reason` instead, per the frozen spec).

    `raw_logits.ndim` MUST be exactly 1 (Blocker 5 repair). A malformed rank
    (a batch dimension, an extra singleton dimension, a bare 0-D scalar) is
    a PROGRAMMER ERROR at the call site, not a shape this function silently
    accommodates -- raises `ValueError` immediately, never flattens or sums
    over the extra dimension(s) on the caller's behalf.
    """
    if raw_logits.ndim != 1:
        raise ValueError(f"raw_logits must be 1-D (a single vocabulary distribution), got shape {tuple(raw_logits.shape)}")
    if raw_logits.numel() == 0:
        return _missing("empty_logits_tensor")
    z = raw_logits.float()
    log_p = torch.log_softmax(z, dim=-1)
    p = torch.exp(log_p)
    entropy = -(p * log_p).sum()
    value = entropy.item()
    if not torch.isfinite(entropy):
        return _missing(NON_FINITE_MISSING_REASON)
    return UncertaintySignal(value=value, missing_reason=None)


def compute_logit_margin(raw_logits: torch.Tensor) -> UncertaintySignal:
    """Top-1 minus top-2 raw-logit margin, in logit units (not a probability
    margin, not a selected-token log-probability, independent of whether
    the sampled token was top-1). Computed in float32.

    `raw_logits.ndim` MUST be exactly 1 (Blocker 5 repair), for the same
    reason as `compute_entropy_nats`: a malformed rank raises `ValueError`
    immediately, never silently flattened or reduced.
    """
    if raw_logits.ndim != 1:
        raise ValueError(f"raw_logits must be 1-D (a single vocabulary distribution), got shape {tuple(raw_logits.shape)}")
    if raw_logits.numel() < 2:
        return _missing(VOCAB_TOO_SMALL_MISSING_REASON)
    z = raw_logits.float()
    top2 = torch.topk(z, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    if not torch.isfinite(margin):
        return _missing(NON_FINITE_MISSING_REASON)
    return UncertaintySignal(value=margin.item(), missing_reason=None)


@dataclass(frozen=True)
class PairUncertaintySignals:
    """Pair-level (candidate `e`, donor `r`) predictors, §8 of the
    correction document. Source values are always retained alongside the
    difference — never only the unauditable difference."""

    entropy_e: UncertaintySignal
    entropy_r: UncertaintySignal
    entropy_diff: float | None
    logit_margin_e: UncertaintySignal
    logit_margin_r: UncertaintySignal
    logit_margin_diff: float | None
    uncertainty_signal_source: str = UNCERTAINTY_SIGNAL_SOURCE


def compute_pair_uncertainty_signals(
    entropy_e: UncertaintySignal,
    entropy_r: UncertaintySignal,
    logit_margin_e: UncertaintySignal,
    logit_margin_r: UncertaintySignal,
) -> PairUncertaintySignals:
    """entropy_diff = entropy_e - entropy_r, logit_margin_diff analogously —
    only when both source values exist; `None` (never a fabricated zero) if
    either side is missing."""
    entropy_diff = (
        entropy_e.value - entropy_r.value
        if entropy_e.is_available and entropy_r.is_available
        else None
    )
    logit_margin_diff = (
        logit_margin_e.value - logit_margin_r.value
        if logit_margin_e.is_available and logit_margin_r.is_available
        else None
    )
    return PairUncertaintySignals(
        entropy_e=entropy_e,
        entropy_r=entropy_r,
        entropy_diff=entropy_diff,
        logit_margin_e=logit_margin_e,
        logit_margin_r=logit_margin_r,
        logit_margin_diff=logit_margin_diff,
    )


PredictionCallKind = Literal["prefill", "decode", "unavailable"]


@dataclass(frozen=True)
class PredictionLogitSource:
    """Where to look up the raw logits that predicted absolute token
    position `j`: the natural R-KV forward call whose last input token is
    `x_{j-1}` — the prefill call (index `j-1` in its per-position output) if
    `j-1` falls inside the prompt, otherwise the single-token decode call
    that consumed `x_{j-1}` (0-indexed among decode calls)."""

    call_kind: PredictionCallKind
    sequence_index: int | None  # index into that call's output logits
    missing_reason: str | None


def resolve_prediction_logit_source(absolute_position: int, prompt_length: int) -> PredictionLogitSource:
    """§5: token position 0 has no preceding prediction distribution and is
    marked unavailable, never invented. For `j >= 1`, the predicting call's
    last input token is `x_{j-1}`; if `j-1 < prompt_length` that call is the
    one N-token prefill call (this repository's frozen batch-1 decode-loop
    shape) and the relevant output index is `j-1`; otherwise it is the
    single-token decode call that consumed `x_{j-1}`, decode-call index
    `(j-1) - prompt_length` (0-indexed).
    """
    if absolute_position < 0:
        raise ValueError(f"absolute_position must be >= 0, got {absolute_position}")
    if prompt_length <= 0:
        raise ValueError(f"prompt_length must be > 0, got {prompt_length}")
    if absolute_position == 0:
        return PredictionLogitSource(
            call_kind="unavailable", sequence_index=None, missing_reason=POSITION_ZERO_MISSING_REASON
        )
    prev = absolute_position - 1
    if prev < prompt_length:
        return PredictionLogitSource(call_kind="prefill", sequence_index=prev, missing_reason=None)
    decode_call_index = prev - prompt_length
    return PredictionLogitSource(call_kind="decode", sequence_index=decode_call_index, missing_reason=None)
