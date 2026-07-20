"""Shared dependency-injection types for the B1B/B1B-R2 CPU harness
architecture (`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9,
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §6, authorized by
CLAUDE.md §1b/§4b -- CPU-side harness ARCHITECTURE only, dependency-injected
synthetic/deterministic components in CPU tests, never a real model).

## Prefill/decode call-boundary split (B1B-R2 §6)

A real model's prompt is fed through ONE prefill forward call (all prompt
positions processed together, R-KV's own per-step schedule/state can behave
differently there than during single-token decode), then every subsequent
token is fed through its own single-token decode call
(`docs/REPLAY_DESIGN.md` §2, CLAUDE.md §4's frozen "batch-1, token-by-token
decode loop ... never `model.generate()`" rule). The prior version of this
harness fed EVERY position -- prompt included -- through one generic
per-token `step_fn`, which cannot distinguish "this is the one-shot prefill
call" from "this is decode call #1"; a real integration plugging into that
seam could not honor the real prefill/decode boundary at all. `PrefillFn`
and `DecodeOneFn` are the two seams a real-model integration would plug
into later (still out of scope here): `PrefillFn` is called EXACTLY ONCE
per pass, with the complete prompt; `DecodeOneFn` is called once per
continuation token thereafter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from kvcot.generation.state import ModelStateSnapshot


@dataclass(frozen=True)
class LayerStepObservation:
    """One layer's bookkeeping for a single decode/prefill-position.
    `pre_event_absolute_position_map`/`observed_kept_absolute_positions`
    are populated only when `had_compaction` is True -- exactly the shape
    `kvcot.discovery.capture.capture_update_kv` and
    `kvcot.generation.provenance.LayerProvenance` already use, reused here
    rather than reinvented."""

    had_compaction: bool
    cache_length_after: int
    pre_event_absolute_position_map: torch.Tensor | None = None
    observed_kept_absolute_positions: torch.Tensor | None = None
    # Needed to identify the protected-recent-window TAIL of
    # `observed_kept_absolute_positions` (the last `window_size` columns,
    # `kvcot.discovery.capture`'s own `torch.cat([topk, recent_window])`
    # ordering) -- those positions are always kept, never competing on
    # score, so Pass 1's candidate/donor pool selection excludes them
    # (`kvcot.discovery.pass1._pools_for_layer_head`).
    window_size: int | None = None


@dataclass(frozen=True)
class NaturalStepResult:
    """Result of feeding one DECODE token into the synthetic (or, in a
    future, out-of-scope integration, real) model for one single-token
    forward call. `next_token_logits` are the raw, 1-D vocabulary logits
    produced by THIS call -- the logits that will be used to select the
    NEXT token, exactly matching `kvcot.discovery.uncertainty`'s
    `raw_next_token_logits_at_token_prediction_time` contract."""

    next_token_logits: torch.Tensor
    new_state: Any
    layer_observations: dict[int, LayerStepObservation]


# (state, token_id) -> NaturalStepResult. Called EXACTLY ONCE per
# continuation token -- never used for the prompt, which goes through
# `PrefillFn` instead. Deliberately model-agnostic, so a CPU test can supply
# a small deterministic toy implementation and this harness never imports or
# downloads a real model.
DecodeOneFn = Callable[[Any, int], NaturalStepResult]

# Backwards-compatible alias: pre-B1B-R2 code referred to this shape as
# `NaturalStepFn`. Kept as a plain alias (never a redefinition) so any
# remaining reference resolves to the identical type as `DecodeOneFn`.
NaturalStepFn = DecodeOneFn


@dataclass(frozen=True)
class PrefillStepResult:
    """Result of feeding the COMPLETE prompt through exactly one prefill
    call. `per_position_logits[i]` are the raw next-token logits produced
    by predicting the token immediately after `prompt_token_ids[i]` (the
    same "logits that predict position i+1" contract `NaturalStepResult`
    uses for decode) -- length always equals `len(prompt_token_ids)`, one
    entry per prompt position, never aggregated or subsampled.
    `per_position_layer_observations[i]` is that same position's per-layer
    compaction bookkeeping, in the identical shape `NaturalStepResult.
    layer_observations` uses -- a real prefill call can still trigger
    mid-prompt compaction events (a long enough prompt exceeds budget before
    the first generated token), and this is how the harness observes them
    without ever breaking the prefill into repeated one-token calls."""

    new_state: Any
    per_position_logits: tuple[torch.Tensor, ...]
    per_position_layer_observations: tuple[dict[int, LayerStepObservation], ...]


# (state, prompt_token_ids) -> PrefillStepResult. Called EXACTLY ONCE per
# pass, with the COMPLETE prompt -- never once per prompt token. A real
# integration plugs in a single `model(input_ids=prompt_token_ids)` forward
# call here; the CPU synthetic tests supply a small deterministic function
# that internally loops over prompt positions but is invoked by Pass 1/
# Pass 2 exactly once, matching the real call-count contract this section
# exists to freeze.
PrefillFn = Callable[[Any, Sequence[int]], PrefillStepResult]


# state -> a complete post-event model-state snapshot
# (`kvcot.generation.state.ModelStateSnapshot`), used by Pass 2 to capture
# ONE pristine, complete snapshot per selected target event
# (`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §5) -- never a
# single layer's returned K/V tensors standing in for the whole model's
# continuation state. A real integration plugs in
# `kvcot.generation.replay.capture_snapshot` (already the primary pipeline's
# own complete-state capture); CPU synthetic tests supply a small
# deterministic builder reading every field off the synthetic harness state.
SnapshotFn = Callable[[Any], ModelStateSnapshot]
