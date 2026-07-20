"""Shared dependency-injection types for the B1B CPU harness architecture
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9, authorized by CLAUDE.md
§1b/§4b -- CPU-side harness ARCHITECTURE only, dependency-injected
synthetic/deterministic components in CPU tests, never a real model).

`NaturalStepFn` is the one seam every real-model integration would plug
into later (out of scope here): given the current per-layer KV-cluster
state and the next token id to feed in, it returns the raw next-token
logits AND, per layer, whether a real R-KV eviction event fired on this
call plus the exact bookkeeping needed to reconstruct absolute-position
survivor identity -- everything Pass 1/Pass 2 need, without either module
importing or assuming anything about how those numbers were produced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class LayerStepObservation:
    """One layer's bookkeeping for a single decode/prefill call.
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
    """Result of feeding one token into the synthetic (or, in a future,
    out-of-scope integration, real) model for one forward call.
    `next_token_logits` are the raw, 1-D vocabulary logits produced by THIS
    call -- the logits that will be used to select the NEXT token, exactly
    matching `kvcot.discovery.uncertainty`'s
    `raw_next_token_logits_at_token_prediction_time` contract."""

    next_token_logits: torch.Tensor
    new_state: Any
    layer_observations: dict[int, LayerStepObservation]


# (state, token_id) -> NaturalStepResult. Mirrors
# `kvcot.discovery.branch_eval.StepFn`'s (logits, new_state) shape, with
# per-layer eviction bookkeeping added -- deliberately model-agnostic, so a
# CPU test can supply a small deterministic toy implementation and this
# harness never imports or downloads a real model.
NaturalStepFn = Callable[[Any, int], NaturalStepResult]
