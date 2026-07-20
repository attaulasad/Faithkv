"""Explicit batch-1, token-by-token decode loop (§4, §6). Never uses
`model.generate()` on the state-critical path — that is a hard requirement,
not a style preference: `generate()`'s internal batching/scheduling is not
guaranteed to preserve the exact prefill-then-single-token call shape this
repository's replay depends on to reproduce R-KV's `self.length` trajectory
(docs/REPLAY_DESIGN.md §2).

Batch size is frozen at 1 (§4) with no padding, so `attention_mask` is
omitted from every forward call here — each attention layer's `is_causal`
flag (set at `modeling.py:216`/`Qwen2Attention_init`) already produces the
correct causal mask for a single unpadded sequence; there is no padding
mask to supply.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from kvcot.generation.sampling import sample_next_token


@dataclass
class GenerationResult:
    generated_token_ids: list[int]
    cap_hit: bool
    wall_time_seconds: float
    final_absolute_position: int
    # Number of real R-KV compaction EVENTS during this generation (one count,
    # NOT events * n_layers). 0 for FullKV / patched-noop-that-never-fired.
    compaction_count: int = 0
    # Absolute token positions (index into the prompt+generated stream, i.e.
    # `self.length` at the forward call that fired the event) at which each
    # compaction event occurred — the same convention replay_and_snapshot
    # records, so base and replay agree. Length == compaction_count.
    compaction_event_steps: list[int] = field(default_factory=list)


def _rkv_layer_event_counts(model) -> list[int]:
    """Per-R-KV-layer count of recorded compaction events
    (`len(kv_cluster.kept_token_indices)`), for layers that actually track it.
    Empty for stock FullKV. All R-KV layers share one compression schedule and
    identical key lengths (modeling.py: one `compression` flag set for every
    layer at once, per-step), so these counts must all be equal — the caller
    asserts that and takes a single representative value rather than summing
    across layers (summing is exactly the ×n_layers inflation bug)."""
    counts: list[int] = []
    for layer in model.model.layers:
        kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
        if kv_cluster is not None and getattr(kv_cluster, "record_kept_token_indices", False):
            counts.append(len(kv_cluster.kept_token_indices))
    return counts


def _representative_event_count(model) -> int:
    counts = _rkv_layer_event_counts(model)
    return counts[0] if counts else 0


def prefill(model, cache, prompt_token_ids: list[int], device: str) -> tuple[torch.Tensor, int]:
    """One N-token forward call over the full prompt. docs/REPLAY_DESIGN.md
    §2: prefill must be a single multi-token call, not N single-token
    calls, to reproduce the correct `self.length` trajectory that the
    original run's replay is trying to match. Returns (last-position
    logits, absolute position after prefill)."""
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    n = len(prompt_token_ids)
    position_ids = torch.arange(0, n, device=device).unsqueeze(0)
    cache_position = torch.arange(0, n, device=device)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
    return out.logits[0, -1, :], n


def decode_step(
    model, cache, last_token_id: int, absolute_position: int, device: str, call_observer=None
) -> torch.Tensor:
    """One single-token forward call. docs/REPLAY_DESIGN.md §2: decode must
    be single-token calls, matching the frozen batch-1 loop, so
    `self.length` increments by exactly 1 per call during decode. Never
    derives `position_ids`/`cache_position` from the (possibly compressed)
    cache length — `absolute_position` is tracked explicitly by the caller
    and threaded through here (§3.3)."""
    input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=device)
    position_ids = torch.tensor([[absolute_position]], device=device)
    cache_position = torch.tensor([absolute_position], device=device)
    if call_observer is not None:
        call_observer("decode", input_ids, position_ids, cache_position)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            cache_position=cache_position,
        )
    return out.logits[0, -1, :]


def generate_base(
    model,
    cache,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generator: torch.Generator,
    eos_token_id: int,
    device: str,
) -> GenerationResult:
    """§4: sampling, temperature 0.6, top-p 0.95, one sequence,
    `max_new_tokens=6144` (never `max_length` — nothing in this function
    accepts or derives a total-sequence-length cap; the only stopping
    conditions are EOS and `max_new_tokens` generated tokens)."""
    start = time.monotonic()
    logits, absolute_position = prefill(model, cache, prompt_token_ids, device)

    # Compaction-event tracking (§8.3/§12). One count, recorded at the absolute
    # position of the forward call that fired it — never events * n_layers. A
    # single forward call can fire at most one event per layer (kept_token_
    # indices grows by 1), and all R-KV layers fire together, so tracking one
    # representative count is sufficient; final consistency is asserted below.
    compaction_event_steps: list[int] = []
    prev_event_count = _representative_event_count(model)  # events fired during prefill
    compaction_event_steps.extend([absolute_position] * prev_event_count)

    generated: list[int] = []
    cap_hit = True
    for _ in range(max_new_tokens):
        next_id = sample_next_token(logits, temperature, top_p, generator)
        token_id = int(next_id.item())
        generated.append(token_id)
        if token_id == eos_token_id:
            cap_hit = False
            break
        logits = decode_step(model, cache, token_id, absolute_position, device)
        absolute_position += 1
        cur_event_count = _representative_event_count(model)
        if cur_event_count > prev_event_count:
            compaction_event_steps.extend([absolute_position] * (cur_event_count - prev_event_count))
            prev_event_count = cur_event_count

    layer_counts = _rkv_layer_event_counts(model)
    if len(set(layer_counts)) > 1:
        raise AssertionError(
            f"R-KV layers disagree on compaction event count ({sorted(set(layer_counts))}) — "
            "they share one per-step compression schedule and must all fire the same events "
            "(modeling.py CausalLM_forward sets the compression flag for every layer at once)."
        )
    compaction_count = layer_counts[0] if layer_counts else 0

    wall_time = time.monotonic() - start
    return GenerationResult(
        generated_token_ids=generated,
        cap_hit=cap_hit,
        wall_time_seconds=wall_time,
        final_absolute_position=absolute_position,
        compaction_count=compaction_count,
        compaction_event_steps=compaction_event_steps,
    )
