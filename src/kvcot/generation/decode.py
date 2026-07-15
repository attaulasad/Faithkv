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
from dataclasses import dataclass

import torch

from kvcot.generation.sampling import sample_next_token, greedy_next_token


@dataclass
class GenerationResult:
    generated_token_ids: list[int]
    cap_hit: bool
    wall_time_seconds: float
    final_absolute_position: int


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


def decode_step(model, cache, last_token_id: int, absolute_position: int, device: str) -> torch.Tensor:
    """One single-token forward call. docs/REPLAY_DESIGN.md §2: decode must
    be single-token calls, matching the frozen batch-1 loop, so
    `self.length` increments by exactly 1 per call during decode. Never
    derives `position_ids`/`cache_position` from the (possibly compressed)
    cache length — `absolute_position` is tracked explicitly by the caller
    and threaded through here (§3.3)."""
    input_ids = torch.tensor([[last_token_id]], dtype=torch.long, device=device)
    position_ids = torch.tensor([[absolute_position]], device=device)
    cache_position = torch.tensor([absolute_position], device=device)
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

    wall_time = time.monotonic() - start
    return GenerationResult(
        generated_token_ids=generated,
        cap_hit=cap_hit,
        wall_time_seconds=wall_time,
        final_absolute_position=absolute_position,
    )


def generate_probe_answer(
    model,
    cache,
    last_token_id: int,
    absolute_position: int,
    max_new_tokens: int,
    eos_token_id: int,
    device: str,
) -> list[int]:
    """Greedy decoding for the 48-token answer probe (§4). `last_token_id`
    is the final token already fed into the model (the last token of the
    teacher-forced control suffix); this function performs the first
    forward call to get its logits' successor, matching the same
    single-token call shape as `decode_step`.
    """
    logits = decode_step(model, cache, last_token_id, absolute_position, device)
    absolute_position += 1

    generated: list[int] = []
    for _ in range(max_new_tokens):
        next_id = greedy_next_token(logits)
        token_id = int(next_id.item())
        generated.append(token_id)
        if token_id == eos_token_id:
            break
        logits = decode_step(model, cache, token_id, absolute_position, device)
        absolute_position += 1
    return generated
