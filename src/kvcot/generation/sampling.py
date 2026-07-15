"""Seeded sampling for base generation and greedy decoding for probes (§4).

Imports torch at module scope by design — only ever imported from a real
generation/replay code path (see pyproject.toml's note on deferred GPU
imports), never from `--dry-run` or `kvcot.analysis`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from kvcot.utils.seeding import derive_seed, seed_all_cpu_rngs


def make_generator(
    global_seed: int, dataset_name: str, problem_index: int, device: str
) -> tuple[torch.Generator, int]:
    """Construct the per-example torch.Generator per §4: seeded via
    SHA-256 of (global_seed, dataset_name, problem_index) —
    kvcot.utils.seeding.derive_seed takes no condition/method parameter, so
    FullKV and R-KV calling this with identical arguments structurally
    receive the identical derived seed for the same example.
    """
    seed = derive_seed(global_seed, dataset_name, problem_index)
    seed_all_cpu_rngs(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen, seed


def sample_next_token(
    logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator
) -> torch.Tensor:
    """Temperature + top-p (nucleus) sampling for base generation (§4:
    temperature=0.6, top_p=0.95, single sequence, batch size 1). `logits`
    is the last-position logits, shape (vocab_size,) or (seq_len,
    vocab_size) — only the final position is used. Returns a 0-dim
    LongTensor token id.
    """
    if logits.dim() == 2:
        logits = logits[-1]
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Standard nucleus-sampling mask: mark tokens whose *inclusion* would
    # push cumulative probability strictly past top_p, then shift the mask
    # right by one position so the token that actually crosses the
    # threshold is kept (only tokens strictly after it are dropped), and
    # always keep the single highest-probability token regardless of top_p.
    remove_mask = cumulative_probs > top_p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    filtered_probs = sorted_probs.masked_fill(remove_mask, 0.0)
    filtered_probs = filtered_probs / filtered_probs.sum()

    sampled_sorted_index = torch.multinomial(filtered_probs, num_samples=1, generator=generator)
    token_id = sorted_indices[sampled_sorted_index]
    return token_id.view(())


def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
    """Deterministic argmax decoding, used only for the 48-token answer
    probe (§4) — never for the long base reasoning generation."""
    if logits.dim() == 2:
        logits = logits[-1]
    return torch.argmax(logits, dim=-1)
