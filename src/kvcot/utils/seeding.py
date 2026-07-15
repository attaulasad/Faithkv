"""Deterministic, order-independent seed derivation.

Per §4 of the build brief: "Sample with a per-example torch.Generator seeded
via SHA-256 of (global_seed, dataset_name, problem_index) — not Python's
randomized hash(), not iteration order." This module produces that integer;
it does not import torch itself, so it stays usable from CPU-only code
(config validation, dry-run planning, unit tests) — the caller constructs a
`torch.Generator` from the returned int wherever torch is actually available.

FullKV and R-KV receive the identical derived seed for a given
(global_seed, dataset_name, problem_index) triple: this function takes no
condition/method argument by design, so it is structurally impossible to
accidentally derive different seeds per condition.
"""
from __future__ import annotations

import hashlib

# torch.Generator.manual_seed accepts the full uint64 range, but we clamp to
# the positive int64 range (63 bits) so the same integer is also safely
# representable as a signed 64-bit value in JSONL records, other languages'
# JSON parsers, and numpy's legacy RandomState (which is 32-bit but is not
# what we use — this headroom is for tooling that reads the records, not for
# torch itself).
_SEED_MASK = (1 << 63) - 1


def derive_seed(global_seed: int, dataset_name: str, problem_index: int) -> int:
    """Deterministically derive a per-example seed.

    Same (global_seed, dataset_name, problem_index) always yields the same
    seed, regardless of process, platform, dict/iteration order, or any other
    incidental state. Different problem_index values yield independent
    (uncorrelated, not just numerically different) seeds because the index
    is hashed, not offset-added.
    """
    if not isinstance(global_seed, int):
        raise TypeError(f"global_seed must be int, got {type(global_seed)!r}")
    if not isinstance(problem_index, int):
        raise TypeError(f"problem_index must be int, got {type(problem_index)!r}")
    canonical = f"kvcot-seed-v1|{global_seed}|{dataset_name}|{problem_index}"
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return raw & _SEED_MASK


def seed_all_cpu_rngs(seed: int) -> None:
    """Seed Python's and numpy's global RNGs. Does not touch torch — call
    this alongside (not instead of) seeding the torch.Generator used for
    sampling, and alongside torch's own global seeding required for
    `torch.use_deterministic_algorithms` per §6.1 of the brief.
    """
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed & 0xFFFFFFFF)  # numpy legacy API is 32-bit
    except ImportError:
        pass
