"""Deterministic sampling utilities for the B1A discovery schema (Part III /
Part VII.17 of `docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`).

Frozen algorithm family, all built on one canonical SHA-256 seed helper.
Each sampling decision uses its own literal seed-suffix so the draws are
independent of one another. Every pool argument is sorted before sampling,
so caller-side dict/set iteration order can never affect the output.

Pure Python (hashlib + random only) — no torch import, so this module is
usable from CPU-only planning/test code exactly like
`kvcot.utils.seeding`.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence


def sha256_seed(*parts: object) -> int:
    """Canonical seed derivation reused by every draw below: pipe-joined
    `str()` of each part, UTF-8 encoded, SHA-256 hashed, first 8 digest
    bytes interpreted as a big-endian unsigned 64-bit integer. `random.Random`
    accepts arbitrary-size Python ints directly, so no additional masking is
    applied (unlike `kvcot.utils.seeding.derive_seed`'s 63-bit clamp, which
    exists only for downstream JSON/other-language interoperability)."""
    canonical = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


@dataclass(frozen=True)
class IdentitySeedParts:
    """The identity tuple every draw in this module is seeded from, besides
    its own literal disambiguating suffix."""

    global_seed: int
    dataset_name: str
    problem_index: int
    model_revision: str
    rkv_revision: str


@dataclass(frozen=True)
class EventSelection:
    selected_events_chronological: tuple[int, int, int]
    chronological_ordinal_by_event: dict[int, int]


def select_events(eligible_event_ids: Sequence[int], identity: IdentitySeedParts) -> EventSelection | None:
    """Select exactly 3 eligible compaction events without replacement.
    Returns None (example ineligible, never "run with fewer") if fewer than
    3 eligible events exist. Input is deduplicated and sorted ascending
    before sampling, so caller-side ordering is irrelevant."""
    eligible_sorted = sorted(set(eligible_event_ids))
    if len(eligible_sorted) < 3:
        return None
    event_seed = sha256_seed(
        identity.global_seed,
        identity.dataset_name,
        identity.problem_index,
        identity.model_revision,
        identity.rkv_revision,
        "b05r21_event",
    )
    rng = random.Random(event_seed)
    selected = rng.sample(eligible_sorted, 3)
    selected_sorted = tuple(sorted(selected))
    ordinal = {event_id: k for k, event_id in enumerate(selected_sorted)}
    return EventSelection(selected_events_chronological=selected_sorted, chronological_ordinal_by_event=ordinal)


@dataclass(frozen=True)
class DepthStratumAssignment:
    """`depth_strata_permutation[k]` is the depth stratum assigned to the
    event at chronological ordinal `k`. Deliberately independent of
    `EventSelection`'s own seed, so which three events are chosen and which
    depth third each lands in are two separate randomized decisions —
    fixing the layer-depth/event-time confound (Part III)."""

    depth_strata_permutation: tuple[int, int, int]
    depth_stratum_by_event: dict[int, int]


def assign_depth_strata(
    selected_events_chronological: Sequence[int], identity: IdentitySeedParts
) -> DepthStratumAssignment:
    """Independently permute {0, 1, 2} (early/middle/late third) and assign
    by chronological ordinal — never assign depth stratum == chronological
    ordinal directly, which would silently confound compaction time with
    layer depth."""
    depth_permutation_seed = sha256_seed(
        identity.global_seed,
        identity.dataset_name,
        identity.problem_index,
        identity.model_revision,
        identity.rkv_revision,
        "b05r22_depth_permutation",
    )
    depth_strata = tuple(random.Random(depth_permutation_seed).sample([0, 1, 2], 3))
    depth_stratum_by_event = {
        event_id: depth_strata[chronological_ordinal]
        for chronological_ordinal, event_id in enumerate(selected_events_chronological)
    }
    return DepthStratumAssignment(
        depth_strata_permutation=depth_strata, depth_stratum_by_event=depth_stratum_by_event
    )


@dataclass(frozen=True)
class LayerSelection:
    layer_index: int
    depth_stratum: int
    lo: int
    hi: int


def select_layer(
    event_id: int, depth_stratum: int, num_hidden_layers: int, identity: IdentitySeedParts
) -> LayerSelection:
    """Draw a layer uniformly within the assigned depth stratum's third of
    the model's depth range — never from the full range (that would not
    actually guarantee coverage; see Part III)."""
    if num_hidden_layers < 3:
        raise ValueError(f"num_hidden_layers must be >= 3 for depth-third partitioning, got {num_hidden_layers}")
    if depth_stratum not in (0, 1, 2):
        raise ValueError(f"depth_stratum must be 0, 1, or 2, got {depth_stratum}")
    lo = (depth_stratum * num_hidden_layers) // 3
    hi = ((depth_stratum + 1) * num_hidden_layers) // 3
    if hi <= lo:
        raise ValueError(
            f"depth stratum {depth_stratum} produced an empty layer range [{lo}, {hi}) "
            f"for num_hidden_layers={num_hidden_layers}"
        )
    layer_seed = sha256_seed(
        identity.global_seed,
        identity.dataset_name,
        identity.problem_index,
        identity.model_revision,
        identity.rkv_revision,
        event_id,
        depth_stratum,
        "b05r22_layer",
    )
    layer_index = lo + (layer_seed % (hi - lo))
    return LayerSelection(layer_index=layer_index, depth_stratum=depth_stratum, lo=lo, hi=hi)


def select_kv_head(event_id: int, num_key_value_heads: int, identity: IdentitySeedParts) -> int:
    """Draw a KV head uniformly over the full range, independent of the
    layer draw — head selection has no depth-coverage requirement."""
    if num_key_value_heads <= 0:
        raise ValueError(f"num_key_value_heads must be > 0, got {num_key_value_heads}")
    head_seed = sha256_seed(
        identity.global_seed,
        identity.dataset_name,
        identity.problem_index,
        identity.model_revision,
        identity.rkv_revision,
        event_id,
        "b05r22_head",
    )
    return head_seed % num_key_value_heads


def cross_product_pairs(evicted_selected: Sequence[int], donor_selected: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """The four (evicted, donor) swap-branch pairs for one selected
    (event, layer, head) — `2 evicted x 2 donors`, nested within one event
    and never treated as four independent samples."""
    return tuple((e, r) for e in evicted_selected for r in donor_selected)


@dataclass(frozen=True)
class CandidateDonorSelection:
    evicted_selected: tuple[int, int]
    donor_selected: tuple[int, int]
    cross_product: tuple[tuple[int, int], ...]


def select_candidates_and_donors(
    evicted_pool: Sequence[int],
    retained_pool: Sequence[int],
    event_id: int,
    layer_index: int,
    kv_head_index: int,
    identity: IdentitySeedParts,
) -> CandidateDonorSelection | None:
    """Sample 2 evicted candidates and 2 retained donors, independently
    seeded, over the sorted (never caller-order-dependent) pools. Returns
    None (the whole event is invalidated, never partially sampled) if
    either pool has fewer than 2 entries. Selection never depends on any
    swap outcome — both draws are made from Pass-1-style pool bookkeeping
    alone, before any branch is constructed or scored, by construction of
    this function's signature (it never takes an outcome/gain argument)."""
    evicted_pool_sorted = sorted(set(evicted_pool))
    retained_pool_sorted = sorted(set(retained_pool))
    if len(evicted_pool_sorted) < 2 or len(retained_pool_sorted) < 2:
        return None

    evicted_seed = sha256_seed(
        identity.global_seed,
        identity.dataset_name,
        identity.problem_index,
        identity.model_revision,
        identity.rkv_revision,
        event_id,
        layer_index,
        kv_head_index,
        "b05r21_evicted",
    )
    evicted_rng = random.Random(evicted_seed)
    evicted_selected = tuple(evicted_rng.sample(evicted_pool_sorted, 2))

    donor_seed = sha256_seed(
        identity.global_seed,
        identity.dataset_name,
        identity.problem_index,
        identity.model_revision,
        identity.rkv_revision,
        event_id,
        layer_index,
        kv_head_index,
        "b05r21_donor",
    )
    donor_rng = random.Random(donor_seed)
    donor_selected = tuple(donor_rng.sample(retained_pool_sorted, 2))

    return CandidateDonorSelection(
        evicted_selected=evicted_selected,
        donor_selected=donor_selected,
        cross_product=cross_product_pairs(evicted_selected, donor_selected),
    )
