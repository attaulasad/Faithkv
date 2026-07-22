"""B2A-R2 forensic pair-record persistence repair
(`docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`).

Pure Python (no torch import, matching `kvcot.discovery.schemas`/`kvcot
.discovery.nll`'s existing discipline) -- a deterministic summary computed
exclusively from already-validated `SwapPairRecord` objects. Never a second,
independently-written swap-gain or identity formula: every statistic here
reads `record.swap_gain`/`record.score_margin_e_minus_r` directly, both
already cross-validated by `SwapPairRecord`'s own model validators.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Sequence

from kvcot.discovery.schemas import SwapPairRecord
from kvcot.utils.hashing import sha256_json

SCIENTIFIC_SUMMARY_SCHEMA_VERSION = "b2a_scientific_summary.v1"

# One documented positive-gain threshold, used nowhere else -- never a
# second, independently-chosen constant.
_MEANINGFUL_GAIN_THRESHOLD = 0.01


def _rank(values: Sequence[float]) -> list[float]:
    """Tie-aware average ranks (1-indexed) -- the standard Spearman
    tie-handling convention: every member of a tied group receives the mean
    of the ordinal ranks the group spans."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def tie_aware_spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Deterministic, dependency-free Spearman rank correlation (Pearson
    correlation of tie-aware average ranks -- the standard equivalence).
    `None` (never a fabricated 0.0 or 1.0) when: fewer than two pairs,
    either series contains a non-finite value, or either series has zero
    variance (a constant series has no rank correlation to report)."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must be the same length")
    if len(xs) < 2:
        return None
    if not all(math.isfinite(v) for v in xs) or not all(math.isfinite(v) for v in ys):
        return None
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return None

    rank_x = _rank(list(xs))
    rank_y = _rank(list(ys))
    n = len(rank_x)
    mean_x = sum(rank_x) / n
    mean_y = sum(rank_y) / n
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(rank_x, rank_y))
    variance_x = sum((a - mean_x) ** 2 for a in rank_x)
    variance_y = sum((b - mean_y) ** 2 for b in rank_y)
    if variance_x == 0.0 or variance_y == 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def build_scientific_summary(pair_records: Sequence[SwapPairRecord]) -> dict[str, Any]:
    """Pure CPU summary derived exclusively from validated pair records.
    `pair_records_sha256` is computed over the FULL population (real + the
    no-op control, in the order given) -- the same canonical hash
    `rkv/pair_records.json` is verified against
    (`kvcot.discovery.attempt_verification`), so a caller can bind this
    summary to the exact file it describes.

    Every mean/median/min/max statistic is computed over VALID (never
    invalid-placeholder) real records only -- `SwapPairRecord` only
    guarantees finite `swap_gain`/NLL values when `valid_flag=True`
    (`_nll_finiteness_for_valid_records`). `None` (never a fabricated 0.0)
    when zero valid real records exist -- an honestly empty statistic, not a
    silently wrong one.
    """
    real_records = [record for record in pair_records if not record.is_noop_control]
    no_op_records = [record for record in pair_records if record.is_noop_control]
    valid_real_records = [record for record in real_records if record.valid_flag]

    gains = [record.swap_gain for record in valid_real_records]
    positive_gain_count = sum(1 for gain in gains if gain > 0.0)
    gain_above_0_01_count = sum(1 for gain in gains if gain > _MEANINGFUL_GAIN_THRESHOLD)

    if gains:
        median_swap_gain: float | None = statistics.median(gains)
        mean_swap_gain: float | None = statistics.mean(gains)
        minimum_swap_gain: float | None = min(gains)
        maximum_swap_gain: float | None = max(gains)
    else:
        median_swap_gain = mean_swap_gain = minimum_swap_gain = maximum_swap_gain = None

    spearman = tie_aware_spearman(
        [record.score_margin_e_minus_r for record in valid_real_records],
        [record.swap_gain for record in valid_real_records],
    )

    return {
        "schema_version": SCIENTIFIC_SUMMARY_SCHEMA_VERSION,
        "pair_records_sha256": sha256_json([record.model_dump(mode="json") for record in pair_records]),
        "real_pair_count": len(real_records),
        "no_op_pair_count": len(no_op_records),
        "positive_gain_count": positive_gain_count,
        "gain_above_0_01_count": gain_above_0_01_count,
        "median_swap_gain": median_swap_gain,
        "mean_swap_gain": mean_swap_gain,
        "minimum_swap_gain": minimum_swap_gain,
        "maximum_swap_gain": maximum_swap_gain,
        "spearman_score_margin_vs_swap_gain": spearman,
    }
