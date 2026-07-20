"""Canonical mean-NLL helper (Blocker 3 repair,
`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`). One documented formula, used
by both the producer (`kvcot.discovery.branch_eval.evaluate_branch`) and
the validator (`kvcot.discovery.schemas.SwapPairRecord`) -- never two
independent implementations of the same arithmetic drifting apart.

Pure Python (no torch import) so `kvcot.discovery.schemas` can use it
without pulling torch into the CPU-only, no-analysis-torch-import-tested
half of this repository's import graph.
"""
from __future__ import annotations

from typing import Sequence


def mean_nll(per_token_nll: Sequence[float]) -> float:
    """Arithmetic mean of a non-empty sequence of per-token NLL values.
    Deliberately does not guard against an empty sequence with a fabricated
    default (e.g. `0.0`) -- an empty horizon is a caller bug, not a
    zero-loss branch, and must raise (`ZeroDivisionError`) rather than be
    silently repaired."""
    return sum(per_token_nll) / len(per_token_nll)
