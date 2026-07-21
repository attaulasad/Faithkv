"""Canonical mismatch record for sequence comparisons (independent-audit
Gate H3). Every replay/trace comparison in `kvcot.discovery.b2a_workers`
used to export only a bare `first_mismatch_index` -- diagnosing WHY two
sequences diverged then required re-running the model to see what the
values actually were at that position. `build_mismatch_record` instead
captures the expected/observed values (and lengths) at the first point of
divergence, so a mismatch is diagnosable directly from the artifact.

Pure Python, no torch import -- usable from CPU tests exactly like the rest
of `kvcot.discovery`'s evidence-construction modules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

MISMATCH_KIND_VALUE_DIFFERS = "value_differs"
MISMATCH_KIND_EXPECTED_ENDS_FIRST = "expected_ends_first"
MISMATCH_KIND_OBSERVED_ENDS_FIRST = "observed_ends_first"
MISMATCH_KIND_NONE = "matched"


@dataclass(frozen=True)
class MismatchRecord:
    matched: bool
    first_mismatch_index: int | None
    expected_value: Any
    observed_value: Any
    expected_length: int
    observed_length: int
    mismatch_kind: str

    def export(self) -> dict[str, Any]:
        return asdict(self)


def build_mismatch_record(expected: Sequence[Any], observed: Sequence[Any]) -> MismatchRecord:
    """Compares `expected` and `observed` element-by-element and reports
    the first point of divergence, if any -- never indexing beyond either
    sequence's own length. A length mismatch with no earlier value
    difference is reported as `expected_ends_first`/`observed_ends_first`
    at the shorter sequence's length (the first index where one sequence
    has a value and the other does not)."""
    expected_length = len(expected)
    observed_length = len(observed)
    shortest = min(expected_length, observed_length)
    for index in range(shortest):
        if expected[index] != observed[index]:
            return MismatchRecord(
                matched=False,
                first_mismatch_index=index,
                expected_value=expected[index],
                observed_value=observed[index],
                expected_length=expected_length,
                observed_length=observed_length,
                mismatch_kind=MISMATCH_KIND_VALUE_DIFFERS,
            )
    if expected_length == observed_length:
        return MismatchRecord(
            matched=True,
            first_mismatch_index=None,
            expected_value=None,
            observed_value=None,
            expected_length=expected_length,
            observed_length=observed_length,
            mismatch_kind=MISMATCH_KIND_NONE,
        )
    if expected_length < observed_length:
        return MismatchRecord(
            matched=False,
            first_mismatch_index=shortest,
            expected_value=None,
            observed_value=observed[shortest],
            expected_length=expected_length,
            observed_length=observed_length,
            mismatch_kind=MISMATCH_KIND_EXPECTED_ENDS_FIRST,
        )
    return MismatchRecord(
        matched=False,
        first_mismatch_index=shortest,
        expected_value=expected[shortest],
        observed_value=None,
        expected_length=expected_length,
        observed_length=observed_length,
        mismatch_kind=MISMATCH_KIND_OBSERVED_ENDS_FIRST,
    )
