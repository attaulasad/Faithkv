"""MATH-500 symbolic-equivalence verifier (Part VI of
`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`).

Separate from `kvcot.utils.answers` by design — GSM8K's existing exact/
numeric matching behavior is unchanged by this module. Wraps the pinned
`math-verify[antlr4_13_2]==0.9.0` package, isolated in a fresh child
process per comparison (`kvcot.utils._math_verify_worker`) so the 5.0
second hard timeout is enforced by killing an actual OS process, never a
thread left running past its deadline. See that module's docstring for why
`math_verify`'s own built-in timeout mechanism is disabled here instead of
reused (it uses `multiprocessing`, which was found to raise a real
`OSError` on this Windows development host).

Never imports torch/transformers — this module has no GPU dependency at
all, matching `kvcot.utils`'s existing discipline
(`tests/unit/test_no_analysis_torch_import.py`).
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from typing import Literal

VERIFIER_TIMEOUT_SECONDS = 5.0

MathVerificationStatus = Literal[
    "equivalent",
    "not_equivalent",
    "prediction_unparseable",
    "gold_unparseable",
    "timeout",
    "verifier_error",
]

_STATUS_TO_IS_EQUIVALENT: dict[str, bool | None] = {
    "equivalent": True,
    "not_equivalent": False,
    "prediction_unparseable": None,
    "gold_unparseable": None,
    "timeout": None,
    "verifier_error": None,
}

_VALID_STATUSES = frozenset(_STATUS_TO_IS_EQUIVALENT)


@dataclass(frozen=True)
class MathVerificationResult:
    is_equivalent: bool | None
    status: MathVerificationStatus
    failure_reason: str | None

    def __post_init__(self) -> None:
        expected_is_equivalent = _STATUS_TO_IS_EQUIVALENT[self.status]
        if self.is_equivalent != expected_is_equivalent:
            raise ValueError(
                f"status={self.status!r} requires is_equivalent={expected_is_equivalent!r}, "
                f"got {self.is_equivalent!r} -- every failure mode must map to None, never to False."
            )


def _invoke_worker(payload: dict, timeout_seconds: float) -> MathVerificationResult:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "kvcot.utils._math_verify_worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        # subprocess.run kills the child process on timeout (not merely a
        # thread) and waits for it -- the symbolic work genuinely stops.
        return MathVerificationResult(
            is_equivalent=None, status="timeout", failure_reason=f"verifier subprocess exceeded {timeout_seconds}s"
        )

    if proc.returncode != 0:
        return MathVerificationResult(
            is_equivalent=None,
            status="verifier_error",
            failure_reason=f"worker exited with code {proc.returncode}: {proc.stderr.strip()[:2000]}",
        )

    stdout = proc.stdout.strip()
    if not stdout:
        return MathVerificationResult(
            is_equivalent=None, status="verifier_error", failure_reason="worker produced no stdout output"
        )

    try:
        # The worker prints exactly one JSON line; take the last line in
        # case any warning text was accidentally printed before it.
        parsed = json.loads(stdout.splitlines()[-1])
    except Exception as exc:
        return MathVerificationResult(
            is_equivalent=None,
            status="verifier_error",
            failure_reason=f"could not parse worker stdout as JSON: {exc!r}: {stdout!r}",
        )

    status = parsed.get("status")
    if status not in _VALID_STATUSES:
        return MathVerificationResult(
            is_equivalent=None,
            status="verifier_error",
            failure_reason=f"worker returned an unrecognized status: {status!r}",
        )

    return MathVerificationResult(
        is_equivalent=_STATUS_TO_IS_EQUIVALENT[status],
        status=status,
        failure_reason=parsed.get("failure_reason"),
    )


def _verify_math_equivalence_raw(
    prediction_text: str,
    gold_text: str,
    *,
    timeout_seconds: float,
    extra_payload: dict | None = None,
) -> MathVerificationResult:
    """Internal entry point taking an explicit timeout and an
    `extra_payload` escape hatch (`sleep_seconds`, test-only) — never call
    this from production code; use `verify_math_equivalence` instead, which
    freezes the timeout at `VERIFIER_TIMEOUT_SECONDS`."""
    payload = {"gold_text": gold_text, "prediction_text": prediction_text}
    if extra_payload:
        payload.update(extra_payload)
    return _invoke_worker(payload, timeout_seconds)


def verify_math_equivalence(prediction_text: str, gold_text: str) -> MathVerificationResult:
    """Compare `prediction_text` against `gold_text` for MATH-style
    symbolic equivalence (fractions, algebraic rearrangement, set/interval
    notation, `x=2`-style assignment). Extracts the last valid `\\boxed{}`
    (or bare) expression from each side via `math_verify.parse` — an
    earlier intermediate box never overrides a later, final one, since
    `math_verify`'s own extraction already implements last-valid-box
    semantics. Every failure mode (`prediction_unparseable`,
    `gold_unparseable`, `timeout`, `verifier_error`) maps `is_equivalent`
    to `None`, never to `False` — an answer is never marked correct or
    incorrect on parsing failure, only "not adjudicable" for that
    comparison. Hard 5.0 second timeout, enforced by killing an isolated
    child OS process, always.
    """
    return _verify_math_equivalence_raw(prediction_text, gold_text, timeout_seconds=VERIFIER_TIMEOUT_SECONDS)
