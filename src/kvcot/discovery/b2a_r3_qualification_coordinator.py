"""B2A-R3 sequential qualification coordinator (Step 3R4 Finding 6,
docs/B2A_R3_STAGE_A_PROTOCOL_ALIGNMENT_AMENDMENT_2026-07-23.md §8).

CPU-testable orchestration only: `fullkv_worker_runner` and `clock` are
always dependency-injected, so this module never initializes CUDA, loads
a model or tokenizer, or imports R-KV -- exactly as every other B2A-R3
Stage-A module. No line in this module writes to a real filesystem path;
it returns the built artifact dict to its caller, who decides whether (and
where) to persist it via
`kvcot.discovery.b2a_r3_artifacts.write_qualification_artifact_atomic` --
this repair round never wires that decision to a real execution path or
CLI command (protocol §14.1/§14.2; CLAUDE.md §1i).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from kvcot.discovery.b2a_r3_artifacts import build_qualification_artifact
from kvcot.discovery.b2a_r3_authorization import (
    AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
    ConsumedAuthorizationContext,
    VerifiedAuthorizationContext,
    _CONSUMED_CONTEXT_TOKEN,
    _VERIFIED_CONTEXT_TOKEN,
)
from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_contract import PER_CANDIDATE_WORKER_TIMEOUT_SECONDS, QUALIFICATION_CANDIDATE_LIMIT
from kvcot.discovery.b2a_r3_qualification import build_qualification_outcome
from kvcot.discovery.b2a_r3_worker_adapter import FullKVWorkerResultR3, adapt_fullkv_worker_result_to_r3_evidence

__all__ = [
    "QualificationCoordinatorRefused",
    "CandidateWorkerTimeout",
    "run_b2a_r3_qualification_coordinator",
]


class QualificationCoordinatorRefused(RuntimeError):
    """Hard, fail-closed coordinator failure -- an unverified authorization
    context, an attempt to exceed the verified context's own limits, an
    authorization/candidate-manifest mismatch, or malformed worker
    evidence. Never raised for a legitimate scientific qualification
    failure (`qualified=False`), which is recorded normally as an
    attempted outcome instead -- conflating the two would let a genuinely
    broken run masquerade as an ordinary negative result."""


class CandidateWorkerTimeout(RuntimeError):
    """Raised BY an injected `fullkv_worker_runner` to signal that this
    specific candidate's worker exceeded its per-candidate timeout --
    distinct from a scientific qualification failure and from malformed
    worker evidence. The coordinator catches only this (and the stdlib
    `TimeoutError`, for injected runners that prefer it) to stop with
    `stopped_reason='candidate_worker_timeout'`."""


def _iso8601(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def run_b2a_r3_qualification_coordinator(
    *,
    candidate_manifest: dict[str, Any],
    expected_config_sha256: str,
    consumed_authorization_context: ConsumedAuthorizationContext,
    fullkv_worker_runner: Callable[[int, int], Any],
    clock: Callable[[], float],
    per_candidate_timeout_seconds: int,
) -> dict[str, Any]:
    """Sequential, first-pass B2A-R3 qualification coordinator.

    `fullkv_worker_runner(candidate_ordinal, timeout_seconds)` must return
    a `kvcot.discovery.b2a_r3_worker_adapter.FullKVWorkerResultR3` (or raise
    `CandidateWorkerTimeout`/`TimeoutError` to signal a per-candidate
    timeout); `clock()` must return a Unix timestamp (float seconds) each
    call, monotonically non-decreasing across the coordinator's lifetime.

    Guarantees:

    1. Requires a `VerifiedAuthorizationContext` actually produced by
       `kvcot.discovery.b2a_r3_authorization.verify_authorization_preconditions`
       for the Stage B (`fullkv_qualification`) stage -- never a
       hand-constructed one, never a Stage C context.
    2. Reads `maximum_candidates`/`phase_wall_time_limit_seconds` ONLY from
       that verified context (itself sourced from the parsed authorization
       document, Step 3R4 Finding 3) -- never from a CLI argument, an
       environment variable, or a hard-coded default.
    3. Verifies the candidate manifest and requires its hash to match the
       one the verified claim actually authorized.
    4. Iterates candidates in exact ordinal order, 0, 1, 2, ....
    5. Never attempts more than `verified_authorization_context.maximum_candidates`.
    6. Never attempts more than the protocol maximum of eight, regardless
       of what any authorization claims.
    7. Checks remaining phase-wide wall time before every worker launch.
    8. Runs exactly one FullKV candidate at a time (a plain `for` loop --
       no concurrency of any kind).
    9. Applies the frozen per-candidate timeout by threading it into the
       injected runner; `per_candidate_timeout_seconds` must equal the one
       frozen protocol constant exactly, never an operator-chosen value.
    10. Converts every worker result through the one canonical adapter.
    11. Builds every outcome through the one authoritative evaluator.
    12. Appends every attempted outcome, pass or fail.
    13. Stops immediately at the first passing candidate.
    14. Never evaluates a candidate after a pass.
    15. Records "no selection" if every authorized candidate fails.
    16. Fails closed (raises `QualificationCoordinatorRefused`) on
        malformed worker evidence -- never silently records it as an
        ordinary scientific rejection.
    17. Never imports or calls R-KV.
    18. Never runs pair evaluation.
    19. Builds the final v2 qualification artifact via
        `build_qualification_artifact` -- the one authoritative builder.
    20. Returns the artifact dict to the caller.
    21. Never writes to any filesystem path itself.
    """
    if not isinstance(consumed_authorization_context, ConsumedAuthorizationContext):
        raise QualificationCoordinatorRefused(
            "consumed_authorization_context must be a ConsumedAuthorizationContext returned by claim_authorization"
        )
    if consumed_authorization_context._consumption_token is not _CONSUMED_CONTEXT_TOKEN:
        raise QualificationCoordinatorRefused("authorization context was not produced by claim_authorization")
    verified_authorization_context = consumed_authorization_context.verified_context
    if not isinstance(verified_authorization_context, VerifiedAuthorizationContext):
        raise QualificationCoordinatorRefused("consumed context does not carry a verified authorization context")
    if verified_authorization_context._verification_token is not _VERIFIED_CONTEXT_TOKEN:
        raise QualificationCoordinatorRefused(
            "verified_authorization_context was not produced by verify_authorization_preconditions"
        )
    claim = verified_authorization_context.claim
    if claim.authorization_stage != AUTHORIZATION_STAGE_FULLKV_QUALIFICATION:
        raise QualificationCoordinatorRefused(
            "verified_authorization_context is not a Stage B (fullkv_qualification) context"
        )

    if per_candidate_timeout_seconds != PER_CANDIDATE_WORKER_TIMEOUT_SECONDS:
        raise QualificationCoordinatorRefused(
            f"per_candidate_timeout_seconds must be the frozen protocol constant "
            f"{PER_CANDIDATE_WORKER_TIMEOUT_SECONDS}, got {per_candidate_timeout_seconds}"
        )

    maximum_candidates = verified_authorization_context.maximum_candidates
    phase_wall_time_limit_seconds = verified_authorization_context.phase_wall_time_limit_seconds
    if maximum_candidates is None or phase_wall_time_limit_seconds is None:
        raise QualificationCoordinatorRefused(
            "verified Stage B context has no maximum_candidates/phase_wall_time_limit_seconds -- these "
            "must come from the parsed authorization document, never a CLI default"
        )
    if not (1 <= maximum_candidates <= QUALIFICATION_CANDIDATE_LIMIT):
        raise QualificationCoordinatorRefused(
            f"maximum_candidates must be in 1..{QUALIFICATION_CANDIDATE_LIMIT}, got {maximum_candidates}"
        )

    manifest = verify_candidate_manifest_structure(
        candidate_manifest, expected_config_sha256=expected_config_sha256
    )
    if manifest.canonical_sha256 != claim.candidate_manifest_canonical_sha256:
        raise QualificationCoordinatorRefused(
            "candidate manifest hash does not match the one the verified authorization claim authorized"
        )

    effective_limit = min(maximum_candidates, QUALIFICATION_CANDIDATE_LIMIT, len(manifest.candidates))

    started_at = clock()
    attempted: list[dict[str, Any]] = []
    stopped_reason = "all_authorized_candidates_exhausted"

    for ordinal in range(effective_limit):
        elapsed_seconds = clock() - started_at
        remaining_phase_seconds = phase_wall_time_limit_seconds - elapsed_seconds
        if remaining_phase_seconds <= 0:
            stopped_reason = "phase_wall_time_exhausted"
            break

        # Step 3R4-Repair-2 Finding 4: the worker is never handed the full
        # frozen per-candidate timeout unconditionally -- it is capped at
        # however much of the AUTHORIZED phase-wide wall time actually
        # remains, so a candidate started near the end of the phase cannot
        # run for up to `PER_CANDIDATE_WORKER_TIMEOUT_SECONDS` regardless of
        # how little authorized time is left.
        effective_worker_timeout = min(per_candidate_timeout_seconds, remaining_phase_seconds)

        try:
            worker_result = fullkv_worker_runner(ordinal, effective_worker_timeout)
        except (CandidateWorkerTimeout, TimeoutError):
            stopped_reason = (
                "phase_wall_time_exhausted"
                if effective_worker_timeout == remaining_phase_seconds
                else "candidate_worker_timeout"
            )
            break

        if (clock() - started_at) >= phase_wall_time_limit_seconds:
            stopped_reason = "phase_wall_time_exhausted"
            break

        try:
            evidence = adapt_fullkv_worker_result_to_r3_evidence(
                worker_result=worker_result,
                candidate_manifest=candidate_manifest,
                candidate_ordinal=ordinal,
                expected_config_sha256=expected_config_sha256,
            )
            outcome = build_qualification_outcome(
                evidence, candidate_manifest=candidate_manifest, expected_config_sha256=expected_config_sha256,
            )
        except Exception as exc:  # noqa: BLE001 -- fail closed; never mis-record as a scientific rejection
            raise QualificationCoordinatorRefused(
                f"candidate ordinal={ordinal} produced malformed worker evidence: {exc}"
            ) from exc

        attempted.append(outcome)
        if outcome["qualified"]:
            stopped_reason = "first_pass"
            break

        # Finding 4: also re-check elapsed phase-wide wall time immediately
        # after every worker finishes -- never rely solely on the pre-launch
        # check ahead of a NEXT candidate that may never come (e.g. this was
        # already the last authorized candidate, in which case the loop
        # would otherwise exit normally and mis-report
        # "all_authorized_candidates_exhausted" even though the phase
        # deadline had already passed).
        if (clock() - started_at) >= phase_wall_time_limit_seconds:
            stopped_reason = "phase_wall_time_exhausted"
            break

    completed_at = clock()

    return build_qualification_artifact(
        attempted_outcomes=attempted,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
        stopped_reason=stopped_reason,
        authorized_maximum_candidates=maximum_candidates,
        authorized_phase_wall_time_limit_seconds=phase_wall_time_limit_seconds,
        consumed_authorization_context=consumed_authorization_context,
        attempt_started_at_utc=_iso8601(started_at),
        attempt_completed_at_utc=_iso8601(completed_at),
    )
