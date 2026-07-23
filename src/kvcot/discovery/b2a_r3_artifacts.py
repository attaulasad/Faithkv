"""B2A-R3 strict artifact verification (Step 3 Stage-A, protocol §12.6).

`kvcot.discovery.b2a_r3_candidates` already verifies the candidate
manifest; `kvcot.discovery.b2a_r3_qualification`'s
`CandidateQualificationOutcomeR3` already verifies one attempted outcome's
own internal consistency. This module is the top-level qualification
ARTIFACT schema (protocol §12.6) and its cross-artifact verifier -- the one
place that checks the artifact's own `canonical_sha256`, its bound
candidate-manifest/config hashes, and that its recorded
first-passing-candidate selection reproduces from its own `attempted` list
via `kvcot.discovery.b2a_r3_qualification.select_first_qualified_r3`
(never a second, independently-written selection rule).

No line in this module ever constructs a REAL qualification artifact --
Stage A exercises it only against synthetic/injected fixtures (protocol
§14.1). No torch, no CUDA, no R-KV import.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kvcot.discovery.b2a_r3_contract import (
    BUDGET,
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_ORDER_PROTOCOL_VERSION,
    CONFIG_PATH,
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SPLIT,
    GENERATION_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
    QUALIFICATION_CANDIDATE_LIMIT,
    QUALIFICATION_PROTOCOL_VERSION,
    RUNTIME_PREDICTOR_VERSION,
    RUNTIME_SOURCE_ARTIFACT_PATH,
    RUNTIME_SOURCE_ARTIFACT_SHA256,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    compute_canonical_sha256,
    require_lowercase_hex64,
    verify_canonical_sha256,
)
from kvcot.discovery.b2a_r3_candidates import atomic_write_json, verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_qualification import (
    CandidateQualificationOutcomeR3,
    rederive_and_verify_qualification_outcome,
    select_first_qualified_r3,
)

__all__ = [
    "ArtifactVerificationRefused",
    "SELECTION_STATUS_SELECTED",
    "SELECTION_STATUS_NONE_QUALIFIED",
    "QualificationArtifactR3",
    "verify_qualification_artifact",
    "QualificationArtifactBuildRefused",
    "QualificationArtifactWriteRefused",
    "ALLOWED_QUALIFICATION_STOPPED_REASONS",
    "STOPPED_REASON_FIRST_PASS",
    "STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED",
    "STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED",
    "STOPPED_REASON_CANDIDATE_WORKER_TIMEOUT",
    "build_qualification_artifact",
    "write_qualification_artifact_atomic",
]

SELECTION_STATUS_SELECTED = "selected"
SELECTION_STATUS_NONE_QUALIFIED = "no_candidate_qualified"
_VALID_SELECTION_STATUSES = (SELECTION_STATUS_SELECTED, SELECTION_STATUS_NONE_QUALIFIED)

# Step 3R4-Repair-2 (repairs independent-re-audit Blocking Finding 6): moved
# above `QualificationArtifactR3` (from their original position further
# below, alongside `build_qualification_artifact`) so the schema's own
# field/model validators can reference them directly -- a stopped reason is
# now validated at the SCHEMA level, never left as a bare, unconstrained
# `str` a caller could set to anything and still pass
# `QualificationArtifactR3.model_validate`.
STOPPED_REASON_FIRST_PASS: Final[str] = "first_pass"
STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED: Final[str] = "all_authorized_candidates_exhausted"
STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED: Final[str] = "phase_wall_time_exhausted"
STOPPED_REASON_CANDIDATE_WORKER_TIMEOUT: Final[str] = "candidate_worker_timeout"
ALLOWED_QUALIFICATION_STOPPED_REASONS: Final[frozenset[str]] = frozenset({
    STOPPED_REASON_FIRST_PASS,
    STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED,
    STOPPED_REASON_PHASE_WALL_TIME_EXHAUSTED,
    STOPPED_REASON_CANDIDATE_WORKER_TIMEOUT,
})


class ArtifactVerificationRefused(ValueError):
    """Any hard rejection during qualification-artifact verification."""


def _parseable_iso8601(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
        return True
    except (TypeError, ValueError):
        return False


class QualificationArtifactR3(BaseModel):
    """Protocol §12.6. Strict, immutable, extra fields forbidden."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    artifact_schema_version: str
    candidate_order_protocol_version: str
    qualification_protocol_version: str
    runtime_predictor_version: str
    candidate_manifest_path: str
    candidate_manifest_canonical_sha256: str
    config_path: str
    config_sha256: str
    generation_config_sha256: str
    dataset_repo: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    budget: int
    runtime_source_artifact_path: str
    runtime_source_artifact_sha256: str
    attempted: list[CandidateQualificationOutcomeR3]
    attempted_candidate_count: int
    first_passing_candidate_ordinal: int | None
    selected_unique_id: str | None
    selection_status: str
    qualification_stopped_reason: str
    # Step 3R4-Repair-2 Finding 6: the AUTHORIZED candidate limit this
    # attempt actually ran under (`VerifiedAuthorizationContext
    # .maximum_candidates`, sourced from the parsed authorization document),
    # persisted so `qualification_stopped_reason ==
    # "all_authorized_candidates_exhausted"` can be independently checked
    # against it rather than trusted as a bare, self-reported string.
    authorized_maximum_candidates: int
    attempt_started_at_utc: str
    attempt_completed_at_utc: str
    canonical_sha256: str

    @field_validator("candidate_manifest_canonical_sha256", "config_sha256", "canonical_sha256")
    @classmethod
    def _hex64(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @field_validator("attempt_started_at_utc", "attempt_completed_at_utc")
    @classmethod
    def _iso8601(cls, v: str, info: Any) -> str:
        if not _parseable_iso8601(v):
            raise ValueError(f"{info.field_name} must be a parseable ISO 8601 timestamp, got {v!r}")
        return v

    @field_validator("qualification_stopped_reason")
    @classmethod
    def _valid_stopped_reason(cls, v: str) -> str:
        if v not in ALLOWED_QUALIFICATION_STOPPED_REASONS:
            raise ValueError(
                f"qualification_stopped_reason must be one of {sorted(ALLOWED_QUALIFICATION_STOPPED_REASONS)}, "
                f"got {v!r}"
            )
        return v

    @field_validator("authorized_maximum_candidates")
    @classmethod
    def _maximum_candidates_in_range(cls, v: int) -> int:
        if isinstance(v, bool) or not (1 <= v <= QUALIFICATION_CANDIDATE_LIMIT):
            raise ValueError(
                f"authorized_maximum_candidates must be an int in 1..{QUALIFICATION_CANDIDATE_LIMIT}, got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> "QualificationArtifactR3":
        if self.artifact_schema_version != QUALIFICATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("artifact_schema_version does not match the frozen value")
        if self.candidate_order_protocol_version != CANDIDATE_ORDER_PROTOCOL_VERSION:
            raise ValueError("candidate_order_protocol_version does not match the frozen value")
        if self.qualification_protocol_version != QUALIFICATION_PROTOCOL_VERSION:
            raise ValueError("qualification_protocol_version does not match the frozen value")
        if self.runtime_predictor_version != RUNTIME_PREDICTOR_VERSION:
            raise ValueError("runtime_predictor_version does not match the frozen value")
        if self.candidate_manifest_path != CANDIDATE_MANIFEST_PATH:
            raise ValueError("candidate_manifest_path does not match the frozen path")
        if self.config_path != CONFIG_PATH:
            raise ValueError("config_path does not match the frozen path")
        if self.generation_config_sha256 != GENERATION_CONFIG_SHA256:
            raise ValueError("generation_config_sha256 does not match the frozen value")
        if (self.dataset_repo, self.dataset_config, self.dataset_split, self.dataset_revision) != (
            DATASET_REPO, DATASET_CONFIG, DATASET_SPLIT, DATASET_REVISION,
        ):
            raise ValueError("dataset identity does not match the frozen values")
        if (self.model_name, self.model_revision) != (MODEL_NAME, MODEL_REVISION):
            raise ValueError("model identity does not match the frozen values")
        if (self.tokenizer_name, self.tokenizer_revision) != (TOKENIZER_NAME, TOKENIZER_REVISION):
            raise ValueError("tokenizer identity does not match the frozen values")
        if self.budget != BUDGET:
            raise ValueError(f"budget must be {BUDGET}")
        if self.runtime_source_artifact_path != RUNTIME_SOURCE_ARTIFACT_PATH:
            raise ValueError("runtime_source_artifact_path does not match the frozen path")
        if self.runtime_source_artifact_sha256 != RUNTIME_SOURCE_ARTIFACT_SHA256:
            raise ValueError("runtime_source_artifact_sha256 does not match the frozen value")

        if self.attempted_candidate_count != len(self.attempted):
            raise ValueError("attempted_candidate_count disagrees with len(attempted)")
        if self.selection_status not in _VALID_SELECTION_STATUSES:
            raise ValueError(f"selection_status must be one of {_VALID_SELECTION_STATUSES}")

        # Step 3R4-Repair-2 Finding 6: never attempt more candidates than
        # this attempt was actually authorized for, and "exhausted" must
        # mean EXACTLY the authorized count was attempted -- never fewer
        # (a stop for some other reason mislabeled as exhaustion) and never
        # more (already impossible via the count check above, restated here
        # for a defect that would otherwise only surface indirectly).
        if self.attempted_candidate_count > self.authorized_maximum_candidates:
            raise ValueError(
                f"attempted_candidate_count={self.attempted_candidate_count} exceeds "
                f"authorized_maximum_candidates={self.authorized_maximum_candidates}"
            )
        if self.qualification_stopped_reason == STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED:
            if self.attempted_candidate_count != self.authorized_maximum_candidates:
                raise ValueError(
                    "qualification_stopped_reason='all_authorized_candidates_exhausted' but "
                    f"attempted_candidate_count={self.attempted_candidate_count} != "
                    f"authorized_maximum_candidates={self.authorized_maximum_candidates}"
                )

        attempted_dicts = [outcome.model_dump(mode="json") for outcome in self.attempted]
        try:
            recomputed_selection = select_first_qualified_r3(attempted_dicts)
        except Exception as exc:
            raise ValueError(f"attempted list fails first-pass selection replay: {exc}") from exc

        if recomputed_selection is None:
            if self.selection_status != SELECTION_STATUS_NONE_QUALIFIED:
                raise ValueError("no candidate qualified, but selection_status != 'no_candidate_qualified'")
            if self.first_passing_candidate_ordinal is not None or self.selected_unique_id is not None:
                raise ValueError("no candidate qualified, but a selected ordinal/unique_id is recorded")
        else:
            if self.selection_status != SELECTION_STATUS_SELECTED:
                raise ValueError("a candidate qualified, but selection_status != 'selected'")
            if self.first_passing_candidate_ordinal != recomputed_selection["candidate_ordinal"]:
                raise ValueError("first_passing_candidate_ordinal does not match the recomputed first pass")
            if self.selected_unique_id != recomputed_selection["unique_id"]:
                raise ValueError("selected_unique_id does not match the recomputed first pass")

        if datetime.fromisoformat(self.attempt_completed_at_utc) < datetime.fromisoformat(
            self.attempt_started_at_utc
        ):
            raise ValueError("attempt_completed_at_utc precedes attempt_started_at_utc")
        return self


def verify_qualification_artifact(
    artifact: dict[str, Any], *, candidate_manifest: dict[str, Any], expected_config_sha256: str
) -> QualificationArtifactR3:
    """Full strict verification: canonical self-hash, schema/cross-field
    invariants, and cross-artifact hash agreement with the candidate
    manifest actually supplied and the config hash actually expected.
    Fails closed -- raises on the first defect found."""
    verify_canonical_sha256(artifact)
    typed = QualificationArtifactR3.model_validate(artifact)

    try:
        verify_candidate_manifest_structure(
            candidate_manifest, expected_config_sha256=expected_config_sha256
        )
    except Exception as exc:
        raise ArtifactVerificationRefused(f"candidate manifest verification failed: {exc}") from exc
    if typed.candidate_manifest_canonical_sha256 != candidate_manifest.get("canonical_sha256"):
        raise ArtifactVerificationRefused(
            "qualification artifact's candidate_manifest_canonical_sha256 does not match the "
            "supplied candidate manifest's own canonical_sha256"
        )
    if typed.config_sha256 != expected_config_sha256:
        raise ArtifactVerificationRefused(
            "qualification artifact's config_sha256 does not match the expected config hash"
        )
    for outcome in typed.attempted:
        try:
            rederive_and_verify_qualification_outcome(
                outcome, candidate_manifest, expected_config_sha256
            )
        except Exception as exc:
            raise ArtifactVerificationRefused(
                f"qualification outcome ordinal={outcome.candidate_ordinal} failed semantic rederivation: {exc}"
            ) from exc
    return typed


# --------------------------------------------------------------------------
# Step 3R4 Finding 6: qualification artifact builder and atomic writer.
# Neither function is ever called against the real
# results/decisions/b2a_r3_qualification.json path by anything in this
# repair round -- Stage A exercises both only against synthetic/injected
# fixtures under tmp_path (protocol §14.1).
# --------------------------------------------------------------------------


class QualificationArtifactBuildRefused(ValueError):
    """Any hard rejection while assembling a qualification artifact from a
    sequence of attempted outcomes -- non-contiguous ordinals, an
    over-authorized candidate count, a stopped_reason/selection
    disagreement, or an outcome that fails semantic rederivation."""


class QualificationArtifactWriteRefused(RuntimeError):
    """Any hard rejection while atomically writing a qualification
    artifact -- an invalid artifact, or an attempt to overwrite an
    existing (and therefore immutable) artifact file."""


def build_qualification_artifact(
    *,
    attempted_outcomes: list[dict[str, Any] | CandidateQualificationOutcomeR3],
    candidate_manifest: dict[str, Any],
    expected_config_sha256: str,
    stopped_reason: str,
    authorized_maximum_candidates: int,
    attempt_started_at_utc: str,
    attempt_completed_at_utc: str,
) -> dict[str, Any]:
    """Builds a complete, hash-verified, v3 qualification artifact
    (protocol §12.6, Step 3R4-Repair-2) from a sequence of already-produced
    attempted outcomes.

    Every outcome is independently semantically re-derived (never trusted
    as pre-validated); ordinals must be exactly `0..len(attempted)-1`
    contiguous; the recorded selection must reproduce from the attempted
    list via the same `select_first_qualified_r3` the artifact schema
    itself replays; `stopped_reason` must be one of the four frozen
    reasons and must agree with whether a candidate actually qualified.
    No caller may supply an arbitrary top-level identity (dataset/model/
    tokenizer/budget/version) -- every one of those is populated here from
    the frozen contract constants and the independently-verified candidate
    manifest, never from a parameter.

    Step 3R4-Repair-2 Finding 6: `authorized_maximum_candidates` is now a
    REQUIRED parameter -- the caller (the qualification coordinator) must
    supply the exact `VerifiedAuthorizationContext.maximum_candidates` this
    attempt actually ran under, never an unconstrained value. Never more
    outcomes than that limit may be attempted, and
    `stopped_reason='all_authorized_candidates_exhausted'` is refused
    outright unless `len(attempted_outcomes)` equals it exactly -- this is
    the same invariant `QualificationArtifactR3`'s own schema validator
    enforces on the constructed payload, checked here too so a caller gets
    a clear `QualificationArtifactBuildRefused` instead of a raw pydantic
    error.
    """
    if stopped_reason not in ALLOWED_QUALIFICATION_STOPPED_REASONS:
        raise QualificationArtifactBuildRefused(
            f"stopped_reason must be one of {sorted(ALLOWED_QUALIFICATION_STOPPED_REASONS)}, got {stopped_reason!r}"
        )
    if isinstance(authorized_maximum_candidates, bool) or not (
        1 <= authorized_maximum_candidates <= QUALIFICATION_CANDIDATE_LIMIT
    ):
        raise QualificationArtifactBuildRefused(
            f"authorized_maximum_candidates must be an int in 1..{QUALIFICATION_CANDIDATE_LIMIT}, "
            f"got {authorized_maximum_candidates!r}"
        )
    if len(attempted_outcomes) > authorized_maximum_candidates:
        raise QualificationArtifactBuildRefused(
            f"attempted {len(attempted_outcomes)} candidates, exceeding the authorized limit of "
            f"{authorized_maximum_candidates}"
        )
    if (
        stopped_reason == STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED
        and len(attempted_outcomes) != authorized_maximum_candidates
    ):
        raise QualificationArtifactBuildRefused(
            "stopped_reason='all_authorized_candidates_exhausted' but "
            f"len(attempted_outcomes)={len(attempted_outcomes)} != "
            f"authorized_maximum_candidates={authorized_maximum_candidates}"
        )

    manifest = verify_candidate_manifest_structure(
        candidate_manifest, expected_config_sha256=expected_config_sha256
    )

    typed_outcomes: list[CandidateQualificationOutcomeR3] = []
    for outcome in attempted_outcomes:
        typed = (
            outcome
            if isinstance(outcome, CandidateQualificationOutcomeR3)
            else CandidateQualificationOutcomeR3.model_validate(outcome)
        )
        try:
            rederive_and_verify_qualification_outcome(typed, candidate_manifest, expected_config_sha256)
        except Exception as exc:
            raise QualificationArtifactBuildRefused(
                f"candidate ordinal={typed.candidate_ordinal} failed semantic rederivation: {exc}"
            ) from exc
        typed_outcomes.append(typed)

    ordinals = [outcome.candidate_ordinal for outcome in typed_outcomes]
    if ordinals != list(range(len(ordinals))):
        raise QualificationArtifactBuildRefused(
            f"attempted candidate ordinals must be exactly 0..{len(ordinals) - 1} contiguous, got {ordinals}"
        )

    attempted_dicts = [outcome.model_dump(mode="json") for outcome in typed_outcomes]
    try:
        selection = select_first_qualified_r3(attempted_dicts)
    except Exception as exc:
        raise QualificationArtifactBuildRefused(f"attempted list fails first-pass selection replay: {exc}") from exc

    if selection is None:
        if stopped_reason == STOPPED_REASON_FIRST_PASS:
            raise QualificationArtifactBuildRefused("stopped_reason='first_pass' but no candidate qualified")
        selection_status = SELECTION_STATUS_NONE_QUALIFIED
        first_ordinal = None
        selected_unique_id = None
    else:
        if stopped_reason != STOPPED_REASON_FIRST_PASS:
            raise QualificationArtifactBuildRefused(
                f"a candidate qualified (ordinal={selection['candidate_ordinal']}) but "
                f"stopped_reason={stopped_reason!r} != 'first_pass'"
            )
        selection_status = SELECTION_STATUS_SELECTED
        first_ordinal = selection["candidate_ordinal"]
        selected_unique_id = selection["unique_id"]

    payload: dict[str, Any] = {
        "artifact_schema_version": QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
        "candidate_order_protocol_version": CANDIDATE_ORDER_PROTOCOL_VERSION,
        "qualification_protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "runtime_predictor_version": RUNTIME_PREDICTOR_VERSION,
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH,
        "candidate_manifest_canonical_sha256": manifest.canonical_sha256,
        "config_path": CONFIG_PATH,
        "config_sha256": expected_config_sha256,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "budget": BUDGET,
        "runtime_source_artifact_path": RUNTIME_SOURCE_ARTIFACT_PATH,
        "runtime_source_artifact_sha256": RUNTIME_SOURCE_ARTIFACT_SHA256,
        "attempted": attempted_dicts,
        "attempted_candidate_count": len(attempted_dicts),
        "first_passing_candidate_ordinal": first_ordinal,
        "selected_unique_id": selected_unique_id,
        "selection_status": selection_status,
        "qualification_stopped_reason": stopped_reason,
        "authorized_maximum_candidates": authorized_maximum_candidates,
        "attempt_started_at_utc": attempt_started_at_utc,
        "attempt_completed_at_utc": attempt_completed_at_utc,
    }
    payload["canonical_sha256"] = compute_canonical_sha256(payload)
    QualificationArtifactR3.model_validate(payload)  # construct-time self-check
    return payload


def write_qualification_artifact_atomic(
    artifact: dict[str, Any],
    *,
    output_path: str | Path,
    candidate_manifest: dict[str, Any],
    expected_config_sha256: str,
) -> None:
    """Atomic write, following the same discipline as
    `kvcot.discovery.b2a_r3_candidates.atomic_write_json`, plus three
    stricter rules specific to this immutable artifact: the artifact is
    FULLY semantically re-verified (not just schema-validated) before
    writing, byte-for-byte re-read and fully semantically re-verified again
    after writing, and an existing file at `output_path` is never
    overwritten -- the protocol treats a written qualification artifact as
    immutable once it exists.

    Step 3R4-Repair-2 Finding 5: the prior version of this function only
    checked the canonical self-hash and top-level schema
    (`QualificationArtifactR3.model_validate`) -- both are transport
    integrity checks. A canonically-rehashed artifact with internally
    self-consistent but semantically FABRICATED outcome fields (e.g. a
    condition map that does not actually reproduce from its own raw
    evidence) would satisfy that shallow check. `candidate_manifest`/
    `expected_config_sha256` are now REQUIRED parameters so this function
    can call the one authoritative semantic verifier,
    `verify_qualification_artifact` (which independently re-derives every
    attempted outcome's 27 conditions via
    `kvcot.discovery.b2a_r3_qualification.rederive_and_verify_qualification_outcome`
    and replays first-pass selection), both BEFORE writing and again on the
    artifact actually read back from disk.

    Never defaults to the real production path -- `output_path` is always
    an explicit, caller-supplied argument; Stage-A tests always pass a
    `tmp_path`-rooted path.
    """
    verify_qualification_artifact(
        artifact, candidate_manifest=candidate_manifest, expected_config_sha256=expected_config_sha256
    )

    target = Path(output_path)
    if target.exists():
        raise QualificationArtifactWriteRefused(
            f"refusing to overwrite an existing qualification artifact at {target} -- "
            "this artifact is immutable once written"
        )

    atomic_write_json(target, artifact)

    written = json.loads(target.read_text(encoding="utf-8"))
    verify_qualification_artifact(
        written, candidate_manifest=candidate_manifest, expected_config_sha256=expected_config_sha256
    )
