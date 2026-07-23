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

from datetime import datetime
from typing import Any

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
    QUALIFICATION_PROTOCOL_VERSION,
    RUNTIME_PREDICTOR_VERSION,
    RUNTIME_SOURCE_ARTIFACT_PATH,
    RUNTIME_SOURCE_ARTIFACT_SHA256,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    require_lowercase_hex64,
    verify_canonical_sha256,
)
from kvcot.discovery.b2a_r3_qualification import CandidateQualificationOutcomeR3, select_first_qualified_r3

__all__ = [
    "ArtifactVerificationRefused",
    "SELECTION_STATUS_SELECTED",
    "SELECTION_STATUS_NONE_QUALIFIED",
    "QualificationArtifactR3",
    "verify_qualification_artifact",
]

SELECTION_STATUS_SELECTED = "selected"
SELECTION_STATUS_NONE_QUALIFIED = "no_candidate_qualified"
_VALID_SELECTION_STATUSES = (SELECTION_STATUS_SELECTED, SELECTION_STATUS_NONE_QUALIFIED)


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

    verify_canonical_sha256(candidate_manifest)
    if typed.candidate_manifest_canonical_sha256 != candidate_manifest.get("canonical_sha256"):
        raise ArtifactVerificationRefused(
            "qualification artifact's candidate_manifest_canonical_sha256 does not match the "
            "supplied candidate manifest's own canonical_sha256"
        )
    if typed.config_sha256 != expected_config_sha256:
        raise ArtifactVerificationRefused(
            "qualification artifact's config_sha256 does not match the expected config hash"
        )
    return typed
