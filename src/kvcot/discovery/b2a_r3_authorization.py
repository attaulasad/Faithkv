"""B2A-R3 atomic authorization-claim mechanism (Step 3 Stage-A, protocol
§14.4).

Implements and tests the future authorization-consumption mechanism
WITHOUT ever creating a real claim: `claims_root` is always an explicit
caller-supplied argument (never a hard-coded default pointing at the real
`results/decisions/b2a_r3_authorization_claims/` directory), and no CLI
command in this repository calls `create_authorization_claim` or
`claim_authorization` at all -- Stage A exercises this module only against
synthetic fixtures under `tmp_path`.

Creation of the exclusively-created filesystem entry IS the consumption
event (protocol §14.4.2) -- never a subsequent successful write, never a
subsequent successful GPU run.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kvcot.discovery.b2a_r3_contract import (
    AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
    REQUIRED_REPOSITORY,
    SELECTED_MANIFEST_HASH_ALGORITHM,
    global_claim_path,
    require_lowercase_hex64,
    validate_authorization_id,
    verify_canonical_sha256,
)

__all__ = [
    "AuthorizationAlreadyConsumed",
    "AuthorizationClaimRefused",
    "AUTHORIZATION_STAGE_FULLKV_QUALIFICATION",
    "AUTHORIZATION_STAGE_B2A_R3_EXECUTION",
    "AuthorizationClaimR3",
    "create_authorization_claim",
    "claim_authorization",
    "plan_authorization_claim_dry_run",
]

AUTHORIZATION_STAGE_FULLKV_QUALIFICATION = "fullkv_qualification"
AUTHORIZATION_STAGE_B2A_R3_EXECUTION = "b2a_r3_execution"
_VALID_STAGES = (AUTHORIZATION_STAGE_FULLKV_QUALIFICATION, AUTHORIZATION_STAGE_B2A_R3_EXECUTION)


class AuthorizationAlreadyConsumed(RuntimeError):
    """Raised when a claim already exists at the deterministic global path
    -- complete, partial, empty, or corrupt, it is ALWAYS consumed
    (protocol §14.4.3); never repaired, deleted, or treated as available
    for retry."""


class AuthorizationClaimRefused(ValueError):
    """Any hard rejection of a claim payload before the atomic filesystem
    operation is even attempted."""


class AuthorizationClaimR3(BaseModel):
    """Protocol §12.8. Strict, immutable, extra fields forbidden."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    artifact_schema_version: str
    authorization_id: str
    authorization_stage: str
    authorization_document_path: str
    authorization_document_sha256: str
    authorized_repository: str
    authorized_branch: str
    authorized_commit_sha: str
    observed_repository: str
    observed_branch: str
    observed_commit_sha: str
    required_ancestor_shas: list[str]
    required_rkv_sha: str
    observed_rkv_sha: str
    candidate_manifest_canonical_sha256: str
    qualification_artifact_canonical_sha256: str | None
    selected_manifest_sha256: str | None
    selected_manifest_hash_algorithm: str | None
    attempt_id: str
    global_claim_path: str
    attempt_directory_path: str
    claimed_at_utc: str
    canonical_sha256: str

    @field_validator("authorization_id")
    @classmethod
    def _valid_authorization_id(cls, v: str) -> str:
        return validate_authorization_id(v)

    @field_validator(
        "authorization_document_sha256", "candidate_manifest_canonical_sha256", "canonical_sha256",
    )
    @classmethod
    def _hex64(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @field_validator("qualification_artifact_canonical_sha256", "selected_manifest_sha256")
    @classmethod
    def _hex64_if_present(cls, v: str | None, info: Any) -> str | None:
        if v is None:
            return None
        return require_lowercase_hex64(v, info.field_name)

    @field_validator("claimed_at_utc")
    @classmethod
    def _iso8601(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"claimed_at_utc must be a parseable ISO 8601 timestamp, got {v!r}") from exc
        return v

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> "AuthorizationClaimR3":
        if self.artifact_schema_version != AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("artifact_schema_version does not match the frozen value")
        if self.authorization_stage not in _VALID_STAGES:
            raise ValueError(f"authorization_stage must be one of {_VALID_STAGES}")
        if self.authorized_repository != REQUIRED_REPOSITORY:
            raise ValueError(f"authorized_repository must be {REQUIRED_REPOSITORY!r}")

        stage_c_fields = (
            self.qualification_artifact_canonical_sha256, self.selected_manifest_sha256,
            self.selected_manifest_hash_algorithm,
        )
        if self.authorization_stage == AUTHORIZATION_STAGE_FULLKV_QUALIFICATION:
            if any(f is not None for f in stage_c_fields):
                raise ValueError(
                    "Stage B (fullkv_qualification) claims must have null "
                    "qualification_artifact_canonical_sha256/selected_manifest_sha256/"
                    "selected_manifest_hash_algorithm"
                )
        else:
            if any(f is None for f in stage_c_fields):
                raise ValueError(
                    "Stage C (b2a_r3_execution) claims require "
                    "qualification_artifact_canonical_sha256, selected_manifest_sha256, and "
                    "selected_manifest_hash_algorithm"
                )
            if self.selected_manifest_hash_algorithm != SELECTED_MANIFEST_HASH_ALGORITHM:
                raise ValueError(
                    f"selected_manifest_hash_algorithm must be {SELECTED_MANIFEST_HASH_ALGORITHM!r}"
                )

        expected_path = global_claim_path(self.authorization_id)
        if self.global_claim_path != expected_path:
            raise ValueError(
                f"global_claim_path {self.global_claim_path!r} does not match the deterministic path "
                f"{expected_path!r} derived from authorization_id"
            )
        return self


def create_authorization_claim(payload: dict[str, Any], *, claims_root: str | Path) -> Path:
    """Steps 2-4 of protocol §14.4.2: derive the deterministic claim path,
    exclusively create it (`O_CREAT | O_EXCL`), then write + flush +
    fsync the payload. The success of the exclusive-create call IS the
    consumption event -- this function's caller must have already
    completed every pre-claim verification (step 1) before calling this.

    Raises `AuthorizationAlreadyConsumed` if a filesystem entry already
    exists at the deterministic path -- complete, partial, or corrupt, it
    is permanently consumed (protocol §14.4.3)."""
    authorization_id = payload.get("authorization_id")
    if not isinstance(authorization_id, str):
        raise AuthorizationClaimRefused("payload has no string authorization_id")
    validate_authorization_id(authorization_id)

    claims_root_path = Path(claims_root)
    claims_root_path.mkdir(parents=True, exist_ok=True)
    claim_path = claims_root_path / f"{authorization_id}.json"

    text = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    try:
        fd = os.open(str(claim_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise AuthorizationAlreadyConsumed(
            f"a filesystem entry already exists at {claim_path} -- authorization_id "
            f"{authorization_id!r} is permanently consumed"
        ) from exc
    try:
        os.write(fd, text)
        os.fsync(fd)
    finally:
        os.close(fd)
    return claim_path


def claim_authorization(payload: dict[str, Any], *, claims_root: str | Path) -> tuple[AuthorizationClaimR3, Path]:
    """Full-schema validation (including the payload's own
    `canonical_sha256`) followed by the atomic claim operation. Returns
    `(typed_claim, claim_path)` on success; raises
    `AuthorizationAlreadyConsumed` if the deterministic path is already
    occupied."""
    verify_canonical_sha256(payload)
    typed = AuthorizationClaimR3.model_validate(payload)
    claim_path = create_authorization_claim(payload, claims_root=claims_root)
    return typed, claim_path


def plan_authorization_claim_dry_run(payload: dict[str, Any]) -> dict[str, Any]:
    """CPU-only planning: validates the payload's shape WITHOUT touching
    any filesystem path -- never creates a claim directory, never creates
    a claim file, never creates an attempt directory."""
    try:
        verify_canonical_sha256(payload)
        AuthorizationClaimR3.model_validate(payload)
        valid = True
        error: str | None = None
    except Exception as exc:  # noqa: BLE001 -- report the reason, never silently swallow
        valid = False
        error = str(exc)
    return {
        "payload_schema_valid": valid,
        "validation_error": error,
        "authorization_claim_created": False,
        "authorization_consumed": False,
    }
