"""B2A-R3 machine-readable authorization document parser (Step 3R4 Finding
3, docs/B2A_R3_STAGE_A_PROTOCOL_ALIGNMENT_AMENDMENT_2026-07-23.md §5).

Before this module, a dated Stage B/C authorization Markdown document was
treated as an opaque byte blob: only its whole-file `sha256_file` hash was
checked (protocol §12.1/§14.4), never its content. The authorization
CLAIM then supplied every enforceable field (branch, commit, ancestors,
R-KV SHA, ...) and the caller wrapped those CLAIM fields into an
`AttemptProvenancePolicy` -- a circular design in which the claim defined
its own policy. This module makes the document itself authoritative: it
must contain one exact, strictly-schema-validated embedded JSON block, and
`policy_from_authorization_document` builds the policy from THAT parsed
document, never from the claim.

No CUDA, no torch, no R-KV import. No line in this module creates a real
authorization document -- only a strict parser/verifier for one a human
operator would commit under a genuinely separate, future, dated
authorization (not this Step 3R4 CPU repair round).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kvcot.discovery.b2a_r3_contract import (
    REQUIRED_REPOSITORY,
    SELECTED_MANIFEST_HASH_ALGORITHM,
    require_lowercase_hex64,
    require_strict_int,
    validate_authorization_id,
)
from kvcot.discovery.b2a_r3_provenance import AttemptProvenancePolicy

__all__ = [
    "AUTHORIZATION_DOCUMENT_SCHEMA_VERSION",
    "AUTHORIZATION_DOCUMENT_BEGIN_MARKER",
    "AUTHORIZATION_DOCUMENT_END_MARKER",
    "STAGE_FULLKV_QUALIFICATION",
    "STAGE_B2A_R3_EXECUTION",
    "AuthorizationDocumentMalformed",
    "AuthorizationDocumentR3",
    "extract_authorization_json_block",
    "parse_authorization_document_text",
    "parse_authorization_document",
    "policy_from_authorization_document",
]

AUTHORIZATION_DOCUMENT_SCHEMA_VERSION: Final[str] = "faithkv-b2a-r3-stage-authorization-document-v2"
AUTHORIZATION_DOCUMENT_BEGIN_MARKER: Final[str] = "<!-- BEGIN B2A-R3 AUTHORIZATION JSON -->"
AUTHORIZATION_DOCUMENT_END_MARKER: Final[str] = "<!-- END B2A-R3 AUTHORIZATION JSON -->"

# Reused verbatim from kvcot.discovery.b2a_r3_authorization -- not
# reimported from there to avoid a circular import (that module will, in
# turn, import THIS one); the two string literals are the single frozen
# source of truth (protocol §14.3) and are cross-checked by a regression
# test asserting byte-for-byte equality with the authorization module's
# own constants.
STAGE_FULLKV_QUALIFICATION: Final[str] = "fullkv_qualification"
STAGE_B2A_R3_EXECUTION: Final[str] = "b2a_r3_execution"
_VALID_STAGES: Final[tuple[str, ...]] = (STAGE_FULLKV_QUALIFICATION, STAGE_B2A_R3_EXECUTION)

_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)\n```", re.DOTALL)
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")


class AuthorizationDocumentMalformed(ValueError):
    """Any hard rejection while extracting or parsing the embedded JSON
    authorization block -- missing markers, multiple blocks, a non-JSON
    fence, stray text, invalid JSON, or a duplicate key."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise AuthorizationDocumentMalformed(f"duplicate JSON key {key!r} in authorization document JSON block")
        seen[key] = value
    return seen


def extract_authorization_json_block(text: str) -> str:
    """Requires exactly one `AUTHORIZATION_DOCUMENT_BEGIN_MARKER`/
    `_END_MARKER` pair, and exactly one fenced ```json ... ``` block
    between them with nothing but whitespace outside the fence."""
    begin_count = text.count(AUTHORIZATION_DOCUMENT_BEGIN_MARKER)
    end_count = text.count(AUTHORIZATION_DOCUMENT_END_MARKER)
    if begin_count == 0 or end_count == 0:
        raise AuthorizationDocumentMalformed(
            "authorization document is missing the BEGIN/END B2A-R3 AUTHORIZATION JSON markers"
        )
    if begin_count > 1 or end_count > 1:
        raise AuthorizationDocumentMalformed(
            "authorization document contains more than one BEGIN/END B2A-R3 AUTHORIZATION JSON marker"
        )
    begin_index = text.index(AUTHORIZATION_DOCUMENT_BEGIN_MARKER) + len(AUTHORIZATION_DOCUMENT_BEGIN_MARKER)
    end_index = text.index(AUTHORIZATION_DOCUMENT_END_MARKER)
    if end_index < begin_index:
        raise AuthorizationDocumentMalformed("the END marker precedes the BEGIN marker")
    between = text[begin_index:end_index]

    fences = list(_FENCE_RE.finditer(between))
    if len(fences) == 0:
        raise AuthorizationDocumentMalformed(
            "no fenced code block was found between the authorization JSON markers"
        )
    if len(fences) > 1:
        raise AuthorizationDocumentMalformed(
            "more than one fenced code block was found between the authorization JSON markers"
        )
    fence = fences[0]
    language = fence.group(1).strip()
    if language != "json":
        raise AuthorizationDocumentMalformed(
            f"the fenced block between the authorization JSON markers must be ```json, got ```{language!r}"
        )
    before, after = between[: fence.start()], between[fence.end() :]
    if before.strip() or after.strip():
        raise AuthorizationDocumentMalformed(
            "text was found between the authorization JSON markers outside the JSON fence"
        )
    return fence.group(2)


class AuthorizationDocumentR3(BaseModel):
    """The one, complete, authoritative Stage B/C authorization-document
    JSON schema (Step 3R4 Finding 3). Strict, extra fields forbidden.
    `authorization_document_sha256` is deliberately NEVER a field here --
    the committed document cannot contain its own final byte hash; the
    claim records that hash separately (protocol §12.1/§12.8)."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    authorization_document_schema_version: str
    authorization_id: str
    authorization_stage: str
    authorized_repository: str
    authorized_branch: str
    authorized_code_commit_sha: str
    required_ancestor_shas: list[str]
    required_rkv_sha: str
    candidate_manifest_canonical_sha256: str
    maximum_candidates: int | None = None
    phase_wall_time_limit_seconds: int | None = None
    qualification_artifact_canonical_sha256: str | None = None
    selected_manifest_sha256: str | None = None
    selected_manifest_hash_algorithm: str | None = None
    created_at_utc: str

    @field_validator("authorization_id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        return validate_authorization_id(v)

    @field_validator("candidate_manifest_canonical_sha256")
    @classmethod
    def _hex64(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @field_validator("qualification_artifact_canonical_sha256", "selected_manifest_sha256")
    @classmethod
    def _hex64_if_present(cls, v: str | None, info: Any) -> str | None:
        if v is None:
            return None
        return require_lowercase_hex64(v, info.field_name)

    @field_validator("authorized_code_commit_sha", "required_rkv_sha")
    @classmethod
    def _hex40(cls, v: str, info: Any) -> str:
        if not isinstance(v, str) or not _HEX40_RE.match(v):
            raise ValueError(f"{info.field_name} must be a lowercase 40-hex-character commit SHA, got {v!r}")
        return v

    @field_validator("required_ancestor_shas")
    @classmethod
    def _hex40_list(cls, v: list[str]) -> list[str]:
        for sha in v:
            if not isinstance(sha, str) or not _HEX40_RE.match(sha):
                raise ValueError(f"every required_ancestor_shas entry must be a lowercase 40-hex commit SHA, got {sha!r}")
        return v

    @field_validator("maximum_candidates", "phase_wall_time_limit_seconds")
    @classmethod
    def _strict_int_if_present(cls, v: int | None, info: Any) -> int | None:
        if v is None:
            return None
        return require_strict_int(v, info.field_name)

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> "AuthorizationDocumentR3":
        if self.authorization_document_schema_version != AUTHORIZATION_DOCUMENT_SCHEMA_VERSION:
            raise ValueError(
                f"authorization_document_schema_version must be {AUTHORIZATION_DOCUMENT_SCHEMA_VERSION!r}"
            )
        if self.authorization_stage not in _VALID_STAGES:
            raise ValueError(f"authorization_stage must be one of {_VALID_STAGES}, got {self.authorization_stage!r}")
        if self.authorized_repository != REQUIRED_REPOSITORY:
            raise ValueError(f"authorized_repository must be {REQUIRED_REPOSITORY!r}")

        stage_b_fields = (self.maximum_candidates, self.phase_wall_time_limit_seconds)
        stage_c_fields = (
            self.qualification_artifact_canonical_sha256,
            self.selected_manifest_sha256,
            self.selected_manifest_hash_algorithm,
        )
        if self.authorization_stage == STAGE_FULLKV_QUALIFICATION:
            if any(f is None for f in stage_b_fields):
                raise ValueError(
                    "Stage B (fullkv_qualification) documents require maximum_candidates and "
                    "phase_wall_time_limit_seconds"
                )
            if any(f is not None for f in stage_c_fields):
                raise ValueError(
                    "Stage B (fullkv_qualification) documents must not contain Stage-C-only fields "
                    "(qualification_artifact_canonical_sha256/selected_manifest_sha256/"
                    "selected_manifest_hash_algorithm)"
                )
            if not (1 <= self.maximum_candidates <= 8):
                raise ValueError(f"maximum_candidates must be in 1..8, got {self.maximum_candidates}")
            if self.phase_wall_time_limit_seconds <= 0:
                raise ValueError(
                    f"phase_wall_time_limit_seconds must be a strict positive integer, "
                    f"got {self.phase_wall_time_limit_seconds}"
                )
        else:
            if any(f is None for f in stage_c_fields):
                raise ValueError(
                    "Stage C (b2a_r3_execution) documents require qualification_artifact_canonical_sha256, "
                    "selected_manifest_sha256, and selected_manifest_hash_algorithm"
                )
            if any(f is not None for f in stage_b_fields):
                raise ValueError(
                    "Stage C (b2a_r3_execution) documents must not contain Stage-B-only fields "
                    "(maximum_candidates/phase_wall_time_limit_seconds)"
                )
            if self.selected_manifest_hash_algorithm != SELECTED_MANIFEST_HASH_ALGORITHM:
                raise ValueError(f"selected_manifest_hash_algorithm must be {SELECTED_MANIFEST_HASH_ALGORITHM!r}")
        return self


def parse_authorization_document_text(text: str) -> AuthorizationDocumentR3:
    json_text = extract_authorization_json_block(text)
    try:
        payload = json.loads(json_text, object_pairs_hook=_reject_duplicate_keys)
    except AuthorizationDocumentMalformed:
        raise
    except (ValueError, json.JSONDecodeError) as exc:
        raise AuthorizationDocumentMalformed(f"invalid JSON in authorization document block: {exc}") from exc
    if not isinstance(payload, dict):
        raise AuthorizationDocumentMalformed("authorization document JSON block must be a JSON object")
    try:
        return AuthorizationDocumentR3.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 -- re-raise as this module's own malformed-document error
        raise AuthorizationDocumentMalformed(f"authorization document JSON failed schema validation: {exc}") from exc


def parse_authorization_document(path: str | Path) -> AuthorizationDocumentR3:
    text = Path(path).read_text(encoding="utf-8")
    return parse_authorization_document_text(text)


def policy_from_authorization_document(
    document: AuthorizationDocumentR3, *, authorization_document_sha256: str
) -> AttemptProvenancePolicy:
    """Builds the enforced `AttemptProvenancePolicy` ENTIRELY from the
    parsed document -- never from a claim. `authorization_document_sha256`
    is supplied separately (the observed `sha256_file` of the committed
    document's bytes) because the document itself never contains its own
    hash (see the module docstring)."""
    from kvcot.discovery.b2a_r3_contract import PROVENANCE_POLICY_VERSION

    return AttemptProvenancePolicy(
        provenance_policy_version=PROVENANCE_POLICY_VERSION,
        required_repository=document.authorized_repository,
        required_branch=document.authorized_branch,
        required_commit_sha=document.authorized_code_commit_sha,
        required_ancestor_shas=tuple(document.required_ancestor_shas),
        required_rkv_sha=document.required_rkv_sha,
        authorization_id=document.authorization_id,
        authorization_document_sha256=authorization_document_sha256,
    )
