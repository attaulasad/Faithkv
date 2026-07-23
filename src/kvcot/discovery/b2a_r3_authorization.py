"""B2A-R3 atomic authorization-claim mechanism (Step 3 Stage-A, protocol
§14.4).

Implements and tests the future authorization-consumption mechanism
WITHOUT ever creating a real claim: `claim_authorization` (Step 3R4
Finding 4) takes `repository_root` only, always deriving the exact claim
path internally via `global_claim_path` -- there is no `claims_root`
parameter a caller could point at an arbitrary directory. No CLI command
in this repository calls `_create_authorization_claim` or
`claim_authorization` at all -- Stage A exercises this module only against
synthetic fixtures under `tmp_path`.

Creation of the exclusively-created filesystem entry IS the consumption
event (protocol §14.4.2) -- never a subsequent successful write, never a
subsequent successful GPU run.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
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
from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_provenance import (
    ActiveAuthorizationPaths,
    AttemptProvenancePolicy,
    GitStateProvider,
    verify_attempt_provenance,
    verify_git_state_bound_to_repository_root,
)
from kvcot.utils.hashing import sha256_file

__all__ = [
    "AuthorizationAlreadyConsumed",
    "AuthorizationClaimRefused",
    "AUTHORIZATION_STAGE_FULLKV_QUALIFICATION",
    "AUTHORIZATION_STAGE_B2A_R3_EXECUTION",
    "AuthorizationClaimR3",
    "VerifiedAuthorizationContext",
    "ConsumedAuthorizationContext",
    "StageBAuthorizationBinding",
    "verify_persisted_stage_b_authorization_binding",
    "verify_authorization_preconditions",
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
        ActiveAuthorizationPaths.from_verified_claim(self)
        return self


_VERIFIED_CONTEXT_TOKEN = object()
_CONSUMED_CONTEXT_TOKEN = object()
_STAGE_B_BINDING_TOKEN = object()


@dataclass(frozen=True)
class VerifiedAuthorizationContext:
    claim: AuthorizationClaimR3
    policy: AttemptProvenancePolicy
    active_paths: ActiveAuthorizationPaths
    # Step 3R4 Finding 3: sourced from the PARSED authorization document,
    # never the claim -- the future qualification coordinator (Finding 6)
    # reads its Stage-B limits only through this verified context, never
    # from a CLI argument or a hard-coded default. Both are `None` for a
    # Stage C (b2a_r3_execution) context.
    maximum_candidates: int | None
    phase_wall_time_limit_seconds: int | None
    _verification_token: object


@dataclass(frozen=True)
class ConsumedAuthorizationContext:
    """Proof that semantic authorization preconditions were verified and
    the deterministic global claim file was exclusively created before any
    execution action. Iterates as `(claim, claim_path)` for compatibility
    with older call sites that unpacked `claim_authorization` directly."""

    claim: AuthorizationClaimR3
    verified_context: VerifiedAuthorizationContext
    claim_path: Path
    authorization_claim_canonical_sha256: str
    _consumption_token: object

    def __iter__(self):
        yield self.claim
        yield self.claim_path


@dataclass(frozen=True)
class StageBAuthorizationBinding:
    """Persisted Stage-B authorization chain verified from the global claim
    file and committed authorization document for downstream processes."""

    claim: AuthorizationClaimR3
    verified_context: VerifiedAuthorizationContext
    claim_path: Path
    authorization_claim_canonical_sha256: str
    _binding_token: object


def verify_authorization_preconditions(
    claim_payload: dict[str, Any],
    *,
    git_state: GitStateProvider,
    authorization_document_path: str | Path,
    candidate_manifest: dict[str, Any],
    expected_config_sha256: str,
    qualification_artifact: dict[str, Any] | None = None,
    selected_manifest: Any | None = None,
    selection_provenance: dict[str, Any] | None = None,
    repository_root: str | Path = ".",
) -> VerifiedAuthorizationContext:
    """Verify the complete document/Git/artifact chain before consumption.

    Step 3R4 Finding 3: the enforced policy is now constructed ENTIRELY
    from the parsed authorization document (never accepted as a caller-
    supplied parameter) -- the claim's fields are then required to equal
    the document's fields, never the reverse. A Markdown document
    containing no machine-readable JSON block (e.g. the historical
    "synthetic Stage B authorization" placeholder text) is rejected
    outright, before any Git state is even inspected.
    """
    from kvcot.discovery.b2a_r3_authorization_document import (
        parse_authorization_document,
        policy_from_authorization_document,
    )

    # Step 3R4-Repair-2 Finding 7: bind `git_state` to the exact
    # `repository_root` BEFORE any other verification -- otherwise Git
    # state could be verified against one filesystem root while the
    # authorization document/claim I/O below happens under a different one.
    verify_git_state_bound_to_repository_root(git_state, repository_root)

    verify_canonical_sha256(claim_payload)
    claim = AuthorizationClaimR3.model_validate(claim_payload)

    import re

    stage_patterns = {
        AUTHORIZATION_STAGE_FULLKV_QUALIFICATION:
            r"docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_[0-9]{4}-[0-9]{2}-[0-9]{2}\.md",
        AUTHORIZATION_STAGE_B2A_R3_EXECUTION:
            r"docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_[0-9]{4}-[0-9]{2}-[0-9]{2}\.md",
    }
    if re.fullmatch(stage_patterns[claim.authorization_stage], claim.authorization_document_path) is None:
        raise AuthorizationClaimRefused("authorization document path does not match the exact stage/date pattern")
    expected_document = (Path(repository_root) / claim.authorization_document_path).resolve()
    supplied_document = Path(authorization_document_path).resolve()
    if supplied_document != expected_document or not supplied_document.is_file():
        raise AuthorizationClaimRefused("authorization document is missing or not the exact claimed path")
    if not git_state.is_path_committed(claim.authorization_document_path):
        raise AuthorizationClaimRefused("authorization document is not committed")
    observed_document_hash = sha256_file(supplied_document)
    if observed_document_hash != claim.authorization_document_sha256:
        raise AuthorizationClaimRefused("authorization document byte hash does not match the claim")

    try:
        document = parse_authorization_document(supplied_document)
    except Exception as exc:
        raise AuthorizationClaimRefused(f"authorization document is not machine-readable: {exc}") from exc

    if document.authorization_id != claim.authorization_id:
        raise AuthorizationClaimRefused("document authorization_id does not match the claim")
    if document.authorization_stage != claim.authorization_stage:
        raise AuthorizationClaimRefused("document authorization_stage does not match the claim")
    if document.authorized_branch != claim.authorized_branch:
        raise AuthorizationClaimRefused("document authorized_branch does not match the claim")
    if document.authorized_commit_sha != claim.authorized_commit_sha:
        raise AuthorizationClaimRefused("document authorized_commit_sha does not match the claim")
    if tuple(document.required_ancestor_shas) != tuple(claim.required_ancestor_shas):
        raise AuthorizationClaimRefused("document required_ancestor_shas does not match the claim")
    if document.required_rkv_sha != claim.required_rkv_sha:
        raise AuthorizationClaimRefused("document required_rkv_sha does not match the claim")
    if document.candidate_manifest_canonical_sha256 != claim.candidate_manifest_canonical_sha256:
        raise AuthorizationClaimRefused("document candidate_manifest_canonical_sha256 does not match the claim")
    if claim.authorization_stage == AUTHORIZATION_STAGE_B2A_R3_EXECUTION:
        if document.qualification_artifact_canonical_sha256 != claim.qualification_artifact_canonical_sha256:
            raise AuthorizationClaimRefused(
                "document qualification_artifact_canonical_sha256 does not match the claim"
            )
        if document.selected_manifest_sha256 != claim.selected_manifest_sha256:
            raise AuthorizationClaimRefused("document selected_manifest_sha256 does not match the claim")
        if document.selected_manifest_hash_algorithm != claim.selected_manifest_hash_algorithm:
            raise AuthorizationClaimRefused("document selected_manifest_hash_algorithm does not match the claim")

    # The policy is built ENTIRELY from the parsed document -- never from
    # the claim (Step 3R4 Finding 3). The claim-vs-document equality
    # checks above already ensure the claim cannot smuggle a divergent
    # value past this point.
    policy = policy_from_authorization_document(document, authorization_document_sha256=observed_document_hash)

    if not (
        claim.observed_repository == claim.authorized_repository == policy.required_repository == REQUIRED_REPOSITORY
    ):
        raise AuthorizationClaimRefused("observed/authorized/required repository identities disagree")
    if not (claim.observed_branch == claim.authorized_branch == policy.required_branch):
        raise AuthorizationClaimRefused("observed/authorized/required branch identities disagree")
    if not (claim.observed_commit_sha == claim.authorized_commit_sha == policy.required_commit_sha):
        raise AuthorizationClaimRefused("observed/authorized/required commit identities disagree")
    if not (claim.observed_rkv_sha == claim.required_rkv_sha == policy.required_rkv_sha):
        raise AuthorizationClaimRefused("observed/required R-KV identities disagree")

    ok, reasons = verify_attempt_provenance(policy, git_state)
    if not ok:
        raise AuthorizationClaimRefused(f"repository pre-claim verification failed: {reasons}")

    candidate = verify_candidate_manifest_structure(
        candidate_manifest, expected_config_sha256=expected_config_sha256
    )
    if claim.candidate_manifest_canonical_sha256 != candidate.canonical_sha256:
        raise AuthorizationClaimRefused("claim candidate-manifest hash does not match the supplied manifest")

    if claim.authorization_stage == AUTHORIZATION_STAGE_FULLKV_QUALIFICATION:
        if any(value is not None for value in (
            qualification_artifact, selected_manifest, selection_provenance
        )):
            raise AuthorizationClaimRefused("Stage B preconditions must not receive Stage C artifacts")
    else:
        if qualification_artifact is None or selected_manifest is None or selection_provenance is None:
            raise AuthorizationClaimRefused("Stage C preconditions require the complete qualification/selection chain")
        from kvcot.discovery.b2a_r3_artifacts import verify_qualification_artifact
        from kvcot.discovery.b2a_r3_freeze import verify_selection_provenance

        stage_b_binding = verify_persisted_stage_b_authorization_binding(
            authorization_id=qualification_artifact["stage_b_authorization_id"],
            repository_root=repository_root,
            git_state=git_state,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=expected_config_sha256,
        )
        qualification = verify_qualification_artifact(
            qualification_artifact,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=expected_config_sha256,
            stage_b_authorization_context=stage_b_binding,
        )
        if claim.qualification_artifact_canonical_sha256 != qualification.canonical_sha256:
            raise AuthorizationClaimRefused("claim qualification-artifact hash does not match")
        verified_selection = verify_selection_provenance(
            selection_provenance,
            selected_manifest=selected_manifest,
            candidate_manifest=candidate_manifest,
            qualification_artifact=qualification_artifact,
            expected_config_sha256=expected_config_sha256,
            stage_b_authorization_context=stage_b_binding,
        )
        if claim.selected_manifest_sha256 != verified_selection.selected_manifest_sha256:
            raise AuthorizationClaimRefused("claim selected-manifest hash does not match")

    return VerifiedAuthorizationContext(
        claim=claim,
        policy=policy,
        active_paths=ActiveAuthorizationPaths.from_verified_claim(claim),
        maximum_candidates=document.maximum_candidates,
        phase_wall_time_limit_seconds=document.phase_wall_time_limit_seconds,
        _verification_token=_VERIFIED_CONTEXT_TOKEN,
    )


def _create_authorization_claim(payload: dict[str, Any], *, claim_path: str | Path) -> Path:
    """Steps 2-4 of protocol §14.4.2: exclusively create the EXACT supplied
    claim path (`O_CREAT | O_EXCL`), then write + flush + fsync the
    payload. The success of the exclusive-create call IS the consumption
    event -- this function's caller must have already completed every
    pre-claim verification (step 1) before calling this.

    Low-level and private: accepts an explicit path so race/concurrency
    tests can exercise the raw exclusive-create primitive directly.
    Production code must never call this with an arbitrary path --
    `claim_authorization` below is the only production entry point, and it
    always derives this path itself via
    `kvcot.discovery.b2a_r3_contract.global_claim_path`, never from a
    caller-supplied root (Step 3R4 Finding 4).

    Raises `AuthorizationAlreadyConsumed` if a filesystem entry already
    exists at the given path -- complete, partial, or corrupt, it is
    permanently consumed (protocol §14.4.3)."""
    authorization_id = payload.get("authorization_id")
    if not isinstance(authorization_id, str):
        raise AuthorizationClaimRefused("payload has no string authorization_id")
    validate_authorization_id(authorization_id)

    import tempfile

    claim_path = Path(claim_path)
    claim_path.parent.mkdir(parents=True, exist_ok=True)

    text = json.dumps(payload, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    fd, tmp_name = tempfile.mkstemp(
        dir=str(claim_path.parent), prefix=f".{authorization_id}.", suffix=".json.tmp"
    )
    try:
        try:
            view = memoryview(text)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise AuthorizationClaimRefused("failed to write authorization claim temp file")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(tmp_name, claim_path)
        except FileExistsError as exc:
            raise AuthorizationAlreadyConsumed(
                f"a filesystem entry already exists at {claim_path} -- authorization_id "
                f"{authorization_id!r} is permanently consumed"
            ) from exc
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    else:
        os.remove(tmp_name)
    with open(claim_path, "r", encoding="utf-8") as f:
        written_payload = json.load(f)
    if written_payload != payload:
        raise AuthorizationClaimRefused("authorization claim did not round-trip after exclusive publication")
    try:
        dir_fd = os.open(str(claim_path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass
    return claim_path


def claim_authorization(
    payload: dict[str, Any],
    *,
    repository_root: str | Path,
    verified_context: VerifiedAuthorizationContext,
    git_state: GitStateProvider,
) -> ConsumedAuthorizationContext:
    """Full-schema validation, exact global-path binding, an immediate
    Git/worktree reverification, then the atomic claim operation (Step 3R4
    Finding 4). No caller may direct consumption at an arbitrary
    directory: the claim path is always
    `repository_root / global_claim_path(authorization_id)`, derived
    internally -- there is no `claims_root` parameter to override it.
    Returns `(typed_claim, claim_path)` on success; raises
    `AuthorizationAlreadyConsumed` if the deterministic path is already
    occupied.

    Step 3R4-Repair-2 Finding 7: `git_state` is bound to the exact
    `repository_root` here too (not only in
    `verify_authorization_preconditions`) -- the claim file is always
    written under `repository_root`, so a `git_state` verifying a different
    root must be refused before the exclusive-create call, never after."""
    verify_git_state_bound_to_repository_root(git_state, repository_root)
    verify_canonical_sha256(payload)
    typed = AuthorizationClaimR3.model_validate(payload)
    if verified_context._verification_token is not _VERIFIED_CONTEXT_TOKEN:
        raise AuthorizationClaimRefused("authorization context was not produced by semantic precondition verification")
    if typed != verified_context.claim:
        raise AuthorizationClaimRefused("verified context does not authorize this exact claim payload")

    relative_claim_path = global_claim_path(typed.authorization_id)
    if typed.global_claim_path != relative_claim_path:
        raise AuthorizationClaimRefused(
            "payload global_claim_path does not match the deterministic path derived from authorization_id"
        )
    if verified_context.claim.global_claim_path != relative_claim_path:
        raise AuthorizationClaimRefused(
            "verified context's claim global_claim_path does not match the deterministic path"
        )

    # Step 3R4 Finding 4: reverify Git/worktree state immediately before
    # the exclusive-create call -- this narrows the window between the
    # earlier verify_authorization_preconditions call and actual
    # consumption. No CUDA/model/tokenizer action may happen before this
    # succeeds and the claim is created.
    ok, reasons = verify_attempt_provenance(verified_context.policy, git_state)
    if not ok:
        raise AuthorizationClaimRefused(
            f"pre-claim Git/worktree reverification failed immediately before consumption: {reasons}"
        )

    absolute_claim_path = Path(repository_root) / relative_claim_path
    claim_path = _create_authorization_claim(payload, claim_path=absolute_claim_path)
    return ConsumedAuthorizationContext(
        claim=typed,
        verified_context=verified_context,
        claim_path=claim_path,
        authorization_claim_canonical_sha256=typed.canonical_sha256,
        _consumption_token=_CONSUMED_CONTEXT_TOKEN,
    )


def verify_persisted_stage_b_authorization_binding(
    *,
    authorization_id: str,
    repository_root: str | Path,
    git_state: GitStateProvider,
    candidate_manifest: dict[str, Any],
    expected_config_sha256: str,
) -> StageBAuthorizationBinding:
    """Reconstruct the Stage-B authorization binding from disk for later
    processes. This verifies the global claim file exists, is canonical,
    points at the committed Stage-B authorization document, and authorizes
    the supplied candidate manifest and config identity."""
    validate_authorization_id(authorization_id)
    claim_path = Path(repository_root) / global_claim_path(authorization_id)
    try:
        with open(claim_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        raise AuthorizationClaimRefused(f"failed to read persisted Stage-B claim {claim_path}: {exc}") from exc
    verified = verify_authorization_preconditions(
        payload,
        git_state=git_state,
        authorization_document_path=Path(repository_root) / payload["authorization_document_path"],
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
        repository_root=repository_root,
    )
    if verified.claim.authorization_stage != AUTHORIZATION_STAGE_FULLKV_QUALIFICATION:
        raise AuthorizationClaimRefused("persisted authorization binding is not Stage B")
    if verified.claim.authorization_id != authorization_id:
        raise AuthorizationClaimRefused("persisted authorization claim id does not match the requested id")
    return StageBAuthorizationBinding(
        claim=verified.claim,
        verified_context=verified,
        claim_path=claim_path,
        authorization_claim_canonical_sha256=verified.claim.canonical_sha256,
        _binding_token=_STAGE_B_BINDING_TOKEN,
    )


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
