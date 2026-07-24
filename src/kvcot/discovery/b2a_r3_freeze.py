"""B2A-R3 selected-row freeze contract (Step 3 Stage-A, protocol §13,
§12.7).

Separate pure-construction and I/O layers, exactly as required: the
construction function never touches a filesystem; the I/O layer (used
ONLY by tests against `tmp_path`) is a thin, separately callable wrapper
around `kvcot.discovery.b2a_r3_candidates.atomic_write_json`. No line in
this module writes to the real
`configs/discovery/b2a_one_example_manifest.json` or
`results/decisions/b2a_r3_selection_provenance.json` paths, and no CLI in
this repository wires this module's construction/write functions to those
real paths under Stage A -- only a `--dry-run` planning command
(`kvcot.cli`) may call `plan_freeze_dry_run` below.

The prompt-rendering step (tokenizing the selected row) is always
performed by an INJECTED `TokenizerRenderer` callable -- there is no
default that loads a real tokenizer, mirroring the qualification
evaluator's "no implicit fallback that performs inference when a
dependency is not injected" rule. CPU tests inject a fake renderer and
never load `transformers`.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace as _dataclass_replace
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kvcot.discovery.b2a_r3_artifacts import SELECTION_STATUS_SELECTED, verify_qualification_artifact
from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_contract import (
    CANDIDATE_MANIFEST_PATH,
    EMBEDDED_ROW_COLUMNS,
    QUALIFICATION_ARTIFACT_PATH,
    SELECTED_MANIFEST_HASH_ALGORITHM,
    SELECTED_MANIFEST_PATH,
    SELECTION_PROTOCOL_VERSION,
    SELECTION_PROVENANCE_ARTIFACT_SCHEMA_VERSION,
    SELECTION_PROVENANCE_PATH,
    PROMPT_ADD_GENERATION,
    PROMPT_MESSAGE_ROLES,
    PROMPT_SPECIAL_TOKENS_NOTE,
    PROMPT_TOKENIZE,
    REQUIRED_REPOSITORY,
    compute_canonical_sha256,
    require_lowercase_hex64,
    verify_canonical_sha256,
)
from kvcot.discovery.manifest import B2AOneExampleManifest, ChatTemplateRenderingConfig
from kvcot.utils.hashing import sha256_bytes, sha256_int_ids, sha256_json

__all__ = [
    "RowFreezeRefusedR3",
    "PromptRenderingResult",
    "TokenizerRenderer",
    "SelectionProvenanceR3",
    "verify_freeze_chain",
    "construct_selected_manifest_and_provenance",
    "write_selected_manifest_and_provenance",
    "verify_selection_provenance",
    "plan_freeze_dry_run",
    "ProductionPublicationRefused",
    "PUBLICATION_STATE_A_INITIAL",
    "PUBLICATION_STATE_B_COMPLETE",
    "PUBLICATION_STATE_C_PROVENANCE_FIRST_PARTIAL",
    "PUBLICATION_STATE_D_MANIFEST_FIRST_PARTIAL",
    "PUBLICATION_STATE_E_INVALID",
    "ProductionFreezePlan",
    "construct_production_freeze_plan",
    "classify_publication_state",
    "verify_git_worktree_safety_for_freeze",
    "publish_production_freeze",
]


class RowFreezeRefusedR3(RuntimeError):
    """Raised whenever a freeze request does not exactly match what the
    qualification artifact and candidate manifest jointly authorize."""


class PromptRenderingResult(BaseModel):
    """What an injected `TokenizerRenderer` must return -- the rendering-
    derived fields only; the row/raw-content hash are already known from
    the candidate manifest and are never re-derived here."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    rendered_user_message_sha256: str
    chat_template_source_sha256: str
    chat_message_payload_sha256: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str
    prompt_token_count: int
    tokenizer_revision_used_for_prompt_hash: str
    prompt_rendering_config: ChatTemplateRenderingConfig

    @field_validator(
        "rendered_user_message_sha256",
        "chat_template_source_sha256",
        "chat_message_payload_sha256",
        "prompt_token_ids_sha256",
    )
    @classmethod
    def _render_hash(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @model_validator(mode="after")
    def _prompt_identity_reproduces(self) -> "PromptRenderingResult":
        if any(type(token_id) is not int for token_id in self.prompt_token_ids):
            raise ValueError("prompt_token_ids must contain strict integers, never bool")
        if self.prompt_token_count != len(self.prompt_token_ids):
            raise ValueError("prompt_token_count disagrees with len(prompt_token_ids)")
        if sha256_int_ids(list(self.prompt_token_ids)) != self.prompt_token_ids_sha256:
            raise ValueError("prompt_token_ids_sha256 does not reproduce from prompt_token_ids")
        config = self.prompt_rendering_config
        if (
            tuple(config.message_roles),
            config.add_generation_prompt,
            config.tokenize,
            config.add_special_tokens_note,
        ) != (
            PROMPT_MESSAGE_ROLES,
            PROMPT_ADD_GENERATION,
            PROMPT_TOKENIZE,
            PROMPT_SPECIAL_TOKENS_NOTE,
        ):
            raise ValueError("prompt_rendering_config does not match the frozen canonical convention")
        return self


TokenizerRenderer = Callable[[dict[str, Any]], PromptRenderingResult]


def _verify_rendering_binding(rendering: PromptRenderingResult, candidate_manifest: dict[str, Any]) -> None:
    if rendering.tokenizer_revision_used_for_prompt_hash != candidate_manifest["tokenizer_revision"]:
        raise RowFreezeRefusedR3("renderer tokenizer revision does not match the candidate manifest")


class SelectionProvenanceR3(BaseModel):
    """Protocol §12.7. Strict, immutable, extra fields forbidden."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    artifact_schema_version: str
    qualification_artifact_path: str
    qualification_artifact_canonical_sha256: str
    candidate_manifest_path: str
    candidate_manifest_canonical_sha256: str
    selected_manifest_path: str
    selected_manifest_sha256: str
    selected_manifest_hash_algorithm: str
    selected_ordinal: int
    selected_unique_id: str
    selection_protocol_version: str
    row_raw_sha256: str
    prompt_token_ids_sha256: str
    tokenizer_revision_used_for_prompt_hash: str
    canonical_sha256: str

    @field_validator(
        "qualification_artifact_canonical_sha256", "candidate_manifest_canonical_sha256",
        "selected_manifest_sha256", "row_raw_sha256", "prompt_token_ids_sha256", "canonical_sha256",
    )
    @classmethod
    def _hex64(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> "SelectionProvenanceR3":
        if self.artifact_schema_version != SELECTION_PROVENANCE_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("artifact_schema_version does not match the frozen value")
        if self.qualification_artifact_path != QUALIFICATION_ARTIFACT_PATH:
            raise ValueError("qualification_artifact_path does not match the frozen path")
        if self.candidate_manifest_path != CANDIDATE_MANIFEST_PATH:
            raise ValueError("candidate_manifest_path does not match the frozen path")
        if self.selected_manifest_path != SELECTED_MANIFEST_PATH:
            raise ValueError("selected_manifest_path does not match the frozen path")
        if self.selected_manifest_hash_algorithm != SELECTED_MANIFEST_HASH_ALGORITHM:
            raise ValueError("selected_manifest_hash_algorithm does not match the frozen value")
        if self.selection_protocol_version != SELECTION_PROTOCOL_VERSION:
            raise ValueError("selection_protocol_version does not match the frozen value")
        if not (0 <= self.selected_ordinal <= 7):
            raise ValueError("selected_ordinal must be in [0, 7] (the qualification window)")
        return self


def verify_freeze_chain(
    *,
    candidate_manifest: dict[str, Any],
    qualification_artifact: dict[str, Any],
    expected_config_sha256: str,
    stage_b_authorization_context: Any,
) -> tuple[Any, Any]:
    """Every check protocol §13 requires BEFORE any freeze proceeds.
    Returns `(candidate_row, qualification_outcome)` -- both already
    strictly typed -- only when every check passes."""
    candidate_typed = verify_candidate_manifest_structure(candidate_manifest)
    qualification_typed = verify_qualification_artifact(
        qualification_artifact,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=stage_b_authorization_context,
    )

    if qualification_typed.selection_status != SELECTION_STATUS_SELECTED:
        raise RowFreezeRefusedR3("qualification artifact has no selected row -- nothing to freeze")

    selected_ordinal = qualification_typed.first_passing_candidate_ordinal
    selected_unique_id = qualification_typed.selected_unique_id

    candidate_matches = [c for c in candidate_typed.candidates if c.candidate_ordinal == selected_ordinal]
    if len(candidate_matches) != 1:
        raise RowFreezeRefusedR3(
            f"candidate manifest does not have exactly one row at ordinal={selected_ordinal} "
            f"(found {len(candidate_matches)})"
        )
    candidate_row = candidate_matches[0]
    if candidate_row.unique_id != selected_unique_id:
        raise RowFreezeRefusedR3(
            f"candidate manifest row at ordinal={selected_ordinal} has unique_id={candidate_row.unique_id!r}, "
            f"but qualification selected unique_id={selected_unique_id!r}"
        )

    outcome_matches = [o for o in qualification_typed.attempted if o.candidate_ordinal == selected_ordinal]
    if len(outcome_matches) != 1:
        raise RowFreezeRefusedR3(
            f"qualification artifact does not have exactly one outcome at ordinal={selected_ordinal}"
        )
    outcome = outcome_matches[0]
    if not outcome.qualified:
        raise RowFreezeRefusedR3(f"outcome at ordinal={selected_ordinal} is recorded as NOT qualified")
    if outcome.unique_id != selected_unique_id:
        raise RowFreezeRefusedR3("outcome unique_id does not match the qualification artifact's own selection")
    if outcome.raw_row_sha256 != candidate_row.raw_row_sha256:
        raise RowFreezeRefusedR3("outcome raw_row_sha256 does not match the candidate manifest row's own hash")
    if outcome.problem_sha256 != candidate_row.problem_sha256:
        raise RowFreezeRefusedR3("outcome problem_sha256 does not match the candidate manifest row's own hash")
    if outcome.gold_answer_sha256 != candidate_row.gold_answer_sha256:
        raise RowFreezeRefusedR3("outcome gold_answer_sha256 does not match the candidate manifest row's own hash")

    return candidate_row, outcome


def construct_selected_manifest_and_provenance(
    *,
    candidate_manifest: dict[str, Any],
    qualification_artifact: dict[str, Any],
    expected_config_sha256: str,
    stage_b_authorization_context: Any,
    tokenizer_renderer: TokenizerRenderer,
) -> tuple[B2AOneExampleManifest, dict[str, Any]]:
    """Pure construction only -- no filesystem I/O. Reads the selected
    candidate's complete pinned row from the candidate manifest's own
    embedded `row` field; NEVER refetches the dataset (protocol §13,
    R3-AUDIT-22's frozen embed-not-refetch rule)."""
    candidate_row, _outcome = verify_freeze_chain(
        candidate_manifest=candidate_manifest, qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=stage_b_authorization_context,
    )

    row = candidate_row.row
    if tuple(row.keys()) != EMBEDDED_ROW_COLUMNS:
        raise RowFreezeRefusedR3(f"embedded row has unexpected columns: {tuple(row.keys())}")
    recomputed_raw_hash = sha256_json(row)
    if recomputed_raw_hash != candidate_row.raw_row_sha256:
        raise RowFreezeRefusedR3(
            f"candidate row's raw content hash does not reproduce: recomputed={recomputed_raw_hash!r} "
            f"stored={candidate_row.raw_row_sha256!r}"
        )

    rendering = PromptRenderingResult.model_validate(tokenizer_renderer(row))
    _verify_rendering_binding(rendering, candidate_manifest)

    new_manifest = B2AOneExampleManifest(
        dataset_repo=candidate_manifest["dataset_repo"],
        dataset_config=candidate_manifest["dataset_config"],
        dataset_split=candidate_manifest["dataset_split"],
        dataset_revision=candidate_manifest["dataset_revision"],
        example_index=candidate_row.source_example_index,
        unique_id=candidate_row.unique_id,
        raw_content_hash=recomputed_raw_hash,
        gold_answer=row["answer"],
        prompt_token_ids_sha256=rendering.prompt_token_ids_sha256,
        tokenizer_revision_used_for_prompt_hash=rendering.tokenizer_revision_used_for_prompt_hash,
        rendered_user_message_sha256=rendering.rendered_user_message_sha256,
        chat_template_source_sha256=rendering.chat_template_source_sha256,
        chat_message_payload_sha256=rendering.chat_message_payload_sha256,
        prompt_rendering_config=rendering.prompt_rendering_config,
        prompt_token_count=rendering.prompt_token_count,
        prompt_token_ids=rendering.prompt_token_ids,
    )
    selected_manifest_sha256 = new_manifest.manifest_hash()

    provenance: dict[str, Any] = {
        "artifact_schema_version": SELECTION_PROVENANCE_ARTIFACT_SCHEMA_VERSION,
        "qualification_artifact_path": QUALIFICATION_ARTIFACT_PATH,
        "qualification_artifact_canonical_sha256": qualification_artifact["canonical_sha256"],
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH,
        "candidate_manifest_canonical_sha256": candidate_manifest["canonical_sha256"],
        "selected_manifest_path": SELECTED_MANIFEST_PATH,
        "selected_manifest_sha256": selected_manifest_sha256,
        "selected_manifest_hash_algorithm": SELECTED_MANIFEST_HASH_ALGORITHM,
        "selected_ordinal": candidate_row.candidate_ordinal,
        "selected_unique_id": candidate_row.unique_id,
        "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
        "row_raw_sha256": recomputed_raw_hash,
        "prompt_token_ids_sha256": rendering.prompt_token_ids_sha256,
        "tokenizer_revision_used_for_prompt_hash": rendering.tokenizer_revision_used_for_prompt_hash,
    }
    provenance["canonical_sha256"] = compute_canonical_sha256(provenance)
    SelectionProvenanceR3.model_validate(provenance)  # construct-time self-check
    return new_manifest, provenance


def write_selected_manifest_and_provenance(
    new_manifest: B2AOneExampleManifest,
    provenance: dict[str, Any],
    *,
    manifest_path: str | Path,
    provenance_path: str | Path,
) -> None:
    """The I/O layer -- used ONLY by tests against `tmp_path` in Stage A.
    No Stage-A CLI command calls this against the real
    `configs/discovery/b2a_one_example_manifest.json` /
    `results/decisions/b2a_r3_selection_provenance.json` paths."""
    from kvcot.discovery.b2a_r3_candidates import atomic_write_json

    atomic_write_json(manifest_path, new_manifest.model_dump(mode="json"))
    atomic_write_json(provenance_path, provenance)


def verify_selection_provenance(
    provenance: dict[str, Any],
    *,
    selected_manifest: B2AOneExampleManifest,
    candidate_manifest: dict[str, Any],
    qualification_artifact: dict[str, Any],
    expected_config_sha256: str,
    stage_b_authorization_context: Any,
) -> SelectionProvenanceR3:
    """Full strict verification against the selected manifest's own
    EXTERNAL hash (`B2AOneExampleManifest.manifest_hash()`, never a
    `canonical_sha256` field added to that historical schema) and the
    candidate-manifest / qualification-artifact hashes actually supplied."""
    verify_canonical_sha256(provenance)
    typed = SelectionProvenanceR3.model_validate(provenance)
    candidate_row, outcome = verify_freeze_chain(
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=stage_b_authorization_context,
    )
    if typed.selected_ordinal != candidate_row.candidate_ordinal:
        raise RowFreezeRefusedR3("selection provenance ordinal does not match the replayed freeze chain")
    if typed.selected_unique_id != candidate_row.unique_id:
        raise RowFreezeRefusedR3("selection provenance unique_id does not match the replayed freeze chain")
    if typed.row_raw_sha256 != candidate_row.raw_row_sha256:
        raise RowFreezeRefusedR3("selection provenance row hash does not match the selected candidate")

    if (
        selected_manifest.dataset_repo,
        selected_manifest.dataset_config,
        selected_manifest.dataset_split,
        selected_manifest.dataset_revision,
        selected_manifest.example_index,
        selected_manifest.unique_id,
        selected_manifest.raw_content_hash,
        selected_manifest.gold_answer,
    ) != (
        candidate_manifest["dataset_repo"],
        candidate_manifest["dataset_config"],
        candidate_manifest["dataset_split"],
        candidate_manifest["dataset_revision"],
        candidate_row.source_example_index,
        candidate_row.unique_id,
        candidate_row.raw_row_sha256,
        candidate_row.row["answer"],
    ):
        raise RowFreezeRefusedR3("selected manifest identity/content does not match the replayed selected row")
    if selected_manifest.prompt_token_ids is None or selected_manifest.prompt_token_count is None:
        raise RowFreezeRefusedR3("selected manifest prompt token identity is incomplete")
    if any(type(token_id) is not int for token_id in selected_manifest.prompt_token_ids):
        raise RowFreezeRefusedR3("selected manifest prompt token IDs are not strict integers")
    if selected_manifest.prompt_token_count != len(selected_manifest.prompt_token_ids):
        raise RowFreezeRefusedR3("selected manifest prompt token count does not reproduce")
    prompt_hash = sha256_int_ids(list(selected_manifest.prompt_token_ids))
    if prompt_hash != selected_manifest.prompt_token_ids_sha256:
        raise RowFreezeRefusedR3("selected manifest prompt token hash does not reproduce")
    if not (
        selected_manifest.prompt_token_ids_sha256
        == outcome.expected_prompt_token_ids_sha256
        == outcome.observed_prompt_token_ids_sha256
    ):
        raise RowFreezeRefusedR3(
            "selected manifest prompt token hash does not match the replayed qualification outcome"
        )
    if selected_manifest.tokenizer_revision_used_for_prompt_hash != candidate_manifest["tokenizer_revision"]:
        raise RowFreezeRefusedR3("selected manifest tokenizer revision does not match the candidate manifest")
    config = selected_manifest.prompt_rendering_config
    if config is None or (
        tuple(config.message_roles), config.add_generation_prompt, config.tokenize, config.add_special_tokens_note
    ) != (PROMPT_MESSAGE_ROLES, PROMPT_ADD_GENERATION, PROMPT_TOKENIZE, PROMPT_SPECIAL_TOKENS_NOTE):
        raise RowFreezeRefusedR3("selected manifest rendering config is not the frozen convention")
    if typed.selected_manifest_sha256 != selected_manifest.manifest_hash():
        raise RowFreezeRefusedR3(
            "selection provenance's selected_manifest_sha256 does not match "
            "B2AOneExampleManifest.manifest_hash() of the supplied manifest"
        )
    if typed.candidate_manifest_canonical_sha256 != candidate_manifest.get("canonical_sha256"):
        raise RowFreezeRefusedR3(
            "selection provenance's candidate_manifest_canonical_sha256 does not match the supplied "
            "candidate manifest's own canonical_sha256"
        )
    if typed.qualification_artifact_canonical_sha256 != qualification_artifact.get("canonical_sha256"):
        raise RowFreezeRefusedR3(
            "selection provenance's qualification_artifact_canonical_sha256 does not match the supplied "
            "qualification artifact's own canonical_sha256"
        )
    if typed.prompt_token_ids_sha256 != selected_manifest.prompt_token_ids_sha256:
        raise RowFreezeRefusedR3("selection provenance prompt hash does not match the selected manifest")
    if typed.tokenizer_revision_used_for_prompt_hash != selected_manifest.tokenizer_revision_used_for_prompt_hash:
        raise RowFreezeRefusedR3("selection provenance tokenizer revision does not match the selected manifest")
    if outcome.unique_id != candidate_row.unique_id:
        raise RowFreezeRefusedR3("qualification outcome does not match the replayed candidate")
    return typed


def plan_freeze_dry_run(
    *,
    candidate_manifest: dict[str, Any],
    qualification_artifact: dict[str, Any],
    expected_config_sha256: str,
    stage_b_authorization_context: Any,
) -> dict[str, Any]:
    """CPU-only planning: verifies the complete freeze chain and reports
    what a real freeze WOULD do -- never touches a tokenizer, never writes
    a file. This is the only function `kvcot freeze-b2a-r3-selected-row
    --dry-run` may call."""
    try:
        candidate_row, _outcome = verify_freeze_chain(
            candidate_manifest=candidate_manifest, qualification_artifact=qualification_artifact,
            expected_config_sha256=expected_config_sha256,
            stage_b_authorization_context=stage_b_authorization_context,
        )
    except RowFreezeRefusedR3 as exc:
        return {
            "would_freeze": False,
            "refusal_reason": str(exc),
            "would_load_tokenizer_for_execution": False,
            "would_write_selected_manifest": False,
            "would_write_selection_provenance": False,
        }
    return {
        "would_freeze": True,
        "selected_ordinal": candidate_row.candidate_ordinal,
        "selected_unique_id": candidate_row.unique_id,
        "would_load_tokenizer_for_execution": False,
        "would_write_selected_manifest": False,
        "would_write_selection_provenance": False,
    }


# ==========================================================================
# Phase 2 (freezer implementation authorization, dated 2026-07-24):
# production freeze-plan construction and the guarded publication state
# machine. Everything above this line is unchanged Stage-A code (pure
# construction, synthetic-only). Everything below writes to the real fixed
# production paths, but ONLY from `publish_production_freeze`, and ONLY
# after `construct_production_freeze_plan` has completed all nine
# construction-order steps entirely in memory.
# ==========================================================================


class ProductionPublicationRefused(RuntimeError):
    """Any hard refusal before, during, or after production publication --
    an invalid/ambiguous live output state, a worktree/Git safety failure,
    a byte mismatch at any verification point, or a post-publication
    full-chain verification failure. Never deletes or overwrites a target
    it cannot first prove is safe to touch."""


PUBLICATION_STATE_A_INITIAL = "state_a_initial"
PUBLICATION_STATE_B_COMPLETE = "state_b_complete"
PUBLICATION_STATE_C_PROVENANCE_FIRST_PARTIAL = "state_c_provenance_first_partial"
PUBLICATION_STATE_D_MANIFEST_FIRST_PARTIAL = "state_d_manifest_first_partial"
PUBLICATION_STATE_E_INVALID = "state_e_invalid"

_PLAN_CONSTRUCTION_TOKEN = object()


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    """The one serialization every production write in this module uses --
    identical to `kvcot.discovery.b2a_r3_candidates.atomic_write_json`'s own
    convention (fixed field order, `ensure_ascii=True`, trailing newline) so
    a byte comparison against a freshly-serialized expected payload is
    exact, never merely semantic."""
    return (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


@dataclass(frozen=True)
class ProductionFreezePlan:
    """Everything a real freeze publication needs, constructed and
    internally verified entirely in memory before
    `publish_production_freeze` is ever called (protocol: construction
    order steps 1-9). Every field here is immutable evidence, never a
    caller-supplied override."""

    repository_root: str
    current_git_sha: str
    candidate_manifest_path: str
    candidate_manifest_canonical_sha256: str
    qualification_artifact_path: str
    qualification_artifact_canonical_sha256: str
    selected_manifest_path: str
    selection_provenance_path: str
    selected_ordinal: int
    selected_unique_id: str
    historical_selected_manifest_git_blob_sha: str
    historical_selected_manifest_bytes_sha256: str
    historical_selected_manifest_bytes: bytes
    expected_new_manifest_bytes: bytes
    expected_new_manifest_sha256: str
    expected_provenance_bytes: bytes
    expected_provenance_canonical_sha256: str
    tokenizer_repository: str
    tokenizer_requested_revision: str
    tokenizer_resolved_revision: str
    tokenizer_local_path: str
    publication_state_before: str
    candidate_manifest: dict[str, Any]
    qualification_artifact: dict[str, Any]
    expected_config_sha256: str
    stage_b_authorization_context: Any
    _construction_token: object


def construct_production_freeze_plan(
    *,
    repository_root: str | Path = ".",
    config_path: str,
    tokenizer_renderer: TokenizerRenderer,
    tokenizer_repository: str,
    tokenizer_requested_revision: str,
    tokenizer_resolved_revision: str,
    tokenizer_local_path: str,
) -> ProductionFreezePlan:
    """Construction order (steps 1-9), entirely in memory, no filesystem
    publication:

    1. Verify persisted Stage-B authorization binding.
    2. Verify candidate manifest.
    3. Verify qualification artifact.
    4. Replay selected-row chain.
    5. Resolve exact local tokenizer (already done by the caller --
       `tokenizer_repository`/`tokenizer_resolved_revision`/
       `tokenizer_local_path` are the already-verified snapshot identity;
       see `kvcot.discovery.b2a_r3_production_tokenizer`).
    6. Render exact prompt (via the injected `tokenizer_renderer`, inside
       step 7/8's call into `construct_selected_manifest_and_provenance`).
    7. Construct selected manifest.
    8. Construct provenance.
    9. Verify the expected complete chain in memory
       (`verify_selection_provenance`, called here against the freshly
       constructed manifest/provenance -- BEFORE any write).
    """
    from kvcot.config import config_identity
    from kvcot.discovery.b2a_r3_authorization import verify_persisted_stage_b_authorization_binding
    from kvcot.discovery.b2a_r3_provenance import SubprocessGitStateProvider

    repository_root = str(repository_root)
    candidate_manifest_path = Path(repository_root) / CANDIDATE_MANIFEST_PATH
    qualification_artifact_path = Path(repository_root) / QUALIFICATION_ARTIFACT_PATH
    selected_manifest_path = Path(repository_root) / SELECTED_MANIFEST_PATH

    with open(candidate_manifest_path, "r", encoding="utf-8") as f:
        candidate_manifest = json.load(f)
    with open(qualification_artifact_path, "r", encoding="utf-8") as f:
        qualification_artifact = json.load(f)
    expected_config_sha256 = config_identity(Path(repository_root) / config_path)

    git_state = SubprocessGitStateProvider(repository_root)

    # Step 1: verify persisted Stage-B authorization binding.
    stage_b_binding = verify_persisted_stage_b_authorization_binding(
        authorization_id=qualification_artifact["stage_b_authorization_id"],
        repository_root=repository_root,
        git_state=git_state,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
    )

    # Steps 2-4, 6-8: candidate/qualification verification, selected-row
    # chain replay, prompt rendering, and construction -- all delegated to
    # the exact same pure-construction function Stage A already exercises
    # against synthetic fixtures. `tokenizer_renderer` performs step 5/6.
    new_manifest, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=stage_b_binding,
        tokenizer_renderer=tokenizer_renderer,
    )
    if provenance["tokenizer_revision_used_for_prompt_hash"] != tokenizer_resolved_revision:
        raise RowFreezeRefusedR3(
            "constructed provenance tokenizer revision does not match the resolved production snapshot"
        )

    # Step 9: verify the expected complete chain in memory, BEFORE any
    # write -- reuses the one authoritative full-chain verifier.
    verify_selection_provenance(
        provenance,
        selected_manifest=new_manifest,
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=stage_b_binding,
    )

    current_git_sha = git_state.current_commit_sha()
    # The "historical" manifest is always the committed bytes at the
    # CURRENT git commit -- never whatever happens to be on disk right now.
    # Publication never commits its own writes, so this stays stable across
    # repeated plan construction (idempotent/recovery calls), but reading
    # live disk bytes here instead would silently drift to whatever the
    # last publish already wrote, making State A/B/C/D classification
    # depend on execution order rather than committed Git history.
    historical_bytes = git_state.file_text_at_commit(SELECTED_MANIFEST_PATH, current_git_sha).encode("utf-8")
    blob_sha = subprocess.run(
        ["git", "rev-parse", f"{current_git_sha}:{SELECTED_MANIFEST_PATH}"],
        cwd=repository_root, text=True, capture_output=True, check=True,
    ).stdout.strip()

    expected_new_manifest_bytes = _canonical_json_bytes(new_manifest.model_dump(mode="json"))
    expected_provenance_bytes = _canonical_json_bytes(provenance)

    plan = ProductionFreezePlan(
        repository_root=repository_root,
        current_git_sha=current_git_sha,
        candidate_manifest_path=str(candidate_manifest_path),
        candidate_manifest_canonical_sha256=candidate_manifest["canonical_sha256"],
        qualification_artifact_path=str(qualification_artifact_path),
        qualification_artifact_canonical_sha256=qualification_artifact["canonical_sha256"],
        selected_manifest_path=str(selected_manifest_path),
        selection_provenance_path=str(Path(repository_root) / SELECTION_PROVENANCE_PATH),
        selected_ordinal=provenance["selected_ordinal"],
        selected_unique_id=provenance["selected_unique_id"],
        historical_selected_manifest_git_blob_sha=blob_sha,
        historical_selected_manifest_bytes_sha256=sha256_bytes(historical_bytes),
        historical_selected_manifest_bytes=historical_bytes,
        expected_new_manifest_bytes=expected_new_manifest_bytes,
        expected_new_manifest_sha256=new_manifest.manifest_hash(),
        expected_provenance_bytes=expected_provenance_bytes,
        expected_provenance_canonical_sha256=provenance["canonical_sha256"],
        tokenizer_repository=tokenizer_repository,
        tokenizer_requested_revision=tokenizer_requested_revision,
        tokenizer_resolved_revision=tokenizer_resolved_revision,
        tokenizer_local_path=tokenizer_local_path,
        publication_state_before=PUBLICATION_STATE_E_INVALID,  # placeholder, replaced below
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=stage_b_binding,
        _construction_token=_PLAN_CONSTRUCTION_TOKEN,
    )
    # classify_publication_state only reads `plan`'s already-frozen fields
    # (never re-derives anything), so it is safe to call once more, now
    # that the plan object exists, to fill in the real starting state.
    real_state = classify_publication_state(plan=plan)
    return _dataclass_replace(plan, publication_state_before=real_state)


def classify_publication_state(*, plan: ProductionFreezePlan) -> str:
    """States A-E, classified purely from the plan's own frozen evidence and
    the CURRENT bytes on disk at the two fixed production output paths --
    never from any other signal."""
    manifest_path = Path(plan.selected_manifest_path)
    provenance_path = Path(plan.selection_provenance_path)

    if not manifest_path.exists():
        return PUBLICATION_STATE_E_INVALID
    manifest_bytes = manifest_path.read_bytes()
    provenance_exists = provenance_path.exists()
    provenance_bytes = provenance_path.read_bytes() if provenance_exists else None

    manifest_is_historical = manifest_bytes == plan.historical_selected_manifest_bytes
    manifest_is_expected_new = manifest_bytes == plan.expected_new_manifest_bytes
    provenance_is_expected = provenance_exists and provenance_bytes == plan.expected_provenance_bytes

    if manifest_is_historical and not provenance_exists:
        return PUBLICATION_STATE_A_INITIAL
    if manifest_is_expected_new and provenance_is_expected:
        return PUBLICATION_STATE_B_COMPLETE
    if manifest_is_historical and provenance_is_expected:
        return PUBLICATION_STATE_C_PROVENANCE_FIRST_PARTIAL
    if manifest_is_expected_new and not provenance_exists:
        return PUBLICATION_STATE_D_MANIFEST_FIRST_PARTIAL
    return PUBLICATION_STATE_E_INVALID


def verify_git_worktree_safety_for_freeze(plan: ProductionFreezePlan) -> None:
    """Phase 2.9: refuses publication on anything but a worktree whose only
    (possible) differences from a clean checkout are the two fixed
    production output paths themselves. This structurally rejects an
    altered candidate manifest, qualification artifact, consumed claim,
    R-KV pin, or any unrelated tracked/untracked change -- any of those
    would show up as an extra dirty/staged/untracked path and get caught
    by the allowlist check below."""
    from kvcot.discovery.b2a_r3_provenance import SubprocessGitStateProvider

    git_state = SubprocessGitStateProvider(plan.repository_root)
    if str(Path(git_state.repository_root).resolve()) != str(Path(plan.repository_root).resolve()):
        raise ProductionPublicationRefused("git_state.repository_root does not match the plan's repository_root")
    if git_state.current_repository() != REQUIRED_REPOSITORY:
        raise ProductionPublicationRefused(
            f"current repository does not match the required {REQUIRED_REPOSITORY!r}"
        )
    claim = plan.stage_b_authorization_context.claim
    if git_state.current_branch() != claim.authorized_branch:
        raise ProductionPublicationRefused(
            f"current branch does not match the Stage-B authorized branch {claim.authorized_branch!r}"
        )
    if git_state.current_commit_sha() != plan.current_git_sha:
        raise ProductionPublicationRefused("current commit SHA has changed since the plan was constructed")
    observed_rkv_sha = git_state.rkv_submodule_sha()
    if observed_rkv_sha != claim.required_rkv_sha:
        raise ProductionPublicationRefused(
            f"R-KV submodule SHA {observed_rkv_sha!r} does not match the required {claim.required_rkv_sha!r}"
        )

    allowed_paths = {SELECTED_MANIFEST_PATH, SELECTION_PROVENANCE_PATH}
    status = git_state.worktree_status()
    unexpected = sorted(status.dirty_paths - allowed_paths)
    if unexpected:
        raise ProductionPublicationRefused(
            f"unrelated worktree change(s) detected, refusing publication: {unexpected}"
        )


def _fsync_directory(path: Path) -> None:
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass


def _atomic_replace_verified_historical(target: Path, *, expected_current_bytes: bytes, new_bytes: bytes) -> None:
    """Historical selected-manifest replacement: atomic `os.replace`, but
    ONLY after the live bytes on disk exactly equal the expected committed
    historical bytes -- never a blind overwrite."""
    if not target.exists():
        raise ProductionPublicationRefused(f"expected historical manifest at {target} does not exist")
    current = target.read_bytes()
    if current != expected_current_bytes:
        raise ProductionPublicationRefused(
            f"live bytes at {target} do not exactly match the expected committed historical bytes -- refusing replace"
        )
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".b2a-r3-freeze-manifest-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(new_bytes)
            f.flush()
            os.fsync(f.fileno())
        if Path(tmp_name).read_bytes() != new_bytes:
            raise ProductionPublicationRefused("temporary manifest file bytes do not match the expected new bytes")
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    _fsync_directory(target.parent)
    if target.read_bytes() != new_bytes:
        raise ProductionPublicationRefused(f"reload of {target} after publish does not match the expected new bytes")


def _exclusive_publish_no_clobber(target: Path, payload_bytes: bytes) -> None:
    """No-clobber provenance publication (protocol §2.8): write+fsync a
    temp file, `os.link` it onto the target (never `os.replace`), fail on
    `FileExistsError`, clean up the temp path either way, fsync the parent
    directory, reload, and require exact byte equality. Identical idiom to
    `b2a_r3_authorization._create_authorization_claim` and
    `b2a_r3_artifacts._exclusive_write_json`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".b2a-r3-freeze-provenance-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload_bytes)
            f.flush()
            os.fsync(f.fileno())
        if Path(tmp_name).read_bytes() != payload_bytes:
            raise ProductionPublicationRefused("temporary provenance file bytes do not match the expected payload")
        try:
            os.link(tmp_name, target)
        except FileExistsError as exc:
            raise ProductionPublicationRefused(
                f"refusing to overwrite an existing file at {target} -- selection provenance is no-clobber"
            ) from exc
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    else:
        os.remove(tmp_name)
    _fsync_directory(target.parent)
    if target.read_bytes() != payload_bytes:
        raise ProductionPublicationRefused(f"reload of {target} after publish does not match the expected payload")


def publish_production_freeze(plan: ProductionFreezePlan) -> dict[str, Any]:
    """The guarded publication state machine (States A-E). Refuses on
    State E without touching either target. From State A, publishes the
    manifest first, then the no-clobber provenance, then runs full-chain
    verification (the normal order) -- States C/D each publish only the
    single missing output before that same final verification; State B is
    an idempotent no-op. Full-chain verification is required once, only
    after both outputs are confirmed present -- never while intentionally
    in a one-file partial state."""
    if plan._construction_token is not _PLAN_CONSTRUCTION_TOKEN:
        raise ProductionPublicationRefused("plan was not produced by construct_production_freeze_plan")

    verify_git_worktree_safety_for_freeze(plan)

    manifest_path = Path(plan.selected_manifest_path)
    provenance_path = Path(plan.selection_provenance_path)

    state_before = classify_publication_state(plan=plan)
    if state_before == PUBLICATION_STATE_E_INVALID:
        raise ProductionPublicationRefused(
            "live production output state is invalid/ambiguous (State E) -- refusing to publish or touch any target"
        )

    already_frozen = state_before == PUBLICATION_STATE_B_COMPLETE
    if state_before == PUBLICATION_STATE_A_INITIAL:
        _atomic_replace_verified_historical(
            manifest_path,
            expected_current_bytes=plan.historical_selected_manifest_bytes,
            new_bytes=plan.expected_new_manifest_bytes,
        )
        _exclusive_publish_no_clobber(provenance_path, plan.expected_provenance_bytes)
    elif state_before == PUBLICATION_STATE_C_PROVENANCE_FIRST_PARTIAL:
        _atomic_replace_verified_historical(
            manifest_path,
            expected_current_bytes=plan.historical_selected_manifest_bytes,
            new_bytes=plan.expected_new_manifest_bytes,
        )
    elif state_before == PUBLICATION_STATE_D_MANIFEST_FIRST_PARTIAL:
        _exclusive_publish_no_clobber(provenance_path, plan.expected_provenance_bytes)
    # State B: already_frozen -- no write of either kind.

    reloaded_manifest = B2AOneExampleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    reloaded_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    verify_selection_provenance(
        reloaded_provenance,
        selected_manifest=reloaded_manifest,
        candidate_manifest=plan.candidate_manifest,
        qualification_artifact=plan.qualification_artifact,
        expected_config_sha256=plan.expected_config_sha256,
        stage_b_authorization_context=plan.stage_b_authorization_context,
    )
    state_after = classify_publication_state(plan=plan)
    if state_after != PUBLICATION_STATE_B_COMPLETE:
        raise ProductionPublicationRefused(
            f"post-publication state is {state_after!r}, not State B complete -- full-chain verification failed"
        )

    return {
        "selected_unique_id": plan.selected_unique_id,
        "selected_ordinal": plan.selected_ordinal,
        "selected_manifest_path": plan.selected_manifest_path,
        "selected_manifest_sha256": plan.expected_new_manifest_sha256,
        "selection_provenance_path": plan.selection_provenance_path,
        "selection_provenance_canonical_sha256": plan.expected_provenance_canonical_sha256,
        "tokenizer_snapshot_revision": plan.tokenizer_resolved_revision,
        "publication_state_before": state_before,
        "publication_state_after": state_after,
        "already_frozen": already_frozen,
        "verification_passed": True,
    }
