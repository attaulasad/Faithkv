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

from dataclasses import dataclass
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
    compute_canonical_sha256,
    require_lowercase_hex64,
    verify_canonical_sha256,
)
from kvcot.discovery.manifest import B2AOneExampleManifest, ChatTemplateRenderingConfig
from kvcot.utils.hashing import sha256_json

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
]


class RowFreezeRefusedR3(RuntimeError):
    """Raised whenever a freeze request does not exactly match what the
    qualification artifact and candidate manifest jointly authorize."""


@dataclass(frozen=True)
class PromptRenderingResult:
    """What an injected `TokenizerRenderer` must return -- the rendering-
    derived fields only; the row/raw-content hash are already known from
    the candidate manifest and are never re-derived here."""

    rendered_user_message_sha256: str
    chat_template_source_sha256: str
    chat_message_payload_sha256: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str
    prompt_token_count: int
    tokenizer_revision_used_for_prompt_hash: str
    prompt_rendering_config: ChatTemplateRenderingConfig


TokenizerRenderer = Callable[[dict[str, Any]], PromptRenderingResult]


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
    *, candidate_manifest: dict[str, Any], qualification_artifact: dict[str, Any], expected_config_sha256: str
) -> tuple[Any, Any]:
    """Every check protocol §13 requires BEFORE any freeze proceeds.
    Returns `(candidate_row, qualification_outcome)` -- both already
    strictly typed -- only when every check passes."""
    candidate_typed = verify_candidate_manifest_structure(candidate_manifest)
    qualification_typed = verify_qualification_artifact(
        qualification_artifact, candidate_manifest=candidate_manifest, expected_config_sha256=expected_config_sha256
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
    tokenizer_renderer: TokenizerRenderer,
) -> tuple[B2AOneExampleManifest, dict[str, Any]]:
    """Pure construction only -- no filesystem I/O. Reads the selected
    candidate's complete pinned row from the candidate manifest's own
    embedded `row` field; NEVER refetches the dataset (protocol §13,
    R3-AUDIT-22's frozen embed-not-refetch rule)."""
    candidate_row, _outcome = verify_freeze_chain(
        candidate_manifest=candidate_manifest, qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
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

    rendering = tokenizer_renderer(row)

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
) -> SelectionProvenanceR3:
    """Full strict verification against the selected manifest's own
    EXTERNAL hash (`B2AOneExampleManifest.manifest_hash()`, never a
    `canonical_sha256` field added to that historical schema) and the
    candidate-manifest / qualification-artifact hashes actually supplied."""
    verify_canonical_sha256(provenance)
    typed = SelectionProvenanceR3.model_validate(provenance)
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
    return typed


def plan_freeze_dry_run(
    *, candidate_manifest: dict[str, Any], qualification_artifact: dict[str, Any], expected_config_sha256: str
) -> dict[str, Any]:
    """CPU-only planning: verifies the complete freeze chain and reports
    what a real freeze WOULD do -- never touches a tokenizer, never writes
    a file. This is the only function `kvcot freeze-b2a-r3-selected-row
    --dry-run` may call."""
    try:
        candidate_row, _outcome = verify_freeze_chain(
            candidate_manifest=candidate_manifest, qualification_artifact=qualification_artifact,
            expected_config_sha256=expected_config_sha256,
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
