"""Freeze the B2A-R2 qualified row into a replacement
`configs/discovery/b2a_one_example_manifest.json` (2026-07-22).

The selected row may ONLY come from an immutable qualification artifact
(`kvcot.discovery.b2a_qualification`) that has already run FullKV-only
qualification against the committed candidate manifest
(`kvcot.discovery.b2a_r2_candidates`) and identified the first candidate,
in committed order, satisfying every frozen qualification condition. This
module's entire purpose is to make every OTHER path fail closed:

  - an arbitrary `example_index` the caller supplies directly -- rejected;
    the index is read ONLY from the candidate manifest, at the ordinal the
    qualification artifact itself selected.
  - a candidate manifest that does not match the qualification artifact's
    recorded hash of it -- rejected (the artifact and the manifest it was
    computed against must be the SAME committed file).
  - a config whose dataset/model/tokenizer revision or R-KV budget disagree
    with what the qualification artifact/candidate manifest recorded --
    rejected.
  - a selected candidate whose OWN recorded qualification conditions are
    not ALL true -- rejected (defense in depth: even if `selected_ordinal`
    were somehow tampered with, the conditions are re-checked here, not
    merely trusted).
  - any ordinal other than the artifact's own `selected_ordinal` -- there
    is no parameter through which a caller can request a different row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kvcot.discovery.manifest_prepare import ManifestPreparationError


class RowFreezeRefused(RuntimeError):
    """Raised whenever the requested freeze does not exactly match what the
    qualification artifact and candidate manifest jointly authorize."""


@dataclass(frozen=True)
class SelectionProvenance:
    """Everything tying the frozen replacement manifest back to the exact
    qualification decision that selected it -- written as a companion file,
    never folded into `B2AOneExampleManifest` itself (that schema's fields
    all remain exactly as they were; this module adds no new field to it)."""

    qualification_artifact_path: str
    qualification_artifact_hash: str
    candidate_manifest_path: str
    candidate_manifest_hash: str
    selected_ordinal: int
    selected_unique_id: str
    selection_protocol_version: str
    row_raw_sha256: str
    prompt_token_ids_sha256: str
    tokenizer_revision_used_for_prompt_hash: str

    def to_json(self) -> dict[str, Any]:
        return {
            "qualification_artifact_path": self.qualification_artifact_path,
            "qualification_artifact_hash": self.qualification_artifact_hash,
            "candidate_manifest_path": self.candidate_manifest_path,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "selected_ordinal": self.selected_ordinal,
            "selected_unique_id": self.selected_unique_id,
            "selection_protocol_version": self.selection_protocol_version,
            "row_raw_sha256": self.row_raw_sha256,
            "prompt_token_ids_sha256": self.prompt_token_ids_sha256,
            "tokenizer_revision_used_for_prompt_hash": self.tokenizer_revision_used_for_prompt_hash,
        }


def _require_selected_and_qualified(qualification_artifact: dict[str, Any]) -> tuple[int, str]:
    selected_ordinal = qualification_artifact.get("selected_ordinal")
    selected_unique_id = qualification_artifact.get("selected_unique_id")
    if selected_ordinal is None or selected_unique_id is None:
        raise RowFreezeRefused("qualification artifact has no selected row -- nothing to freeze.")

    matches = [
        outcome for outcome in qualification_artifact["attempted"]
        if outcome["candidate_ordinal"] == selected_ordinal
    ]
    if len(matches) != 1:
        raise RowFreezeRefused(
            f"qualification artifact's selected_ordinal={selected_ordinal} does not match exactly one "
            f"attempted candidate (found {len(matches)})."
        )
    outcome = matches[0]
    if outcome["unique_id"] != selected_unique_id:
        raise RowFreezeRefused(
            f"qualification artifact is internally inconsistent: selected_unique_id={selected_unique_id!r} "
            f"but the outcome at selected_ordinal={selected_ordinal} has unique_id={outcome['unique_id']!r}."
        )
    if not outcome["qualified"]:
        raise RowFreezeRefused(
            f"candidate ordinal={selected_ordinal} is recorded as NOT qualified "
            f"(failed_conditions={outcome['failed_conditions']!r}) -- refusing to freeze it."
        )
    if not all(outcome["conditions"].values()):
        raise RowFreezeRefused(
            f"candidate ordinal={selected_ordinal}'s own condition map has a false entry "
            f"({outcome['conditions']!r}) despite qualified=True -- refusing an internally inconsistent artifact."
        )
    return selected_ordinal, selected_unique_id


def _require_matching_candidate_row(
    candidate_manifest: dict[str, Any], *, selected_ordinal: int, selected_unique_id: str
) -> dict[str, Any]:
    matches = [
        c for c in candidate_manifest["candidates"]
        if c["candidate_ordinal"] == selected_ordinal
    ]
    if len(matches) != 1:
        raise RowFreezeRefused(
            f"candidate manifest does not have exactly one candidate at ordinal={selected_ordinal} "
            f"(found {len(matches)})."
        )
    row = matches[0]
    if row["unique_id"] != selected_unique_id:
        raise RowFreezeRefused(
            f"candidate manifest's row at ordinal={selected_ordinal} has unique_id={row['unique_id']!r}, "
            f"but the qualification artifact selected unique_id={selected_unique_id!r} -- refusing to freeze "
            "a mismatched row."
        )
    return row


def validate_freeze_request(
    *,
    config: Any,
    qualification_artifact: dict[str, Any],
    candidate_manifest: dict[str, Any],
    candidate_manifest_path: str,
) -> tuple[dict[str, Any], int, str]:
    """Every rejection this module promises, checked BEFORE anything is
    fetched, tokenized, or written. Returns `(candidate_row, selected_ordinal,
    selected_unique_id)` only when every check passes."""
    if qualification_artifact["candidate_manifest_hash"] != candidate_manifest["canonical_sha256"]:
        raise RowFreezeRefused(
            "qualification artifact's candidate_manifest_hash "
            f"({qualification_artifact['candidate_manifest_hash']!r}) does not match the candidate manifest's "
            f"own canonical_sha256 ({candidate_manifest['canonical_sha256']!r}) -- refusing to freeze against "
            "a candidate manifest the qualification artifact was not actually computed from."
        )

    for name, artifact_value, config_value in (
        ("dataset_revision", qualification_artifact["dataset_revision"], config.dataset.revision),
        ("model_revision", qualification_artifact["model_revision"], config.model.revision),
        ("tokenizer_revision", qualification_artifact["tokenizer_revision"], config.model.tokenizer_revision),
        ("budget", qualification_artifact["budget"], config.rkv.budget),
    ):
        if artifact_value != config_value:
            raise RowFreezeRefused(
                f"qualification artifact {name}={artifact_value!r} does not match config {name}={config_value!r}."
            )

    selected_ordinal, selected_unique_id = _require_selected_and_qualified(qualification_artifact)
    candidate_row = _require_matching_candidate_row(
        candidate_manifest, selected_ordinal=selected_ordinal, selected_unique_id=selected_unique_id
    )
    return candidate_row, selected_ordinal, selected_unique_id


def freeze_qualified_row(
    *,
    config: Any,
    qualification_artifact: dict[str, Any],
    qualification_artifact_path: str,
    qualification_artifact_hash: str,
    candidate_manifest: dict[str, Any],
    candidate_manifest_path: str,
    manifest_path: Any = None,
) -> tuple[Any, SelectionProvenance]:
    """The one function that may write a replacement
    `configs/discovery/b2a_one_example_manifest.json`. Every identity/hash
    check in this module runs first; only then is the tokenizer touched
    (CPU-only, no CUDA) to resolve the new row's prompt identity, reusing
    `kvcot.discovery.manifest_prepare._render_and_tokenize` directly -- the
    SAME rendering call the original `example_index=0` manifest and every
    B2A execution re-verification already use, never a second,
    independently-invented rendering path."""
    from kvcot.discovery.discovery_config import MATH500_DATASET_REPO
    from kvcot.discovery.manifest import B2AOneExampleManifest, DEFAULT_MANIFEST_PATH
    from kvcot.discovery.manifest_prepare import _atomic_write_manifest, _render_and_tokenize, _verify_row_schema
    from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

    if manifest_path is None:
        manifest_path = DEFAULT_MANIFEST_PATH

    candidate_row, selected_ordinal, selected_unique_id = validate_freeze_request(
        config=config, qualification_artifact=qualification_artifact, candidate_manifest=candidate_manifest,
        candidate_manifest_path=candidate_manifest_path,
    )

    row = candidate_row["row"]
    _verify_row_schema(row)
    recomputed_raw_hash = sha256_json(row)
    if recomputed_raw_hash != candidate_row["raw_row_sha256"]:
        raise RowFreezeRefused(
            f"candidate row's raw content hash does not reproduce: recomputed={recomputed_raw_hash!r} "
            f"stored={candidate_row['raw_row_sha256']!r}."
        )

    tokenizer, user_message, messages, token_ids = _render_and_tokenize(
        row, config.model.tokenizer_name, config.model.tokenizer_revision,
    )
    from kvcot.discovery.manifest import ChatTemplateRenderingConfig

    new_manifest = B2AOneExampleManifest(
        dataset_repo=MATH500_DATASET_REPO,
        dataset_config=config.dataset.config,
        dataset_split=config.dataset.split,
        dataset_revision=config.dataset.revision,
        example_index=candidate_row["source_example_index"],
        unique_id=selected_unique_id,
        raw_content_hash=recomputed_raw_hash,
        gold_answer=row["answer"],
        prompt_token_ids_sha256=sha256_int_ids(token_ids),
        tokenizer_revision_used_for_prompt_hash=config.model.tokenizer_revision,
        rendered_user_message_sha256=sha256_text(user_message),
        chat_template_source_sha256=sha256_text(tokenizer.chat_template),
        chat_message_payload_sha256=sha256_json(messages),
        prompt_rendering_config=ChatTemplateRenderingConfig(
            message_roles=("user",), add_generation_prompt=True, tokenize=True,
            add_special_tokens_note=(
                "special-token behavior is entirely delegated to the tokenizer's own chat_template "
                "(Jinja) -- apply_chat_template was called with no separate add_special_tokens override"
            ),
        ),
        prompt_token_count=len(token_ids),
        prompt_token_ids=tuple(token_ids),
    )

    provenance = SelectionProvenance(
        qualification_artifact_path=qualification_artifact_path,
        qualification_artifact_hash=qualification_artifact_hash,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_hash=candidate_manifest["canonical_sha256"],
        selected_ordinal=selected_ordinal,
        selected_unique_id=selected_unique_id,
        selection_protocol_version=candidate_manifest["protocol_version"],
        row_raw_sha256=recomputed_raw_hash,
        prompt_token_ids_sha256=new_manifest.prompt_token_ids_sha256,
        tokenizer_revision_used_for_prompt_hash=config.model.tokenizer_revision,
    )

    _atomic_write_manifest(new_manifest, manifest_path)
    return new_manifest, provenance
