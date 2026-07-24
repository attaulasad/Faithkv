"""B2A-R3 production tokenizer renderer (Phase 2, freezer implementation
authorization dated 2026-07-24).

Deliberately kept OUT of `kvcot.discovery.b2a_r3_freeze` -- that module's
own AST-level import-safety test
(`tests/unit/test_cli_b2a_r3.py::test_source_scan_new_modules_never_import_forbidden_modules_anywhere`)
statically forbids `transformers`/`torch` anywhere in that file, module-level
or deferred. This module is the one, single place a real tokenizer is ever
loaded for B2A-R3 freeze publication; it implements the `TokenizerRenderer`
callable contract `kvcot.discovery.b2a_r3_freeze.TokenizerRenderer` already
defines, so it is injected into `construct_selected_manifest_and_provenance`
exactly like a test's fake renderer -- no second, independently-invented
prompt-rendering convention.

Boundary enforced here, never weakened:

- Exact local snapshot resolution only
  (`kvcot.discovery.snapshot_boundary.resolve_local_snapshot`,
  `asset_type="tokenizer"`, `local_files_only=True` enforced by that
  function) -- no network fallback, ever.
- Requested revision must equal resolved revision exactly.
- No `AutoModel` import, no direct `torch` import, no CUDA inspection or
  initialization -- only `transformers.AutoTokenizer`.
- No dataset access -- the row is always supplied by the caller (already
  resolved from the candidate manifest's own embedded row, never refetched
  here).
- Reuses `kvcot.discovery.manifest_prepare.render_with_loaded_tokenizer`
  directly for the actual chat-template call, so this is the SAME frozen
  rendering convention every other B2A prompt-identity path already uses --
  never a second one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from kvcot.discovery.b2a_r3_contract import (
    PROMPT_ADD_GENERATION,
    PROMPT_MESSAGE_ROLES,
    PROMPT_SPECIAL_TOKENS_NOTE,
    PROMPT_TOKENIZE,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
)
from kvcot.discovery.b2a_r3_freeze import PromptRenderingResult, RowFreezeRefusedR3
from kvcot.discovery.manifest import ChatTemplateRenderingConfig
from kvcot.discovery.snapshot_boundary import SnapshotBoundaryError, VerifiedLocalSnapshot, resolve_local_snapshot
from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

__all__ = [
    "ProductionTokenizerResolutionRefused",
    "resolve_production_tokenizer_snapshot",
    "render_production_prompt",
    "build_production_tokenizer_renderer",
]


class ProductionTokenizerResolutionRefused(RuntimeError):
    """Any hard rejection while resolving or loading the production
    tokenizer snapshot -- never a silent fallback to network resolution."""


def resolve_production_tokenizer_snapshot(*, cache_dir: str | Path | None = None) -> VerifiedLocalSnapshot:
    """Resolve the exact, frozen local tokenizer snapshot for B2A-R3 -- the
    same `TOKENIZER_NAME`/`TOKENIZER_REVISION` bound in
    `kvcot.discovery.b2a_r3_contract` (identical to `MODEL_NAME`/
    `MODEL_REVISION`; this function only ever touches the tokenizer side of
    that identity)."""
    try:
        snapshot = resolve_local_snapshot(
            repository_id=TOKENIZER_NAME,
            revision=TOKENIZER_REVISION,
            asset_type="tokenizer",
            cache_dir=cache_dir,
        )
    except SnapshotBoundaryError as exc:
        raise ProductionTokenizerResolutionRefused(
            f"exact local tokenizer snapshot unavailable for {TOKENIZER_NAME}@{TOKENIZER_REVISION}: {exc}"
        ) from exc
    if snapshot.asset_type != "tokenizer":
        raise ProductionTokenizerResolutionRefused("resolved snapshot asset_type is not 'tokenizer'")
    if snapshot.requested_revision != TOKENIZER_REVISION:
        raise ProductionTokenizerResolutionRefused("resolved snapshot requested_revision does not match TOKENIZER_REVISION")
    if snapshot.resolved_revision != TOKENIZER_REVISION:
        raise ProductionTokenizerResolutionRefused(
            f"resolved tokenizer revision {snapshot.resolved_revision!r} does not exactly equal the "
            f"requested/frozen revision {TOKENIZER_REVISION!r}"
        )
    if snapshot.local_files_only is not True:
        raise ProductionTokenizerResolutionRefused("resolved snapshot did not enforce local_files_only")
    return snapshot


def render_production_prompt(row: dict[str, Any], *, snapshot: VerifiedLocalSnapshot) -> PromptRenderingResult:
    """Load the tokenizer from the already-verified exact local snapshot
    and render the frozen B2A prompt convention. Never called with an
    unverified snapshot -- `build_production_tokenizer_renderer` below is
    the only production entry point and always resolves+verifies the
    snapshot first."""
    from transformers import AutoTokenizer

    from kvcot.discovery.manifest_prepare import render_with_loaded_tokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot.local_path, local_files_only=True, use_fast=True)
    if not tokenizer.chat_template:
        raise RowFreezeRefusedR3(
            f"production tokenizer {TOKENIZER_NAME}@{snapshot.resolved_revision} has no (or an empty) "
            "chat_template -- refusing to invent one."
        )

    user_message, messages, token_ids = render_with_loaded_tokenizer(tokenizer, row)
    if len(token_ids) == 0:
        raise RowFreezeRefusedR3("production prompt rendering produced zero tokens -- refusing an empty prompt.")

    return PromptRenderingResult(
        rendered_user_message_sha256=sha256_text(user_message),
        chat_template_source_sha256=sha256_text(tokenizer.chat_template),
        chat_message_payload_sha256=sha256_json(messages),
        prompt_token_ids=tuple(token_ids),
        prompt_token_ids_sha256=sha256_int_ids(token_ids),
        prompt_token_count=len(token_ids),
        tokenizer_revision_used_for_prompt_hash=snapshot.resolved_revision,
        prompt_rendering_config=ChatTemplateRenderingConfig(
            message_roles=PROMPT_MESSAGE_ROLES,
            add_generation_prompt=PROMPT_ADD_GENERATION,
            tokenize=PROMPT_TOKENIZE,
            add_special_tokens_note=PROMPT_SPECIAL_TOKENS_NOTE,
        ),
    )


def build_production_tokenizer_renderer(*, cache_dir: str | Path | None = None):
    """Returns a `TokenizerRenderer` callable (resolves the exact local
    snapshot once, eagerly, before returning -- so a caller building a
    freeze plan learns immediately if the snapshot is unavailable, never
    only at first prompt-rendering call) bound to that verified snapshot."""
    snapshot = resolve_production_tokenizer_snapshot(cache_dir=cache_dir)

    def _renderer(row: dict[str, Any]) -> PromptRenderingResult:
        return render_production_prompt(row, snapshot=snapshot)

    return _renderer
