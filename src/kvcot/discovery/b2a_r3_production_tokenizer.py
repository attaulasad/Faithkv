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

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from kvcot.discovery.b2a_r3_contract import (
    PROMPT_ADD_GENERATION,
    PROMPT_MESSAGE_ROLES,
    PROMPT_SPECIAL_TOKENS_NOTE,
    PROMPT_TOKENIZE,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
)
from kvcot.discovery.b2a_r3_freeze import PromptRenderingResult
from kvcot.discovery.snapshot_boundary import SnapshotBoundaryError, VerifiedLocalSnapshot, resolve_local_snapshot

__all__ = [
    "ProductionTokenizerResolutionRefused",
    "resolve_production_tokenizer_snapshot",
    "render_production_prompt",
    "render_production_prompt_with_audit",
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
    """Render through the strict subprocess boundary and return only the
    validated prompt identity."""
    result, _audit = render_production_prompt_with_audit(row, snapshot=snapshot)
    return result


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "USE_TORCH": "0",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return env


def render_production_prompt_with_audit(
    row: dict[str, Any], *, snapshot: VerifiedLocalSnapshot
) -> tuple[PromptRenderingResult, dict[str, Any]]:
    """Render the selected row in a fresh Python subprocess.

    The child process installs a Torch import guard before importing
    Transformers, refuses to open model-weight/index files, and reports
    proof fields that are validated here before the prompt identity is
    accepted by the parent.
    """
    payload = {
        "row": row,
        "snapshot": {
            "repository_id": snapshot.repository_id,
            "requested_revision": snapshot.requested_revision,
            "resolved_revision": snapshot.resolved_revision,
            "asset_type": snapshot.asset_type,
            "local_path": snapshot.local_path,
            "local_files_only": snapshot.local_files_only,
        },
    }
    completed = subprocess.run(
        [sys.executable, "-m", "kvcot.discovery.b2a_r3_tokenizer_worker"],
        input=json.dumps(payload, sort_keys=True),
        text=True,
        capture_output=True,
        env=_worker_env(),
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise ProductionTokenizerResolutionRefused(
            "production tokenizer subprocess refused rendering: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    try:
        worker_payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ProductionTokenizerResolutionRefused(
            f"production tokenizer subprocess returned non-JSON output: {completed.stdout!r}"
        ) from exc
    audit = worker_payload.get("audit")
    if not isinstance(audit, dict):
        raise ProductionTokenizerResolutionRefused("production tokenizer subprocess omitted audit evidence")
    if audit.get("torch_modules_at_start") != [] or audit.get("torch_modules_at_exit") != []:
        raise ProductionTokenizerResolutionRefused("torch modules entered the production tokenizer subprocess")
    if audit.get("cuda_visible_devices") != "":
        raise ProductionTokenizerResolutionRefused("production tokenizer subprocess had CUDA_VISIBLE_DEVICES set")
    if audit.get("model_weight_open_attempts") != []:
        raise ProductionTokenizerResolutionRefused("production tokenizer subprocess opened a model-weight file")
    result_payload = dict(worker_payload["result"])
    result_payload["prompt_token_ids"] = tuple(result_payload["prompt_token_ids"])
    return PromptRenderingResult.model_validate(result_payload), audit


def build_production_tokenizer_renderer(
    *, cache_dir: str | Path | None = None, snapshot: VerifiedLocalSnapshot | None = None
):
    """Returns a `TokenizerRenderer` callable (resolves the exact local
    snapshot once, eagerly, before returning -- so a caller building a
    freeze plan learns immediately if the snapshot is unavailable, never
    only at first prompt-rendering call) bound to that verified snapshot."""
    if snapshot is None:
        snapshot = resolve_production_tokenizer_snapshot(cache_dir=cache_dir)

    def _renderer(row: dict[str, Any]) -> PromptRenderingResult:
        return render_production_prompt(row, snapshot=snapshot)

    return _renderer
