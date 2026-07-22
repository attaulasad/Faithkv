"""CPU-only real resolution of the B2A one-example manifest's prompt
identity (B1B-R3 §5). `kvcot prepare-b2a-manifest` is the only caller.

Downloads exactly two kinds of thing, both explicitly permitted by
CLAUDE.md's B1B-R3 constraints:

1. One pinned MATH-500 row, fetched directly from the dataset repo's raw
   file at the frozen immutable revision
   (`https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/<revision>/test.jsonl`)
   via plain `urllib.request` -- deliberately NOT the `datasets` library
   (broken in this repository's own dev environment: `pyarrow._compute`'s
   native extension is blocked by a local Windows Application Control
   policy, unrelated to this task) and NOT the mutable `datasets-server`
   convenience API (`/rows`, `/first-rows` -- these serve from an
   auto-converted parquet branch that is not necessarily pinned to the
   exact git revision requested, confirmed by direct comparison during this
   pass: the convenience API's row content hashed differently from the
   revision-pinned raw file even though both reported the identical
   `unique_id`). Fetching `resolve/<revision>/...` is the only method that
   is actually revision-pinned.
2. The pinned tokenizer's `tokenizer.json`/`tokenizer_config.json` only, via
   `transformers.AutoTokenizer.from_pretrained(..., revision=...)` --
   NEVER a model weight shard. `_assert_no_weight_files_requested` below is
   a belt-and-suspenders guard (verified after the fact against
   `huggingface_hub`'s local cache) that no `*.safetensors`/
   `pytorch_model*`/`*.bin` file was ever written for this repo.

## Audit finding: the previously-committed `raw_content_hash` was not
## reproducible

`configs/discovery/b2a_one_example_manifest.json`'s `raw_content_hash`
(from B1B-R2) was compared, during this pass, against a fresh
`sha256_json` of the SAME row (`example_index=0`, `unique_id=
"test/precalculus/807.json"`) fetched via three independent methods (the
`datasets-server` `/rows` API, the `/first-rows` API, and the revision-
pinned raw `test.jsonl` line) -- none reproduced the committed hash, while
the row content, unique_id, and dataset revision AGREED across all three
fetches. The committed hash could not be reproduced by any of a dozen
reasonable canonicalization variants tried (raw JSON encoding, field
subsets, `row_idx` inclusion, list-vs-dict encoding). This module treats
that as a genuine, unresolved discrepancy in prior work, not something to
paper over: `prepare_manifest` always recomputes `raw_content_hash` fresh
from the pinned-revision fetch and requires `--force` (with the old/new
hashes printed) to overwrite an already-populated value that disagrees.
"""
from __future__ import annotations

import json
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kvcot.discovery.discovery_config import MATH500_DATASET_REPO, MATH500_DATASET_REVISION, DiscoveryConfig
from kvcot.discovery.manifest import (
    DEFAULT_MANIFEST_PATH,
    B2AOneExampleManifest,
    ChatTemplateRenderingConfig,
)
from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

EXPECTED_MATH500_COLUMNS = ("problem", "solution", "answer", "subject", "level", "unique_id")
_HF_DATASET_RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/test.jsonl"
_FORBIDDEN_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth")
_FORBIDDEN_WEIGHT_PREFIXES = ("pytorch_model", "model-", "model.safetensors")


class ManifestPreparationError(RuntimeError):
    pass


class ManifestPreparationRefused(RuntimeError):
    """Raised for a refusal that is not itself an error in the fetched
    data -- e.g. an already-resolved manifest without `--force`."""


@dataclass(frozen=True)
class FetchedDatasetRow:
    row: dict[str, Any]
    raw_content_hash: str


def _fetch_pinned_dataset_row(dataset_repo: str, dataset_revision: str, example_index: int) -> FetchedDatasetRow:
    """Fetch ONE line of the revision-pinned `test.jsonl`, by streaming and
    counting newlines -- never the whole file materialized as a Python list
    beyond what's needed, and never a mutable convenience API (see module
    docstring)."""
    url = _HF_DATASET_RESOLVE_URL.format(repo=dataset_repo, revision=dataset_revision)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 -- fixed https:// HF URL, not user input
            for line_index, raw_line in enumerate(resp):
                if line_index == example_index:
                    text = raw_line.decode("utf-8").rstrip("\r\n")
                    row = json.loads(text)
                    return FetchedDatasetRow(row=row, raw_content_hash=sha256_json(row))
    except urllib.error.URLError as exc:
        raise ManifestPreparationError(f"failed to fetch {url}: {exc}") from exc
    raise ManifestPreparationError(f"{url} has fewer than {example_index + 1} rows")


def _verify_row_schema(row: dict[str, Any]) -> None:
    got = tuple(row.keys())
    if got != EXPECTED_MATH500_COLUMNS:
        raise ManifestPreparationError(
            f"MATH-500 row has unexpected columns: expected {EXPECTED_MATH500_COLUMNS}, got {got}"
        )


def _snapshot_weight_shaped_files(tokenizer_name: str, tokenizer_revision: str) -> frozenset[str]:
    """B1B-R4 §15 repair: a pure OBSERVATION of which weight-shaped files
    are currently present in the huggingface_hub local cache for this exact
    repo/revision -- never itself a pass/fail judgment. Used as a before/
    after pair by `_assert_no_new_weight_files_introduced` below, scoped
    ONLY to `prepare-b2a-manifest`'s own tokenizer-loading call
    (`resolve_prompt_identity`) -- generic prompt rendering/tokenization
    (`kvcot.discovery.b2a_execute._verify_resolved_prompt_identity`'s reuse
    of `_render_and_tokenize`) must NEVER inspect or reject unrelated
    pre-existing model weights, so this function is deliberately not called
    from `_render_and_tokenize` itself."""
    from huggingface_hub import scan_cache_dir

    try:
        cache_info = scan_cache_dir()
    except Exception:
        return frozenset()  # no local cache at all yet -- nothing to snapshot
    found: set[str] = set()
    for repo in cache_info.repos:
        if repo.repo_id != tokenizer_name:
            continue
        for revision in repo.revisions:
            if revision.commit_hash != tokenizer_revision:
                continue
            for file in revision.files:
                name = file.file_path.name
                if name.endswith(_FORBIDDEN_WEIGHT_SUFFIXES) or name.startswith(_FORBIDDEN_WEIGHT_PREFIXES):
                    found.add(str(file.file_path))
    return frozenset(found)


def _assert_no_new_weight_files_introduced(before: frozenset[str], after: frozenset[str]) -> None:
    """B1B-R4 §15: fail ONLY if a weight-shaped file is NEW (present after
    but not before) -- a pre-existing weight file (e.g. from a prior,
    separately-authorized GPU run on the same host) must never fail this
    command; only weight files THIS command's tokenizer-loading call
    actually introduced are a violation."""
    new_files = after - before
    if new_files:
        raise ManifestPreparationError(
            f"prepare-b2a-manifest introduced new weight-shaped file(s) during this command: "
            f"{sorted(new_files)!r} -- this command must never download model weights, only tokenizer files. "
            "Pre-existing weight files already in the cache before this command ran are NOT themselves a failure."
        )


@dataclass(frozen=True)
class PreparedManifestPlan:
    """What `--dry-run` prints and what `--execute` actually does -- the
    same plan object drives both, so dry-run can never silently diverge
    from what execution would do."""

    dataset_repo: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    example_index: int
    tokenizer_name: str
    tokenizer_revision: str
    existing_manifest_path: Path
    existing_manifest_is_prompt_resolved: bool
    force: bool


def build_plan(config: DiscoveryConfig, manifest_path: Path = DEFAULT_MANIFEST_PATH, force: bool = False) -> PreparedManifestPlan:
    existing_resolved = False
    if manifest_path.exists():
        try:
            existing = B2AOneExampleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            existing_resolved = existing.prompt_identity_is_resolved
        except Exception:
            existing_resolved = False
    if config.dataset.revision is None:
        raise ManifestPreparationError("config.dataset.revision is not frozen -- cannot resolve a manifest against it.")
    return PreparedManifestPlan(
        dataset_repo=MATH500_DATASET_REPO,
        dataset_config=config.dataset.config,
        dataset_split=config.dataset.split,
        dataset_revision=config.dataset.revision,
        example_index=0,
        tokenizer_name=config.model.tokenizer_name,
        tokenizer_revision=config.model.tokenizer_revision,
        existing_manifest_path=manifest_path,
        existing_manifest_is_prompt_resolved=existing_resolved,
        force=force,
    )


def _render_and_tokenize(
    row: dict[str, Any], tokenizer_name: str, tokenizer_revision: str, *, local_only_path: str | None = None
):
    """The exact frozen chat-template call
    (`kvcot.cli.cmd_generate`'s own call shape, byte-for-byte identical --
    see module docstring of `kvcot.discovery.manifest`): one user message,
    `add_generation_prompt=True`, `tokenize=True`, no other arguments.

    B1B-R4 §15: this function is shared by BOTH `prepare-b2a-manifest`
    (`resolve_prompt_identity` below) AND generic B2A prompt-identity
    re-verification (`kvcot.discovery.b2a_execute
    ._verify_resolved_prompt_identity`, which reuses it directly to avoid a
    second, independently-written verification path) -- it must therefore
    NEVER inspect or reject pre-existing model weights (a valid GPU host
    that has already downloaded weights for a real, separately-authorized
    run must not fail generic prompt verification). The weight-cache safety
    guard lives ONLY around `resolve_prompt_identity`'s own call site,
    scoped to manifest preparation specifically.

    Independent-audit Gate H4.5 repair: `local_only_path`, when supplied
    (`b2a_execute._verify_resolved_prompt_identity`'s call, after it has
    already independently resolved and verified the exact local tokenizer
    snapshot via `kvcot.discovery.snapshot_boundary.resolve_local_snapshot`),
    loads the tokenizer from that EXACT verified local directory with
    `local_files_only=True` -- the same strict local-asset boundary the
    workers themselves use -- instead of `tokenizer_name`/`tokenizer_revision`
    resolved through `huggingface_hub`'s normal (potentially network-
    touching) cache lookup. `prepare-b2a-manifest`'s own call
    (`resolve_prompt_identity` below) never passes this -- it is the one
    place a live, network-capable tokenizer resolution remains explicitly
    authorized (CLAUDE.md's tokenizer-only allowance), unchanged by this
    repair. Defaults to `None`, preserving prior behavior exactly for that
    caller."""
    from transformers import AutoTokenizer

    from kvcot.probes.templates import render_base_user_message

    if local_only_path is not None:
        tokenizer = AutoTokenizer.from_pretrained(local_only_path, local_files_only=True, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, revision=tokenizer_revision, use_fast=True)

    if tokenizer.chat_template is None:
        raise ManifestPreparationError(
            f"tokenizer {tokenizer_name}@{tokenizer_revision} has no chat_template -- refusing to invent one."
        )

    user_message = render_base_user_message(row["problem"])
    messages = [{"role": "user", "content": user_message}]
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    token_ids = list(token_ids)
    if len(token_ids) == 0:
        raise ManifestPreparationError("apply_chat_template returned zero tokens -- refusing an empty prompt.")

    return tokenizer, user_message, messages, token_ids


@dataclass(frozen=True)
class ResolvedPromptIdentity:
    raw_content_hash: str
    row: dict[str, Any]
    rendered_user_message_sha256: str
    chat_template_source_sha256: str
    chat_message_payload_sha256: str
    prompt_token_ids: tuple[int, ...]
    prompt_token_ids_sha256: str
    prompt_token_count: int
    tokenizer_revision_used_for_prompt_hash: str
    prompt_rendering_config: ChatTemplateRenderingConfig


def resolve_prompt_identity(plan: PreparedManifestPlan) -> ResolvedPromptIdentity:
    fetched = _fetch_pinned_dataset_row(plan.dataset_repo, plan.dataset_revision, plan.example_index)
    _verify_row_schema(fetched.row)

    # B1B-R4 §15: the weight-cache safety guard is scoped to THIS command's
    # own tokenizer-loading call only -- before/after snapshot, fail only on
    # a NEW weight-shaped file this command's own call introduced. A
    # pre-existing weight file (from a prior, separately-authorized GPU run
    # on the same host) must never fail this command.
    before = _snapshot_weight_shaped_files(plan.tokenizer_name, plan.tokenizer_revision)
    tokenizer, user_message, messages, token_ids = _render_and_tokenize(
        fetched.row, plan.tokenizer_name, plan.tokenizer_revision
    )
    after = _snapshot_weight_shaped_files(plan.tokenizer_name, plan.tokenizer_revision)
    _assert_no_new_weight_files_introduced(before, after)

    return ResolvedPromptIdentity(
        raw_content_hash=fetched.raw_content_hash,
        row=fetched.row,
        rendered_user_message_sha256=sha256_text(user_message),
        chat_template_source_sha256=sha256_text(tokenizer.chat_template),
        chat_message_payload_sha256=sha256_json(messages),
        prompt_token_ids=tuple(token_ids),
        prompt_token_ids_sha256=sha256_int_ids(token_ids),
        prompt_token_count=len(token_ids),
        tokenizer_revision_used_for_prompt_hash=plan.tokenizer_revision,
        prompt_rendering_config=ChatTemplateRenderingConfig(
            message_roles=("user",),
            add_generation_prompt=True,
            tokenize=True,
            add_special_tokens_note=(
                "special-token behavior is entirely delegated to the tokenizer's own chat_template "
                "(Jinja) -- apply_chat_template was called with no separate add_special_tokens override"
            ),
        ),
    )


def _atomic_write_manifest(manifest: B2AOneExampleManifest, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".manifest-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
            f.write("\n")
        os.replace(tmp_path, path)  # atomic on both POSIX and Windows (same filesystem)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def prepare_manifest(
    config: DiscoveryConfig, manifest_path: Path = DEFAULT_MANIFEST_PATH, force: bool = False
) -> B2AOneExampleManifest:
    """The one function `kvcot prepare-b2a-manifest --execute` calls.
    Validates, fetches, renders, tokenizes, hashes, and atomically writes --
    refuses to overwrite an already-resolved manifest unless `force=True`,
    and even with `force=True` requires every frozen upstream identity
    field (repo/config/split/revision/example_index/unique_id) to stay
    unchanged (only the derived hash/prompt fields may be corrected)."""
    plan = build_plan(config, manifest_path, force=force)

    old_manifest: B2AOneExampleManifest | None = None
    if manifest_path.exists():
        if not force:
            raise ManifestPreparationRefused(
                f"{manifest_path} already exists -- refusing to overwrite an existing manifest without --force."
            )
        try:
            old_manifest = B2AOneExampleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            # The existing file does not even parse under the current schema
            # (e.g. an older schema version) -- there is no old upstream
            # identity to preserve-and-compare against, so proceed to
            # regenerate it fresh; this is still gated behind --force above.
            old_manifest = None

    resolved = resolve_prompt_identity(plan)

    if old_manifest is not None:
        for field, old_value, new_value in (
            ("dataset_repo", old_manifest.dataset_repo, plan.dataset_repo),
            ("dataset_config", old_manifest.dataset_config, plan.dataset_config),
            ("dataset_split", old_manifest.dataset_split, plan.dataset_split),
            ("dataset_revision", old_manifest.dataset_revision, plan.dataset_revision),
            ("example_index", old_manifest.example_index, plan.example_index),
            ("unique_id", old_manifest.unique_id, resolved.row["unique_id"]),
        ):
            if old_value != new_value:
                raise ManifestPreparationError(
                    f"refusing to change frozen upstream identity field {field!r}: {old_value!r} -> {new_value!r}. "
                    "prepare-b2a-manifest may correct derived hashes, never the pinned dataset identity itself."
                )

    new_manifest = B2AOneExampleManifest(
        dataset_repo=plan.dataset_repo,
        dataset_config=plan.dataset_config,
        dataset_split=plan.dataset_split,
        dataset_revision=plan.dataset_revision,
        example_index=plan.example_index,
        unique_id=resolved.row["unique_id"],
        raw_content_hash=resolved.raw_content_hash,
        gold_answer=resolved.row["answer"],
        prompt_token_ids_sha256=resolved.prompt_token_ids_sha256,
        tokenizer_revision_used_for_prompt_hash=resolved.tokenizer_revision_used_for_prompt_hash,
        rendered_user_message_sha256=resolved.rendered_user_message_sha256,
        chat_template_source_sha256=resolved.chat_template_source_sha256,
        chat_message_payload_sha256=resolved.chat_message_payload_sha256,
        prompt_rendering_config=resolved.prompt_rendering_config,
        prompt_token_count=resolved.prompt_token_count,
        prompt_token_ids=resolved.prompt_token_ids,
    )

    if old_manifest is not None and force:
        old_hash = old_manifest.manifest_hash()
        new_hash = new_manifest.manifest_hash()
        print(f"prepare-b2a-manifest --force: old manifest_hash={old_hash}")
        print(f"prepare-b2a-manifest --force: new manifest_hash={new_hash}")
        if old_manifest.raw_content_hash != new_manifest.raw_content_hash:
            print(
                f"prepare-b2a-manifest --force: raw_content_hash CORRECTED "
                f"{old_manifest.raw_content_hash} -> {new_manifest.raw_content_hash}"
            )

    _atomic_write_manifest(new_manifest, manifest_path)
    return new_manifest
