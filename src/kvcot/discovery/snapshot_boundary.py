"""Immutable, local-only Hugging Face snapshot boundary for discovery."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.h5")


class SnapshotBoundaryError(RuntimeError):
    pass


@dataclass(frozen=True)
class VerifiedLocalSnapshot:
    repository_id: str
    requested_revision: str
    resolved_revision: str
    asset_type: Literal["tokenizer", "model"]
    local_path: str
    files: tuple[str, ...]
    total_bytes: int
    required_free_bytes: int
    free_bytes: int
    local_files_only: bool = True


def _inventory(path: Path) -> tuple[str, ...]:
    return tuple(sorted(file.relative_to(path).as_posix() for file in path.rglob("*") if file.is_file()))


def _assert_no_incomplete(path: Path, files: tuple[str, ...]) -> None:
    bad = [name for name in files if ".incomplete" in name or name.endswith(".lock")]
    if bad:
        raise SnapshotBoundaryError(f"snapshot contains incomplete/lock files: {bad}")


def _validate_tokenizer_files(path: Path, files: tuple[str, ...]) -> None:
    if "tokenizer_config.json" not in files:
        raise SnapshotBoundaryError("tokenizer snapshot is missing tokenizer_config.json")
    vocabulary = {"tokenizer.json", "tokenizer.model", "vocab.json"}
    if not vocabulary.intersection(files):
        raise SnapshotBoundaryError("tokenizer snapshot has no tokenizer vocabulary/model file")


def _validate_model_files(path: Path, files: tuple[str, ...]) -> None:
    if "config.json" not in files:
        raise SnapshotBoundaryError("model snapshot is missing config.json")
    indexes = [name for name in files if name.endswith(".safetensors.index.json")]
    for name in indexes:
        try:
            payload = json.loads((path / name).read_text(encoding="utf-8"))
            shards = set(payload["weight_map"].values())
        except Exception as exc:
            raise SnapshotBoundaryError(f"invalid safetensors index {name}: {exc}") from exc
        missing = sorted(shard for shard in shards if shard not in files)
        if missing:
            raise SnapshotBoundaryError(f"safetensors index references missing shards: {missing}")
    has_weight = any(any(Path(name).match(pattern) for pattern in WEIGHT_PATTERNS) for name in files)
    if not has_weight:
        raise SnapshotBoundaryError("model snapshot contains no recognized local weight file")


def resolve_local_snapshot(
    repository_id: str,
    revision: str,
    asset_type: Literal["tokenizer", "model"],
    *,
    cache_dir: str | Path | None = None,
    required_free_bytes: int = 0,
) -> VerifiedLocalSnapshot:
    """Resolve an exact cached snapshot with public huggingface_hub APIs."""
    if not FULL_SHA_RE.fullmatch(revision):
        raise SnapshotBoundaryError("revision must be a full lowercase immutable 40-character commit SHA")
    from huggingface_hub import scan_cache_dir, snapshot_download

    try:
        local = Path(
            snapshot_download(
                repo_id=repository_id,
                revision=revision,
                cache_dir=None if cache_dir is None else str(cache_dir),
                local_files_only=True,
            )
        ).resolve()
    except Exception as exc:
        raise SnapshotBoundaryError(
            f"exact local snapshot {repository_id}@{revision} is unavailable; network/floating fallback is forbidden"
        ) from exc
    if not local.is_dir():
        raise SnapshotBoundaryError(f"resolved snapshot path does not exist: {local}")

    # Public cache metadata is the identity authority, not private
    # transformers `_commit_hash` fields and not path-name inference alone.
    try:
        cache = scan_cache_dir(cache_dir=cache_dir)
    except TypeError:
        cache = scan_cache_dir(cache_dir)
    matches = [
        rev
        for repo in cache.repos
        if repo.repo_id == repository_id
        for rev in repo.revisions
        if rev.commit_hash == revision and Path(rev.snapshot_path).resolve() == local
    ]
    if len(matches) != 1:
        raise SnapshotBoundaryError("public cache metadata does not identify exactly the requested immutable revision")
    files = _inventory(local)
    _assert_no_incomplete(local, files)
    if asset_type == "tokenizer":
        _validate_tokenizer_files(local, files)
    elif asset_type == "model":
        _validate_model_files(local, files)
    else:
        raise SnapshotBoundaryError(f"unsupported asset_type: {asset_type!r}")
    total_bytes = sum((local / name).stat().st_size for name in files)
    free = shutil.disk_usage(local).free
    if free < required_free_bytes:
        raise SnapshotBoundaryError(
            f"insufficient disk space: required {required_free_bytes} free bytes, observed {free}"
        )
    return VerifiedLocalSnapshot(
        repository_id=repository_id,
        requested_revision=revision,
        resolved_revision=matches[0].commit_hash,
        asset_type=asset_type,
        local_path=str(local),
        files=files,
        total_bytes=total_bytes,
        required_free_bytes=required_free_bytes,
        free_bytes=free,
    )


def contains_weight_files(path: str | Path) -> tuple[str, ...]:
    root = Path(path)
    return tuple(
        sorted(
            file.relative_to(root).as_posix()
            for file in root.rglob("*")
            if file.is_file()
            and (
                any(file.match(pattern) for pattern in WEIGHT_PATTERNS)
                or file.name.endswith(".index.json")
                and ("model" in file.name or "weight" in file.name)
            )
        )
    )
