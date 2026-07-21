"""Immutable, local-only Hugging Face snapshot boundary for discovery."""
from __future__ import annotations

import hashlib
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
    # F8 (final independent-audit repair): immutable integrity evidence so a
    # coordinator can revalidate the FULL resolver contract from the
    # worker-exported dict alone -- including the model index/shard
    # validation the resolver itself performed at load time.
    file_count: int = 0
    file_inventory: tuple[tuple[str, int], ...] = ()
    inventory_sha256: str = ""
    weight_index_files: tuple[str, ...] = ()
    weight_index_sha256: tuple[tuple[str, str], ...] = ()
    referenced_shards: tuple[str, ...] = ()
    missing_referenced_shards: tuple[str, ...] = ()
    recognized_weight_files: tuple[str, ...] = ()


def _inventory(path: Path) -> tuple[str, ...]:
    return tuple(sorted(file.relative_to(path).as_posix() for file in path.rglob("*") if file.is_file()))


def compute_inventory_sha256(file_inventory) -> str:
    """Canonical hash over the sorted `(relative_path, size_bytes)` pairs."""
    canonical = json.dumps([[str(name), int(size)] for name, size in file_inventory], sort_keys=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _recognized_weight_files(files) -> tuple[str, ...]:
    return tuple(
        sorted(name for name in files if any(Path(str(name)).match(pattern) for pattern in WEIGHT_PATTERNS))
    )


def _parse_weight_indexes(path: Path, files) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    """Returns `(index_files, index_content_hashes, referenced_shards)` --
    raises `SnapshotBoundaryError` on a malformed index."""
    indexes = tuple(sorted(name for name in files if str(name).endswith(".safetensors.index.json")))
    hashes = []
    shards: set[str] = set()
    for name in indexes:
        raw = (path / name).read_bytes()
        hashes.append((name, hashlib.sha256(raw).hexdigest()))
        try:
            payload = json.loads(raw.decode("utf-8"))
            shards.update(set(payload["weight_map"].values()))
        except Exception as exc:
            raise SnapshotBoundaryError(f"invalid safetensors index {name}: {exc}") from exc
    return indexes, tuple(hashes), tuple(sorted(shards))


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
    _, _, referenced = _parse_weight_indexes(path, files)
    missing = sorted(shard for shard in referenced if shard not in files)
    if missing:
        raise SnapshotBoundaryError(f"safetensors index references missing shards: {missing}")
    if not _recognized_weight_files(files):
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
    file_inventory = tuple((name, int((local / name).stat().st_size)) for name in files)
    total_bytes = sum(size for _, size in file_inventory)
    free = shutil.disk_usage(local).free
    if free < required_free_bytes:
        raise SnapshotBoundaryError(
            f"insufficient disk space: required {required_free_bytes} free bytes, observed {free}"
        )
    if asset_type == "model":
        index_files, index_hashes, referenced = _parse_weight_indexes(local, files)
    else:
        index_files, index_hashes, referenced = (), (), ()
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
        file_count=len(files),
        file_inventory=file_inventory,
        inventory_sha256=compute_inventory_sha256(file_inventory),
        weight_index_files=index_files,
        weight_index_sha256=index_hashes,
        referenced_shards=referenced,
        missing_referenced_shards=tuple(shard for shard in referenced if shard not in files),
        recognized_weight_files=_recognized_weight_files(files),
    )


def verify_snapshot_evidence_raw(
    evidence: dict | None,
    *,
    expected_repository_id: str,
    expected_revision: str,
    asset_type: Literal["tokenizer", "model"],
) -> bool:
    """Independent-audit Gate H4.4/H4.6 repair: the final coordinator gate
    used to trust `worker.snapshot_evidence.get("verified") is True` plus a
    single `resolved_revision` comparison -- accepting whatever a worker's
    JSON output claimed without independently re-checking the CONTENT of
    that claim (repository identity, asset type, exact-SHA revision
    request/resolution agreement, `local_files_only`, a non-empty file
    inventory, and the required config/tokenizer/weight files actually
    being present in that inventory).

    This re-validates a worker-reported `VerifiedLocalSnapshot.__dict__`
    (as received over JSON, so a plain `dict`, never re-touching the
    filesystem or the network) against every field `resolve_local_snapshot`
    itself would have enforced at load time -- a schema drift or a
    malformed/tampered worker report is caught here independently, never
    assumed correct merely because the worker's own load succeeded."""
    if not isinstance(evidence, dict):
        return False
    if evidence.get("repository_id") != expected_repository_id:
        return False
    if evidence.get("asset_type") != asset_type:
        return False
    requested = evidence.get("requested_revision")
    resolved = evidence.get("resolved_revision")
    if not isinstance(requested, str) or not FULL_SHA_RE.fullmatch(requested):
        return False
    if requested != expected_revision:
        return False
    if not isinstance(resolved, str) or not FULL_SHA_RE.fullmatch(resolved):
        return False
    if resolved != requested:
        return False
    if evidence.get("local_files_only") is not True:
        return False
    files = evidence.get("files")
    if not isinstance(files, (list, tuple)) or len(files) == 0:
        return False
    files = tuple(files)
    if any(".incomplete" in str(name) or str(name).endswith(".lock") for name in files):
        return False
    total_bytes = evidence.get("total_bytes")
    if not isinstance(total_bytes, (int, float)) or isinstance(total_bytes, bool) or total_bytes <= 0:
        return False
    if asset_type == "tokenizer":
        if "tokenizer_config.json" not in files:
            return False
        if not {"tokenizer.json", "tokenizer.model", "vocab.json"}.intersection(files):
            return False
    else:
        if "config.json" not in files:
            return False
        has_weight = any(
            any(Path(str(name)).match(pattern) for pattern in WEIGHT_PATTERNS) for name in files
        )
        if not has_weight:
            return False

    # F8: full resolver-equivalent integrity revalidation from the exported
    # evidence alone -- inventory hash/size internal consistency, index
    # accounting, and referenced-shard completeness, never trusting the
    # worker's own "verified" outcome.
    local_path = evidence.get("local_path")
    if not isinstance(local_path, str) or not local_path:
        return False
    file_inventory = evidence.get("file_inventory")
    if not isinstance(file_inventory, (list, tuple)) or len(file_inventory) == 0:
        return False
    try:
        inventory_pairs = [(str(name), int(size)) for name, size in file_inventory]
    except (TypeError, ValueError):
        return False
    if evidence.get("file_count") != len(files) or len(inventory_pairs) != len(files):
        return False
    if sorted(name for name, _ in inventory_pairs) != sorted(str(name) for name in files):
        return False
    if any(size < 0 for _, size in inventory_pairs):
        return False
    if sum(size for _, size in inventory_pairs) != total_bytes:
        return False
    if evidence.get("inventory_sha256") != compute_inventory_sha256(inventory_pairs):
        return False
    if tuple(evidence.get("missing_referenced_shards") or ()) != ():
        return False
    if asset_type == "model":
        index_files = tuple(str(name) for name in (evidence.get("weight_index_files") or ()))
        expected_indexes = tuple(sorted(str(name) for name in files if str(name).endswith(".safetensors.index.json")))
        if index_files != expected_indexes:
            return False
        index_hashes = evidence.get("weight_index_sha256") or ()
        if tuple(sorted(str(name) for name, _ in index_hashes)) != expected_indexes:
            return False
        referenced = [str(name) for name in (evidence.get("referenced_shards") or ())]
        if any(shard not in {str(name) for name in files} for shard in referenced):
            return False
        recognized = [str(name) for name in (evidence.get("recognized_weight_files") or ())]
        expected_recognized = list(_recognized_weight_files([str(name) for name in files]))
        if not recognized or recognized != expected_recognized:
            return False
    return True


def revalidate_snapshot_evidence_against_directory(evidence: dict) -> bool:
    """F8: when the evidence's `local_path` exists on the machine doing the
    verification, recompute the inventory, per-file sizes, inventory hash,
    and (for models) index parses and referenced shards directly from disk
    and require exact agreement with the exported evidence. Never touches
    the network. Returns `True` vacuously when the directory is absent
    (a coordinator on a different filesystem cannot re-read it)."""
    local_path = evidence.get("local_path")
    if not isinstance(local_path, str) or not local_path:
        return False
    root = Path(local_path)
    if not root.is_dir():
        return True
    files = _inventory(root)
    if sorted(str(name) for name in (evidence.get("files") or ())) != list(files):
        return False
    file_inventory = tuple((name, int((root / name).stat().st_size)) for name in files)
    try:
        exported = tuple((str(name), int(size)) for name, size in (evidence.get("file_inventory") or ()))
    except (TypeError, ValueError):
        return False
    if tuple(sorted(exported)) != tuple(sorted(file_inventory)):
        return False
    if evidence.get("inventory_sha256") != compute_inventory_sha256(file_inventory):
        return False
    if evidence.get("total_bytes") != sum(size for _, size in file_inventory):
        return False
    if evidence.get("asset_type") == "model":
        try:
            index_files, index_hashes, referenced = _parse_weight_indexes(root, files)
        except SnapshotBoundaryError:
            return False
        if tuple(str(name) for name in (evidence.get("weight_index_files") or ())) != index_files:
            return False
        if tuple((str(n), str(h)) for n, h in (evidence.get("weight_index_sha256") or ())) != index_hashes:
            return False
        if tuple(str(name) for name in (evidence.get("referenced_shards") or ())) != referenced:
            return False
        if any(shard not in files for shard in referenced):
            return False
    return True


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
