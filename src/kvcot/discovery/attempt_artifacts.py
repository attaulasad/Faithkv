"""Immutable, atomic B2A attempt directories and progress journals."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
import importlib.metadata
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ATTEMPT_ROOT = Path("results/decisions")


class AttemptArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class AttemptDirectory:
    attempt_id: str
    path: Path


def create_attempt_directory(
    *, root: Path = DEFAULT_ATTEMPT_ROOT, now: datetime | None = None, attempt_id: str | None = None
) -> AttemptDirectory:
    now = now or datetime.now(timezone.utc)
    attempt_id = attempt_id or uuid.uuid4().hex
    name = f"b2a_attempt_{now.strftime('%Y%m%dT%H%M%S%fZ')}_{attempt_id}"
    path = root / name
    try:
        path.mkdir(parents=True, exist_ok=False)
        for role in ("fullkv", "rkv"):
            (path / role).mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise AttemptArtifactError(f"refusing to overwrite attempt directory {path}") from exc
    return AttemptDirectory(attempt_id=attempt_id, path=path)


def atomic_write_json(path: Path, payload: Any) -> Path:
    if path.exists():
        raise AttemptArtifactError(f"refusing to overwrite immutable artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def atomic_write_text(path: Path, content: str) -> Path:
    if path.exists():
        raise AttemptArtifactError(f"refusing to overwrite immutable artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return path


def append_progress(
    path: Path, *, attempt_id: str, worker_role: str, stage: str, status: str, counters=None, detail=None
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        "attempt_id": attempt_id,
        "worker_role": worker_role,
        "counters": counters or {},
        "detail": detail,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_attempt_references(attempt: AttemptDirectory) -> dict[str, Any]:
    references = []
    for path in sorted(file for file in attempt.path.rglob("*") if file.is_file()):
        references.append({
            "relative_path": path.relative_to(attempt.path).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return {"attempt_id": attempt.attempt_id, "attempt_directory": attempt.path.name, "files": references}


def collect_execution_provenance(*, repository: Path, expected_rkv_sha: str, artifact_root: Path) -> dict[str, Any]:
    """Collect CPU-safe provenance without recording credentials or tokens."""
    repository = repository.resolve()

    def git(*args: str, check: bool = True) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repository, check=check, capture_output=True, text=True,
        )
        return completed.stdout.strip()

    raw_status_lines = [line for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines() if line]
    try:
        attempt_relative = artifact_root.resolve().relative_to(repository).as_posix()
    except ValueError:
        attempt_relative = ""
    status_lines = [
        line for line in raw_status_lines
        if not attempt_relative or not line[3:].replace("\\", "/").startswith(attempt_relative + "/")
    ]
    staged = [line[3:] for line in status_lines if line[:1] not in (" ", "?")]
    unstaged = [line[3:] for line in status_lines if len(line) > 1 and line[1] not in (" ", "?")]
    untracked = [line[3:] for line in status_lines if line.startswith("??")]
    submodule_line = git("submodule", "status", "third_party/R-KV")
    observed_rkv_sha = submodule_line.lstrip("-+U ").split()[0] if submodule_line else None
    starting_sha = "3c853cff34e52d792cd0e5a96d1a5369f17f8047"
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", starting_sha, "HEAD"], cwd=repository,
        check=False, capture_output=True,
    ).returncode == 0

    package_names = (
        "torch", "transformers", "accelerate", "flash-attn", "datasets",
        "huggingface-hub", "pydantic", "numpy",
    )
    software: dict[str, str | None] = {"python": platform.python_version()}
    for name in package_names:
        try:
            software[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            software[name] = None

    artifact_disk = shutil.disk_usage(artifact_root.resolve().anchor or artifact_root.resolve())
    model_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_probe = model_cache if model_cache.exists() else model_cache.parent
    model_disk = shutil.disk_usage(cache_probe.resolve().anchor or cache_probe.resolve())
    return {
        "git": {
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "base_main_sha": git("rev-parse", "origin/main", check=False) or None,
            "dirty": bool(status_lines),
            "status": status_lines,
            "staged_paths": staged,
            "unstaged_paths": unstaged,
            "untracked_paths": untracked,
            "starting_ancestor": starting_sha,
            "starting_ancestor_verified": ancestor,
            "rkv_submodule_sha": observed_rkv_sha,
            "expected_rkv_sha": expected_rkv_sha,
            "rkv_submodule_match": observed_rkv_sha == expected_rkv_sha,
        },
        "software": software,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "artifact_free_disk_bytes": artifact_disk.free,
            "model_cache_free_disk_bytes": model_disk.free,
        },
    }


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
