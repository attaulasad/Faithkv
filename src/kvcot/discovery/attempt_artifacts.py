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


SEMANTIC_ROLE_BY_RELATIVE_PATH: dict[str, str] = {
    "invocation.json": "invocation",
    "preflight.json": "preflight",
    "provenance.json": "provenance",
    "completion.json": "completion",
    "process_outcome.json": "process_outcome",
    "failure.json": "failure",
    "final_write_failure.json": "final_write_failure",
}
_SEMANTIC_ROLE_BY_WORKER_FILE: dict[str, str] = {
    "command.json": "worker_command",
    "stdout.log": "worker_stdout",
    "stderr.log": "worker_stderr",
    "progress.jsonl": "worker_progress",
    "result.json": "worker_result",
    "result.json.envelope.json": "worker_envelope",
    "envelope.json": "worker_envelope",
    "timing.json": "worker_timing",
    "memory.json": "worker_memory",
    "termination.json": "worker_termination",
    "pair_identities.json": "pair_identities",
    "semantic_swaps.json": "semantic_swaps",
    "replay_evidence.json": "replay_evidence",
}


def semantic_role_for(relative_path: str) -> str:
    if relative_path in SEMANTIC_ROLE_BY_RELATIVE_PATH:
        return SEMANTIC_ROLE_BY_RELATIVE_PATH[relative_path]
    parts = relative_path.split("/")
    if len(parts) == 2 and parts[0] in ("fullkv", "rkv") and parts[1] in _SEMANTIC_ROLE_BY_WORKER_FILE:
        return _SEMANTIC_ROLE_BY_WORKER_FILE[parts[1]]
    return "unknown"


def build_attempt_references(attempt: AttemptDirectory, *, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """F4.6/F5: the immutable pre-final reference manifest -- one entry per
    file with relative path, semantic role, size, and SHA-256. `exclude`
    names relative paths never listed (only ever `final.json`, which cannot
    reference itself)."""
    references = []
    for path in sorted(file for file in attempt.path.rglob("*") if file.is_file()):
        relative = path.relative_to(attempt.path).as_posix()
        if relative in exclude:
            continue
        references.append({
            "relative_path": relative,
            "semantic_role": semantic_role_for(relative),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        })
    return {"attempt_id": attempt.attempt_id, "attempt_directory": attempt.path.name, "files": references}


# F6: the B1 repair-round start authorities -- 419bbc0 is the round-four
# starting commit; the two earlier execution-boundary commits remain
# required ancestors. Never only 3c853cf.
B1_REPAIR_ROUND4_STARTING_COMMIT = "419bbc0020b374d6c4a2085a7a04ff293d7ec680"
B1_REQUIRED_ANCESTOR_SHAS: tuple[str, ...] = (
    B1_REPAIR_ROUND4_STARTING_COMMIT,
    "7ef13ae566e7c3e699e5143405baf76a81078edf",
    "3c853cff34e52d792cd0e5a96d1a5369f17f8047",
)


def total_physical_ram_bytes() -> int | None:
    """CPU-safe total physical RAM. `None` (honestly unavailable) when no
    supported mechanism exists -- never a fabricated value."""
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        names = getattr(os, "sysconf_names", {})
        if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
            return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except Exception:
            pass
    return None


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
    branch = git("branch", "--show-current")

    def is_ancestor(sha: str) -> bool:
        return subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "HEAD"], cwd=repository,
            check=False, capture_output=True,
        ).returncode == 0

    ancestry = {sha: is_ancestor(sha) for sha in B1_REQUIRED_ANCESTOR_SHAS}

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
            "branch": branch,
            "head": git("rev-parse", "HEAD"),
            "base_main_sha": git("rev-parse", "origin/main", check=False) or None,
            "origin_branch_sha": (git("rev-parse", f"origin/{branch}", check=False) or None) if branch else None,
            "dirty": bool(status_lines),
            "status": status_lines,
            "staged_paths": staged,
            "unstaged_paths": unstaged,
            "untracked_paths": untracked,
            "starting_commit": B1_REPAIR_ROUND4_STARTING_COMMIT,
            "required_ancestry": ancestry,
            "all_required_ancestry_verified": all(ancestry.values()),
            # Retained for backward compatibility with earlier artifacts --
            # 3c853cf is no longer the SOLE start authority.
            "starting_ancestor": "3c853cff34e52d792cd0e5a96d1a5369f17f8047",
            "starting_ancestor_verified": ancestry["3c853cff34e52d792cd0e5a96d1a5369f17f8047"],
            "rkv_submodule_sha": observed_rkv_sha,
            "expected_rkv_sha": expected_rkv_sha,
            "rkv_submodule_match": observed_rkv_sha == expected_rkv_sha,
        },
        "software": software,
        "system": {
            "os": platform.system(),
            "platform": platform.platform(),
            "kernel_release": platform.release(),
            "kernel_version": platform.version(),
            "architecture": platform.machine(),
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "total_physical_ram_bytes": total_physical_ram_bytes(),
            "artifact_free_disk_bytes": artifact_disk.free,
            "model_cache_free_disk_bytes": model_disk.free,
        },
        # Retained for backward compatibility with earlier consumers.
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "artifact_free_disk_bytes": artifact_disk.free,
            "model_cache_free_disk_bytes": model_disk.free,
        },
        # F6: GPU/driver/CUDA/cuDNN evidence deliberately does NOT live here
        # (this collector is CPU-safe) -- cross-references to where that
        # evidence durably lives inside the same attempt directory.
        "gpu_evidence_cross_references": [
            "preflight.json:device",
            "fullkv/result.json:device_evidence",
            "rkv/result.json:device_evidence",
        ],
    }


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
