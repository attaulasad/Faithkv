"""Immutable B2A result artifacts, pass OR fail (B1B-R3 §16). Every B2A
attempt writes one artifact, even when the gate fails or the run raises
before completing -- the prior behavior of writing only after a pass is
prohibited.

Path shape: `results/decisions/b2a_<timestamp>_<config-hash-prefix>_
<manifest-hash-prefix>.json` -- never a fixed filename (so a second attempt
can never silently clobber a prior one), write-temp-then-atomic-rename,
refuse a pre-existing path outright (astronomically unlikely with a
timestamp+hash-prefix name, but checked anyway rather than assumed)."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DECISIONS_DIR = Path("results/decisions")


class ArtifactAlreadyExistsError(RuntimeError):
    pass


@dataclass(frozen=True)
class B2AArtifactPaths:
    directory: Path = RESULTS_DECISIONS_DIR


def build_artifact_path(
    config_hash: str, manifest_hash: str, *, directory: Path = RESULTS_DECISIONS_DIR, now: datetime | None = None
) -> Path:
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    return directory / f"b2a_{timestamp}_{config_hash[:12]}_{manifest_hash[:12]}.json"


def write_b2a_artifact(payload: dict[str, Any], path: Path) -> Path:
    """Atomic write-temp-then-rename; refuses to overwrite an existing
    path (never a fixed filename in practice, so this should never
    legitimately collide -- if it does, that is itself worth surfacing
    loudly rather than silently overwriting evidence)."""
    if path.exists():
        raise ArtifactAlreadyExistsError(f"refusing to overwrite existing artifact at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=".b2a-artifact-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True, default=str)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def build_and_write_b2a_artifact(
    payload: dict[str, Any],
    config_hash: str,
    manifest_hash: str,
    *,
    directory: Path = RESULTS_DECISIONS_DIR,
    now: datetime | None = None,
) -> Path:
    path = build_artifact_path(config_hash, manifest_hash, directory=directory, now=now)
    return write_b2a_artifact(payload, path)
