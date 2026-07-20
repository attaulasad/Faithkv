"""Immutable B2A result artifacts, pass OR fail (B1B-R4 §17, superseding
B1B-R3's version of this module). Every B2A attempt writes one artifact,
even when the gate fails or the run raises before completing -- the prior
behavior of writing only after a pass is prohibited.

## B1B-R4 §17 repair: collision-resistant naming

B1B-R3's path shape (`b2a_<second-resolution-timestamp>_<config-hash-
prefix>_<manifest-hash-prefix>.json`) could collide: two attempts against
the SAME config/manifest pair, started within the same wall-clock second
(e.g. an immediate retry after a fast failure), would derive the identical
path and the second write would be silently refused by
`write_b2a_artifact`'s pre-existing overwrite guard -- losing the second
attempt's evidence rather than merely being "astronomically unlikely".

Path shape: `results/decisions/b2a_<UTC timestamp with microseconds>_
<random UUID4 hex>_<config-hash-prefix>_<manifest-hash-prefix>.json` --
microsecond resolution collapses the same-second collision window to
same-microsecond, and the random suffix makes even a same-microsecond
collision cryptographically negligible. Write-temp-then-atomic-rename,
refuse a pre-existing path outright (never silently overwrites evidence)."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
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
    config_hash: str,
    manifest_hash: str,
    *,
    directory: Path = RESULTS_DECISIONS_DIR,
    now: datetime | None = None,
    random_suffix: str | None = None,
) -> Path:
    """`random_suffix` is dependency-injected (defaults to a fresh
    `uuid.uuid4().hex`) so CPU tests can assert on an exact, deterministic
    path rather than a randomly-generated one."""
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
    random_suffix = random_suffix if random_suffix is not None else uuid.uuid4().hex
    return directory / f"b2a_{timestamp}_{random_suffix}_{config_hash[:12]}_{manifest_hash[:12]}.json"


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
    random_suffix: str | None = None,
) -> Path:
    path = build_artifact_path(config_hash, manifest_hash, directory=directory, now=now, random_suffix=random_suffix)
    return write_b2a_artifact(payload, path)
