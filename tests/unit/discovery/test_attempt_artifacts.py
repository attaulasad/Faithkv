from __future__ import annotations

from datetime import datetime, timezone

import pytest

from kvcot.discovery.attempt_artifacts import (
    AttemptArtifactError,
    append_progress,
    atomic_write_json,
    build_attempt_references,
    create_attempt_directory,
)


def test_attempt_directory_is_unique_immutable_atomic_and_hashed(tmp_path):
    attempt = create_attempt_directory(
        root=tmp_path,
        now=datetime(2026, 7, 20, tzinfo=timezone.utc),
        attempt_id="a" * 32,
    )
    assert attempt.path.name.startswith("b2a_attempt_20260720T000000000000Z_")
    artifact = atomic_write_json(attempt.path / "preflight.json", {"passed": False})
    with pytest.raises(AttemptArtifactError, match="overwrite"):
        atomic_write_json(artifact, {"passed": True})
    append_progress(
        attempt.path / "rkv" / "progress.jsonl",
        attempt_id=attempt.attempt_id,
        worker_role="rkv",
        stage="startup",
        status="started",
    )
    references = build_attempt_references(attempt)
    paths = {item["relative_path"] for item in references["files"]}
    assert paths == {"preflight.json", "rkv/progress.jsonl"}
    assert all(len(item["sha256"]) == 64 for item in references["files"])


def test_attempt_directory_refuses_collision(tmp_path):
    kwargs = dict(root=tmp_path, now=datetime(2026, 7, 20, tzinfo=timezone.utc), attempt_id="b" * 32)
    create_attempt_directory(**kwargs)
    with pytest.raises(AttemptArtifactError, match="overwrite"):
        create_attempt_directory(**kwargs)
