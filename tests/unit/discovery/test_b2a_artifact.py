import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kvcot.discovery.b2a_artifact import (
    ArtifactAlreadyExistsError,
    build_and_write_b2a_artifact,
    build_artifact_path,
    write_b2a_artifact,
)


def test_artifact_path_includes_timestamp_and_hash_prefixes():
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    path = build_artifact_path("a" * 64, "b" * 64, directory=Path("x"), now=now)
    assert path.name == "b2a_20260720T120000Z_aaaaaaaaaaaa_bbbbbbbbbbbb.json"


def test_write_and_read_back_pass_artifact(tmp_path):
    payload = {"passed": True, "gate_result": {"a": 1}}
    path = build_and_write_b2a_artifact(payload, "a" * 64, "b" * 64, directory=tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["passed"] is True


def test_write_and_read_back_fail_artifact(tmp_path):
    payload = {"passed": False, "failed_conditions": ["token_identical_replay"]}
    path = build_and_write_b2a_artifact(payload, "c" * 64, "d" * 64, directory=tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["passed"] is False
    assert "token_identical_replay" in loaded["failed_conditions"]


def test_two_writes_produce_unique_paths(tmp_path):
    p1 = build_and_write_b2a_artifact({"n": 1}, "a" * 64, "b" * 64, directory=tmp_path,
                                       now=datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc))
    p2 = build_and_write_b2a_artifact({"n": 2}, "a" * 64, "b" * 64, directory=tmp_path,
                                       now=datetime(2026, 7, 20, 12, 0, 1, tzinfo=timezone.utc))
    assert p1 != p2
    assert p1.exists() and p2.exists()


def test_overwrite_refused(tmp_path):
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    path = build_artifact_path("a" * 64, "b" * 64, directory=tmp_path, now=now)
    write_b2a_artifact({"n": 1}, path)
    with pytest.raises(ArtifactAlreadyExistsError):
        write_b2a_artifact({"n": 2}, path)


def test_atomic_write_leaves_no_tmp_file_on_success(tmp_path):
    build_and_write_b2a_artifact({"n": 1}, "a" * 64, "b" * 64, directory=tmp_path)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []
