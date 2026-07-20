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


def test_artifact_path_includes_microsecond_timestamp_random_suffix_and_hash_prefixes():
    now = datetime(2026, 7, 20, 12, 0, 0, 123456, tzinfo=timezone.utc)
    path = build_artifact_path("a" * 64, "b" * 64, directory=Path("x"), now=now, random_suffix="deadbeef")
    assert path.name == "b2a_20260720T120000123456Z_deadbeef_aaaaaaaaaaaa_bbbbbbbbbbbb.json"


def test_default_random_suffix_is_generated_when_not_supplied():
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    p1 = build_artifact_path("a" * 64, "b" * 64, directory=Path("x"), now=now)
    p2 = build_artifact_path("a" * 64, "b" * 64, directory=Path("x"), now=now)
    assert p1 != p2  # random suffix differs even with an identical timestamp/hash pair


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


def test_two_writes_at_the_same_second_produce_unique_paths(tmp_path):
    """B1B-R4 §17 regression: the OLD second-resolution-only naming
    scheme would collide for two attempts started within the same
    wall-clock second -- microseconds plus a random suffix must keep them
    distinct even when `now` is identical down to the second."""
    same_second = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    p1 = build_and_write_b2a_artifact({"n": 1}, "a" * 64, "b" * 64, directory=tmp_path, now=same_second)
    p2 = build_and_write_b2a_artifact({"n": 2}, "a" * 64, "b" * 64, directory=tmp_path, now=same_second)
    assert p1 != p2
    assert p1.exists() and p2.exists()


def test_two_writes_at_the_same_microsecond_still_produce_unique_paths(tmp_path):
    """Even a same-microsecond collision (two attempts launched back-to-back
    on a fast machine) must not collide -- the random suffix alone must
    disambiguate them."""
    same_instant = datetime(2026, 7, 20, 12, 0, 0, 999999, tzinfo=timezone.utc)
    p1 = build_and_write_b2a_artifact({"n": 1}, "a" * 64, "b" * 64, directory=tmp_path, now=same_instant)
    p2 = build_and_write_b2a_artifact({"n": 2}, "a" * 64, "b" * 64, directory=tmp_path, now=same_instant)
    assert p1 != p2
    assert p1.exists() and p2.exists()


def test_overwrite_refused_when_random_suffix_is_forced_identical(tmp_path):
    now = datetime(2026, 7, 20, 12, 0, 0, tzinfo=timezone.utc)
    path = build_artifact_path("a" * 64, "b" * 64, directory=tmp_path, now=now, random_suffix="fixed")
    write_b2a_artifact({"n": 1}, path)
    with pytest.raises(ArtifactAlreadyExistsError):
        write_b2a_artifact({"n": 2}, path)


def test_atomic_write_leaves_no_tmp_file_on_success(tmp_path):
    build_and_write_b2a_artifact({"n": 1}, "a" * 64, "b" * 64, directory=tmp_path)
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == []
