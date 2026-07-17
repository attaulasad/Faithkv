"""Unit tests for kvcot.runtime.git_is_dirty (2026-07-19 review): a real,
throwaway git repository (via the `git` CLI, in a tmp_path) is used rather
than mocking subprocess, so this actually exercises `--untracked-files=no`
against real git behavior instead of a guessed mock. Skips cleanly if `git`
is not on PATH (should never happen in this repo's own CI/dev environment,
but keeps this test honest rather than assuming).
"""
from __future__ import annotations

import shutil
import subprocess

import pytest

from kvcot.runtime import git_is_dirty

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git CLI not available")


def _run(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _init_repo(tmp_path):
    _run(tmp_path, "init", "-q")
    _run(tmp_path, "config", "user.email", "test@example.com")
    _run(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.txt").write_text("original content\n", encoding="utf-8")
    _run(tmp_path, "add", "tracked.txt")
    _run(tmp_path, "commit", "-q", "-m", "initial commit")


def test_clean_repo_is_not_dirty(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert git_is_dirty() is False


def test_new_untracked_file_does_not_count_as_dirty(tmp_path, monkeypatch):
    # The exact scenario the 2026-07-19 review found: a new, intentionally-
    # committed-later output artifact (e.g. results/run_manifests/*.json)
    # appearing as untracked must NOT make git_dirty report True.
    _init_repo(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "some_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert git_is_dirty() is False


def test_modified_tracked_file_counts_as_dirty(tmp_path, monkeypatch):
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("modified content\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert git_is_dirty() is True


def test_staged_new_file_counts_as_dirty(tmp_path, monkeypatch):
    # A file staged with `git add` (e.g. a freshly regenerated
    # requirements-lock.txt about to be committed) is a tracked-index
    # change, not merely an untracked file, and must still count as dirty.
    _init_repo(tmp_path)
    (tmp_path / "new_tracked.txt").write_text("content\n", encoding="utf-8")
    _run(tmp_path, "add", "new_tracked.txt")
    monkeypatch.chdir(tmp_path)
    assert git_is_dirty() is True


def test_modified_and_committed_lock_file_is_clean_again(tmp_path, monkeypatch):
    # Models the documented setup_vast.sh workflow: requirements-lock.txt is
    # regenerated (a tracked-file modification, correctly dirty), then
    # committed -- after which the tree is clean again, as intended.
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("regenerated lock content\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert git_is_dirty() is True
    _run(tmp_path, "add", "tracked.txt")
    _run(tmp_path, "commit", "-q", "-m", "regenerate lock file")
    assert git_is_dirty() is False
