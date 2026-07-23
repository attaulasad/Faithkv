"""B2A-R3 versioned provenance-policy tests. Entirely synthetic
`GitStateProvider` implementations -- no real `git` subprocess call, no
torch, no CUDA."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from kvcot.discovery.b2a_r3_contract import PROVENANCE_POLICY_VERSION, REQUIRED_REPOSITORY
from kvcot.discovery.b2a_r3_provenance import (
    ActiveAuthorizationPaths,
    AttemptProvenancePolicy,
    WorktreeStatus,
    verify_attempt_provenance,
)

REQUIRED_BRANCH = "research/b2a-r3-runtime-qualified-calibration"
REQUIRED_COMMIT = "a" * 40
ANCESTOR_1 = "b" * 40
ANCESTOR_2 = "c" * 40
RKV_SHA = "45eaa7d69d20b7388321f077020a610d9afb65bd"


@dataclass(frozen=True)
class FakeGitState:
    repository: str = REQUIRED_REPOSITORY
    branch: str = REQUIRED_BRANCH
    commit_sha: str = REQUIRED_COMMIT
    ancestors: frozenset[str] = frozenset({ANCESTOR_1, ANCESTOR_2})
    rkv_sha: str = RKV_SHA
    status: WorktreeStatus = WorktreeStatus(staged_paths=(), unstaged_paths=(), untracked_paths=())

    def current_repository(self) -> str:
        return self.repository

    def current_branch(self) -> str:
        return self.branch

    def current_commit_sha(self) -> str:
        return self.commit_sha

    def is_ancestor(self, ancestor_sha: str, commit_sha: str) -> bool:
        return ancestor_sha in self.ancestors and commit_sha == self.commit_sha

    def rkv_submodule_sha(self) -> str:
        return self.rkv_sha

    def worktree_status(self) -> WorktreeStatus:
        return self.status

    def is_path_committed(self, path: str) -> bool:
        return True


def _policy(**overrides) -> AttemptProvenancePolicy:
    fields = dict(
        provenance_policy_version=PROVENANCE_POLICY_VERSION,
        required_repository=REQUIRED_REPOSITORY,
        required_branch=REQUIRED_BRANCH,
        required_commit_sha=REQUIRED_COMMIT,
        required_ancestor_shas=(ANCESTOR_1, ANCESTOR_2),
        required_rkv_sha=RKV_SHA,
        authorization_id="stage-b-2026-08-01",
        authorization_document_sha256="d" * 64,
    )
    fields.update(overrides)
    return AttemptProvenancePolicy(**fields)


def _active_paths(**overrides) -> ActiveAuthorizationPaths:
    values = dict(
        authorization_id="stage-b-2026-08-01",
        global_claim_path="results/decisions/b2a_r3_authorization_claims/stage-b-2026-08-01.json",
        attempt_id="deadbeef",
        attempt_directory_path="results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef",
    )
    values.update(overrides)
    return ActiveAuthorizationPaths.from_verified_claim(SimpleNamespace(**values))


def test_policy_rejects_wrong_version():
    with pytest.raises(ValueError):
        _policy(provenance_policy_version="wrong-v0")


def test_policy_rejects_wrong_repository():
    with pytest.raises(ValueError):
        _policy(required_repository="wrong/repo")


def test_matching_state_passes():
    ok, reasons = verify_attempt_provenance(_policy(), FakeGitState())
    assert ok is True
    assert reasons == ()


def test_wrong_repository_fails():
    ok, reasons = verify_attempt_provenance(_policy(), FakeGitState(repository="someone/else"))
    assert ok is False
    assert any("repository" in r for r in reasons)


def test_wrong_branch_fails():
    ok, reasons = verify_attempt_provenance(_policy(), FakeGitState(branch="main"))
    assert ok is False
    assert any("branch" in r for r in reasons)


def test_wrong_commit_fails():
    ok, reasons = verify_attempt_provenance(_policy(), FakeGitState(commit_sha="f" * 40))
    assert ok is False
    assert any("commit" in r for r in reasons)


def test_missing_ancestor_fails():
    ok, reasons = verify_attempt_provenance(_policy(), FakeGitState(ancestors=frozenset({ANCESTOR_1})))
    assert ok is False
    assert any("ancestor" in r for r in reasons)


def test_wrong_rkv_sha_fails():
    ok, reasons = verify_attempt_provenance(_policy(), FakeGitState(rkv_sha="0" * 40))
    assert ok is False
    assert any("R-KV" in r or "rkv" in r.lower() for r in reasons)


def test_dirty_pre_claim_worktree_fails():
    dirty = WorktreeStatus(staged_paths=("foo.py",), unstaged_paths=(), untracked_paths=())
    ok, reasons = verify_attempt_provenance(_policy(), FakeGitState(status=dirty))
    assert ok is False
    assert any("dirty" in r for r in reasons)


def test_post_claim_allowlist_permits_exact_paths():
    from kvcot.discovery.b2a_r3_contract import global_claim_path

    claim_path = global_claim_path("stage-b-2026-08-01")
    attempt_dir = "results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef"
    dirty = WorktreeStatus(staged_paths=(), unstaged_paths=(claim_path,), untracked_paths=(attempt_dir,))
    ok, reasons = verify_attempt_provenance(
        _policy(), FakeGitState(status=dirty), active_authorization_paths=_active_paths(),
    )
    assert ok is True
    assert reasons == ()


def test_post_claim_allowlist_rejects_any_other_path():
    from kvcot.discovery.b2a_r3_contract import global_claim_path

    claim_path = global_claim_path("stage-b-2026-08-01")
    dirty = WorktreeStatus(staged_paths=(), unstaged_paths=(claim_path, "configs/lock.yaml"), untracked_paths=())
    ok, reasons = verify_attempt_provenance(
        _policy(), FakeGitState(status=dirty), active_authorization_paths=_active_paths(),
    )
    assert ok is False
    assert any("configs/lock.yaml" in r for r in reasons)


def test_never_claims_configs_directory_as_expected():
    dirty = WorktreeStatus(staged_paths=(), unstaged_paths=(), untracked_paths=("configs/some_new_file.json",))
    ok, reasons = verify_attempt_provenance(
        _policy(), FakeGitState(status=dirty), active_authorization_paths=_active_paths(),
    )
    assert ok is False
    assert any("configs/some_new_file.json" in reason for reason in reasons)


def test_child_of_active_attempt_directory_is_accepted():
    child = "results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef/result.json"
    dirty = WorktreeStatus((), (), (child,))
    ok, reasons = verify_attempt_provenance(
        _policy(), FakeGitState(status=dirty), active_authorization_paths=_active_paths()
    )
    assert ok is True
    assert reasons == ()


@pytest.mark.parametrize(
    "dirty_path",
    [
        "results/decisions/b2a_r3_authorization_claims/different.json",
        "results/decisions/b2a_r3_attempt_20260801T000000000000Z_other",
        "results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef-sibling/file.json",
        "results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef/../escape.json",
        "src/kvcot/discovery/change.py",
    ],
)
def test_post_claim_paths_reject_siblings_traversal_and_source(dirty_path):
    dirty = WorktreeStatus((), (), (dirty_path,))
    ok, _reasons = verify_attempt_provenance(
        _policy(), FakeGitState(status=dirty), active_authorization_paths=_active_paths()
    )
    assert ok is False


# --------------------------------------------------------------------- historical regression


def test_does_not_touch_historical_required_branch_constant():
    from kvcot.discovery.attempt_verification import REQUIRED_BRANCH as HISTORICAL_REQUIRED_BRANCH

    assert HISTORICAL_REQUIRED_BRANCH == "research/b1b-r4-final-b2a-closure"


def test_attempt_provenance_policy_is_independent_of_historical_constant():
    from kvcot.discovery.attempt_verification import REQUIRED_BRANCH as HISTORICAL_REQUIRED_BRANCH

    policy = _policy(required_branch="research/b2a-r3-runtime-qualified-calibration")
    assert policy.required_branch != HISTORICAL_REQUIRED_BRANCH
