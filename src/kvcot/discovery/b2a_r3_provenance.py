"""B2A-R3 versioned provenance policy (Step 3 Stage-A, protocol §14.5).

`src/kvcot/discovery/attempt_verification.py`'s module-level
`REQUIRED_BRANCH = "research/b1b-r4-final-b2a-closure"` is a historical
constant tied to the branch B2A-R1/B2A-R2 executed on -- this module NEVER
imports, edits, or reinterprets it. `AttemptProvenancePolicy` is a
separately-constructed, in-memory policy object, populated from a future
B2A-R3 dated authorization document, carrying its own required
repository/branch/commit/ancestors/R-KV SHA/authorization identity. Git
inspection is injected behind `GitStateProvider` so CPU tests exercise
this entirely with synthetic repository states -- no real `git` subprocess
call, no torch, no CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from kvcot.discovery.b2a_r3_contract import PROVENANCE_POLICY_VERSION, REQUIRED_REPOSITORY

__all__ = [
    "AttemptProvenancePolicy",
    "WorktreeStatus",
    "GitStateProvider",
    "verify_attempt_provenance",
]


@dataclass(frozen=True)
class AttemptProvenancePolicy:
    """A separately-constructed policy object (protocol §14.5) -- never
    derived from, and never feeding back into,
    `kvcot.discovery.attempt_verification.REQUIRED_BRANCH` or any other
    historical B2A-R1/R2 verification global."""

    provenance_policy_version: str
    required_repository: str
    required_branch: str
    required_commit_sha: str
    required_ancestor_shas: tuple[str, ...]
    required_rkv_sha: str
    authorization_id: str
    authorization_document_sha256: str

    def __post_init__(self) -> None:
        if self.provenance_policy_version != PROVENANCE_POLICY_VERSION:
            raise ValueError(
                f"provenance_policy_version must be {PROVENANCE_POLICY_VERSION!r}, "
                f"got {self.provenance_policy_version!r}"
            )
        if self.required_repository != REQUIRED_REPOSITORY:
            raise ValueError(f"required_repository must be {REQUIRED_REPOSITORY!r}")


@dataclass(frozen=True)
class WorktreeStatus:
    staged_paths: tuple[str, ...]
    unstaged_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]

    @property
    def dirty_paths(self) -> frozenset[str]:
        return frozenset(self.staged_paths) | frozenset(self.unstaged_paths) | frozenset(self.untracked_paths)


class GitStateProvider(Protocol):
    """The injection seam -- CPU tests implement this against a synthetic
    in-memory repository state; production would eventually implement it
    against real `git` subprocess calls, but no Stage-A code constructs a
    real implementation or invokes one."""

    def current_repository(self) -> str: ...
    def current_branch(self) -> str: ...
    def current_commit_sha(self) -> str: ...
    def is_ancestor(self, ancestor_sha: str, commit_sha: str) -> bool: ...
    def rkv_submodule_sha(self) -> str: ...
    def worktree_status(self) -> WorktreeStatus: ...


def verify_attempt_provenance(
    policy: AttemptProvenancePolicy,
    git_state: GitStateProvider,
    *,
    post_claim_allowlist: frozenset[str] = frozenset(),
) -> tuple[bool, tuple[str, ...]]:
    """Verifies observed Git/worktree state against `policy`. When
    `post_claim_allowlist` is empty (the pre-claim case), the worktree must
    be completely clean -- no staged, unstaged, or untracked path is ever
    accepted. When non-empty (the post-claim case), ONLY the exact active
    global claim path and the exact active attempt-directory root may be
    recognized as expected -- never a broader "anything under
    results/decisions/ is fine" rule (protocol §14.4.6)."""
    reasons: list[str] = []

    observed_repository = git_state.current_repository()
    if observed_repository != policy.required_repository:
        reasons.append(
            f"observed repository {observed_repository!r} != required {policy.required_repository!r}"
        )

    observed_branch = git_state.current_branch()
    if observed_branch != policy.required_branch:
        reasons.append(f"observed branch {observed_branch!r} != required {policy.required_branch!r}")

    observed_commit_sha = git_state.current_commit_sha()
    if observed_commit_sha != policy.required_commit_sha:
        reasons.append(f"observed commit {observed_commit_sha!r} != required {policy.required_commit_sha!r}")

    for ancestor_sha in policy.required_ancestor_shas:
        if not git_state.is_ancestor(ancestor_sha, policy.required_commit_sha):
            reasons.append(f"required ancestor {ancestor_sha!r} does not verify as an ancestor of HEAD")

    observed_rkv_sha = git_state.rkv_submodule_sha()
    if observed_rkv_sha != policy.required_rkv_sha:
        reasons.append(f"observed R-KV submodule SHA {observed_rkv_sha!r} != required {policy.required_rkv_sha!r}")

    status = git_state.worktree_status()
    unexpected = status.dirty_paths - post_claim_allowlist
    if unexpected:
        reasons.append(f"unexpected dirty/staged/untracked path(s): {sorted(unexpected)}")

    return (len(reasons) == 0), tuple(reasons)
