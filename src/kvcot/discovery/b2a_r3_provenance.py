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
from pathlib import PurePosixPath
import re
import subprocess
from typing import Any, Protocol

from kvcot.discovery.b2a_r3_contract import (
    AUTHORIZATION_CLAIMS_DIR,
    PROVENANCE_POLICY_VERSION,
    REQUIRED_REPOSITORY,
    global_claim_path,
    validate_authorization_id,
)

__all__ = [
    "AttemptProvenancePolicy",
    "WorktreeStatus",
    "ActiveAuthorizationPaths",
    "GitStateProvider",
    "SubprocessGitStateProvider",
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


_ATTEMPT_PATH_RE = re.compile(
    r"^results/decisions/b2a_r3_attempt_[0-9]{8}T[0-9]{12}Z_([A-Za-z0-9][A-Za-z0-9._-]{0,127})$"
)


def _normalized_repo_path(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a non-empty repository-relative POSIX path")
    path = PurePosixPath(value.rstrip("/"))
    if path.is_absolute() or ".." in path.parts or str(path) != value.rstrip("/"):
        raise ValueError(f"{name} is not a normalized repository-relative path: {value!r}")
    return str(path)


@dataclass(frozen=True)
class ActiveAuthorizationPaths:
    """The only paths an already-verified active claim may introduce."""

    global_claim_path: str
    attempt_directory_root: str

    @classmethod
    def from_verified_claim(cls, claim: Any) -> "ActiveAuthorizationPaths":
        authorization_id = validate_authorization_id(claim.authorization_id)
        claim_path = _normalized_repo_path(claim.global_claim_path, name="global_claim_path")
        if claim_path != global_claim_path(authorization_id):
            raise ValueError("active global claim path is not the deterministic path for authorization_id")
        if PurePosixPath(claim_path).parent != PurePosixPath(AUTHORIZATION_CLAIMS_DIR):
            raise ValueError("active global claim path is outside the exact claims root")

        attempt_path = _normalized_repo_path(
            claim.attempt_directory_path, name="attempt_directory_path"
        )
        match = _ATTEMPT_PATH_RE.fullmatch(attempt_path)
        if match is None or match.group(1) != claim.attempt_id:
            raise ValueError("attempt directory does not match the exact B2A-R3 naming convention/attempt_id")
        validate_authorization_id(claim.attempt_id)
        if PurePosixPath(attempt_path).parent != PurePosixPath("results/decisions"):
            raise ValueError("attempt directory is outside results/decisions")
        return cls(global_claim_path=claim_path, attempt_directory_root=attempt_path)


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
    def is_path_committed(self, path: str) -> bool: ...


class SubprocessGitStateProvider:
    """CPU-only production provider used by semantic verification CLIs."""

    def __init__(self, repository_root: str = ".") -> None:
        self.repository_root = repository_root

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=self.repository_root, text=True, capture_output=True, check=check
        )

    def current_repository(self) -> str:
        remote = self._run("config", "--get", "remote.origin.url").stdout.strip()
        remote = remote.removesuffix(".git")
        if remote.startswith("git@github.com:"):
            return remote.split(":", 1)[1]
        marker = "github.com/"
        if marker in remote:
            return remote.split(marker, 1)[1]
        return remote

    def current_branch(self) -> str:
        return self._run("branch", "--show-current").stdout.strip()

    def current_commit_sha(self) -> str:
        return self._run("rev-parse", "HEAD").stdout.strip()

    def is_ancestor(self, ancestor_sha: str, commit_sha: str) -> bool:
        return self._run("merge-base", "--is-ancestor", ancestor_sha, commit_sha, check=False).returncode == 0

    def rkv_submodule_sha(self) -> str:
        fields = self._run("ls-tree", "HEAD", "third_party/R-KV").stdout.split()
        if len(fields) < 3:
            raise ValueError("third_party/R-KV gitlink is missing")
        return fields[2]

    def worktree_status(self) -> WorktreeStatus:
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        for line in self._run("status", "--porcelain=v1", "--untracked-files=all").stdout.splitlines():
            code, path = line[:2], line[3:]
            if " -> " in path:
                path = path.split(" -> ", 1)[1]
            if code == "??":
                untracked.append(path)
            else:
                if code[0] != " ":
                    staged.append(path)
                if code[1] != " ":
                    unstaged.append(path)
        return WorktreeStatus(tuple(staged), tuple(unstaged), tuple(untracked))

    def is_path_committed(self, path: str) -> bool:
        return self._run("ls-files", "--error-unmatch", "--", path, check=False).returncode == 0


def verify_attempt_provenance(
    policy: AttemptProvenancePolicy,
    git_state: GitStateProvider,
    *,
    active_authorization_paths: ActiveAuthorizationPaths | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Verifies observed Git/worktree state against `policy`. When
    `active_authorization_paths` is absent (the pre-claim case), the worktree must
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
    unexpected: set[str] = set()
    for dirty_path in status.dirty_paths:
        try:
            normalized = _normalized_repo_path(dirty_path, name="dirty worktree path")
        except ValueError:
            unexpected.add(dirty_path)
            continue
        allowed = False
        if active_authorization_paths is not None:
            claim_path = _normalized_repo_path(
                active_authorization_paths.global_claim_path, name="active global claim path"
            )
            attempt_root = _normalized_repo_path(
                active_authorization_paths.attempt_directory_root, name="active attempt root"
            )
            allowed = normalized == claim_path or normalized == attempt_root or normalized.startswith(
                attempt_root + "/"
            )
        if not allowed:
            unexpected.add(dirty_path)
    if unexpected:
        reasons.append(f"unexpected dirty/staged/untracked path(s): {sorted(unexpected)}")

    return (len(reasons) == 0), tuple(reasons)
