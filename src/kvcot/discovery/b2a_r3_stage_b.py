"""Production B2A-R3 Stage-B qualification execution path.

This module is the narrow bridge from an already-authorized Stage-B claim
to the existing qualification coordinator. The public entry point uses
only frozen protocol paths/constants; tests may inject the worker runner
and clock, but production uses a subprocess per candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

from kvcot.config import config_identity
from kvcot.discovery import b2a_r3_contract as c
from kvcot.discovery.b2a_r3_artifacts import write_qualification_artifact_atomic
from kvcot.discovery.b2a_r3_authorization import (
    ConsumedAuthorizationContext,
    claim_authorization,
    verify_authorization_preconditions,
)
from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_provenance import GitStateProvider, SubprocessGitStateProvider
from kvcot.discovery.b2a_r3_qualification_coordinator import (
    CandidateWorkerTimeout,
    run_b2a_r3_qualification_coordinator,
)
from kvcot.discovery.b2a_r3_worker_adapter import FullKVWorkerResultR3
from kvcot.discovery.b2a_workers import _worker_subprocess_env
from kvcot.discovery.discovery_config import load_discovery_config

__all__ = [
    "StageBQualificationRunResult",
    "StageBQualificationExecutionRefused",
    "run_b2a_r3_stage_b_qualification",
    "run_fullkv_r3_worker_subprocess",
]


class StageBQualificationExecutionRefused(RuntimeError):
    """Fail-closed production Stage-B execution failure."""


@dataclass(frozen=True)
class StageBQualificationRunResult:
    artifact: dict[str, Any]
    qualification_artifact_path: Path
    authorization_claim_path: Path
    consumed_authorization_context: ConsumedAuthorizationContext


def _load_fixed_candidate_manifest(repository_root: Path, expected_config_sha256: str) -> dict[str, Any]:
    manifest_path = repository_root / c.CANDIDATE_MANIFEST_PATH
    with open(manifest_path, "r", encoding="utf-8") as f:
        candidate_manifest = json.load(f)
    verify_candidate_manifest_structure(candidate_manifest, expected_config_sha256=expected_config_sha256)
    return candidate_manifest


def _subprocess_worker_entry(candidate_ordinal_text: str, output_path_text: str) -> int:
    """Internal child-process body used by `run_fullkv_r3_worker_subprocess`."""
    try:
        from kvcot.discovery.b2a_r3_qualification_worker import run_fullkv_r3_qualification_worker

        candidate_ordinal = int(candidate_ordinal_text)
        repository_root = Path.cwd()
        expected_config_sha256 = config_identity(c.CONFIG_PATH)
        config = load_discovery_config(repository_root / c.CONFIG_PATH)
        candidate_manifest = _load_fixed_candidate_manifest(repository_root, expected_config_sha256)
        manifest = verify_candidate_manifest_structure(
            candidate_manifest, expected_config_sha256=expected_config_sha256
        )
        if not (0 <= candidate_ordinal < len(manifest.candidates)):
            raise StageBQualificationExecutionRefused(f"candidate ordinal {candidate_ordinal} is out of range")
        candidate = manifest.candidates[candidate_ordinal]
        result = run_fullkv_r3_qualification_worker(config, candidate, config_path=c.CONFIG_PATH)
        output_path = Path(output_path_text)
        output_path.write_text(
            json.dumps(FullKVWorkerResultR3.model_validate(result).model_dump(mode="json"), ensure_ascii=True),
            encoding="utf-8",
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"b2a-r3 Stage-B worker subprocess failed: {exc}", file=sys.stderr)
        return 2


def run_fullkv_r3_worker_subprocess(
    candidate_ordinal: int,
    timeout_seconds: float,
    *,
    repository_root: str | Path = ".",
    subprocess_run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> FullKVWorkerResultR3:
    """Run one fixed-path FullKV R3 worker in a subprocess.

    Candidate order and timeout are supplied only by the coordinator. The
    worker loads the frozen config and frozen candidate manifest from the
    repository root and writes to an internal temp file that is deleted by
    the parent.
    """
    if not isinstance(candidate_ordinal, int) or candidate_ordinal < 0:
        raise StageBQualificationExecutionRefused("candidate_ordinal must be a non-negative int")
    if timeout_seconds <= 0 or timeout_seconds > c.PER_CANDIDATE_WORKER_TIMEOUT_SECONDS:
        raise StageBQualificationExecutionRefused(
            "timeout_seconds must be positive and no greater than the frozen per-candidate timeout"
        )

    repository_root = Path(repository_root)
    fd, output_name = tempfile.mkstemp(prefix="b2a-r3-stage-b-worker-", suffix=".json")
    try:
        import os

        os.close(fd)
    except OSError:
        pass
    Path(output_name).unlink(missing_ok=True)

    code = (
        "from kvcot.discovery.b2a_r3_stage_b import _subprocess_worker_entry; "
        "import sys; "
        "raise SystemExit(_subprocess_worker_entry(sys.argv[1], sys.argv[2]))"
    )
    try:
        try:
            completed = subprocess_run(
                [sys.executable, "-c", code, str(candidate_ordinal), output_name],
                cwd=str(repository_root),
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
                env=_worker_subprocess_env(c.CONFIG_PATH),
            )
        except subprocess.TimeoutExpired as exc:
            raise CandidateWorkerTimeout(f"candidate ordinal={candidate_ordinal} worker timed out") from exc

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            raise StageBQualificationExecutionRefused(
                f"candidate ordinal={candidate_ordinal} worker subprocess failed with exit "
                f"{completed.returncode}: {stderr}"
            )
        with open(output_name, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return FullKVWorkerResultR3.model_validate(payload)
    finally:
        Path(output_name).unlink(missing_ok=True)


def run_b2a_r3_stage_b_qualification(
    *,
    claim_payload: dict[str, Any],
    repository_root: str | Path = ".",
    git_state: GitStateProvider | None = None,
    fullkv_worker_runner: Callable[[int, float], Any] | None = None,
    clock: Callable[[], float] | None = None,
) -> StageBQualificationRunResult:
    """Consume a Stage-B claim and write the immutable qualification artifact.

    Production callers cannot choose candidate order, limits, timeouts,
    output paths, auth roots, config path, or candidate manifest path; all
    of those are fixed by the B2A-R3 contract and the parsed authorization
    document. Tests may inject `fullkv_worker_runner` and `clock` to avoid
    CUDA/model work while still exercising this orchestration path.
    """
    repository_root = Path(repository_root)
    git_state = git_state or SubprocessGitStateProvider(str(repository_root))
    expected_config_sha256 = config_identity(repository_root / c.CONFIG_PATH)
    candidate_manifest = _load_fixed_candidate_manifest(repository_root, expected_config_sha256)

    authorization_document_path = repository_root / claim_payload["authorization_document_path"]
    verified_context = verify_authorization_preconditions(
        claim_payload,
        git_state=git_state,
        authorization_document_path=authorization_document_path,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
        repository_root=repository_root,
    )
    consumed = claim_authorization(
        claim_payload,
        repository_root=repository_root,
        verified_context=verified_context,
        git_state=git_state,
    )
    runner = fullkv_worker_runner or (
        lambda ordinal, timeout: run_fullkv_r3_worker_subprocess(
            ordinal,
            timeout,
            repository_root=repository_root,
        )
    )
    artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
        consumed_authorization_context=consumed,
        fullkv_worker_runner=runner,
        clock=clock or time.time,
        per_candidate_timeout_seconds=c.PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
    )
    output_path = repository_root / c.QUALIFICATION_ARTIFACT_PATH
    write_qualification_artifact_atomic(
        artifact,
        output_path=output_path,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=consumed,
    )
    return StageBQualificationRunResult(
        artifact=artifact,
        qualification_artifact_path=output_path,
        authorization_claim_path=consumed.claim_path,
        consumed_authorization_context=consumed,
    )
