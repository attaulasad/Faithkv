"""CPU tests for the production B2A-R3 Stage-B execution path."""
from __future__ import annotations

import inspect
import itertools
import json
from pathlib import Path
import subprocess
import sys

from kvcot.discovery import b2a_r3_contract as c
from kvcot.discovery.b2a_r3_stage_b import (
    run_b2a_r3_stage_b_qualification,
    run_fullkv_r3_worker_subprocess,
)
from kvcot.discovery.b2a_r3_worker_adapter import FullKVWorkerResultR3
from kvcot.cli import build_parser

from tests.unit.discovery.test_b2a_r3_authorization import _verified_stage_b
from tests.unit.discovery.test_b2a_r3_worker_adapter import _valid_worker_result


def _install_fixed_files(tmp_path: Path, candidate_manifest: dict) -> None:
    manifest_path = tmp_path / c.CANDIDATE_MANIFEST_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(candidate_manifest, ensure_ascii=True), encoding="utf-8")
    config_path = tmp_path / c.CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("synthetic config bytes\n", encoding="utf-8")


def test_stage_b_entry_consumes_claim_and_writes_fixed_artifact(tmp_path, monkeypatch):
    from kvcot.discovery import b2a_r3_stage_b

    claim_payload, _verified, _document, candidate_manifest, git_state = _verified_stage_b(tmp_path)
    _install_fixed_files(tmp_path, candidate_manifest)
    monkeypatch.setattr(b2a_r3_stage_b, "config_identity", lambda _path: candidate_manifest["config_sha256"])

    calls: list[tuple[int, float]] = []

    def runner(ordinal: int, timeout_seconds: float) -> FullKVWorkerResultR3:
        calls.append((ordinal, timeout_seconds))
        return _valid_worker_result(ordinal)

    ticks = (1_800_000_000.0 + i for i in itertools.count())
    result = run_b2a_r3_stage_b_qualification(
        claim_payload=claim_payload,
        repository_root=tmp_path,
        git_state=git_state,
        fullkv_worker_runner=runner,
        clock=lambda: next(ticks),
    )

    assert calls == [(0, 3599.0)]
    assert result.authorization_claim_path == tmp_path / claim_payload["global_claim_path"]
    assert result.qualification_artifact_path == tmp_path / c.QUALIFICATION_ARTIFACT_PATH
    assert result.qualification_artifact_path.is_file()
    assert result.artifact["stage_b_authorization_id"] == claim_payload["authorization_id"]
    assert result.artifact["authorization_claim_canonical_sha256"] == claim_payload["canonical_sha256"]


def test_stage_b_public_entry_does_not_accept_runtime_policy_overrides():
    signature = inspect.signature(run_b2a_r3_stage_b_qualification)
    for forbidden in (
        "candidate_order",
        "maximum_candidates",
        "phase_wall_time_limit_seconds",
        "per_candidate_timeout_seconds",
        "output_path",
        "claims_root",
        "config_path",
        "candidate_manifest_path",
    ):
        assert forbidden not in signature.parameters

    parser = build_parser()
    args = parser.parse_args(["run-b2a-r3-stage-b-qualification", "--claim", "claim.json"])
    for forbidden in ("candidates", "artifact", "config", "repository_root", "output"):
        assert not hasattr(args, forbidden)


def test_subprocess_worker_wrapper_uses_internal_temp_output(tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        output_path = Path(cmd[-1])
        output_path.write_text(
            json.dumps(_valid_worker_result(0).model_dump(mode="json"), ensure_ascii=True),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    result = run_fullkv_r3_worker_subprocess(
        0,
        c.PER_CANDIDATE_WORKER_TIMEOUT_SECONDS,
        repository_root=tmp_path,
        subprocess_run=fake_run,
    )

    assert result.unique_id == _valid_worker_result(0).unique_id
    cmd, kwargs = calls[0]
    assert cmd[:2] == [sys.executable, "-c"]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["timeout"] == c.PER_CANDIDATE_WORKER_TIMEOUT_SECONDS
    assert kwargs["env"]["PYTHONHASHSEED"] == "13"
    assert kwargs["env"]["TOKENIZERS_PARALLELISM"] == "false"
    assert not Path(cmd[-1]).exists()
