"""CPU-only mocked end-to-end test of `kvcot.discovery.b2a_execute
.run_b2a_calibration`'s COMPLETE control flow (B1B-R3 §17) -- prompt-
identity verification and the subprocess launch are both faked (never a
real network fetch, tokenizer load, Python subprocess, or GPU access), but
the REAL evidence producer (`kvcot.discovery.b2a_evidence`), gate evaluator
(`kvcot.discovery.b2a_contract.evaluate_b2a_gate`), and artifact writer
(`kvcot.discovery.b2a_artifact`) all execute for real. This does NOT mock
by returning a preconstructed passing `B2AGateResult` -- every field is
derived from the fake workers' JSON payloads exactly as a real run would."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcot.discovery import b2a_execute
from kvcot.discovery.discovery_config import load_discovery_config
from kvcot.discovery.manifest import load_b2a_one_example_manifest

CONFIG_PATH = "configs/discovery/llama8b_math500_b1024.yaml"
MANIFEST_PATH = "configs/discovery/b2a_one_example_manifest.json"


@pytest.fixture
def config():
    return load_discovery_config(CONFIG_PATH)


@pytest.fixture
def manifest():
    return load_b2a_one_example_manifest(MANIFEST_PATH)


def _passing_payloads(manifest, config):
    fullkv = dict(
        role="fullkv", model_revision=config.model.revision, tokenizer_revision=config.model.tokenizer_revision,
        dataset_repo=manifest.dataset_repo, dataset_revision=manifest.dataset_revision,
        manifest_hash=manifest.manifest_hash(), prompt_token_ids_sha256=manifest.prompt_token_ids_sha256,
        natural_generated_token_ids=[1, 2, 3], natural_answer="4", natural_answer_status="correct",
        wall_seconds=12.0, peak_cuda_allocated_bytes=1_000_000, peak_cuda_reserved_bytes=2_000_000,
        every_parameter_on_cuda=True, batch_size=1, software_versions={"torch": "2.0"},
    )
    rkv = dict(
        role="rkv", model_revision=config.model.revision, tokenizer_revision=config.model.tokenizer_revision,
        dataset_repo=manifest.dataset_repo, dataset_revision=manifest.dataset_revision,
        manifest_hash=manifest.manifest_hash(), prompt_token_ids_sha256=manifest.prompt_token_ids_sha256,
        rkv_upstream_revision=config.rkv.upstream_revision, runtime_rkv_config_hash="h" * 64,
        frozen_rkv_config_hash="h" * 64, example_valid=True, event_count=3, observed_retention_ratio=0.4,
        no_op_numerical_parity=True, natural_answer_status="correct", wall_seconds_pass1=5.0,
        wall_seconds_pass2=5.0, wall_seconds_targeted_capture=0.5, wall_seconds_cache_clone_restore=0.1,
        wall_seconds_one_swap=0.01, wall_seconds_bridge_plus_48_scored=1.0, peak_cuda_allocated_bytes=1_500_000,
        peak_cuda_reserved_bytes=2_500_000, every_parameter_on_cuda=True, batch_size=1,
        software_versions={"torch": "2.0"},
    )
    return fullkv, rkv


def _fake_runner_writing(fullkv_payload, rkv_payload, fail_role=None):
    def runner(argv):
        role = argv[argv.index("--role") + 1]
        output_path = Path(argv[argv.index("--output") + 1])
        if role == fail_role:
            return SimpleNamespace(returncode=1, stdout="", stderr="simulated failure")
        payload = fullkv_payload if role == "fullkv" else rkv_payload
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return runner


def test_full_coordinator_flow_passes_and_writes_pass_artifact(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)

    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)

    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact as real_writer

    def patched_writer(payload, config_hash, manifest_hash, **kwargs):
        return real_writer(payload, config_hash, manifest_hash, directory=tmp_path)

    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", patched_writer)

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )

    assert artifact.gate_result.passed is True
    assert artifact.artifact_path.exists()
    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["shared_identity_ok"] is True
    assert payload["measurement"]["projected_complete_pilot_gpu_hours"] > 0.0


def test_gate_fails_when_rkv_example_invalid_but_artifact_still_written(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["example_valid"] = False
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)

    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact as real_writer

    def patched_writer(payload, config_hash, manifest_hash, **kwargs):
        return real_writer(payload, config_hash, manifest_hash, directory=tmp_path)

    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", patched_writer)

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )

    assert artifact.gate_result.passed is False
    assert "token_identical_replay" in artifact.gate_result.failed_conditions
    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False


def test_worker_failure_still_writes_a_fail_artifact(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload, fail_role="rkv")

    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact as real_writer

    def patched_writer(payload, config_hash, manifest_hash, **kwargs):
        return real_writer(payload, config_hash, manifest_hash, directory=tmp_path)

    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", patched_writer)

    with pytest.raises(Exception, match="rkv worker exited"):
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=runner,
        )

    written = list(tmp_path.glob("b2a_*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["passed"] is False
    assert "rkv worker exited" in payload["failure_reason"]


def test_shared_identity_mismatch_fails_the_gate(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["manifest_hash"] = "totally-different-hash".ljust(64, "0")
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)

    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact as real_writer

    def patched_writer(payload, config_hash, manifest_hash, **kwargs):
        return real_writer(payload, config_hash, manifest_hash, directory=tmp_path)

    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", patched_writer)

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert artifact.gate_result.passed is False
    assert "manifest_hash_match" in artifact.gate_result.failed_conditions


def test_prompt_identity_refusal_before_any_worker_launch_still_writes_fail_artifact(config, manifest, monkeypatch, tmp_path):
    def _boom(c, m):
        raise b2a_execute.B2AExecutionRefused("simulated prompt identity mismatch")

    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", _boom)

    def runner_that_must_not_be_called(argv):
        raise AssertionError("subprocess must never be launched when prompt identity verification fails first")

    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact as real_writer

    def patched_writer(payload, config_hash, manifest_hash, **kwargs):
        return real_writer(payload, config_hash, manifest_hash, directory=tmp_path)

    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", patched_writer)

    with pytest.raises(b2a_execute.B2AExecutionRefused):
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=runner_that_must_not_be_called,
        )

    written = list(tmp_path.glob("b2a_*.json"))
    assert len(written) == 1
