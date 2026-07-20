"""B1B-R4 §16/§19/§20 tests for `kvcot.discovery.b2a_worker_entry.main()` --
exercised directly, in-process, with injected fake config/manifest loaders
and fake `run_fullkv_worker`/`run_rkv_worker` functions. Never launches a
real subprocess, never imports torch/transformers.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from kvcot.discovery import b2a_worker_entry


class _FakeConfig:
    class _Model:
        revision = "modelrev"
        tokenizer_revision = "tokrev"

    model = _Model()


class _FakeManifest:
    dataset_revision = "d" * 40

    def manifest_hash(self):
        return "m" * 64


def _patch_loaders(monkeypatch):
    monkeypatch.setattr(
        "kvcot.discovery.discovery_config.load_discovery_config", lambda path: _FakeConfig()
    )
    monkeypatch.setattr(
        "kvcot.discovery.manifest.load_b2a_one_example_manifest", lambda path: _FakeManifest()
    )


def _fake_success_result(role: str) -> dict:
    return {
        "role": role,
        "determinism_policy": {"framework_seed": 13},
        "runtime_identity": {"resolved_model_revision": "modelrev", "resolved_tokenizer_revision": "tokrev"},
    }


def test_main_writes_result_and_success_envelope_for_fullkv(monkeypatch, tmp_path):
    _patch_loaders(monkeypatch)
    monkeypatch.setattr(
        "kvcot.discovery.b2a_workers.run_fullkv_worker", lambda config, manifest: _fake_success_result("fullkv")
    )
    output_path = tmp_path / "fullkv_result.json"

    exit_code = b2a_worker_entry.main(
        ["--role", "fullkv", "--config", "cfg.yaml", "--manifest", "manifest.json", "--output", str(output_path)]
    )

    assert exit_code == 0
    assert output_path.exists()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["role"] == "fullkv"

    envelope_path = tmp_path / "fullkv_result.json.envelope.json"
    assert envelope_path.exists()
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["success"] is True
    assert envelope["role"] == "fullkv"
    assert envelope["error_type"] is None
    assert envelope["resolved_identities"]["resolved_model_revision"] == "modelrev"


def test_main_calls_canonical_run_rkv_worker_never_a_stub(monkeypatch, tmp_path):
    """B1B-R4 §19: the 'rkv' role must call `kvcot.discovery.b2a_workers
    .run_rkv_worker` -- never `kvcot.discovery.b2a_execute.run_rkv_worker_body`
    (the B1B-R3 split this repair removes)."""
    _patch_loaders(monkeypatch)
    called = {"run_rkv_worker": False}

    def fake_run_rkv_worker(config, manifest):
        called["run_rkv_worker"] = True
        return _fake_success_result("rkv")

    monkeypatch.setattr("kvcot.discovery.b2a_workers.run_rkv_worker", fake_run_rkv_worker)
    output_path = tmp_path / "rkv_result.json"

    exit_code = b2a_worker_entry.main(
        ["--role", "rkv", "--config", "cfg.yaml", "--manifest", "manifest.json", "--output", str(output_path)]
    )

    assert exit_code == 0
    assert called["run_rkv_worker"] is True
    assert not hasattr(b2a_worker_entry, "run_rkv_worker_body")


def test_main_writes_only_failure_envelope_not_a_result_file_on_exception(monkeypatch, tmp_path, capsys):
    _patch_loaders(monkeypatch)

    def boom(config, manifest):
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr("kvcot.discovery.b2a_workers.run_fullkv_worker", boom)
    output_path = tmp_path / "fullkv_result.json"

    exit_code = b2a_worker_entry.main(
        ["--role", "fullkv", "--config", "cfg.yaml", "--manifest", "manifest.json", "--output", str(output_path)]
    )

    assert exit_code == 1
    assert not output_path.exists()  # no result file on failure -- unchanged contract

    envelope_path = tmp_path / "fullkv_result.json.envelope.json"
    assert envelope_path.exists()  # B1B-R4 §16: envelope is ALWAYS written, even on failure
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["success"] is False
    assert envelope["error_type"] == "RuntimeError"
    assert "simulated worker crash" in envelope["error_message"]
    assert envelope["traceback"] is not None

    captured = capsys.readouterr()
    assert "simulated worker crash" in captured.err


def test_main_requires_role_config_manifest_output_arguments():
    with pytest.raises(SystemExit):
        b2a_worker_entry.main(["--role", "fullkv"])  # missing required args


def test_main_rejects_unrecognized_role():
    with pytest.raises(SystemExit):
        b2a_worker_entry.main(
            ["--role", "bogus", "--config", "c.yaml", "--manifest", "m.json", "--output", "o.json"]
        )
