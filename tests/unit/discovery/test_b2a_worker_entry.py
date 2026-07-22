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


def test_main_does_not_duplicate_live_progress_events_on_success(monkeypatch, tmp_path):
    """Independent-audit Gate H6.4: the worker body already appends a
    ("<stage>", "completed") progress event LIVE, as each phase finishes
    (via `_production_progress_callback`, exactly like the real
    `run_fullkv_worker`/`run_rkv_worker` bodies do through `measured()`).
    `main()`'s success path must NOT materialize a second, redundant
    "completed" event for the same stage by replaying the final timing
    list -- each named stage must appear in `progress.jsonl` exactly
    once."""
    from kvcot.discovery.b2a_workers import _production_progress_callback

    _patch_loaders(monkeypatch)

    def fake_run_fullkv_worker(config, manifest):
        # Simulates the REAL body's live-progress behavior: obtains the
        # same production progress callback the real worker uses, and
        # emits "completed" events for each named phase as it "finishes"
        # them -- exactly what `kvcot.discovery.b2a_workers.measured` does
        # through `_progress_stage_for_phase`.
        emit = _production_progress_callback("fullkv")
        for stage in ("snapshot resolution", "tokenizer load", "model-load completion", "runtime verification"):
            if emit is not None:
                emit(stage, "completed", None)
        return {
            "role": "fullkv",
            "determinism_policy": {"framework_seed": 13},
            "runtime_identity": {"resolved_model_revision": "modelrev", "resolved_tokenizer_revision": "tokrev"},
            "timing_evidence": [
                {"phase": "snapshot_tokenizer_resolution", "duration_seconds": 0.1, "completed": True},
                {"phase": "tokenizer_load", "duration_seconds": 0.1, "completed": True},
                {"phase": "model_load", "duration_seconds": 1.0, "completed": True},
                {"phase": "post_load_validation", "duration_seconds": 0.05, "completed": True},
            ],
        }

    monkeypatch.setattr("kvcot.discovery.b2a_workers.run_fullkv_worker", fake_run_fullkv_worker)
    output_path = tmp_path / "fullkv_result.json"

    exit_code = b2a_worker_entry.main([
        "--role", "fullkv", "--config", "cfg.yaml", "--manifest", "manifest.json",
        "--output", str(output_path), "--attempt-id", "dup-test-attempt",
    ])
    assert exit_code == 0

    progress_path = tmp_path / "progress.jsonl"
    assert progress_path.is_file()
    events = [json.loads(line) for line in progress_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed_stage_counts: dict[str, int] = {}
    for event in events:
        if event["status"] == "completed":
            completed_stage_counts[event["stage"]] = completed_stage_counts.get(event["stage"], 0) + 1

    for stage in ("snapshot resolution", "tokenizer load", "model-load completion", "runtime verification"):
        assert completed_stage_counts.get(stage) == 1, (
            f"stage {stage!r} was recorded {completed_stage_counts.get(stage)} time(s), expected exactly 1 "
            f"(live-tracked and NOT re-materialized after success)"
        )


def test_main_threads_worker_body_failure_partial_evidence_into_envelope(monkeypatch, tmp_path, capsys):
    """Independent-audit Gate H1: when the worker body raises a
    `WorkerBodyFailure` (real production behavior after the Gate H1 repair
    to `run_fullkv_worker`/`run_rkv_worker`), the failure envelope must
    carry the real partial evidence it accumulated -- never the bare
    `partial_measurements=None, determinism_policy=None` this branch used
    unconditionally before the repair."""
    from kvcot.discovery.worker_partial_evidence import PartialWorkerEvidence, WorkerBodyFailure

    _patch_loaders(monkeypatch)

    def boom(config, manifest):
        evidence = PartialWorkerEvidence(
            role="rkv",
            failing_stage="real_pair:0:5:9",
            last_completed_stage="model-load completion",
            determinism_policy={"framework_seed": 13},
            timing_evidence=[{"phase": "model_load", "duration_seconds": 2.0, "completed": True}],
            peak_cuda_allocated_bytes=123456,
            is_oom=True,
        )
        raise WorkerBodyFailure(evidence, RuntimeError("CUDA out of memory during pair evaluation"))

    monkeypatch.setattr("kvcot.discovery.b2a_workers.run_rkv_worker", boom)
    output_path = tmp_path / "rkv_result.json"

    exit_code = b2a_worker_entry.main(
        ["--role", "rkv", "--config", "cfg.yaml", "--manifest", "manifest.json", "--output", str(output_path)]
    )

    assert exit_code == 1
    assert not output_path.exists()

    envelope_path = tmp_path / "rkv_result.json.envelope.json"
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    assert envelope["success"] is False
    assert envelope["is_oom"] is True
    assert envelope["is_timeout"] is False
    assert envelope["failure_stage"] == "real_pair:0:5:9"
    assert envelope["last_completed_stage"] == "model-load completion"
    # `error_type`/`error_message` report the ORIGINAL cause, never the
    # `WorkerBodyFailure` wrapper's own composed message.
    assert envelope["error_type"] == "RuntimeError"
    assert "CUDA out of memory" in envelope["error_message"]
    assert envelope["partial_measurements"]["determinism_policy"] == {"framework_seed": 13}
    assert envelope["partial_measurements"]["timing_evidence"] == [
        {"phase": "model_load", "duration_seconds": 2.0, "completed": True}
    ]
    assert envelope["partial_measurements"]["peak_cuda_allocated_bytes"] == 123456
    assert envelope["determinism_policy"] == {"framework_seed": 13}

    captured = capsys.readouterr()
    assert "CUDA out of memory" in captured.err


def test_main_reports_no_partial_evidence_for_a_pre_body_failure(monkeypatch, tmp_path):
    """A failure BEFORE any worker body ever started (e.g. `run_fullkv_worker`
    itself is never even reached) genuinely has no partial evidence to
    report -- the envelope's new typed fields must honestly default to
    `None`/`False`, never a fabricated value."""
    _patch_loaders(monkeypatch)

    def boom(config, manifest):
        raise RuntimeError("simulated worker crash")

    monkeypatch.setattr("kvcot.discovery.b2a_workers.run_fullkv_worker", boom)
    output_path = tmp_path / "fullkv_result.json"

    exit_code = b2a_worker_entry.main(
        ["--role", "fullkv", "--config", "cfg.yaml", "--manifest", "manifest.json", "--output", str(output_path)]
    )
    assert exit_code == 1
    envelope = json.loads((tmp_path / "fullkv_result.json.envelope.json").read_text(encoding="utf-8"))
    assert envelope["partial_measurements"] is None
    assert envelope["determinism_policy"] is None
    assert envelope["failure_stage"] is None
    assert envelope["is_oom"] is False
    assert envelope["is_timeout"] is False


def test_main_requires_role_config_manifest_output_arguments():
    with pytest.raises(SystemExit):
        b2a_worker_entry.main(["--role", "fullkv"])  # missing required args


def test_main_rejects_unrecognized_role():
    with pytest.raises(SystemExit):
        b2a_worker_entry.main(
            ["--role", "bogus", "--config", "c.yaml", "--manifest", "m.json", "--output", "o.json"]
        )
