"""Independent-audit Gate H6 tests for
`kvcot.discovery.attempt_verification`. Builds a genuinely internally-
consistent fake attempt directory by hand (never a real subprocess/model),
proves the full verifier accepts it, then mutates one artifact at a time
and proves each mutation is caught -- content verification, not mere
existence."""
from __future__ import annotations

import json
from pathlib import Path

from kvcot.discovery.attempt_verification import verify_attempt_artifacts, verify_worker_envelopes
from kvcot.utils.hashing import sha256_json


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _result_payload(role: str) -> dict:
    return {
        "role": role,
        "timing_evidence": [{"phase": f"{role}_worker_startup", "duration_seconds": 1.0, "completed": True}],
        "memory_phase_evidence": [{"phase": "model_load", "peak_allocated": 100, "completed": True}],
        "attempted_pair_identities": [{"pair_kind": "real"}] if role == "rkv" else None,
        "completed_pair_identities": [{"pair_kind": "real"}] if role == "rkv" else None,
        "failed_pair_identities": [] if role == "rkv" else None,
        "no_op_identity": {"pair_kind": "no_op"} if role == "rkv" else None,
        "semantic_mutation_reports": [{"pair_identity": {"pair_kind": "real"}, "attempted": True}] if role == "rkv" else None,
        "replay_evidence": {"pass1_token_ids": [1, 2, 3]} if role == "rkv" else None,
    }


def _envelope_payload(role: str, result_payload: dict, *, attempt_id: str = "attempt-1") -> dict:
    return {
        "role": role, "attempt_id": attempt_id, "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00", "success": True,
        "requested_identities": {}, "resolved_identities": {}, "partial_measurements": result_payload,
        "determinism_policy": {"framework_seed": 13}, "software_versions": {"torch": "2.0"},
        "hardware_metadata": {}, "error_type": None, "error_message": None, "traceback": None,
        "result_sha256": sha256_json(result_payload),
    }


def _command_payload(role: str) -> dict:
    return {
        "argv": ["python", "-m", "kvcot.discovery.b2a_worker_entry", "--role", role, "--config", "c.yaml"],
        "timeout_seconds": 7200, "check": False, "capture_output": True,
    }


def _progress_lines(role: str) -> str:
    events = [
        {"timestamp": "2026-01-01T00:00:00+00:00", "stage": "startup", "status": "completed", "attempt_id": "attempt-1", "worker_role": role, "counters": {}, "detail": None},
        {"timestamp": "2026-01-01T00:00:30+00:00", "stage": "model_load", "status": "completed", "attempt_id": "attempt-1", "worker_role": role, "counters": {}, "detail": None},
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _build_valid_attempt(tmp_path: Path) -> tuple[Path, dict, dict]:
    attempt_dir = tmp_path / "attempt"
    fullkv_result = _result_payload("fullkv")
    rkv_result = _result_payload("rkv")

    _write(attempt_dir / "invocation.json", {"attempt_id": "attempt-1"})
    _write(attempt_dir / "preflight.json", {"passed": True})
    _write(attempt_dir / "provenance.json", {"git": {"dirty": False, "rkv_submodule_match": True}})

    for role, result in (("fullkv", fullkv_result), ("rkv", rkv_result)):
        _write(attempt_dir / role / "command.json", _command_payload(role))
        _write(attempt_dir / role / "stdout.log", "ok\n")
        _write(attempt_dir / role / "stderr.log", "")
        _write(attempt_dir / role / "progress.jsonl", _progress_lines(role))
        _write(attempt_dir / role / "result.json", result)
        _write(attempt_dir / role / "envelope.json", _envelope_payload(role, result))
        _write(attempt_dir / role / "timing.json", result["timing_evidence"])
        _write(attempt_dir / role / "memory.json", result["memory_phase_evidence"])

    _write(attempt_dir / "rkv" / "pair_identities.json", {
        "attempted": rkv_result["attempted_pair_identities"],
        "completed": rkv_result["completed_pair_identities"],
        "failed": rkv_result["failed_pair_identities"],
        "no_op": rkv_result["no_op_identity"],
    })
    _write(attempt_dir / "rkv" / "semantic_swaps.json", rkv_result["semantic_mutation_reports"])
    _write(attempt_dir / "rkv" / "replay_evidence.json", rkv_result["replay_evidence"])

    return attempt_dir, fullkv_result, rkv_result


def test_verify_attempt_artifacts_accepts_a_genuinely_consistent_attempt(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is True
    assert reasons == ()
    assert verify_worker_envelopes(attempt_dir) is True


def test_verify_attempt_artifacts_fails_on_missing_file(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    (attempt_dir / "rkv" / "replay_evidence.json").unlink()
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("missing required attempt files" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_malformed_json(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    (attempt_dir / "fullkv" / "result.json").write_text("{not valid json", encoding="utf-8")
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("does not parse as valid JSON" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_malformed_jsonl_progress(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    (attempt_dir / "fullkv" / "progress.jsonl").write_text("{not valid json\n", encoding="utf-8")
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("progress.jsonl" in r and "does not parse" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_envelope_result_hash_mismatch(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    envelope = json.loads((attempt_dir / "rkv" / "envelope.json").read_text(encoding="utf-8"))
    envelope["result_sha256"] = "0" * 64
    _write(attempt_dir / "rkv" / "envelope.json", envelope)
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("result_sha256 does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_timing_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "rkv" / "timing.json", [{"phase": "tampered", "duration_seconds": 999.0, "completed": True}])
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("timing.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_memory_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "fullkv" / "memory.json", [{"phase": "tampered", "peak_allocated": 1, "completed": True}])
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("memory.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_pair_identity_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "rkv" / "pair_identities.json", {"attempted": [], "completed": [], "failed": [], "no_op": None})
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("pair_identities.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_replay_token_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "rkv" / "replay_evidence.json", {"pass1_token_ids": [9, 9, 9]})
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("replay_evidence.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_semantic_mutation_report_tampering(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "rkv" / "semantic_swaps.json", [])
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("semantic_swaps.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_command_role_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "fullkv" / "command.json", _command_payload("rkv"))
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("argv does not name role" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_return_code_or_check_true(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    bad_command = _command_payload("fullkv")
    bad_command["check"] = True
    _write(attempt_dir / "fullkv" / "command.json", bad_command)
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("check' must be False" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_timeout_state_or_nonzero_success_envelope(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    envelope = json.loads((attempt_dir / "fullkv" / "envelope.json").read_text(encoding="utf-8"))
    envelope["success"] = False
    envelope["error_type"] = "RuntimeError"
    _write(attempt_dir / "fullkv" / "envelope.json", envelope)
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("not a success envelope" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_attempt_id_disagreement(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    envelope = json.loads((attempt_dir / "rkv" / "envelope.json").read_text(encoding="utf-8"))
    envelope["attempt_id"] = "different-attempt-id"
    envelope["result_sha256"] = sha256_json(rkv_result)  # keep hash valid, isolate the ID check
    _write(attempt_dir / "rkv" / "envelope.json", envelope)
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("disagree on attempt_id" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_empty_progress_journal(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "fullkv" / "progress.jsonl", "")
    verified, reasons = verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result)
    assert verified is False
    assert any("no events" in r for r in reasons)


def test_verify_worker_envelopes_fails_on_missing_or_malformed_or_failed_envelope(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    assert verify_worker_envelopes(attempt_dir) is True

    (attempt_dir / "fullkv" / "envelope.json").unlink()
    assert verify_worker_envelopes(attempt_dir) is False

    attempt_dir2, _, _ = _build_valid_attempt(tmp_path.parent / (tmp_path.name + "-2"))
    (attempt_dir2 / "rkv" / "envelope.json").write_text("not json", encoding="utf-8")
    assert verify_worker_envelopes(attempt_dir2) is False
