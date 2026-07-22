"""Independent-audit Gate H6 + F4 tests for
`kvcot.discovery.attempt_verification`. Builds a genuinely internally-
consistent fake attempt directory by hand (never a real subprocess/model),
proves the full verifier accepts it, then mutates one artifact at a time
and proves each mutation is caught -- content verification, not mere
existence."""
from __future__ import annotations

import json
from pathlib import Path

from kvcot.discovery.attempt_artifacts import B1_REPAIR_ROUND4_STARTING_COMMIT, B1_REQUIRED_ANCESTOR_SHAS
from kvcot.discovery.attempt_verification import REQUIRED_BRANCH, verify_attempt_artifacts, verify_worker_envelopes
from kvcot.discovery.constants import B2A_WORKER_TIMEOUT_SECONDS
from kvcot.discovery.discovery_config import PINNED_RKV_UPSTREAM_REVISION
from kvcot.utils.hashing import sha256_json

ATTEMPT_ID = "attempt-1"


def _valid_provenance_payload() -> dict:
    return {
        "git": {
            "branch": REQUIRED_BRANCH,
            "head": "a" * 40,
            "origin_branch_sha": "a" * 40,
            "dirty": False,
            "staged_paths": [],
            "unstaged_paths": [],
            "untracked_paths": [],
            "starting_commit": B1_REPAIR_ROUND4_STARTING_COMMIT,
            "required_ancestry": {sha: True for sha in B1_REQUIRED_ANCESTOR_SHAS},
            "all_required_ancestry_verified": True,
            "rkv_submodule_sha": PINNED_RKV_UPSTREAM_REVISION,
            "expected_rkv_sha": PINNED_RKV_UPSTREAM_REVISION,
            "rkv_submodule_match": True,
        },
        "software": {"python": "3.11.15", "torch": "2.5.1"},
        "system": {
            "os": "Linux", "platform": "Linux-x86_64", "kernel_release": "6.8.0", "architecture": "x86_64",
            "cpu": "x86_64", "logical_cpu_count": 8, "total_physical_ram_bytes": 34359738368,
        },
        "gpu_evidence_cross_references": [
            "preflight.json:device", "fullkv/result.json:device_evidence", "rkv/result.json:device_evidence",
        ],
    }


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


def _envelope_payload(role: str, result_payload: dict, *, attempt_id: str = ATTEMPT_ID) -> dict:
    return {
        "role": role, "attempt_id": attempt_id, "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:01:00+00:00", "success": True,
        "requested_identities": {}, "resolved_identities": {}, "partial_measurements": result_payload,
        "determinism_policy": {"framework_seed": 13}, "software_versions": {"torch": "2.0"},
        "hardware_metadata": {}, "error_type": None, "error_message": None, "traceback": None,
        "result_sha256": sha256_json(result_payload),
    }


def _command_payload(role: str, attempt_dir: Path) -> dict:
    return {
        "argv": [
            "python", "-m", "kvcot.discovery.b2a_worker_entry",
            "--role", role, "--config", "c.yaml", "--manifest", "m.json",
            "--output", str(attempt_dir / role / "result.json"), "--attempt-id", ATTEMPT_ID,
        ],
        "timeout_seconds": B2A_WORKER_TIMEOUT_SECONDS, "check": False, "capture_output": True, "text": True,
    }


def _event(stage: str, status: str, second: int, role: str, counters=None) -> dict:
    return {
        "timestamp": f"2026-01-01T00:00:{second:02d}+00:00", "stage": stage, "status": status,
        "attempt_id": ATTEMPT_ID, "worker_role": role, "counters": counters or {}, "detail": None,
    }


def _progress_lines(role: str) -> str:
    """A production-shaped journal: ordered singleton completions, a
    started model-load event before its completion, and (rkv) exactly 12
    unique real-pair completions plus one no-op completion."""
    events = [
        _event("startup", "completed", 0, role),
        _event("config validation", "completed", 1, role),
        _event("manifest validation", "completed", 2, role),
        _event("before_model_load", "completed", 3, role),
        _event("snapshot resolution", "completed", 4, role),
        _event("tokenizer load", "completed", 5, role),
        _event("model-load start", "started", 6, role),
        _event("model-load completion", "completed", 7, role),
        _event("runtime verification", "completed", 8, role),
        _event("post_load_baseline", "completed", 9, role),
    ]
    second = 10
    if role == "rkv":
        events.append(_event("Pass 1", "completed", second, role)); second += 1
        events.append(_event("Pass 2", "completed", second, role)); second += 1
        events.append(_event("compact-target conversion", "completed", second, role)); second += 1
        for pair_index in range(12):
            events.append(_event(
                "each real pair", "completed", second, role,
                counters={"timing_phase": f"real_pair:{pair_index}:1:2"},
            ))
            second += 1
        events.append(_event("no-op", "completed", second, role, counters={"timing_phase": "no_op_pair:0:1:1"}))
        second += 1
    else:
        events.append(_event("fullkv_complete_natural_generation", "completed", second, role))
        second += 1
    events.append(_event("result construction", "completed", second, role)); second += 1
    events.append(_event("envelope construction", "completed", second, role))
    return "\n".join(json.dumps(event) for event in events) + "\n"


def _build_valid_attempt(tmp_path: Path) -> tuple[Path, dict, dict]:
    attempt_dir = tmp_path / "attempt"
    fullkv_result = _result_payload("fullkv")
    rkv_result = _result_payload("rkv")

    _write(attempt_dir / "invocation.json", {
        "attempt_id": ATTEMPT_ID,
        "started_at": "2026-01-01T00:00:00+00:00",
        "argv": ["python", "-m", "kvcot", "b2a-calibrate", "--execute"],
        "config_path": "c.yaml",
        "manifest_path": "m.json",
    })
    _write(attempt_dir / "preflight.json", {
        "passed": True,
        "config_hash": "c" * 64,
        "manifest_hash": "m" * 64,
        "device": {
            "visible_gpu_count": 1, "gpu_name": "NVIDIA GeForce RTX 3090", "device_index": 0,
            "requested_device": "cuda:0", "total_vram_bytes": 24 * 1024**3,
            "compute_capability": [8, 6], "driver_version": "550.00", "cuda_runtime": "12.1",
            "cudnn_version": "8902", "policy_satisfied": True, "verified": True,
        },
    })
    _write(attempt_dir / "provenance.json", _valid_provenance_payload())
    _write(attempt_dir / "process_outcome.json", {
        "attempt_id": ATTEMPT_ID,
        "return_codes": {"fullkv": 0, "rkv": 0},
        "timeout_state": {"fullkv": False, "rkv": False},
        "partial_success": False,
        "coordinator_observed_process_seconds": {"fullkv": 1.5, "rkv": 2.5},
    })

    for role, result in (("fullkv", fullkv_result), ("rkv", rkv_result)):
        _write(attempt_dir / role / "command.json", _command_payload(role, attempt_dir))
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


def _verify(attempt_dir, fullkv_result, rkv_result, **kwargs):
    return verify_attempt_artifacts(attempt_dir, fullkv_result=fullkv_result, rkv_result=rkv_result, **kwargs)


def test_verify_attempt_artifacts_accepts_a_genuinely_consistent_attempt(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert reasons == ()
    assert verified is True
    assert verify_worker_envelopes(attempt_dir) is True


def test_verify_attempt_artifacts_fails_on_missing_file(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    (attempt_dir / "rkv" / "replay_evidence.json").unlink()
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("missing required attempt files" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_missing_process_outcome(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    (attempt_dir / "process_outcome.json").unlink()
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("process_outcome.json" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_malformed_json(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    (attempt_dir / "fullkv" / "result.json").write_text("{not valid json", encoding="utf-8")
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("does not parse as valid JSON" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_malformed_jsonl_progress(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    (attempt_dir / "fullkv" / "progress.jsonl").write_text("{not valid json\n", encoding="utf-8")
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("progress.jsonl" in r and "does not parse" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_envelope_result_hash_mismatch(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    envelope = json.loads((attempt_dir / "rkv" / "envelope.json").read_text(encoding="utf-8"))
    envelope["result_sha256"] = "0" * 64
    _write(attempt_dir / "rkv" / "envelope.json", envelope)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("result_sha256 does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_saved_result_vs_coordinator_mismatch(tmp_path):
    """F4.3: the saved result.json must BE the coordinator-supplied result
    -- a divergent (even internally-consistent) saved copy is rejected."""
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    divergent = dict(fullkv_result)
    divergent["role"] = "fullkv"
    divergent["extra_field_never_reported_to_coordinator"] = True
    _write(attempt_dir / "fullkv" / "result.json", divergent)
    envelope = _envelope_payload("fullkv", divergent)
    _write_over(attempt_dir / "fullkv" / "envelope.json", envelope)
    _write_over(attempt_dir / "fullkv" / "timing.json", divergent["timing_evidence"])
    _write_over(attempt_dir / "fullkv" / "memory.json", divergent["memory_phase_evidence"])
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("does not match the coordinator-supplied" in r for r in reasons)


def _write_over(path: Path, payload) -> None:
    path.unlink(missing_ok=True)
    _write(path, payload)


def test_verify_attempt_artifacts_fails_on_timing_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write_over(attempt_dir / "rkv" / "timing.json", [{"phase": "tampered", "duration_seconds": 999.0, "completed": True}])
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("timing.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_memory_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write_over(attempt_dir / "fullkv" / "memory.json", [{"phase": "tampered", "peak_allocated": 1, "completed": True}])
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("memory.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_pair_identity_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write_over(attempt_dir / "rkv" / "pair_identities.json", {"attempted": [], "completed": [], "failed": [], "no_op": None})
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("pair_identities.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_replay_token_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write_over(attempt_dir / "rkv" / "replay_evidence.json", {"pass1_token_ids": [9, 9, 9]})
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("replay_evidence.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_semantic_mutation_report_tampering(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write_over(attempt_dir / "rkv" / "semantic_swaps.json", [])
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("semantic_swaps.json does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_command_role_mutation(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write_over(attempt_dir / "fullkv" / "command.json", _command_payload("rkv", attempt_dir))
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("argv does not name role" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_duplicated_or_contradictory_command_flags(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    command = _command_payload("fullkv", attempt_dir)
    command["argv"] = command["argv"] + ["--config", "other.yaml"]
    _write_over(attempt_dir / "fullkv" / "command.json", command)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("duplicates flag" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_command_missing_text_mode_or_wrong_timeout(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    command = _command_payload("rkv", attempt_dir)
    del command["text"]
    command["timeout_seconds"] = 1
    _write_over(attempt_dir / "rkv" / "command.json", command)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("'text' must be True" in r for r in reasons)
    assert any("timeout_seconds" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_return_code_or_check_true(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    bad_command = _command_payload("fullkv", attempt_dir)
    bad_command["check"] = True
    _write_over(attempt_dir / "fullkv" / "command.json", bad_command)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("check' must be False" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_nonzero_return_code_or_timeout(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    outcome = json.loads((attempt_dir / "process_outcome.json").read_text(encoding="utf-8"))
    outcome["return_codes"] = {"fullkv": 0, "rkv": 1}
    outcome["timeout_state"] = {"fullkv": False, "rkv": True}
    _write_over(attempt_dir / "process_outcome.json", outcome)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("return codes are not both 0" in r for r in reasons)
    assert any("timeout state is not both False" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_timeout_state_or_nonzero_success_envelope(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    envelope = json.loads((attempt_dir / "fullkv" / "envelope.json").read_text(encoding="utf-8"))
    envelope["success"] = False
    envelope["error_type"] = "RuntimeError"
    _write_over(attempt_dir / "fullkv" / "envelope.json", envelope)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("not a success envelope" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_attempt_id_disagreement(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    envelope = json.loads((attempt_dir / "rkv" / "envelope.json").read_text(encoding="utf-8"))
    envelope["attempt_id"] = "different-attempt-id"
    envelope["result_sha256"] = sha256_json(rkv_result)  # keep hash valid, isolate the ID check
    _write_over(attempt_dir / "rkv" / "envelope.json", envelope)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("disagree on attempt_id" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_preflight_hash_disagreement(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _verify(
        attempt_dir, fullkv_result, rkv_result,
        expected_config_hash="f" * 64, expected_manifest_hash="m" * 64,
    )
    assert verified is False
    assert any("preflight.json config_hash does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_unsanitized_argv(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    invocation = json.loads((attempt_dir / "invocation.json").read_text(encoding="utf-8"))
    invocation["argv"] = invocation["argv"] + ["--hf-token", "hf_abc123"]
    _write_over(attempt_dir / "invocation.json", invocation)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("not sanitized" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_missing_started_at(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    invocation = json.loads((attempt_dir / "invocation.json").read_text(encoding="utf-8"))
    del invocation["started_at"]
    _write_over(attempt_dir / "invocation.json", invocation)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("started_at" in r for r in reasons)


def _valid_completion_payload() -> dict:
    return {
        "attempt_id": ATTEMPT_ID, "finished_at": "2026-01-01T00:30:00+00:00",
        "outcome": "gate_passed", "exit_code": 0, "gate_passed": True,
        "intended_final_relative_path": "final.json",
        "config_hash": "c" * 64, "manifest_hash": "m" * 64,
    }


def test_verify_attempt_artifacts_validates_completion_agreement_when_present(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "completion.json", _valid_completion_payload())
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert reasons == ()
    assert verified is True

    bad = _valid_completion_payload()
    bad["exit_code"] = 2
    bad["gate_passed"] = False
    _write_over(attempt_dir / "completion.json", bad)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("disagrees with exit_code" in r for r in reasons)


# ---------------------------------------------------------------------------
# R2 (residual independent-audit repair): completion.json must fail closed --
# a missing/null/malformed/mismatched required field is always rejected,
# never silently accepted, and a self-consistent but non-"gate_passed"
# outcome (e.g. "gate_failed") is rejected by this successful-attempt
# verifier too.
# ---------------------------------------------------------------------------


def test_verify_attempt_artifacts_fails_on_missing_completion_attempt_id(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    completion = _valid_completion_payload()
    del completion["attempt_id"]
    _write(attempt_dir / "completion.json", completion)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("completion.json has no attempt_id" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_null_completion_attempt_id(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    completion = _valid_completion_payload()
    completion["attempt_id"] = None
    _write(attempt_dir / "completion.json", completion)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("completion.json has no attempt_id" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_mismatched_completion_attempt_id(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    completion = _valid_completion_payload()
    completion["attempt_id"] = "different-attempt-id"
    _write(attempt_dir / "completion.json", completion)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("completion.json attempt_id does not match invocation.json" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_missing_or_mismatched_completion_hashes(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)

    missing_config = _valid_completion_payload()
    del missing_config["config_hash"]
    _write(attempt_dir / "completion.json", missing_config)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("completion.json has no config_hash" in r for r in reasons)

    missing_manifest = _valid_completion_payload()
    del missing_manifest["manifest_hash"]
    _write_over(attempt_dir / "completion.json", missing_manifest)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("completion.json has no manifest_hash" in r for r in reasons)

    _write_over(attempt_dir / "completion.json", _valid_completion_payload())
    verified, reasons = _verify(
        attempt_dir, fullkv_result, rkv_result,
        expected_config_hash="f" * 64, expected_manifest_hash="m" * 64,
    )
    assert verified is False
    assert any("completion.json config_hash does not match" in r for r in reasons)

    verified, reasons = _verify(
        attempt_dir, fullkv_result, rkv_result,
        expected_config_hash="c" * 64, expected_manifest_hash="z" * 64,
    )
    assert verified is False
    assert any("completion.json manifest_hash does not match" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_missing_or_wrong_intended_final_relative_path(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)

    missing = _valid_completion_payload()
    del missing["intended_final_relative_path"]
    _write(attempt_dir / "completion.json", missing)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("intended_final_relative_path" in r and "final.json" in r for r in reasons)

    wrong = _valid_completion_payload()
    wrong["intended_final_relative_path"] = "not_final.json"
    _write_over(attempt_dir / "completion.json", wrong)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("intended_final_relative_path" in r and "final.json" in r for r in reasons)


def test_verify_attempt_artifacts_rejects_self_consistent_non_gate_passed_outcome(tmp_path):
    """A `gate_failed` (or `exception`) completion.json can be internally
    self-consistent (exit_code/gate_passed genuinely agree with `outcome`)
    and still must be rejected -- this is the successful-attempt verifier,
    never a generic self-consistency checker."""
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    gate_failed = _valid_completion_payload()
    gate_failed["outcome"] = "gate_failed"
    gate_failed["exit_code"] = 2
    gate_failed["gate_passed"] = False
    _write(attempt_dir / "completion.json", gate_failed)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("is not 'gate_passed'" in r for r in reasons)
    # Self-consistency itself is not violated -- only the stricter
    # successful-attempt requirement is.
    assert not any("disagrees with exit_code" in r for r in reasons)


# ---------------------------------------------------------------------------
# R3 (residual independent-audit repair): the collected provenance's
# experiment-identity and repository-integrity fields must be enforced, not
# only the narrow dirty/rkv_submodule_match/head checks that predated this
# repair.
# ---------------------------------------------------------------------------


def _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, **overrides):
    provenance = _valid_provenance_payload()
    provenance["git"].update(overrides)
    _write_over(attempt_dir / "provenance.json", provenance)
    return _verify(attempt_dir, fullkv_result, rkv_result)


def test_verify_attempt_artifacts_fails_on_wrong_branch(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, branch="main")
    assert verified is False
    assert any("branch !=" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_origin_sha_mismatch(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, origin_branch_sha="b" * 40)
    assert verified is False
    assert any("origin_branch_sha does not match head" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_non_hex_head(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, head="z" * 40)
    assert verified is False
    assert any("no 40-hex HEAD SHA" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_missing_ancestry_entry(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    incomplete = {sha: True for sha in B1_REQUIRED_ANCESTOR_SHAS}
    del incomplete[B1_REQUIRED_ANCESTOR_SHAS[0]]
    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, required_ancestry=incomplete)
    assert verified is False
    assert any("required_ancestry does not attest" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_false_ancestry_result(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    false_one = {sha: True for sha in B1_REQUIRED_ANCESTOR_SHAS}
    false_one[B1_REQUIRED_ANCESTOR_SHAS[1]] = False
    verified, reasons = _mutate_git(
        tmp_path, attempt_dir, fullkv_result, rkv_result,
        required_ancestry=false_one, all_required_ancestry_verified=False,
    )
    assert verified is False
    assert any("required_ancestry does not attest" in r for r in reasons)
    assert any("all_required_ancestry_verified is not true" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_wrong_starting_commit(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, starting_commit="f" * 40)
    assert verified is False
    assert any("starting_commit !=" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_wrong_expected_rkv_sha(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, expected_rkv_sha="d" * 40)
    assert verified is False
    assert any("expected_rkv_sha !=" in r for r in reasons)
    # rkv_submodule_sha (still the real pinned value) now disagrees with the
    # tampered expected_rkv_sha too -- both are independently reported.
    assert any("rkv_submodule_sha does not match expected_rkv_sha" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_nonempty_staged_unstaged_untracked_paths(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, staged_paths=["dirty_file.py"])
    assert verified is False
    assert any("staged_paths is not empty" in r for r in reasons)

    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, unstaged_paths=["x.py"])
    assert verified is False
    assert any("unstaged_paths is not empty" in r for r in reasons)

    verified, reasons = _mutate_git(tmp_path, attempt_dir, fullkv_result, rkv_result, untracked_paths=["y.py"])
    assert verified is False
    assert any("untracked_paths is not empty" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_missing_system_block(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    provenance = _valid_provenance_payload()
    del provenance["system"]
    _write_over(attempt_dir / "provenance.json", provenance)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("has no system evidence" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_malformed_total_ram(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    provenance = _valid_provenance_payload()
    provenance["system"]["total_physical_ram_bytes"] = -1
    _write_over(attempt_dir / "provenance.json", provenance)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("total_physical_ram_bytes is neither a positive int nor null" in r for r in reasons)

    provenance["system"]["total_physical_ram_bytes"] = None
    _write_over(attempt_dir / "provenance.json", provenance)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert reasons == ()
    assert verified is True  # honestly-null RAM is accepted, never rejected


def test_verify_attempt_artifacts_fails_on_missing_gpu_evidence_cross_reference(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    provenance = _valid_provenance_payload()
    provenance["gpu_evidence_cross_references"] = ["preflight.json:device"]
    _write_over(attempt_dir / "provenance.json", provenance)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("gpu_evidence_cross_references does not exactly name" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_missing_software_mapping(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    provenance = _valid_provenance_payload()
    provenance["software"] = {}
    _write_over(attempt_dir / "provenance.json", provenance)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("has no software version mapping" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_wrong_exit_code_or_gate_passed_despite_matching_outcome(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    bad = _valid_completion_payload()
    bad["exit_code"] = 1
    _write(attempt_dir / "completion.json", bad)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("exit_code" in r and "!= 0" in r for r in reasons)

    bad2 = _valid_completion_payload()
    bad2["gate_passed"] = False
    _write_over(attempt_dir / "completion.json", bad2)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("gate_passed" in r and "is not True" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_completion_before_invocation_start(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write(attempt_dir / "completion.json", {
        "attempt_id": ATTEMPT_ID, "finished_at": "2025-12-31T23:00:00+00:00",
        "outcome": "gate_passed", "exit_code": 0, "gate_passed": True,
    })
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("finished_at precedes" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_typed_result_validation(tmp_path):
    """The minimal fixture results are deliberately NOT schema-complete
    worker results -- `typed_results=True` (the production coordinator
    setting) must reject them."""
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result, typed_results=True)
    assert verified is False
    assert any("does not validate as a typed" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_duplicate_singleton_progress_stage(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    text = (attempt_dir / "fullkv" / "progress.jsonl").read_text(encoding="utf-8")
    duplicate = json.dumps(_event("model-load completion", "completed", 59, "fullkv")) + "\n"
    (attempt_dir / "fullkv" / "progress.jsonl").write_text(text + duplicate, encoding="utf-8")
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("completed 2 times" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_out_of_order_progress_stages(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    events = [
        json.loads(line) for line in
        (attempt_dir / "fullkv" / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    # Swap "startup" completion behind "envelope construction".
    startup = next(e for e in events if e["stage"] == "startup")
    events.remove(startup)
    events.append(startup)
    (attempt_dir / "fullkv" / "progress.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("monotonic order" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_wrong_real_pair_completion_count(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    events = [
        json.loads(line) for line in
        (attempt_dir / "rkv" / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    events = [e for e in events if not (e["stage"] == "each real pair" and e["counters"].get("timing_phase") == "real_pair:11:1:2")]
    (attempt_dir / "rkv" / "progress.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("exactly 12 unique real-pair completions" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_failure_event_in_progress(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    text = (attempt_dir / "rkv" / "progress.jsonl").read_text(encoding="utf-8")
    failure = json.dumps(_event("failed", "failed", 58, "rkv")) + "\n"
    (attempt_dir / "rkv" / "progress.jsonl").write_text(text + failure, encoding="utf-8")
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("failure event" in r for r in reasons)


def test_verify_attempt_artifacts_fails_on_incomplete_progress_stage_coverage(tmp_path):
    """Independent-audit Gate H7.4: a progress journal that is non-empty
    but missing one of the required named stages must fail."""
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    truncated = json.dumps(_event("startup", "completed", 0, "rkv")) + "\n"
    _write_over(attempt_dir / "rkv" / "progress.jsonl", truncated)
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert verified is False
    assert any("missing required stage" in r for r in reasons)


def test_verify_progress_stage_completeness_reports_missing_stages_directly():
    from kvcot.discovery.attempt_verification import verify_progress_stage_completeness

    events = [{"stage": "startup", "status": "completed"}]
    complete, missing = verify_progress_stage_completeness(events, role="fullkv")
    assert complete is False
    assert "tokenizer load" in missing
    assert "model-load completion" in missing


def test_verify_attempt_artifacts_fails_on_empty_progress_journal(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    _write_over(attempt_dir / "fullkv" / "progress.jsonl", "")
    verified, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
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


# ---------------------------------------------------------------------------
# F4.6/F5: final reference manifest verification
# ---------------------------------------------------------------------------


def _finalize_attempt(attempt_dir: Path) -> None:
    from kvcot.discovery.attempt_artifacts import AttemptDirectory, build_attempt_references

    _write(attempt_dir / "completion.json", {
        "attempt_id": ATTEMPT_ID, "finished_at": "2026-01-01T00:30:00+00:00",
        "outcome": "gate_passed", "exit_code": 0, "gate_passed": True,
    })
    attempt = AttemptDirectory(attempt_id=ATTEMPT_ID, path=attempt_dir)
    manifest = build_attempt_references(attempt, exclude=("final.json",))
    _write(attempt_dir / "final.json", {"passed": True, "attempt_artifacts": manifest})


def test_final_reference_manifest_accepts_and_references_completion(tmp_path):
    from kvcot.discovery.attempt_verification import verify_final_reference_manifest

    attempt_dir, _, _ = _build_valid_attempt(tmp_path)
    _finalize_attempt(attempt_dir)
    ok, reasons = verify_final_reference_manifest(attempt_dir)
    assert reasons == ()
    assert ok is True
    manifest = json.loads((attempt_dir / "final.json").read_text(encoding="utf-8"))["attempt_artifacts"]
    listed = [item["relative_path"] for item in manifest["files"]]
    assert "completion.json" in listed
    assert "final.json" not in listed


def test_final_reference_manifest_rejects_mutated_log_byte(tmp_path):
    from kvcot.discovery.attempt_verification import verify_final_reference_manifest

    attempt_dir, _, _ = _build_valid_attempt(tmp_path)
    _finalize_attempt(attempt_dir)
    (attempt_dir / "fullkv" / "stdout.log").write_text("tampered\n", encoding="utf-8")
    ok, reasons = verify_final_reference_manifest(attempt_dir)
    assert ok is False
    assert any("content hash changed" in r for r in reasons)


def test_final_reference_manifest_rejects_mutated_completion_record(tmp_path):
    from kvcot.discovery.attempt_verification import verify_final_reference_manifest

    attempt_dir, _, _ = _build_valid_attempt(tmp_path)
    _finalize_attempt(attempt_dir)
    (attempt_dir / "completion.json").write_text(
        json.dumps({"attempt_id": ATTEMPT_ID, "finished_at": "2026-01-01T00:30:00+00:00",
                    "outcome": "gate_passed", "exit_code": 0, "gate_passed": False}),
        encoding="utf-8",
    )
    ok, reasons = verify_final_reference_manifest(attempt_dir)
    assert ok is False
    assert any("completion.json" in r and "hash changed" in r for r in reasons)


def test_final_reference_manifest_rejects_unknown_and_missing_references(tmp_path):
    from kvcot.discovery.attempt_verification import verify_final_reference_manifest

    attempt_dir, _, _ = _build_valid_attempt(tmp_path)
    _finalize_attempt(attempt_dir)
    (attempt_dir / "unlisted_extra.json").write_text("{}", encoding="utf-8")
    ok, reasons = verify_final_reference_manifest(attempt_dir)
    assert ok is False
    assert any("unreferenced files" in r for r in reasons)

    (attempt_dir / "unlisted_extra.json").unlink()
    (attempt_dir / "rkv" / "semantic_swaps.json").unlink()
    ok, reasons = verify_final_reference_manifest(attempt_dir)
    assert ok is False
    assert any("is missing" in r for r in reasons)


def test_final_reference_manifest_rejects_progress_line_mutation(tmp_path):
    from kvcot.discovery.attempt_verification import verify_final_reference_manifest

    attempt_dir, _, _ = _build_valid_attempt(tmp_path)
    _finalize_attempt(attempt_dir)
    path = attempt_dir / "rkv" / "progress.jsonl"
    path.write_text(path.read_text(encoding="utf-8").replace("Pass 1", "Pass X"), encoding="utf-8")
    ok, reasons = verify_final_reference_manifest(attempt_dir)
    assert ok is False
    assert any("progress.jsonl" in r and "hash changed" in r for r in reasons)


# ---------------------------------------------------------------------------
# B2A-R2 repair (2026-07-22): the real B2A-R2 execute attempt
# (results/decisions/b2a_attempt_20260722T101253300941Z_..., preserved in
# docs/evidence/B2A_R2_RESULT_2026-07-22.md) failed `attempt_artifacts_
# verified` on two genuine stage names --
# `kvcot.discovery.orchestrator.run_example`'s own `operation_runner` calls
# ("pass1_plan_construction", "minimized_target_evidence_construction")
# that this module's known-stage list had never been extended to
# recognize. Confirmed by direct source inspection (orchestrator.py), not
# assumed.
# ---------------------------------------------------------------------------


def test_rkv_known_stages_recognize_the_real_orchestrator_derivation_stages():
    from kvcot.discovery.attempt_verification import RKV_KNOWN_PROGRESS_STAGES

    assert "pass1_plan_construction" in RKV_KNOWN_PROGRESS_STAGES
    assert "minimized_target_evidence_construction" in RKV_KNOWN_PROGRESS_STAGES


def test_progress_journal_accepts_the_two_previously_unknown_rkv_stages(tmp_path):
    attempt_dir, fullkv_result, rkv_result = _build_valid_attempt(tmp_path)
    path = attempt_dir / "rkv" / "progress.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    extra = [
        json.dumps(_event("pass1_plan_construction", "completed", 50, "rkv")),
        json.dumps(_event("minimized_target_evidence_construction", "completed", 51, "rkv")),
    ]
    path.write_text("\n".join(lines + extra) + "\n", encoding="utf-8")

    _, reasons = _verify(attempt_dir, fullkv_result, rkv_result)
    assert not any("unknown stage" in r for r in reasons), reasons
