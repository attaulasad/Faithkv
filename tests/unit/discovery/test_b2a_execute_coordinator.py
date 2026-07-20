"""CPU-only mocked end-to-end test of `kvcot.discovery.b2a_execute
.run_b2a_calibration`'s COMPLETE control flow (B1B-R4 §8-§12/§16/§21) --
prompt-identity verification and the subprocess launch are both faked
(never a real network fetch, tokenizer load, Python subprocess, or GPU
access), but the REAL evidence producer (`kvcot.discovery.b2a_evidence`),
gate evaluator (`kvcot.discovery.b2a_contract.evaluate_b2a_gate`), and
artifact writer (`kvcot.discovery.b2a_artifact`) all execute for real. This
does NOT mock by returning a preconstructed passing `B2AGateResult` -- every
field is derived from the fake workers' JSON payloads exactly as a real run
would."""
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

_DETERMINISM_POLICY = dict(
    framework_seed=13, python_random_seeded=True, torch_cpu_seeded=True, torch_cuda_seeded=True,
    cudnn_deterministic_requested=True, attention_backend="flash_attention_2",
    bitwise_determinism_guaranteed=False, tolerance_note="note",
)
_RUNTIME_GENERATION = dict(
    generation_mode="greedy", do_sample=False, temperature=None, top_p=None, batch_size=1, max_new_tokens=48,
    eos_token_id=99, eos_append_feed_policy="p", one_prefill_policy="p", single_token_decode_policy="p",
    attention_backend="flash_attention_2", cache_implementation="DynamicCache", framework_seed=13,
    prompt_token_count=3,
)
_PARAMETER_PLACEMENT = dict(
    unique_device_types=["cuda"], every_parameter_on_cuda=True, hf_device_map=None, no_offload_verified=True,
    parameter_count=100,
)
_RUNTIME_IDENTITY_MATCH = dict(
    requested_model_revision="modelrev", resolved_model_revision="modelrev", model_revision_match=True,
    requested_tokenizer_revision="tokrev", resolved_tokenizer_revision="tokrev", tokenizer_revision_match=True,
)
_MEMORY = dict(
    allocated_before_reset_bytes=100, reserved_before_reset_bytes=200, peak_allocated_bytes=1_500_000,
    peak_reserved_bytes=2_500_000, reset_point="after_model_and_tokenizer_load_before_measured_inference",
)


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
        prompt_token_count=manifest.prompt_token_count,
        natural_generated_token_ids=[1, 2, 3], natural_answer="4", natural_answer_status="correct",
        cap_hit=False, prefill_call_count=1, decode_call_count=3, call_boundary_trace_hash="t" * 64,
        wall_seconds=12.0,
        determinism_policy=_DETERMINISM_POLICY, runtime_generation=_RUNTIME_GENERATION,
        runtime_generation_config_hash="g" * 64, parameter_placement=_PARAMETER_PLACEMENT,
        runtime_identity=_RUNTIME_IDENTITY_MATCH, memory=_MEMORY,
        peak_cuda_allocated_bytes=1_000_000, peak_cuda_reserved_bytes=2_000_000,
        every_parameter_on_cuda=True, batch_size=1, software_versions={"torch": "2.0"},
    )
    rkv = dict(
        role="rkv", model_revision=config.model.revision, tokenizer_revision=config.model.tokenizer_revision,
        dataset_repo=manifest.dataset_repo, dataset_revision=manifest.dataset_revision,
        manifest_hash=manifest.manifest_hash(), prompt_token_ids_sha256=manifest.prompt_token_ids_sha256,
        prompt_token_count=manifest.prompt_token_count,
        rkv_upstream_revision=config.rkv.upstream_revision, runtime_rkv_config_hash="h" * 64,
        frozen_rkv_config_hash="h" * 64, rkv_config_hash_match=True,
        example_valid=True, natural_answer_status="correct",
        token_identical_replay=True, prefill_decode_boundary_parity=True, compaction_position_equality=True,
        capture_gather_parity=True, absolute_position_parity=True, no_op_numerical_parity=True,
        pass1_call_boundary={"prefill_call_count": 1, "prefill_token_count": 3, "decode_call_count": 300, "ordered_trace_hash": "a" * 64},
        pass2_call_boundary={"prefill_call_count": 1, "prefill_token_count": 3, "decode_call_count": 300, "ordered_trace_hash": "a" * 64},
        observed_total_compaction_events=5, eligible_compaction_events=3, selected_compaction_events=3,
        events_with_at_least_one_completed_real_pair=3,
        events_with_all_four_real_pairs_completed=3, attempted_real_pair_count=12, completed_real_pair_count=12,
        failed_real_pair_count=0, attempted_no_op_pair_count=1, completed_no_op_pair_count=1,
        pair_failure_details=[],
        selected_event_count_exact=True, real_pair_count_exact=True, no_op_count_exact=True,
        all_required_pair_evaluations_completed=True,
        observed_retention_ratio=0.4,
        wall_seconds_pass1=5.0, wall_seconds_pass2=5.0, wall_seconds_targeted_capture=0.5,
        real_pair_wall_seconds=[1.0] * 12, no_op_pair_wall_seconds=[0.5],
        determinism_policy=_DETERMINISM_POLICY, runtime_generation=_RUNTIME_GENERATION,
        runtime_generation_config_hash="g" * 64, parameter_placement=_PARAMETER_PLACEMENT,
        runtime_identity=_RUNTIME_IDENTITY_MATCH, memory=_MEMORY, minimized_target_evidence=[],
        peak_cuda_allocated_bytes=1_500_000, peak_cuda_reserved_bytes=2_500_000, every_parameter_on_cuda=True,
        batch_size=1, software_versions={"torch": "2.0"},
    )
    return fullkv, rkv


def _fake_runner_writing(fullkv_payload, rkv_payload, fail_role=None):
    def runner(argv, **kwargs):
        role = argv[argv.index("--role") + 1]
        output_path = Path(argv[argv.index("--output") + 1])
        if role == fail_role:
            return SimpleNamespace(returncode=1, stdout="", stderr="simulated failure")
        payload = fullkv_payload if role == "fullkv" else rkv_payload
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return runner


def _patched_writer(tmp_path):
    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact as real_writer

    def patched_writer(payload, config_hash, manifest_hash, **kwargs):
        return real_writer(payload, config_hash, manifest_hash, directory=tmp_path)

    return patched_writer


def test_full_coordinator_flow_passes_and_writes_pass_artifact(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)

    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

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
    assert payload["measurement"]["per_real_pair_seconds"] == 1.0  # max() of 12 identical 1.0s pairs
    assert payload["b2a_real_pair_evaluations_total"] == 12
    assert payload["b2a_noop_pair_evaluations_total"] == 1


def test_gate_fails_when_rkv_trajectory_parity_false(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["token_identical_replay"] = False
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )

    assert artifact.gate_result.passed is False
    assert "token_identical_replay" in artifact.gate_result.failed_conditions
    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    assert payload["passed"] is False


def test_gate_fails_when_pair_counts_are_not_exact(config, manifest, monkeypatch, tmp_path):
    """B1B-R4 §21 regression: a worker that reports fewer than 12 completed
    real pairs must fail the gate even if every OTHER condition passes."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["completed_real_pair_count"] = 11
    rkv_payload["real_pair_count_exact"] = False
    rkv_payload["all_required_pair_evaluations_completed"] = False
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert artifact.gate_result.passed is False
    assert "real_pair_count_exact" in artifact.gate_result.failed_conditions
    assert "all_required_pair_evaluations_completed" in artifact.gate_result.failed_conditions


def test_gate_fails_on_a_reported_semantic_swap_parity_failure_even_with_every_count_exact(config, manifest, monkeypatch, tmp_path):
    """B1B-R4.1 §18/§30 regression: a worker reporting one
    `semantic_swap_parity_failure` entry in `pair_failure_details` must fail
    the gate's dedicated `semantic_swap_parity` condition -- proven with
    every OTHER condition (including the exact-count ones) left passing, so
    this is not merely riding on `all_required_pair_evaluations_completed`
    already being false for an unrelated reason."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["pair_failure_details"] = [
        {
            "compaction_event_id": 0, "layer_index": 1, "kv_head_index": 0,
            "evicted_absolute_position": 10, "donor_absolute_position": 20,
            "pair_kind": "real", "stage": "semantic_swap_parity_failure",
            "detail": "semantic_swap_parity_provenance_not_updated", "elapsed_seconds": 0.1,
        }
    ]
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )

    assert artifact.gate_result.passed is False
    assert "semantic_swap_parity" in artifact.gate_result.failed_conditions
    # Every count-based condition is untouched by this failure mode --
    # proving `semantic_swap_parity` is independently derived, not a proxy
    # for a count mismatch that would already fail the gate on its own.
    assert "real_pair_count_exact" not in artifact.gate_result.failed_conditions
    assert "all_required_pair_evaluations_completed" not in artifact.gate_result.failed_conditions


def test_worker_failure_still_writes_a_fail_artifact(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload, fail_role="rkv")
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

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
    # B1B-R4 §16: partial FullKV evidence must be preserved.
    assert "partial_fullkv_worker" in payload
    assert payload["partial_fullkv_worker"]["role"] == "fullkv"


def test_fullkv_failure_before_any_output_writes_fail_artifact_without_partial(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload, fail_role="fullkv")
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    with pytest.raises(Exception, match="fullkv worker exited"):
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=runner,
        )
    written = list(tmp_path.glob("b2a_*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert "partial_fullkv_worker" not in payload


def test_shared_identity_mismatch_fails_the_gate(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["manifest_hash"] = "totally-different-hash".ljust(64, "0")
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert artifact.gate_result.passed is False
    assert "manifest_hash_match" in artifact.gate_result.failed_conditions


def test_vram_gate_uses_max_of_allocated_and_reserved_across_both_workers(config, manifest, monkeypatch, tmp_path):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    over_bytes = int(23 * 1024**3)  # over the 22 GiB threshold
    rkv_payload["peak_cuda_reserved_bytes"] = over_bytes
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert artifact.gate_result.passed is False
    assert "peak_vram_within_limit" in artifact.gate_result.failed_conditions


def test_runtime_identity_mismatch_fails_model_revision_match(config, manifest, monkeypatch, tmp_path):
    """B1B-R4 §9 regression: `model_revision_match` now derives from BOTH
    workers' resolved-vs-requested runtime identity, never a structural
    assumption."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["runtime_identity"] = dict(_RUNTIME_IDENTITY_MATCH, resolved_model_revision=None, model_revision_match=False)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert artifact.gate_result.passed is False
    assert "model_revision_match" in artifact.gate_result.failed_conditions


def test_both_workers_agreeing_with_each_other_but_not_the_manifest_still_fails_the_gate(config, manifest, monkeypatch, tmp_path):
    """B1B-R4 §9 self-review finding: `dataset_revision_match` etc. must
    check EACH worker against the coordinator's own expected (manifest)
    value, not merely that the two workers agree with each other -- two
    workers that agree on a WRONG shared value must still fail."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    wrong_hash = "wrong-manifest-hash".ljust(64, "0")
    fullkv_payload["manifest_hash"] = wrong_hash
    rkv_payload["manifest_hash"] = wrong_hash  # BOTH workers agree with each other...
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    # ...but neither matches the manifest's own real hash, so the gate must
    # still fail.
    assert artifact.gate_result.passed is False
    assert "manifest_hash_match" in artifact.gate_result.failed_conditions


def test_no_offload_gate_uses_the_stronger_device_map_check_not_just_every_parameter_on_cuda(config, manifest, monkeypatch, tmp_path):
    """B1B-R4 §24 self-review finding: a worker whose per-parameter walk
    reports every parameter on cuda (`every_parameter_on_cuda=True`) but
    whose `hf_device_map` reveals an offloaded entry
    (`parameter_placement.no_offload_verified=False`) must still fail the
    `no_offload_verified` gate condition -- the weaker top-level field must
    never be used in its place."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["parameter_placement"] = dict(
        _PARAMETER_PLACEMENT, no_offload_verified=False, hf_device_map={"layer.27": "cpu"},
    )
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert artifact.gate_result.passed is False
    assert "no_offload_verified" in artifact.gate_result.failed_conditions


def test_generation_config_hash_mismatch_fails_the_gate(config, manifest, monkeypatch, tmp_path):
    """B1B-R4 §10 regression: `generation_config_hash_match` is now a real
    comparison between the two workers' independently-computed hashes,
    never a literal `True`."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["runtime_generation_config_hash"] = "different" + "g" * 56
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    assert artifact.gate_result.passed is False
    assert "generation_config_hash_match" in artifact.gate_result.failed_conditions


def test_prompt_identity_refusal_before_any_worker_launch_still_writes_fail_artifact(config, manifest, monkeypatch, tmp_path):
    def _boom(c, m):
        raise b2a_execute.B2AExecutionRefused("simulated prompt identity mismatch")

    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", _boom)

    def runner_that_must_not_be_called(argv, **kwargs):
        raise AssertionError("subprocess must never be launched when prompt identity verification fails first")

    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    with pytest.raises(b2a_execute.B2AExecutionRefused):
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=runner_that_must_not_be_called,
        )

    written = list(tmp_path.glob("b2a_*.json"))
    assert len(written) == 1
