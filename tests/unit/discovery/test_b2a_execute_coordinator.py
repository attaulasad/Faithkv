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
from kvcot.discovery.schemas import SwapPairRecord

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


def _swap_pair_record(*, event: int, layer: int, candidate: int, donor: int, is_noop: bool = False) -> SwapPairRecord:
    """B2A-R2 forensic repair: a valid, self-consistent SwapPairRecord for
    the coordinator fixture's real/no-op pairs -- reuses SwapPairRecord's
    own validators (never a second, hand-rolled consistency check)."""
    t = 100 + event
    baseline = [1.0] * 48
    swapped = list(baseline) if is_noop else [0.9] * 48
    swap_gain = 0.0 if is_noop else (sum(baseline) / len(baseline)) - (sum(swapped) / len(swapped))
    return SwapPairRecord(
        example_id="b2a-coordinator-fixture",
        model_revision="modelrev",
        rkv_revision="r" * 40,
        compaction_event_id=event,
        chronological_event_ordinal=event,
        depth_stratum=event,
        layer_index=layer,
        kv_head_index=0,
        event_token_absolute_position=t,
        bridge_token_absolute_position=t + 1,
        first_affected_forward_input_absolute_position=t + 1,
        first_affected_logit_target_absolute_position=t + 2,
        first_scored_absolute_position=t + 2,
        evicted_absolute_token_position=candidate,
        evicted_pre_storage_position=5,
        retained_absolute_token_position=donor,
        retained_pre_storage_position=8,
        retained_post_storage_position=8,
        score_e=0.4,
        score_r=0.6,
        score_margin_e_minus_r=-0.2,
        attention_component_diff=0.01,
        similarity_component_diff=-0.02,
        recency_diff=candidate - donor,
        key_norm_diff=0.1,
        value_norm_diff=-0.1,
        entropy_e=1.2, entropy_e_missing_reason=None,
        entropy_r=0.4, entropy_r_missing_reason=None,
        entropy_diff=0.8,
        logit_margin_e=3.0, logit_margin_e_missing_reason=None,
        logit_margin_r=5.5, logit_margin_r_missing_reason=None,
        logit_margin_diff=-2.5,
        parity_check_passed=True,
        parity_failure_reason=None,
        is_noop_control=is_noop,
        net_physical_bytes_changed=0,
        cap_hit_flag=False,
        valid_flag=True,
        invalid_reason=None,
        reference_horizon_sha256="a" * 64,
        swap_gain=swap_gain,
        baseline_per_token_nll=baseline,
        swapped_per_token_nll=swapped,
    )


def _passing_pair_records() -> list[dict]:
    real = [
        _swap_pair_record(event=event, layer=event, candidate=candidate, donor=donor)
        for event in range(3) for candidate in (10, 11) for donor in (20, 21)
    ]
    no_op = _swap_pair_record(event=0, layer=0, candidate=20, donor=20, is_noop=True)
    return [record.model_dump(mode="json") for record in real + [no_op]]


def _passing_payloads(manifest, config):
    actual_calls = [{
        "call_kind": "prefill", "input_ids_shape": [1, manifest.prompt_token_count],
        "batch_size": 1, "sequence_length": manifest.prompt_token_count, "device": "cuda:0",
        "dtype": "torch.int64", "position_ids_shape": [1, manifest.prompt_token_count],
        "cache_position_shape": [manifest.prompt_token_count],
    }]
    row_identity = {
        "dataset_repo": manifest.dataset_repo, "dataset_revision": manifest.dataset_revision,
        "example_index": manifest.example_index, "unique_id": manifest.unique_id,
        "raw_content_hash": manifest.raw_content_hash, "manifest_canonical_hash": manifest.manifest_hash(),
        "rendered_user_message_sha256": manifest.rendered_user_message_sha256,
        "chat_template_source_sha256": manifest.chat_template_source_sha256,
        "prompt_token_ids_sha256": manifest.prompt_token_ids_sha256,
        "prompt_token_count": manifest.prompt_token_count,
    }
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
        every_parameter_on_cuda=True, batch_size=1, actual_batch_size_verified=True,
        actual_call_evidence=actual_calls, dataset_row_identity=row_identity,
        timing_evidence=[
            {"phase": "fullkv_worker_startup", "duration_seconds": 0.1, "completed": True},
            {"phase": "snapshot_tokenizer_resolution", "duration_seconds": 0.2, "completed": True},
            {"phase": "tokenizer_load", "duration_seconds": 0.3, "completed": True},
            {"phase": "model_load", "duration_seconds": 2.0, "completed": True},
            {"phase": "post_load_validation", "duration_seconds": 0.05, "completed": True},
        ],
        software_versions={"torch": "2.0"},
    )
    selected_events = [
        {"compaction_event_id": event, "absolute_event_position": 100 + event, "layer_index": event, "kv_head_index": 0}
        for event in range(3)
    ]
    real_identities = [
        {
            "compaction_event_id": event, "layer_index": event, "kv_head_index": 0,
            "candidate_absolute_position": candidate, "donor_absolute_position": donor, "pair_kind": "real",
        }
        for event in range(3) for candidate in (10, 11) for donor in (20, 21)
    ]
    noop_identity = {
        "compaction_event_id": 0, "layer_index": 0, "kv_head_index": 0,
        "candidate_absolute_position": 20, "donor_absolute_position": 20, "pair_kind": "no_op",
    }
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
        semantic_swap_checks_required=12, semantic_swap_checks_attempted=12, semantic_swap_checks_passed=12,
        semantic_swap_checks_failed=0,
        unique_completed_real_pair_count=12, events_with_exactly_four_unique_real_pairs=3,
        has_duplicate_real_pair_identity=False, has_duplicate_no_op_pair_identity=False,
        selected_event_count_exact=True, real_pair_count_exact=True, no_op_count_exact=True,
        all_required_pair_evaluations_completed=True,
        observed_retention_ratio=0.4,
        wall_seconds_pass1=5.0, wall_seconds_pass2=5.0, wall_seconds_targeted_capture=0.5,
        real_pair_wall_seconds=[1.0] * 12, no_op_pair_wall_seconds=[0.5],
        determinism_policy=_DETERMINISM_POLICY, runtime_generation=_RUNTIME_GENERATION,
        runtime_generation_config_hash="g" * 64, parameter_placement=_PARAMETER_PLACEMENT,
        runtime_identity=_RUNTIME_IDENTITY_MATCH, memory=_MEMORY, minimized_target_evidence=[],
        peak_cuda_allocated_bytes=1_500_000, peak_cuda_reserved_bytes=2_500_000, every_parameter_on_cuda=True,
        batch_size=1, actual_batch_size_verified=True, actual_call_evidence=actual_calls,
        dataset_row_identity=row_identity, selected_event_evidence=selected_events,
        attempted_pair_identities=real_identities + [noop_identity],
        completed_pair_identities=real_identities + [noop_identity], failed_pair_identities=[],
        no_op_identity=noop_identity,
        timing_evidence=[
            {"phase": "rkv_worker_startup", "duration_seconds": 0.1, "completed": True},
            {"phase": "snapshot_tokenizer_resolution", "duration_seconds": 0.2, "completed": True},
            {"phase": "tokenizer_load", "duration_seconds": 0.3, "completed": True},
            {"phase": "model_load", "duration_seconds": 2.0, "completed": True},
            {"phase": "post_load_validation", "duration_seconds": 0.05, "completed": True},
        ],
        software_versions={"torch": "2.0"},
        # B2A-R2 forensic repair: RKVWorkerResultV2 requires this field --
        # 12 real + 1 no-op, matching real_identities/noop_identity above.
        pair_records=_passing_pair_records(),
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


def _fake_fetched_row(manifest):
    from kvcot.discovery.manifest_prepare import FetchedDatasetRow

    return FetchedDatasetRow(
        row={"problem": "fake problem", "answer": manifest.gold_answer, "unique_id": manifest.unique_id},
        raw_content_hash=manifest.raw_content_hash,
    )


def test_verify_resolved_prompt_identity_fails_closed_when_local_tokenizer_snapshot_is_unavailable(
    config, manifest, monkeypatch
):
    """Independent-audit Gate H4.5: the B2A execute path must fail if the
    exact local tokenizer snapshot is unavailable -- never silently fall
    back to resolving the tokenizer through a network-capable path."""
    from kvcot.discovery import manifest_prepare, snapshot_boundary

    monkeypatch.setattr(manifest_prepare, "_fetch_pinned_dataset_row", lambda *a, **k: _fake_fetched_row(manifest))
    monkeypatch.setattr(manifest_prepare, "_verify_row_schema", lambda row: None)

    def boom(*args, **kwargs):
        raise snapshot_boundary.SnapshotBoundaryError("exact local snapshot is unavailable")

    monkeypatch.setattr(snapshot_boundary, "resolve_local_snapshot", boom)

    with pytest.raises(b2a_execute.B2AExecutionRefused, match="exact local tokenizer snapshot unavailable"):
        b2a_execute._verify_resolved_prompt_identity(config, manifest)


def test_verify_resolved_prompt_identity_loads_tokenizer_from_the_exact_verified_local_snapshot(
    config, manifest, monkeypatch
):
    """The tokenizer must be loaded from the EXACT verified local snapshot
    path (`local_only_path`), not `tokenizer_name`/`tokenizer_revision`
    resolved through an ordinary (potentially network-touching) lookup."""
    from kvcot.discovery import manifest_prepare, snapshot_boundary

    monkeypatch.setattr(manifest_prepare, "_fetch_pinned_dataset_row", lambda *a, **k: _fake_fetched_row(manifest))
    monkeypatch.setattr(manifest_prepare, "_verify_row_schema", lambda row: None)

    fake_snapshot = SimpleNamespace(local_path="/verified/local/tokenizer/snapshot")
    monkeypatch.setattr(snapshot_boundary, "resolve_local_snapshot", lambda *a, **k: fake_snapshot)

    captured = {}

    class _StopEarly(Exception):
        pass

    def spying_render_and_tokenize(row, tokenizer_name, tokenizer_revision, *, local_only_path=None):
        captured["local_only_path"] = local_only_path
        captured["tokenizer_name"] = tokenizer_name
        raise _StopEarly()

    monkeypatch.setattr(manifest_prepare, "_render_and_tokenize", spying_render_and_tokenize)

    with pytest.raises(_StopEarly):
        b2a_execute._verify_resolved_prompt_identity(config, manifest)

    assert captured["local_only_path"] == "/verified/local/tokenizer/snapshot"
    assert captured["tokenizer_name"] == config.model.tokenizer_name


def test_startup_and_load_projection_sums_all_five_one_time_phases(config, manifest, monkeypatch, tmp_path):
    """Independent-audit Gate H2.4: the startup/load projection component
    used to sum only `{role}_worker_startup` + `model_load` (2.1s here:
    0.1 + 2.0) -- it must now include `snapshot_tokenizer_resolution`
    (0.2s), `tokenizer_load` (0.3s), and `post_load_validation` (0.05s) too,
    for a total of 2.65s per worker, never silently undercounting one-time
    setup cost incurred before measured inference."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    projection = payload["runtime_projection"]
    assert projection["fullkv_startup_and_model_load_seconds"] == pytest.approx(2.65)
    assert projection["rkv_startup_and_model_load_seconds"] == pytest.approx(2.65)


def test_startup_and_load_projection_fails_closed_when_a_one_time_phase_is_missing(config, manifest, monkeypatch, tmp_path):
    """A worker payload missing one of the five required one-time setup
    phases (here, `post_load_validation`) must refuse execution rather than
    silently projecting an undercounted total."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    fullkv_payload["timing_evidence"] = [
        record for record in fullkv_payload["timing_evidence"] if record["phase"] != "post_load_validation"
    ]
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    with pytest.raises(b2a_execute.B2AExecutionRefused, match="post_load_validation"):
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=runner,
        )


def test_process_overhead_diagnostic_is_exported_without_double_counting(config, manifest, monkeypatch, tmp_path):
    """Independent-audit Gate H2.5: coordinator-observed process duration,
    worker-internal startup/inference durations, and the derived
    unattributed overhead are all exported as a separate diagnostic --
    never folded into `runtime_projection` itself (which stays exactly the
    frozen §12 formula)."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )
    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    diagnostic = payload["worker_processes"]["process_overhead_diagnostic"]
    assert set(diagnostic) == {"fullkv", "rkv"}
    for role in ("fullkv", "rkv"):
        entry = diagnostic[role]
        assert entry["coordinator_observed_process_seconds"] >= 0.0
        assert entry["worker_internal_startup_and_load_seconds"] == pytest.approx(2.65)
        assert entry["unattributed_process_overhead_seconds"] == pytest.approx(
            entry["coordinator_observed_process_seconds"]
            - entry["worker_internal_startup_and_load_seconds"]
            - entry["worker_internal_inference_seconds"]
        )
    # `runtime_projection` itself must be untouched by this diagnostic --
    # still exactly the frozen §12 formula's fields, nothing summed twice.
    # (B2A-R1 zero-event repair, 2026-07-22, adds the explicit
    # available/unavailable_reason/observed-vs-required-count fields and
    # promotes `projected_complete_pilot_gpu_hours` into the projection
    # itself -- it does not touch or duplicate any of the original ten.)
    assert set(payload["runtime_projection"]) == {
        "fullkv_startup_and_model_load_seconds", "rkv_startup_and_model_load_seconds",
        "fullkv_natural_generation_seconds", "rkv_pass1_seconds", "rkv_pass2_seconds",
        "per_example_inference_seconds", "example_count", "conservative_real_pair_seconds",
        "real_pair_count", "projected_total_seconds",
        "available", "unavailable_reason", "observed_real_pair_duration_count",
        "required_real_pair_duration_count", "projected_complete_pilot_gpu_hours",
    }


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
    """B1B-R4.1 §18/§30, tightened by B1 execution-boundary closure §12: a
    worker reporting one failed semantic-swap check (`semantic_swap_checks_
    passed < semantic_swap_checks_attempted`) must fail the gate's
    dedicated `semantic_swap_parity` condition -- proven with every OTHER
    condition (including the exact-count ones) left passing, so this is not
    merely riding on `all_required_pair_evaluations_completed` already
    being false for an unrelated reason. The gate now derives this from
    POSITIVE counts, not from scanning `pair_failure_details` for a
    specific stage string -- both the failure-detail record AND the
    corresponding count are set here, matching what a real worker would
    report together."""
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
    rkv_payload["semantic_swap_checks_passed"] = 11
    rkv_payload["semantic_swap_checks_failed"] = 1
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


def test_gate_fails_when_semantic_swap_checks_were_never_attempted_despite_no_failure_record(config, manifest, monkeypatch, tmp_path):
    """B1 execution-boundary closure §12: a worker that reports ZERO
    attempted semantic-swap checks (e.g. every real pair failed earlier, at
    candidate/donor pool lookup, before the swap was ever attempted) must
    fail `semantic_swap_parity` even though `pair_failure_details` contains
    no `semantic_swap_parity_failure` entry at all -- the old
    absence-of-failure derivation would have vacuously PASSED this case."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["semantic_swap_checks_attempted"] = 0
    rkv_payload["semantic_swap_checks_passed"] = 0
    rkv_payload["semantic_swap_checks_failed"] = 0
    assert rkv_payload["pair_failure_details"] == []  # no failure record naming this check at all
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )

    assert artifact.gate_result.passed is False
    assert "semantic_swap_parity" in artifact.gate_result.failed_conditions
    assert "all_required_pair_evaluations_completed" not in artifact.gate_result.failed_conditions


def test_gate_fails_on_duplicate_pair_identity_even_though_the_bare_count_is_exact(config, manifest, monkeypatch, tmp_path):
    """B1 execution-boundary closure §13: a worker reporting
    `has_duplicate_real_pair_identity=True` must fail the gate's dedicated
    `unique_real_pair_count_exact`/`no_duplicate_pair_identity` conditions
    even while every bare COUNT (`attempted_real_pair_count`,
    `completed_real_pair_count`, `real_pair_count_exact`) stays at the
    'passing' value of 12 -- the exact scenario the prior `count >= 4`
    derivation could not detect (four records per event, but not four
    DISTINCT identities)."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["unique_completed_real_pair_count"] = 11  # one duplicate among the 12 records
    rkv_payload["events_with_exactly_four_unique_real_pairs"] = 2  # the event with the duplicate has only 3 unique
    rkv_payload["has_duplicate_real_pair_identity"] = True
    duplicate = dict(rkv_payload["completed_pair_identities"][0])
    rkv_payload["completed_pair_identities"][1] = duplicate
    rkv_payload["attempted_pair_identities"][1] = duplicate
    runner = _fake_runner_writing(fullkv_payload, rkv_payload)
    monkeypatch.setattr("kvcot.discovery.b2a_artifact.build_and_write_b2a_artifact", _patched_writer(tmp_path))

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner,
    )

    assert artifact.gate_result.passed is False
    assert "unique_real_pair_count_exact" in artifact.gate_result.failed_conditions
    assert "events_with_four_unique_pairs_exact" in artifact.gate_result.failed_conditions
    assert "no_duplicate_pair_identity" in artifact.gate_result.failed_conditions
    # The bare-count condition this replaces/augments is untouched --
    # proving the new conditions are independently derived.
    assert "real_pair_count_exact" not in artifact.gate_result.failed_conditions


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


# ---------------------------------------------------------------------------
# R4 (residual independent-audit repair): production must independently
# verify the COMPLETED `final.json` after writing it, not merely trust that
# the write itself succeeded.
# ---------------------------------------------------------------------------


def _real_attempt_directory(tmp_path) -> Path:
    from kvcot.discovery.attempt_artifacts import create_attempt_directory

    return create_attempt_directory(root=tmp_path / "decisions").path


def _install_fake_coordination(monkeypatch, fullkv_payload, rkv_payload):
    """Bypasses `run_both_workers_via_subprocess`'s full atomic-envelope
    subprocess contract (already covered elsewhere, e.g.
    `test_b2a_workers.py`) -- this test is scoped to R4's post-final-write
    verification integration, not worker-subprocess plumbing. Still writes
    a real `process_outcome.json` into the real `attempt_directory` so the
    rest of the coordinator's attempt-directory bookkeeping runs for real."""
    from kvcot.discovery import b2a_workers
    from kvcot.discovery.attempt_artifacts import atomic_write_json
    from kvcot.discovery.b2a_workers import FullKVWorkerResult, RKVWorkerResult, WorkerCoordinationResult

    def fake_run_both_workers_via_subprocess(config_path, manifest_path, *, python_executable, subprocess_runner, attempt_directory):
        atomic_write_json(Path(attempt_directory) / "process_outcome.json", {
            "attempt_id": Path(attempt_directory).name.rsplit("_", 1)[-1],
            "return_codes": {"fullkv": 0, "rkv": 0},
            "timeout_state": {"fullkv": False, "rkv": False},
            "partial_success": False,
            "coordinator_observed_process_seconds": {"fullkv": 1.0, "rkv": 1.0},
        })
        return WorkerCoordinationResult(
            fullkv=FullKVWorkerResult.model_validate(fullkv_payload),
            rkv=RKVWorkerResult.model_validate(rkv_payload),
            shared_identity_ok=True, shared_identity_mismatches=(),
            attempt_directory=str(attempt_directory), return_codes={"fullkv": 0, "rkv": 0},
            timeout_state={"fullkv": False, "rkv": False}, partial_success=False,
            coordinator_observed_process_seconds={"fullkv": 1.0, "rkv": 1.0},
        )

    monkeypatch.setattr(b2a_workers, "run_both_workers_via_subprocess", fake_run_both_workers_via_subprocess)


def test_final_manifest_is_verified_and_a_valid_artifact_succeeds(config, manifest, monkeypatch, tmp_path):
    """Proves production actually CALLS the final verifier after writing
    `final.json` (via a spy) and that a genuinely valid final artifact
    still succeeds -- the new call is not merely present but inert."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    _install_fake_coordination(monkeypatch, fullkv_payload, rkv_payload)

    from kvcot.discovery import attempt_verification

    real_verify = attempt_verification.verify_final_reference_manifest
    calls = {"n": 0}

    def spying_verify(directory):
        calls["n"] += 1
        assert directory == attempt_directory
        return real_verify(directory)

    monkeypatch.setattr(attempt_verification, "verify_final_reference_manifest", spying_verify)

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=None, attempt_directory=attempt_directory,
    )

    assert calls["n"] == 1
    assert artifact.artifact_path == attempt_directory / "final.json"
    assert not (attempt_directory / "final_verification_failure.json").exists()


def test_final_verification_failure_raises_writes_failure_record_and_preserves_prior_artifacts(
    config, manifest, monkeypatch, tmp_path
):
    """A `final.json` that fails post-write verification (simulated here via
    a controlled fake -- `verify_final_reference_manifest` itself is
    already exhaustively tested against real corruption in
    `test_attempt_verification.py`) must: raise (never return a successful
    `B2ACalibrationArtifact`), write `final_verification_failure.json`, and
    leave every prior artifact -- including `final.json` and
    `completion.json` themselves -- byte-for-byte unchanged."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    _install_fake_coordination(monkeypatch, fullkv_payload, rkv_payload)

    from kvcot.discovery import attempt_verification

    monkeypatch.setattr(
        attempt_verification, "verify_final_reference_manifest",
        lambda directory: (False, ("injected: final.json reference manifest does not match disk",)),
    )

    with pytest.raises(b2a_execute.B2AFinalVerificationError, match="injected: final.json reference manifest"):
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=None, attempt_directory=attempt_directory,
        )

    final_path = attempt_directory / "final.json"
    completion_path = attempt_directory / "completion.json"
    assert final_path.is_file()
    assert completion_path.is_file()
    final_bytes_before = final_path.read_bytes()
    completion_bytes_before = completion_path.read_bytes()

    failure_path = attempt_directory / "final_verification_failure.json"
    assert failure_path.is_file()
    failure_record = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure_record["reasons"] == ["injected: final.json reference manifest does not match disk"]
    assert failure_record["intended_final_relative_path"] == "final.json"

    # Nothing already written was overwritten by the failure handling.
    assert final_path.read_bytes() == final_bytes_before
    assert completion_path.read_bytes() == completion_bytes_before


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


# ---------------------------------------------------------------------------
# B2A-R1 zero-event coordinator repair (2026-07-22): the actual consumed
# B2A-R1 attempt (results/decisions/b2a_attempt_20260722T072823470986Z_...,
# preserved in docs/evidence/B2A_R1_ATTEMPT_INDEX_2026-07-22.json) produced
# zero compaction events (prompt=105 tokens, generated=449 tokens, processed
# length ~554, well under R-KV budget=1024) and crashed the coordinator with
# an uncaught `ValueError: runtime projection requires exactly 12 B2A
# real-pair durations` -- writing only `failure.json`, never `completion
# .json`/`final.json`. This must now be a clean, internally consistent
# gate-failed attempt: `completion.json` with outcome="gate_failed",
# exit_code=2, gate_passed=False; `final.json` written and independently
# verified; the runtime projection explicitly unavailable, never fabricated.
# ---------------------------------------------------------------------------


def _zero_compaction_event_payloads(manifest, config):
    """Same base fixture as `_passing_payloads`, with every compaction/
    event/pair-count field zeroed out exactly as a real zero-event attempt
    reports them -- Pass 1/Pass 2 replay integrity, device/snapshot
    identity, and generation identity are all UNCHANGED (those genuinely
    still succeeded; only the swap/pair/event counts are zero because no
    eviction ever fired)."""
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload.update(
        observed_total_compaction_events=0,
        eligible_compaction_events=0,
        selected_compaction_events=0,
        events_with_at_least_one_completed_real_pair=0,
        events_with_all_four_real_pairs_completed=0,
        attempted_real_pair_count=0,
        completed_real_pair_count=0,
        failed_real_pair_count=0,
        attempted_no_op_pair_count=0,
        completed_no_op_pair_count=0,
        pair_failure_details=[],
        semantic_swap_checks_required=0,
        semantic_swap_checks_attempted=0,
        semantic_swap_checks_passed=0,
        semantic_swap_checks_failed=0,
        unique_completed_real_pair_count=0,
        events_with_exactly_four_unique_real_pairs=0,
        has_duplicate_real_pair_identity=False,
        has_duplicate_no_op_pair_identity=False,
        selected_event_count_exact=False,
        real_pair_count_exact=False,
        no_op_count_exact=False,
        all_required_pair_evaluations_completed=False,
        # No compression occurred at all -- the realistic retention ratio
        # for zero evictions is 1.0 (nothing evicted), not the passing
        # fixture's 0.4.
        observed_retention_ratio=1.0,
        real_pair_wall_seconds=[],
        no_op_pair_wall_seconds=[],
        selected_event_evidence=[],
        attempted_pair_identities=[],
        completed_pair_identities=[],
        failed_pair_identities=[],
        no_op_identity=None,
        semantic_mutation_reports=[],
        no_op_evidence={},
        # B2A-R2 forensic repair: zero events selected means zero pairs were
        # ever attempted -- an honestly empty population, never the passing
        # fixture's 13 fake records left dangling against zeroed identities.
        pair_records=[],
    )
    return fullkv_payload, rkv_payload


def test_zero_compaction_events_produce_clean_gate_failed_completion_not_exception(
    config, manifest, monkeypatch, tmp_path
):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _zero_compaction_event_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    _install_fake_coordination(monkeypatch, fullkv_payload, rkv_payload)

    # The defect this repairs: this call used to raise an uncaught
    # `ValueError` here. It must now return normally.
    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=None, attempt_directory=attempt_directory,
    )

    assert artifact.gate_result.passed is False
    assert "semantic_swap_parity" in artifact.gate_result.failed_conditions
    assert "real_pair_count_exact" in artifact.gate_result.failed_conditions
    assert "no_op_count_exact" in artifact.gate_result.failed_conditions
    assert "selected_event_count_exact" in artifact.gate_result.failed_conditions
    assert "sufficient_eligible_events" in artifact.gate_result.failed_conditions
    assert "meaningful_compression_observed" in artifact.gate_result.failed_conditions
    assert "all_required_pair_evaluations_completed" in artifact.gate_result.failed_conditions
    assert "runtime_within_limit" in artifact.gate_result.failed_conditions

    completion_path = attempt_directory / "completion.json"
    assert completion_path.is_file()
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["outcome"] == "gate_failed"
    assert completion["exit_code"] == 2
    assert completion["gate_passed"] is False

    final_path = attempt_directory / "final.json"
    assert final_path.is_file()
    assert artifact.artifact_path == final_path

    from kvcot.discovery.attempt_verification import verify_final_reference_manifest

    final_verified, final_reasons = verify_final_reference_manifest(attempt_directory)
    assert final_verified is True, final_reasons
    assert not (attempt_directory / "final_verification_failure.json").exists()
    assert not (attempt_directory / "failure.json").exists()


def test_zero_compaction_events_runtime_projection_is_unavailable_never_fabricated(
    config, manifest, monkeypatch, tmp_path
):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _zero_compaction_event_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    _install_fake_coordination(monkeypatch, fullkv_payload, rkv_payload)

    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=None, attempt_directory=attempt_directory,
    )

    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    projection = payload["runtime_projection"]
    assert projection["available"] is False
    assert projection["unavailable_reason"] == "insufficient_real_pair_durations"
    assert projection["observed_real_pair_duration_count"] == 0
    assert projection["required_real_pair_duration_count"] == 12
    # Never fabricated as 0.0, inf, or the 4.00-hour limit itself.
    assert projection["conservative_real_pair_seconds"] is None
    assert projection["projected_total_seconds"] is None
    assert projection["projected_complete_pilot_gpu_hours"] is None
    assert payload["measurement"]["per_real_pair_seconds"] is None
    assert payload["measurement"]["projected_complete_pilot_gpu_hours"] is None


def test_malformed_real_pair_durations_still_raise_and_write_failure_not_gate_failed(
    config, manifest, monkeypatch, tmp_path
):
    """Contrast case: a genuinely malformed measurement (a present but
    negative real-pair duration -- never actually producible by a correct
    worker) remains a hard coordinator exception, never silently
    downgraded to a clean gate-failed completion."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    rkv_payload["real_pair_wall_seconds"] = [1.0] * 11 + [-5.0]
    attempt_directory = _real_attempt_directory(tmp_path)
    _install_fake_coordination(monkeypatch, fullkv_payload, rkv_payload)

    with pytest.raises(ValueError, match="finite and positive"):
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=None, attempt_directory=attempt_directory,
        )

    assert (attempt_directory / "failure.json").is_file()
    assert not (attempt_directory / "completion.json").exists()
    assert not (attempt_directory / "final.json").exists()


# ---------------------------------------------------------------------------
# B2A-R2 forensic pair-record persistence repair, audit round 2
# (docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md): proves the
# pair-record gate is MANDATORY at the coordinator level -- overall_passed,
# exit_code, and completion.json outcome -- never only the standalone
# verifier's own (False, reasons) return. Each test forces
# `final_gate_result.passed=True` via a monkeypatched
# `evaluate_final_gates` -- this is NOT a real end-to-end "everything
# passes" fixture (no test in this file builds `invocation.json`/
# `preflight.json`/`provenance.json`/`progress.jsonl`, since those are
# written by a still-higher CLI layer this function assumes already ran --
# see `test_verify_attempt_artifacts_accepts_a_genuinely_consistent_attempt`
# in `test_attempt_verification.py` for that boundary's OWN, separately-
# scoped coverage). Patching `evaluate_final_gates` isolates exactly the
# causal claim under test: with the OTHER 31 final-gate conditions and the
# legacy 28-condition gate (`_passing_payloads`, already proven passing by
# `test_full_coordinator_flow_passes_and_writes_pass_artifact` above) held
# constant, does a defective pair-record artifact alone flip
# `overall_passed` to `False` and `exit_code` to `2`?
# ---------------------------------------------------------------------------


def _force_final_gate_passing(monkeypatch):
    from kvcot.discovery import final_contract

    def fake_evaluate_final_gates(conditions):
        return final_contract.FinalGateResult(passed=True, conditions=dict(conditions), failed_conditions=())

    monkeypatch.setattr(final_contract, "evaluate_final_gates", fake_evaluate_final_gates)


def _fake_runner_writing_with_envelopes(fullkv_payload, rkv_payload, *, attempt_id="forensic-r2-test"):
    """Like `_fake_runner_writing`, but ALSO writes the mandatory atomic
    envelope sibling file (`kvcot.discovery.worker_envelope`) -- required
    once `attempt_directory` triggers `preserve=True` in the real
    `run_both_workers_via_subprocess`, hashed from the SAME canonical
    (typed, round-tripped) form the coordinator itself recomputes, never
    the raw payload dict."""

    def runner(argv, **kwargs):
        role = argv[argv.index("--role") + 1]
        output_path = Path(argv[argv.index("--output") + 1])
        payload = fullkv_payload if role == "fullkv" else rkv_payload

        from kvcot.discovery.b2a_workers import FullKVWorkerResult, RKVWorkerResult
        from kvcot.discovery.worker_envelope import build_success_envelope, write_worker_envelope

        model = FullKVWorkerResult if role == "fullkv" else RKVWorkerResult
        canonical_payload = model.model_validate(payload).model_dump(mode="json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(canonical_payload), encoding="utf-8")
        envelope = build_success_envelope(
            role=role, attempt_id=attempt_id, started_at="2026-01-01T00:00:00+00:00",
            requested_identities={}, resolved_identities={}, result_payload=canonical_payload,
            determinism_policy=None, software_versions={"torch": "2.0"}, hardware_metadata={},
        )
        write_worker_envelope(envelope, output_path)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return runner


def _corrupt_after_real_write(monkeypatch, attempt_directory: Path, **corruptions: bool):
    """Wraps the REAL `run_both_workers_via_subprocess` (never a fake
    return value) so `rkv/pair_records.json`/`rkv/scientific_summary.json`
    are genuinely, correctly written by production code first, THEN
    corrupted -- proving the coordinator's own gate catches post-write
    corruption, not merely a hand-built bad fixture."""
    from kvcot.discovery import b2a_workers

    real_fn = b2a_workers.run_both_workers_via_subprocess

    def wrapper(*args, **kwargs):
        result = real_fn(*args, **kwargs)
        if corruptions.get("delete_pair_records"):
            (attempt_directory / "rkv" / "pair_records.json").unlink()
        if corruptions.get("delete_summary"):
            (attempt_directory / "rkv" / "scientific_summary.json").unlink()
        if corruptions.get("corrupt_summary"):
            summary_path = attempt_directory / "rkv" / "scientific_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["mean_swap_gain"] = -999.0
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return result

    monkeypatch.setattr(b2a_workers, "run_both_workers_via_subprocess", wrapper)


def _run_forced_isolated(config, manifest, monkeypatch, attempt_directory, fullkv_payload, rkv_payload):
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    _force_final_gate_passing(monkeypatch)
    runner = _fake_runner_writing_with_envelopes(fullkv_payload, rkv_payload)
    return b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner, attempt_directory=attempt_directory,
    )


def _assert_gate_failed_on_pair_records(artifact, attempt_directory, *, reason_substring: str | None = None):
    assert artifact.gate_result.passed is True  # legacy gate genuinely passed
    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    assert payload["scientific_pair_artifacts_verified"] is False
    completion = json.loads((attempt_directory / "completion.json").read_text(encoding="utf-8"))
    assert completion["outcome"] == "gate_failed"
    assert completion["exit_code"] == 2
    assert completion["gate_passed"] is False
    if reason_substring is not None:
        assert any(reason_substring in r for r in payload["pair_record_verification"]["reasons"])
    return payload


def test_valid_pair_artifacts_do_not_spuriously_fail_the_new_gate(config, manifest, monkeypatch, tmp_path):
    """Control case: with everything else forced-passing and genuinely
    correct pair-record artifacts, the new gate does not itself cause a
    failure -- `scientific_pair_artifacts_verified` is True.

    Also forces `verify_attempt_artifacts` (pre-final verification)
    passing -- this test's fake runner does not produce
    `invocation.json`/`preflight.json`/`provenance.json`/`progress.jsonl`
    (written by a still-higher CLI layer no test in this file replicates,
    see the section docstring above); without this, an otherwise-True
    `overall_passed` would trip the UNRELATED
    `if overall_passed and not prefinal_ok: raise` guard for reasons that
    have nothing to do with pair records."""
    from kvcot.discovery import attempt_verification

    monkeypatch.setattr(attempt_verification, "verify_attempt_artifacts", lambda *a, **k: (True, ()))
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    artifact = _run_forced_isolated(config, manifest, monkeypatch, attempt_directory, fullkv_payload, rkv_payload)
    payload = json.loads(artifact.artifact_path.read_text(encoding="utf-8"))
    assert payload["scientific_pair_artifacts_verified"] is True
    assert payload["pair_record_verification"]["verified"] is True
    completion = json.loads((attempt_directory / "completion.json").read_text(encoding="utf-8"))
    assert completion["outcome"] == "gate_passed"
    assert completion["exit_code"] == 0
    assert completion["gate_passed"] is True


def test_missing_pair_records_file_forces_gate_failure(config, manifest, monkeypatch, tmp_path):
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    _force_final_gate_passing(monkeypatch)
    _corrupt_after_real_write(monkeypatch, attempt_directory, delete_pair_records=True)
    runner = _fake_runner_writing_with_envelopes(fullkv_payload, rkv_payload)
    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner, attempt_directory=attempt_directory,
    )
    _assert_gate_failed_on_pair_records(artifact, attempt_directory, reason_substring="pair_records.json is missing")


def test_missing_scientific_summary_file_forces_gate_failure(config, manifest, monkeypatch, tmp_path):
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    _force_final_gate_passing(monkeypatch)
    _corrupt_after_real_write(monkeypatch, attempt_directory, delete_summary=True)
    runner = _fake_runner_writing_with_envelopes(fullkv_payload, rkv_payload)
    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner, attempt_directory=attempt_directory,
    )
    _assert_gate_failed_on_pair_records(
        artifact, attempt_directory, reason_substring="scientific_summary.json is missing"
    )


def test_incomplete_pair_record_population_forces_gate_failure(config, manifest, monkeypatch, tmp_path):
    """Drops one real record from `pair_records` while leaving the legacy
    count/identity fields (`completed_real_pair_count`,
    `completed_pair_identities`, etc.) untouched -- isolates a defect the
    LEGACY gate structurally cannot see (it never inspects `pair_records`
    at all), proving the new gate catches something genuinely new."""
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    incomplete = rkv_payload["pair_records"][:-2] + rkv_payload["pair_records"][-1:]  # drop one real record
    rkv_payload = dict(rkv_payload, pair_records=incomplete)
    attempt_directory = _real_attempt_directory(tmp_path)
    artifact = _run_forced_isolated(config, manifest, monkeypatch, attempt_directory, fullkv_payload, rkv_payload)
    _assert_gate_failed_on_pair_records(artifact, attempt_directory, reason_substring="real records, expected 12")


def test_duplicate_pair_record_identity_forces_gate_failure(config, manifest, monkeypatch, tmp_path):
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    duplicated = list(rkv_payload["pair_records"])
    duplicated[-1] = dict(duplicated[0])  # overwrite the no-op slot with a duplicate real record
    rkv_payload = dict(rkv_payload, pair_records=duplicated)
    attempt_directory = _real_attempt_directory(tmp_path)
    artifact = _run_forced_isolated(config, manifest, monkeypatch, attempt_directory, fullkv_payload, rkv_payload)
    _assert_gate_failed_on_pair_records(artifact, attempt_directory, reason_substring="duplicate pair identities")


def test_completed_identity_mismatch_forces_gate_failure(config, manifest, monkeypatch, tmp_path):
    """Mutates `layer_index` (never `compaction_event_id`) on one
    `completed_pair_identities` entry -- deliberately chosen so the LEGACY
    gate's own identity conditions (`unique_real_pair_count_exact`,
    `events_with_four_unique_pairs_exact`, which group by
    `compaction_event_id` only) remain unaffected and still genuinely
    pass, isolating the NEW cross-artifact identity check specifically."""
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    mismatched_identities = [dict(item) for item in rkv_payload["completed_pair_identities"]]
    mismatched_identities[0]["layer_index"] = 999
    rkv_payload = dict(rkv_payload, completed_pair_identities=mismatched_identities)
    attempt_directory = _real_attempt_directory(tmp_path)
    artifact = _run_forced_isolated(config, manifest, monkeypatch, attempt_directory, fullkv_payload, rkv_payload)
    assert "unique_real_pair_count_exact" not in artifact.gate_result.failed_conditions
    assert "events_with_four_unique_pairs_exact" not in artifact.gate_result.failed_conditions
    _assert_gate_failed_on_pair_records(artifact, attempt_directory, reason_substring="do not exactly match")


def test_corrupt_scientific_summary_forces_gate_failure(config, manifest, monkeypatch, tmp_path):
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    attempt_directory = _real_attempt_directory(tmp_path)
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    _force_final_gate_passing(monkeypatch)
    _corrupt_after_real_write(monkeypatch, attempt_directory, corrupt_summary=True)
    runner = _fake_runner_writing_with_envelopes(fullkv_payload, rkv_payload)
    artifact = b2a_execute.run_b2a_calibration(
        config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
        python_executable="fake-python", subprocess_runner=runner, attempt_directory=attempt_directory,
    )
    _assert_gate_failed_on_pair_records(artifact, attempt_directory, reason_substring="does not recompute exactly")


def test_v2_schema_version_with_missing_pair_records_never_reaches_completion(config, manifest, monkeypatch, tmp_path):
    """A worker payload explicitly labeled `schema_version=
    "rkv_worker_result.v2"` but missing `pair_records` must never be
    silently accepted (as V1, or otherwise) -- the coordinator's existing,
    unmodified `RKVWorkerResult.model_validate_json` (strict V2) rejects it
    outright, before completion.json is ever written, and the attempt is
    recorded as a hard failure, never a false success."""
    monkeypatch.setattr(b2a_execute, "_verify_resolved_prompt_identity", lambda c, m: None)
    fullkv_payload, rkv_payload = _passing_payloads(manifest, config)
    del rkv_payload["pair_records"]
    rkv_payload["schema_version"] = "rkv_worker_result.v2"
    attempt_directory = _real_attempt_directory(tmp_path)

    def runner(argv, **kwargs):
        role = argv[argv.index("--role") + 1]
        output_path = Path(argv[argv.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = fullkv_payload if role == "fullkv" else rkv_payload
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    with pytest.raises(Exception):  # pydantic ValidationError, propagated as a hard coordinator failure
        b2a_execute.run_b2a_calibration(
            config, manifest, config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH,
            python_executable="fake-python", subprocess_runner=runner, attempt_directory=attempt_directory,
        )

    assert not (attempt_directory / "completion.json").exists()
    assert not (attempt_directory / "final.json").exists()


def test_parse_rkv_worker_result_rejects_v2_schema_version_missing_pair_records():
    """Unit-level companion to the coordinator test above, directly against
    `parse_rkv_worker_result` (the standalone historical-blob parser) --
    must raise, never silently fall back to V1."""
    from pydantic import ValidationError

    from kvcot.discovery.b2a_workers import parse_rkv_worker_result

    with pytest.raises(ValidationError):
        parse_rkv_worker_result({"role": "rkv", "schema_version": "rkv_worker_result.v2"})


def test_parse_rkv_worker_result_rejects_unknown_schema_version():
    from kvcot.discovery.b2a_workers import UnknownRKVWorkerResultSchemaVersion, parse_rkv_worker_result

    with pytest.raises(UnknownRKVWorkerResultSchemaVersion):
        parse_rkv_worker_result({"role": "rkv", "schema_version": "rkv_worker_result.v99_never_existed"})
