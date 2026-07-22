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
