"""Focused regressions for the Step 3 Stage-A independent-audit repairs.

Every artifact is synthetic and canonically rehashed after tampering.  The
candidate manifest is the committed, fully strict Stage-A manifest; no tiny
self-hashed stand-in is used except in the explicit fake-manifest rejection
test.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from kvcot.config import config_identity
from kvcot.analysis.rkv_schedule import predicted_compaction_event_positions
from kvcot.discovery.b2a_r3_artifacts import SELECTION_STATUS_SELECTED, verify_qualification_artifact
from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_contract import (
    BUDGET,
    CANDIDATE_MANIFEST_PATH,
    CANDIDATE_ORDER_PROTOCOL_VERSION,
    CONFIG_PATH,
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SPLIT,
    GENERATION_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    QUALIFICATION_MEMORY_LIMIT_BYTES,
    QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
    QUALIFICATION_PROTOCOL_VERSION,
    RUNTIME_PREDICTOR_VERSION,
    RUNTIME_SOURCE_ARTIFACT_PATH,
    RUNTIME_SOURCE_ARTIFACT_SHA256,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
)
from kvcot.discovery.b2a_r3_qualification import (
    B2AR3FullKVQualificationEvidence,
    build_qualification_outcome,
    evaluate_b2a_r3_qualification_conditions,
)
from kvcot.discovery.b2a_r3_runtime import predict_runtime
from kvcot.discovery.pass1 import eligible_event_positions
from kvcot.utils.hashing import sha256_int_ids, sha256_json


CONFIG_SHA = "de8ac65a348c307c4f00089da07914666332935981bcaa7c98a150a9e7e778b3"
DIVIDE_LENGTH = 128


def test_frozen_config_byte_identity_has_platform_independent_checkout_contract():
    attributes = Path(".gitattributes").read_text(encoding="utf-8")
    assert "configs/discovery/llama8b_math500_b1024.yaml text eol=crlf" in attributes.splitlines()
    config_bytes = Path(CONFIG_PATH).read_bytes()
    assert b"\r\n" in config_bytes
    assert b"\n" not in config_bytes.replace(b"\r\n", b"")
    assert config_identity(CONFIG_PATH) == CONFIG_SHA == _manifest()["config_sha256"]


def _manifest() -> dict:
    return json.loads(Path(CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))


def _rehash(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload.pop("canonical_sha256", None)
    payload["canonical_sha256"] = sha256_json(payload)
    return payload


def _timing() -> list[dict]:
    """Step 3R4-Repair-2 (repairs independent-re-audit Blocking Finding 2):
    the TIMING evidence the REAL `run_fullkv_worker` body actually emits --
    `before_model_load` and `post_load_baseline` genuinely appear here (its
    `measured()` helper times AND memory-samples in one call), and
    `answer_verification` is nested BEFORE `fullkv_complete_natural_generation`
    (called from inside the `run_natural_pass1` call that phase wraps),
    never after. See `_memory()` below for memory-phase evidence, validated
    separately."""
    # Seven singleton phases before the decode loop, one second apart.
    phases = (
        "before_model_load",
        "fullkv_worker_startup",
        "snapshot_tokenizer_resolution",
        "tokenizer_load",
        "model_load",
        "post_load_validation",
        "post_load_baseline",
        "fullkv_prefill",
    )
    records = []
    for index, phase in enumerate(phases):
        records.append(
            {
                "phase": phase,
                "started_at": float(index),
                "ended_at": float(index + 1),
                "duration_seconds": 1.0,
                "synchronize_before_start": True,
                "synchronize_before_end": True,
                "completed": True,
                "failure_type": None,
                "failure_message": None,
            }
        )
    records.append(
        {
            "phase": "fullkv_decode",
            "started_at": 8.0,
            "ended_at": 9.0,
            "duration_seconds": 1.0,
            "synchronize_before_start": True,
            "synchronize_before_end": True,
            "completed": True,
            "failure_type": None,
            "failure_message": None,
        }
    )
    records.extend(
        [
            {
                "phase": "answer_verification",
                "started_at": 9.0,
                "ended_at": 10.0,
                "duration_seconds": 1.0,
                "synchronize_before_start": True,
                "synchronize_before_end": True,
                "completed": True,
                "failure_type": None,
                "failure_message": None,
            },
            {
                "phase": "fullkv_complete_natural_generation",
                "started_at": 7.0,
                "ended_at": 10.0,
                "duration_seconds": 3.0,
                "synchronize_before_start": True,
                "synchronize_before_end": True,
                "completed": True,
                "failure_type": None,
                "failure_message": None,
            },
            {
                "phase": "fullkv_complete_worker",
                "started_at": 0.0,
                "ended_at": 10.0,
                "duration_seconds": 10.0,
                "synchronize_before_start": True,
                "synchronize_before_end": True,
                "completed": True,
                "failure_type": None,
                "failure_message": None,
            },
        ]
    )
    return records


def _memory() -> list[dict]:
    """Step 3R4-Repair-2: the FullKV MEMORY-phase evidence the REAL
    `run_fullkv_worker` body actually emits (7 phases -- `tokenizer_load`
    and `post_load_validation` are genuinely memory-sampled too, not just
    the 5 the historical two-worker gate requires), validated separately
    from timing above via `fullkv_qualification_memory_complete`."""
    phases = (
        "before_model_load",
        "tokenizer_load",
        "model_load",
        "post_load_validation",
        "post_load_baseline",
        "fullkv_complete_natural_generation",
        "fullkv_complete_worker",
    )
    return [
        {
            "phase": phase,
            "allocated_before": 0,
            "reserved_before": 0,
            "peak_allocated": 1_000,
            "peak_reserved": 2_000,
            "allocated_after": 500,
            "reserved_after": 1_000,
            "reset_point": "after_model_and_tokenizer_load_before_measured_inference",
            "synchronized_before": True,
            "synchronized_after": True,
            "completed": True,
        }
        for phase in phases
    ]


def _evidence(**overrides) -> B2AR3FullKVQualificationEvidence:
    manifest = _manifest()
    candidate = manifest["candidates"][0]
    generated = list(range(2000))
    total_len = 100 + len(generated)
    positions = predicted_compaction_event_positions(100, total_len - 1, BUDGET, DIVIDE_LENGTH)
    eligible = eligible_event_positions(positions, prompt_length=100, total_len=total_len)
    fields = {
        "candidate_ordinal": candidate["candidate_ordinal"],
        "source_example_index": candidate["source_example_index"],
        "unique_id": candidate["unique_id"],
        "row": candidate["row"],
        "raw_row_sha256": candidate["raw_row_sha256"],
        "problem_sha256": candidate["problem_sha256"],
        "gold_answer_sha256": candidate["gold_answer_sha256"],
        "worker_dataset_repo": DATASET_REPO,
        "worker_dataset_config": DATASET_CONFIG,
        "worker_dataset_split": DATASET_SPLIT,
        "worker_dataset_revision": DATASET_REVISION,
        "worker_model_name": MODEL_NAME,
        "worker_model_revision": MODEL_REVISION,
        "worker_tokenizer_name": TOKENIZER_NAME,
        "worker_tokenizer_revision": TOKENIZER_REVISION,
        "expected_prompt_token_ids_sha256": "1" * 64,
        "observed_prompt_token_ids_sha256": "1" * 64,
        "prompt_token_count": 100,
        "natural_generated_token_ids": generated,
        "generated_token_count": len(generated),
        "generated_token_ids_sha256": sha256_int_ids(generated),
        "cap_hit": False,
        "extracted_answer": candidate["row"]["answer"],
        "answer_verification_status": "correct",
        "think_parse_status": "ok",
        "think_start_index": 0,
        "think_end_index": 100,
        "generation_prompt_preopened_think": False,
        "fullkv_wall_seconds": 3.0,
        "fullkv_timing_evidence": _timing(),
        "requested_device": "cuda:0",
        "parameter_placement_evidence": {
            "requested_device": "cuda:0",
            "every_parameter_on_cuda": True,
            "no_offload_verified": True,
            "parameter_count": 100,
            "unique_device_types": ["cuda"],
            "unique_devices": ["cuda:0"],
            "hf_device_map": None,
        },
        "actual_batch_size": 1,
        "peak_cuda_allocated_bytes": 1_000,
        "peak_cuda_reserved_bytes": 2_000,
        "predicted_compaction_event_positions": positions,
        "predicted_event_count": len(positions),
        "eligible_event_indices": eligible,
        "eligible_event_count": len(eligible),
        "worker_generation_config_sha256": GENERATION_CONFIG_SHA256,
        "runtime_prediction": predict_runtime(total_len).to_json(),
        "candidate_manifest_canonical_sha256": manifest["canonical_sha256"],
        "config_sha256": CONFIG_SHA,
    }
    fields.update(overrides)
    return B2AR3FullKVQualificationEvidence.model_validate(fields)


def _artifact() -> dict:
    manifest = _manifest()
    outcome = build_qualification_outcome(
        _evidence(), candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA
    )
    payload = {
        "artifact_schema_version": QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
        "candidate_order_protocol_version": CANDIDATE_ORDER_PROTOCOL_VERSION,
        "qualification_protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "runtime_predictor_version": RUNTIME_PREDICTOR_VERSION,
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH,
        "candidate_manifest_canonical_sha256": manifest["canonical_sha256"],
        "config_path": CONFIG_PATH,
        "config_sha256": CONFIG_SHA,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "dataset_repo": DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": DATASET_REVISION,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "budget": BUDGET,
        "runtime_source_artifact_path": RUNTIME_SOURCE_ARTIFACT_PATH,
        "runtime_source_artifact_sha256": RUNTIME_SOURCE_ARTIFACT_SHA256,
        "attempted": [outcome],
        "attempted_candidate_count": 1,
        "first_passing_candidate_ordinal": 0,
        "selected_unique_id": outcome["unique_id"],
        "selection_status": SELECTION_STATUS_SELECTED,
        "qualification_stopped_reason": "first_pass",
        "authorized_maximum_candidates": 8,
        "attempt_started_at_utc": "2026-07-23T00:00:00+00:00",
        "attempt_completed_at_utc": "2026-07-23T00:01:00+00:00",
    }
    return _rehash(payload)


def test_canonically_rehashed_fabricated_condition_map_is_rejected():
    artifact = _artifact()
    artifact["attempted"][0]["answer_verification_status"] = "incorrect"
    artifact = _rehash(artifact)
    with pytest.raises(Exception):
        verify_qualification_artifact(
            artifact, candidate_manifest=_manifest(), expected_config_sha256=CONFIG_SHA
        )


def test_canonically_rehashed_fabricated_schedule_is_rejected():
    artifact = _artifact()
    artifact["attempted"][0]["predicted_compaction_event_positions"][2] += 1
    artifact = _rehash(artifact)
    with pytest.raises(Exception):
        verify_qualification_artifact(
            artifact, candidate_manifest=_manifest(), expected_config_sha256=CONFIG_SHA
        )


def test_minimal_self_hashed_candidate_manifest_is_rejected_by_evaluator():
    fake = {"dataset_revision": DATASET_REVISION, "x": 1}
    fake["canonical_sha256"] = sha256_json(fake)
    evidence = _evidence(candidate_manifest_canonical_sha256=fake["canonical_sha256"])
    with pytest.raises(Exception):
        evaluate_b2a_r3_qualification_conditions(
            evidence, candidate_manifest=fake, expected_config_sha256=CONFIG_SHA
        )


def test_generated_token_hash_mismatch_is_provenance_corruption():
    with pytest.raises(Exception):
        _evidence(generated_token_ids_sha256="0" * 64)


def test_alternate_self_consistent_dataset_identity_is_rejected():
    manifest = _manifest()
    manifest["dataset_repo"] = "alternate/example"
    manifest = _rehash(manifest)
    with pytest.raises(Exception):
        verify_candidate_manifest_structure(manifest)


def test_claim_consumption_without_verified_preconditions_is_rejected(tmp_path):
    import inspect

    from kvcot.discovery.b2a_r3_authorization import claim_authorization

    signature = inspect.signature(claim_authorization)
    assert "verified_context" in signature.parameters
    # Step 3R4 Finding 4: claims_root no longer exists as a parameter at
    # all -- the claim path is always derived internally from
    # repository_root + the deterministic global_claim_path.
    assert "claims_root" not in signature.parameters
    assert "repository_root" in signature.parameters
    assert "git_state" in signature.parameters
    with pytest.raises(TypeError):
        claim_authorization({}, repository_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_arbitrary_post_claim_config_allowlist_is_rejected():
    from kvcot.discovery.b2a_r3_provenance import WorktreeStatus, verify_attempt_provenance
    from tests.unit.discovery.test_b2a_r3_provenance import FakeGitState, _policy

    dirty = WorktreeStatus((), (), ("configs/smuggled.json",))
    ok, _reasons = verify_attempt_provenance(_policy(), FakeGitState(status=dirty))
    assert ok is False


def test_renderer_prompt_hash_mismatch_is_rejected():
    from kvcot.discovery.b2a_r3_freeze import PromptRenderingResult, construct_selected_manifest_and_provenance
    from tests.unit.discovery.test_b2a_r3_freeze import _fake_renderer, _valid_chain

    candidate_manifest, artifact = _valid_chain()

    def bad_renderer(row):
        valid = _fake_renderer(row)
        values = valid.__dict__.copy()
        values["prompt_token_ids_sha256"] = "0" * 64
        return PromptRenderingResult(**values)

    with pytest.raises(Exception):
        construct_selected_manifest_and_provenance(
            candidate_manifest=candidate_manifest,
            qualification_artifact=artifact,
            expected_config_sha256=candidate_manifest["config_sha256"],
            tokenizer_renderer=bad_renderer,
        )


def test_selection_provenance_semantic_ordinal_mismatch_is_rejected():
    from kvcot.discovery.b2a_r3_freeze import (
        construct_selected_manifest_and_provenance,
        verify_selection_provenance,
    )
    from tests.unit.discovery.test_b2a_r3_freeze import _fake_renderer, _valid_chain

    candidate_manifest, artifact = _valid_chain()
    selected_manifest, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=artifact,
        expected_config_sha256=candidate_manifest["config_sha256"],
        tokenizer_renderer=_fake_renderer,
    )
    provenance["selected_ordinal"] = 1
    provenance = _rehash(provenance)
    with pytest.raises(Exception):
        verify_selection_provenance(
            provenance,
            selected_manifest=selected_manifest,
            candidate_manifest=candidate_manifest,
            qualification_artifact=artifact,
            expected_config_sha256=candidate_manifest["config_sha256"],
        )


@pytest.mark.parametrize(
    "case",
    ["cap_hit", "batch_size", "peak_memory", "prompt_hash", "runtime_projection"],
)
def test_canonically_rehashed_raw_gate_contradictions_are_rejected(case):
    artifact = _artifact()
    outcome = artifact["attempted"][0]
    if case == "cap_hit":
        outcome["cap_hit"] = True
    elif case == "batch_size":
        outcome["actual_batch_size"] = 2
    elif case == "peak_memory":
        value = QUALIFICATION_MEMORY_LIMIT_BYTES + 1
        outcome["peak_cuda_allocated_bytes"] = value
        outcome["peak_cuda_tracked_bytes"] = value
    elif case == "prompt_hash":
        outcome["observed_prompt_token_ids_sha256"] = "2" * 64
    else:
        runtime = predict_runtime(2776).to_json()
        for field in (
            "reference_seconds_per_token",
            "predicted_example_seconds",
            "predicted_pair_seconds",
            "projected_total_seconds",
            "projected_gpu_hours",
            "safety_multiplier",
            "runtime_predictor_version",
        ):
            outcome[field] = runtime[field]
    with pytest.raises(Exception):
        verify_qualification_artifact(
            _rehash(artifact), candidate_manifest=_manifest(), expected_config_sha256=CONFIG_SHA
        )


@pytest.mark.parametrize(
    "case",
    ["changed", "reordered", "count", "fabricated_eligible", "ineligible_future", "alternate_consistent"],
)
def test_every_canonically_rehashed_schedule_corruption_is_rejected(case):
    artifact = _artifact()
    outcome = artifact["attempted"][0]
    if case == "changed":
        outcome["predicted_compaction_event_positions"][2] += 1
    elif case == "reordered":
        outcome["predicted_compaction_event_positions"][1:3] = reversed(
            outcome["predicted_compaction_event_positions"][1:3]
        )
    elif case == "count":
        outcome["predicted_event_count"] += 1
    elif case == "fabricated_eligible":
        outcome["eligible_event_indices"][0] = 0
    elif case == "ineligible_future":
        outcome["eligible_event_indices"].append(outcome["predicted_event_count"] - 1)
        outcome["eligible_event_count"] += 1
    else:
        count = outcome["predicted_event_count"]
        outcome["predicted_compaction_event_positions"] = list(range(2000, 2000 + count))
    with pytest.raises(Exception):
        verify_qualification_artifact(
            _rehash(artifact), candidate_manifest=_manifest(), expected_config_sha256=CONFIG_SHA
        )


@pytest.mark.parametrize(
    "case",
    [
        "generic",
        "missing",
        "duplicate",
        "negative",
        "nan",
        "infinity",
        "backwards",
        "duration",
        "incomplete",
        "failure",
        "sync",
        "order",
        "wall",
    ],
)
def test_canonically_rehashed_incomplete_or_corrupt_timing_is_rejected(case):
    artifact = _artifact()
    outcome = artifact["attempted"][0]
    timing = outcome["fullkv_timing_evidence"]
    if case == "generic":
        timing[:] = [{**timing[0], "phase": "generation"}]
    elif case == "missing":
        timing[:] = [row for row in timing if row["phase"] != "model_load"]
    elif case == "duplicate":
        timing.insert(4, copy.deepcopy(next(row for row in timing if row["phase"] == "model_load")))
    elif case == "negative":
        timing[0]["duration_seconds"] = -1.0
    elif case == "nan":
        timing[0]["duration_seconds"] = float("nan")
    elif case == "infinity":
        timing[0]["duration_seconds"] = float("inf")
    elif case == "backwards":
        timing[0]["ended_at"] = timing[0]["started_at"] - 1.0
    elif case == "duration":
        timing[0]["duration_seconds"] = 0.5
    elif case == "incomplete":
        timing[0]["completed"] = False
    elif case == "failure":
        timing[0]["failure_type"] = "RuntimeError"
        timing[0]["failure_message"] = "boom"
    elif case == "sync":
        timing[0]["synchronize_before_end"] = False
    elif case == "order":
        timing[1], timing[2] = timing[2], timing[1]
    else:
        outcome["fullkv_wall_seconds"] += 1.0
    with pytest.raises(Exception):
        verify_qualification_artifact(
            _rehash(artifact), candidate_manifest=_manifest(), expected_config_sha256=CONFIG_SHA
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_repo", "alternate/example"),
        ("dataset_revision", "0" * 40),
        ("model_revision", "0" * 40),
        ("tokenizer_revision", "0" * 40),
        ("budget", 2048),
        ("config_path", "configs/alternate.yaml"),
        ("config_sha256", "0" * 64),
        ("config_sha256", "A" * 64),
        ("budget", "1024"),
    ],
)
def test_canonically_rehashed_frozen_manifest_identity_substitution_is_rejected(field, value):
    manifest = _manifest()
    manifest[field] = value
    with pytest.raises(Exception):
        verify_candidate_manifest_structure(_rehash(manifest), expected_config_sha256=CONFIG_SHA)


@pytest.mark.parametrize("case", ["id", "hash", "count", "reorder", "empty", "bool"])
def test_canonically_rehashed_generated_token_corruption_is_rejected(case):
    artifact = _artifact()
    outcome = artifact["attempted"][0]
    if case == "id":
        outcome["natural_generated_token_ids"][0] += 1
    elif case == "hash":
        outcome["generated_token_ids_sha256"] = "0" * 64
    elif case == "count":
        outcome["generated_token_count"] += 1
        outcome["total_processed_tokens"] += 1
    elif case == "reorder":
        outcome["natural_generated_token_ids"][0:2] = reversed(outcome["natural_generated_token_ids"][0:2])
    elif case == "empty":
        outcome["natural_generated_token_ids"] = []
        outcome["generated_token_count"] = 0
        outcome["total_processed_tokens"] = outcome["prompt_token_count"]
    else:
        outcome["natural_generated_token_ids"][0] = True
    with pytest.raises(Exception):
        verify_qualification_artifact(
            _rehash(artifact), candidate_manifest=_manifest(), expected_config_sha256=CONFIG_SHA
        )
