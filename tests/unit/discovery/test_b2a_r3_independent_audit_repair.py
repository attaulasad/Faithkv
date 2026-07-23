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


def _manifest() -> dict:
    return json.loads(Path(CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))


def _rehash(payload: dict) -> dict:
    payload = copy.deepcopy(payload)
    payload.pop("canonical_sha256", None)
    payload["canonical_sha256"] = sha256_json(payload)
    return payload


def _timing() -> list[dict]:
    phases = (
        "before_model_load",
        "fullkv_worker_startup",
        "snapshot_tokenizer_resolution",
        "tokenizer_load",
        "model_load",
        "post_load_validation",
        "post_load_baseline",
        "fullkv_prefill",
        "fullkv_decode",
        "answer_verification",
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
    records.extend(
        [
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
                "ended_at": 11.0,
                "duration_seconds": 11.0,
                "synchronize_before_start": True,
                "synchronize_before_end": True,
                "completed": True,
                "failure_type": None,
                "failure_message": None,
            },
        ]
    )
    return records


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
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
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
