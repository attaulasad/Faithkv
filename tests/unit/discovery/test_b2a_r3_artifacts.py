"""B2A-R3 qualification-artifact verification tests. Synthetic fixtures
only -- no torch, no CUDA, no real qualification artifact."""
from __future__ import annotations

import pytest

from kvcot.discovery.b2a_r3_artifacts import (
    SELECTION_STATUS_NONE_QUALIFIED,
    SELECTION_STATUS_SELECTED,
    STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED,
    STOPPED_REASON_FIRST_PASS,
    ArtifactVerificationRefused,
    QualificationArtifactR3,
    verify_qualification_artifact,
)
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
    QUALIFICATION_CANDIDATE_LIMIT,
    QUALIFICATION_PROTOCOL_VERSION,
    RUNTIME_PREDICTOR_VERSION,
    RUNTIME_SOURCE_ARTIFACT_PATH,
    RUNTIME_SOURCE_ARTIFACT_SHA256,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
)
from kvcot.discovery.b2a_r3_qualification import build_qualification_outcome
from kvcot.discovery.b2a_r3_runtime import predict_runtime
from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

from tests.unit.discovery.test_b2a_r3_qualification import _valid_evidence

CONFIG_SHA = "de8ac65a348c307c4f00089da07914666332935981bcaa7c98a150a9e7e778b3"


def _uid(ordinal: int) -> str:
    return _candidate_manifest()["candidates"][ordinal]["unique_id"]


def _candidate_manifest():
    import json
    from pathlib import Path

    return json.loads(Path(CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))


def _outcome(ordinal: int, *, qualified: bool):
    overrides = {"candidate_ordinal": ordinal}
    if not qualified:
        overrides["answer_verification_status"] = "incorrect"
    manifest = _candidate_manifest()
    overrides["candidate_manifest_canonical_sha256"] = manifest["canonical_sha256"]
    overrides["config_sha256"] = CONFIG_SHA
    evidence = _valid_evidence(**overrides)
    return build_qualification_outcome(evidence, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)


def _artifact(attempted, *, selection_status, first_ordinal, selected_unique_id):
    manifest = _candidate_manifest()
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
        "attempted": attempted,
        "attempted_candidate_count": len(attempted),
        "first_passing_candidate_ordinal": first_ordinal,
        "selected_unique_id": selected_unique_id,
        "selection_status": selection_status,
        "qualification_stopped_reason": (
            STOPPED_REASON_FIRST_PASS if first_ordinal is not None else STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED
        ),
        "authorized_maximum_candidates": (
            QUALIFICATION_CANDIDATE_LIMIT if first_ordinal is not None else len(attempted)
        ),
        "attempt_started_at_utc": "2026-07-23T00:00:00+00:00",
        "attempt_completed_at_utc": "2026-07-23T00:10:00+00:00",
    }
    payload["canonical_sha256"] = sha256_json(payload)
    return payload, manifest


def test_valid_artifact_with_selected_row_verifies():
    attempted = [_outcome(0, qualified=False), _outcome(1, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=1,
        selected_unique_id=_uid(1),
    )
    typed = verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)
    assert isinstance(typed, QualificationArtifactR3)
    assert typed.selected_unique_id == _uid(1)


def test_valid_artifact_with_no_candidate_qualified_verifies():
    attempted = [_outcome(i, qualified=False) for i in range(3)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_NONE_QUALIFIED, first_ordinal=None, selected_unique_id=None,
    )
    typed = verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)
    assert typed.selection_status == SELECTION_STATUS_NONE_QUALIFIED


def test_rejects_unknown_field():
    attempted = [_outcome(0, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,
        selected_unique_id=_uid(0),
    )
    artifact = dict(artifact)
    artifact["extra"] = 1
    with pytest.raises(Exception):
        verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)


def test_rejects_wrong_candidate_manifest_hash():
    attempted = [_outcome(0, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,
        selected_unique_id=_uid(0),
    )
    wrong_manifest = dict(manifest, x=999)
    wrong_manifest["canonical_sha256"] = sha256_json({k: v for k, v in wrong_manifest.items() if k != "canonical_sha256"})
    with pytest.raises(ArtifactVerificationRefused):
        verify_qualification_artifact(artifact, candidate_manifest=wrong_manifest, expected_config_sha256=CONFIG_SHA)


def test_rejects_wrong_config_hash():
    attempted = [_outcome(0, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,
        selected_unique_id=_uid(0),
    )
    with pytest.raises(ArtifactVerificationRefused):
        verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256="0" * 64)


def test_rejects_tampered_canonical_hash():
    attempted = [_outcome(0, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,
        selected_unique_id=_uid(0),
    )
    artifact = dict(artifact, canonical_sha256="0" * 64)
    with pytest.raises(Exception):
        verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)


def test_rejects_inconsistent_selected_ordinal():
    attempted = [_outcome(0, qualified=False), _outcome(1, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,  # WRONG -- 0 failed
        selected_unique_id=_uid(1),
    )
    artifact["canonical_sha256"] = sha256_json({k: v for k, v in artifact.items() if k != "canonical_sha256"})
    with pytest.raises(Exception):
        verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)


def test_rejects_attempted_count_mismatch():
    attempted = [_outcome(0, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,
        selected_unique_id=_uid(0),
    )
    artifact = dict(artifact)
    artifact["attempted_candidate_count"] = 5
    artifact["canonical_sha256"] = sha256_json({k: v for k, v in artifact.items() if k != "canonical_sha256"})
    with pytest.raises(Exception):
        verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)


def test_rejects_bad_timestamp_order():
    attempted = [_outcome(0, qualified=True)]
    artifact, manifest = _artifact(
        attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,
        selected_unique_id=_uid(0),
    )
    artifact = dict(artifact)
    artifact["attempt_started_at_utc"], artifact["attempt_completed_at_utc"] = (
        artifact["attempt_completed_at_utc"], artifact["attempt_started_at_utc"],
    )
    artifact["canonical_sha256"] = sha256_json({k: v for k, v in artifact.items() if k != "canonical_sha256"})
    with pytest.raises(Exception):
        verify_qualification_artifact(artifact, candidate_manifest=manifest, expected_config_sha256=CONFIG_SHA)
