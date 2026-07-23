"""B2A-R3 atomic authorization-claim tests. Every claim in this file is
written under `tmp_path` -- no real claim is ever created at
`results/decisions/b2a_r3_authorization_claims/`. No torch, no CUDA."""
from __future__ import annotations

import threading

import pytest

from kvcot.discovery.b2a_r3_authorization import (
    AUTHORIZATION_STAGE_B2A_R3_EXECUTION,
    AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
    AuthorizationAlreadyConsumed,
    AuthorizationClaimR3,
    claim_authorization,
    create_authorization_claim,
    plan_authorization_claim_dry_run,
)
from kvcot.discovery.b2a_r3_contract import (
    AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
    REQUIRED_REPOSITORY,
    SELECTED_MANIFEST_HASH_ALGORITHM,
    global_claim_path,
)
from kvcot.utils.hashing import sha256_json


def _stage_b_payload(**overrides):
    payload = {
        "artifact_schema_version": AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
        "authorization_id": "stage-b-2026-08-01",
        "authorization_stage": AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
        "authorization_document_path": "docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-08-01.md",
        "authorization_document_sha256": "a" * 64,
        "authorized_repository": REQUIRED_REPOSITORY,
        "authorized_branch": "research/b2a-r3-runtime-qualified-calibration",
        "authorized_commit_sha": "b" * 40,
        "observed_repository": REQUIRED_REPOSITORY,
        "observed_branch": "research/b2a-r3-runtime-qualified-calibration",
        "observed_commit_sha": "b" * 40,
        "required_ancestor_shas": ["c" * 40],
        "required_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "observed_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "candidate_manifest_canonical_sha256": "d" * 64,
        "qualification_artifact_canonical_sha256": None,
        "selected_manifest_sha256": None,
        "selected_manifest_hash_algorithm": None,
        "attempt_id": "deadbeef",
        "global_claim_path": global_claim_path("stage-b-2026-08-01"),
        "attempt_directory_path": "results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef",
        "claimed_at_utc": "2026-08-01T00:00:00+00:00",
    }
    payload.update(overrides)
    payload["canonical_sha256"] = sha256_json(payload)
    return payload


def _stage_c_payload(**overrides):
    base = _stage_b_payload()
    base.update(
        authorization_id="stage-c-2026-08-05",
        authorization_stage=AUTHORIZATION_STAGE_B2A_R3_EXECUTION,
        authorization_document_path="docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-08-05.md",
        qualification_artifact_canonical_sha256="e" * 64,
        selected_manifest_sha256="f" * 64,
        selected_manifest_hash_algorithm=SELECTED_MANIFEST_HASH_ALGORITHM,
        global_claim_path=global_claim_path("stage-c-2026-08-05"),
    )
    del base["canonical_sha256"]
    base.update(overrides)
    base["canonical_sha256"] = sha256_json(base)
    return base


def test_stage_b_payload_validates():
    payload = _stage_b_payload()
    typed = AuthorizationClaimR3.model_validate(payload)
    assert typed.authorization_stage == AUTHORIZATION_STAGE_FULLKV_QUALIFICATION


def test_stage_c_payload_validates():
    payload = _stage_c_payload()
    typed = AuthorizationClaimR3.model_validate(payload)
    assert typed.authorization_stage == AUTHORIZATION_STAGE_B2A_R3_EXECUTION


def test_stage_b_rejects_non_null_stage_c_fields():
    payload = _stage_b_payload(qualification_artifact_canonical_sha256="e" * 64)
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


def test_stage_c_rejects_missing_stage_c_field():
    payload = _stage_c_payload(selected_manifest_sha256=None)
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


def test_stage_c_rejects_wrong_hash_algorithm_name():
    payload = _stage_c_payload(selected_manifest_hash_algorithm="wrong-v1")
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


def test_wrong_repository_rejected():
    payload = _stage_b_payload(authorized_repository="someone-else/Faithkv")
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


def test_unknown_stage_rejected():
    payload = _stage_b_payload(authorization_stage="not_a_real_stage")
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


def test_global_claim_path_mismatch_rejected():
    payload = _stage_b_payload(global_claim_path="results/decisions/b2a_r3_authorization_claims/wrong.json")
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


@pytest.mark.parametrize(
    "bad_id",
    ["", "../escape", "a/b", "a\\b", "a..b", "a" * 129],
)
def test_invalid_authorization_id_rejected(bad_id):
    payload = _stage_b_payload(authorization_id=bad_id, global_claim_path="whatever")
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


def test_extra_field_rejected():
    payload = _stage_b_payload()
    payload = dict(payload, extra_field="nope")
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


def test_uppercase_hash_rejected():
    payload = _stage_b_payload(authorization_document_sha256="A" * 64)
    with pytest.raises(Exception):
        AuthorizationClaimR3.model_validate(payload)


# --------------------------------------------------------------------- atomic creation


def test_first_claim_succeeds(tmp_path):
    payload = _stage_b_payload()
    path = create_authorization_claim(payload, claims_root=tmp_path)
    assert path.exists()
    assert path.name == "stage-b-2026-08-01.json"


def test_second_claim_for_same_id_fails(tmp_path):
    payload = _stage_b_payload()
    create_authorization_claim(payload, claims_root=tmp_path)
    with pytest.raises(AuthorizationAlreadyConsumed):
        create_authorization_claim(payload, claims_root=tmp_path)


def test_claim_authorization_full_validation_and_creation(tmp_path):
    payload = _stage_b_payload()
    typed, path = claim_authorization(payload, claims_root=tmp_path)
    assert isinstance(typed, AuthorizationClaimR3)
    assert path.exists()


def test_claim_authorization_rejects_invalid_payload_before_touching_disk(tmp_path):
    payload = _stage_b_payload(authorized_repository="wrong/repo")
    with pytest.raises(Exception):
        claim_authorization(payload, claims_root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_empty_partial_or_corrupt_claim_remains_consumed(tmp_path):
    """A pre-existing filesystem entry at the deterministic path -- even
    empty or corrupt -- means permanently consumed. No repair, no
    deletion, no retry."""
    payload = _stage_b_payload()
    claim_path = tmp_path / f"{payload['authorization_id']}.json"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text("", encoding="utf-8")  # empty/corrupt claim, left by e.g. a crash

    with pytest.raises(AuthorizationAlreadyConsumed):
        create_authorization_claim(payload, claims_root=tmp_path)


def test_dry_run_creates_no_claim_directory_or_file(tmp_path):
    payload = _stage_b_payload()
    plan = plan_authorization_claim_dry_run(payload)
    assert plan["authorization_claim_created"] is False
    assert plan["authorization_consumed"] is False
    assert plan["payload_schema_valid"] is True
    # Dry-run planning takes no claims_root at all -- it cannot touch a
    # filesystem path by construction.
    import inspect

    assert "claims_root" not in inspect.signature(plan_authorization_claim_dry_run).parameters


def test_dry_run_reports_invalid_payload():
    payload = _stage_b_payload(authorized_repository="wrong/repo")
    plan = plan_authorization_claim_dry_run(payload)
    assert plan["payload_schema_valid"] is False
    assert plan["authorization_claim_created"] is False


def test_no_production_claims_directory_touched_by_dry_run():
    from kvcot.discovery.b2a_r3_contract import AUTHORIZATION_CLAIMS_DIR
    import os

    before = os.path.exists(AUTHORIZATION_CLAIMS_DIR)
    plan_authorization_claim_dry_run(_stage_b_payload())
    after = os.path.exists(AUTHORIZATION_CLAIMS_DIR)
    assert before == after
    assert after is False, "the real authorization-claims directory must not exist in this repository"


# --------------------------------------------------------------------- mandatory concurrency test


@pytest.mark.parametrize("trial", range(20))
def test_concurrent_claim_exactly_one_winner_one_refusal(tmp_path, trial):
    """Two concurrent claim attempts for the SAME authorization_id must
    produce exactly one successful exclusive creation and one refusal --
    repeated many times to expose any scan-then-write race defect. A
    barrier synchronizes both threads so they race over the SAME
    `os.open(..., O_EXCL)` call as tightly as possible."""
    claims_root = tmp_path / f"trial-{trial}"
    payload = _stage_b_payload(authorization_id=f"race-{trial}", global_claim_path=global_claim_path(f"race-{trial}"))

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            create_authorization_claim(payload, claims_root=claims_root)
            outcome = "won"
        except AuthorizationAlreadyConsumed:
            outcome = "lost"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sorted(outcomes) == ["lost", "won"]
    claim_path = claims_root / f"{payload['authorization_id']}.json"
    assert claim_path.exists()
    # No overwrite: exactly one claim file, with genuine content (not empty).
    assert claim_path.stat().st_size > 0
