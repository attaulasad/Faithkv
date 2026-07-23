"""B2A-R3 atomic authorization-claim tests. Every claim in this file is
written under `tmp_path` -- no real claim is ever created at
`results/decisions/b2a_r3_authorization_claims/`. No torch, no CUDA."""
from __future__ import annotations

import threading
import multiprocessing

import pytest

from kvcot.discovery.b2a_r3_authorization import (
    AUTHORIZATION_STAGE_B2A_R3_EXECUTION,
    AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
    AuthorizationAlreadyConsumed,
    AuthorizationClaimR3,
    _create_authorization_claim as create_authorization_claim,
    claim_authorization,
    plan_authorization_claim_dry_run,
    verify_authorization_preconditions,
)
from kvcot.discovery.b2a_r3_contract import (
    AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
    REQUIRED_REPOSITORY,
    SELECTED_MANIFEST_HASH_ALGORITHM,
    global_claim_path,
)
from kvcot.utils.hashing import sha256_json
from kvcot.utils.hashing import sha256_file


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


def _document_payload(*, authorization_id, authorization_stage, authorized_branch, authorized_commit_sha,
                       required_ancestor_shas, required_rkv_sha, candidate_manifest_canonical_sha256,
                       maximum_candidates=None, phase_wall_time_limit_seconds=None,
                       qualification_artifact_canonical_sha256=None, selected_manifest_sha256=None,
                       selected_manifest_hash_algorithm=None):
    from kvcot.discovery.b2a_r3_authorization_document import AUTHORIZATION_DOCUMENT_SCHEMA_VERSION

    return {
        "authorization_document_schema_version": AUTHORIZATION_DOCUMENT_SCHEMA_VERSION,
        "authorization_id": authorization_id,
        "authorization_stage": authorization_stage,
        "authorized_repository": REQUIRED_REPOSITORY,
        "authorized_branch": authorized_branch,
        "authorized_commit_sha": authorized_commit_sha,
        "required_ancestor_shas": list(required_ancestor_shas),
        "required_rkv_sha": required_rkv_sha,
        "candidate_manifest_canonical_sha256": candidate_manifest_canonical_sha256,
        "maximum_candidates": maximum_candidates,
        "phase_wall_time_limit_seconds": phase_wall_time_limit_seconds,
        "qualification_artifact_canonical_sha256": qualification_artifact_canonical_sha256,
        "selected_manifest_sha256": selected_manifest_sha256,
        "selected_manifest_hash_algorithm": selected_manifest_hash_algorithm,
        "created_at_utc": "2026-08-01T00:00:00+00:00",
    }


def _write_document(path, payload: dict) -> None:
    import json as _json

    from kvcot.discovery.b2a_r3_authorization_document import (
        AUTHORIZATION_DOCUMENT_BEGIN_MARKER,
        AUTHORIZATION_DOCUMENT_END_MARKER,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    body = _json.dumps(payload, indent=2)
    text = (
        "# Synthetic authorization document (test fixture)\n\n"
        f"{AUTHORIZATION_DOCUMENT_BEGIN_MARKER}\n```json\n{body}\n```\n{AUTHORIZATION_DOCUMENT_END_MARKER}\n"
    )
    path.write_text(text, encoding="utf-8")


def _verified_stage_b(tmp_path, *, document_overrides=None, **payload_overrides):
    import json
    from pathlib import Path

    from kvcot.discovery.b2a_r3_contract import CANDIDATE_MANIFEST_PATH
    from tests.unit.discovery.test_b2a_r3_provenance import FakeGitState

    document_rel = "docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-08-01.md"
    document = tmp_path / document_rel
    candidate_manifest = json.loads(Path(CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))
    document_fields = dict(
        authorization_id="stage-b-2026-08-01",
        authorization_stage=AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
        authorized_branch="research/b2a-r3-runtime-qualified-calibration",
        authorized_commit_sha="b" * 40,
        required_ancestor_shas=("c" * 40,),
        required_rkv_sha="45eaa7d69d20b7388321f077020a610d9afb65bd",
        candidate_manifest_canonical_sha256=candidate_manifest["canonical_sha256"],
        maximum_candidates=8,
        phase_wall_time_limit_seconds=3600,
    )
    document_fields.update(document_overrides or {})
    _write_document(document, _document_payload(**document_fields))
    payload = _stage_b_payload(
        authorization_document_sha256=sha256_file(document),
        candidate_manifest_canonical_sha256=candidate_manifest["canonical_sha256"],
        **payload_overrides,
    )
    git_state = FakeGitState(commit_sha="b" * 40, ancestors=frozenset({"c" * 40}), repository_root=str(tmp_path))
    context = verify_authorization_preconditions(
        payload,
        git_state=git_state,
        authorization_document_path=document,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=candidate_manifest["config_sha256"],
        repository_root=tmp_path,
    )
    return payload, context, document, candidate_manifest, git_state


def _stage_c_inputs(tmp_path, **payload_overrides):
    from tests.unit.discovery.test_b2a_r3_freeze import (
        CONFIG_SHA,
        _fake_renderer,
        _valid_chain,
    )
    from tests.unit.discovery.test_b2a_r3_provenance import FakeGitState
    from kvcot.discovery.b2a_r3_freeze import construct_selected_manifest_and_provenance

    document_rel = "docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_2026-08-05.md"
    document = tmp_path / document_rel
    candidate_manifest, qualification_artifact = _valid_chain()
    selected_manifest, selection_provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=CONFIG_SHA,
        tokenizer_renderer=_fake_renderer,
    )
    _write_document(document, _document_payload(
        authorization_id="stage-c-2026-08-05",
        authorization_stage=AUTHORIZATION_STAGE_B2A_R3_EXECUTION,
        authorized_branch="research/b2a-r3-runtime-qualified-calibration",
        authorized_commit_sha="b" * 40,
        required_ancestor_shas=("c" * 40,),
        required_rkv_sha="45eaa7d69d20b7388321f077020a610d9afb65bd",
        candidate_manifest_canonical_sha256=candidate_manifest["canonical_sha256"],
        qualification_artifact_canonical_sha256=qualification_artifact["canonical_sha256"],
        selected_manifest_sha256=selected_manifest.manifest_hash(),
        selected_manifest_hash_algorithm=SELECTED_MANIFEST_HASH_ALGORITHM,
    ))
    payload_values = {
        "authorization_document_sha256": sha256_file(document),
        "candidate_manifest_canonical_sha256": candidate_manifest["canonical_sha256"],
        "qualification_artifact_canonical_sha256": qualification_artifact["canonical_sha256"],
        "selected_manifest_sha256": selected_manifest.manifest_hash(),
    }
    payload_values.update(payload_overrides)
    payload = _stage_c_payload(**payload_values)
    git_state = FakeGitState(commit_sha="b" * 40, ancestors=frozenset({"c" * 40}), repository_root=str(tmp_path))
    return {
        "claim_payload": payload,
        "git_state": git_state,
        "authorization_document_path": document,
        "candidate_manifest": candidate_manifest,
        "expected_config_sha256": CONFIG_SHA,
        "qualification_artifact": qualification_artifact,
        "selected_manifest": selected_manifest,
        "selection_provenance": selection_provenance,
        "repository_root": tmp_path,
    }


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
    claim_path = tmp_path / f"{payload['authorization_id']}.json"
    path = create_authorization_claim(payload, claim_path=claim_path)
    assert path.exists()
    assert path.name == "stage-b-2026-08-01.json"


def test_second_claim_for_same_id_fails(tmp_path):
    payload = _stage_b_payload()
    claim_path = tmp_path / f"{payload['authorization_id']}.json"
    create_authorization_claim(payload, claim_path=claim_path)
    with pytest.raises(AuthorizationAlreadyConsumed):
        create_authorization_claim(payload, claim_path=claim_path)


def test_claim_authorization_full_validation_and_creation(tmp_path):
    payload, context, _document, _manifest, git_state = _verified_stage_b(tmp_path)
    typed, path = claim_authorization(
        payload, repository_root=tmp_path, verified_context=context, git_state=git_state
    )
    assert isinstance(typed, AuthorizationClaimR3)
    assert path.exists() and path.stat().st_size > 0


def test_claim_authorization_uses_exact_global_path_under_repository_root(tmp_path):
    """Step 3R4 Finding 4: the actual created path equals
    repository_root / claim.global_claim_path -- no arbitrary directory."""
    payload, context, _document, _manifest, git_state = _verified_stage_b(tmp_path)
    _typed, path = claim_authorization(
        payload, repository_root=tmp_path, verified_context=context, git_state=git_state
    )
    expected = tmp_path / "results" / "decisions" / "b2a_r3_authorization_claims" / f"{payload['authorization_id']}.json"
    assert path == expected
    assert str(path.relative_to(tmp_path)).replace("\\", "/") == payload["global_claim_path"]


def test_claim_authorization_no_longer_accepts_claims_root_parameter():
    import inspect

    assert "claims_root" not in inspect.signature(claim_authorization).parameters
    assert "repository_root" in inspect.signature(claim_authorization).parameters
    assert "git_state" in inspect.signature(claim_authorization).parameters


def test_claim_authorization_rejects_invalid_payload_before_touching_disk(tmp_path):
    payload = _stage_b_payload(authorized_repository="wrong/repo")
    with pytest.raises(Exception):
        claim_authorization(payload, repository_root=tmp_path, verified_context=object(), git_state=object())
    assert list(tmp_path.iterdir()) == []


def test_claim_authorization_reverifies_worktree_immediately_before_claim(tmp_path):
    """A worktree that becomes dirty between the earlier
    verify_authorization_preconditions call and claim_authorization's own
    reverification must fail -- the claim must not be created."""
    from dataclasses import replace as _dc_replace

    from kvcot.discovery.b2a_r3_provenance import WorktreeStatus

    payload, context, _document, _manifest, git_state = _verified_stage_b(tmp_path)
    dirtied_git_state = _dc_replace(
        git_state, status=WorktreeStatus(staged_paths=(), unstaged_paths=(), untracked_paths=("configs/smuggled.json",))
    )
    with pytest.raises(Exception):
        claim_authorization(payload, repository_root=tmp_path, verified_context=context, git_state=dirtied_git_state)
    claim_path = tmp_path / "results" / "decisions" / "b2a_r3_authorization_claims" / f"{payload['authorization_id']}.json"
    assert not claim_path.exists()


def test_preconditions_reject_authorization_document_byte_change(tmp_path):
    payload, _context, document, candidate_manifest, git_state = _verified_stage_b(tmp_path)
    document.write_text("tampered bytes\n", encoding="utf-8")
    with pytest.raises(Exception):
        verify_authorization_preconditions(
            payload,
            git_state=git_state,
            authorization_document_path=document,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=candidate_manifest["config_sha256"],
            repository_root=tmp_path,
        )


def test_preconditions_reject_observed_repository_disagreement(tmp_path):
    with pytest.raises(Exception):
        _verified_stage_b(tmp_path, observed_repository="someone/else")


@pytest.mark.parametrize(
    "field,value",
    [
        ("observed_branch", "main"),
        ("observed_commit_sha", "0" * 40),
        ("observed_rkv_sha", "0" * 40),
        ("candidate_manifest_canonical_sha256", "0" * 64),
    ],
)
def test_preconditions_reject_claim_identity_disagreements(tmp_path, field, value):
    with pytest.raises(Exception):
        _verified_stage_b(tmp_path, **{field: value})


def test_preconditions_reject_missing_authorization_document(tmp_path):
    payload, _context, document, candidate_manifest, git_state = _verified_stage_b(tmp_path)
    document.unlink()
    with pytest.raises(Exception):
        verify_authorization_preconditions(
            payload,
            git_state=git_state,
            authorization_document_path=document,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=candidate_manifest["config_sha256"],
            repository_root=tmp_path,
        )


def test_preconditions_reject_uncommitted_authorization_document(tmp_path):
    payload, _context, document, candidate_manifest, git_state = _verified_stage_b(tmp_path)

    class UncommittedDocumentGitState:
        def __getattr__(self, name):
            return getattr(git_state, name)

        def is_path_committed(self, path):
            return False

    with pytest.raises(Exception):
        verify_authorization_preconditions(
            payload,
            git_state=UncommittedDocumentGitState(),
            authorization_document_path=document,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=candidate_manifest["config_sha256"],
            repository_root=tmp_path,
        )


def test_preconditions_reject_missing_required_ancestor(tmp_path):
    from tests.unit.discovery.test_b2a_r3_provenance import FakeGitState

    payload, _context, document, candidate_manifest, _git_state = _verified_stage_b(tmp_path)
    with pytest.raises(Exception):
        verify_authorization_preconditions(
            payload,
            git_state=FakeGitState(commit_sha="b" * 40, ancestors=frozenset(), repository_root=str(tmp_path)),
            authorization_document_path=document,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=candidate_manifest["config_sha256"],
            repository_root=tmp_path,
        )


def test_git_state_repository_root_mismatch_rejected_by_preconditions(tmp_path):
    """Step 3R4-Repair-2 Finding 7: a GitStateProvider whose own
    `repository_root` disagrees with the separately-supplied
    `repository_root` argument must be refused -- otherwise Git state could
    be verified against one filesystem root while the authorization
    document is read from (and a future claim written under) a different
    one."""
    from dataclasses import replace as _dc_replace

    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    _payload, _context, document, candidate_manifest, git_state = _verified_stage_b(tmp_path)
    mismatched_git_state = _dc_replace(git_state, repository_root=str(other_root))

    with pytest.raises(Exception):
        verify_authorization_preconditions(
            _payload,
            git_state=mismatched_git_state,
            authorization_document_path=document,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=candidate_manifest["config_sha256"],
            repository_root=tmp_path,
        )


def test_git_state_repository_root_mismatch_rejected_by_claim_authorization(tmp_path):
    """Same defect, checked again at `claim_authorization` -- consumption
    must never proceed with a `git_state` bound to a different root than
    the one the claim is actually written under."""
    from dataclasses import replace as _dc_replace

    other_root = tmp_path / "elsewhere"
    other_root.mkdir()
    payload, context, _document, _manifest, git_state = _verified_stage_b(tmp_path)
    mismatched_git_state = _dc_replace(git_state, repository_root=str(other_root))

    with pytest.raises(Exception):
        claim_authorization(
            payload, repository_root=tmp_path, verified_context=context, git_state=mismatched_git_state
        )
    claim_path = tmp_path / "results" / "decisions" / "b2a_r3_authorization_claims" / f"{payload['authorization_id']}.json"
    assert not claim_path.exists()


def test_stage_c_full_semantic_preconditions_succeed(tmp_path):
    inputs = _stage_c_inputs(tmp_path)
    context = verify_authorization_preconditions(**inputs)
    assert context.claim.authorization_stage == AUTHORIZATION_STAGE_B2A_R3_EXECUTION


@pytest.mark.parametrize(
    "field,value",
    [
        ("qualification_artifact_canonical_sha256", "0" * 64),
        ("selected_manifest_sha256", "0" * 64),
    ],
)
def test_stage_c_claim_hash_mismatches_are_rejected(tmp_path, field, value):
    inputs = _stage_c_inputs(tmp_path, **{field: value})
    with pytest.raises(Exception):
        verify_authorization_preconditions(**inputs)


def test_stage_c_selection_provenance_mismatch_is_rejected_after_rehash(tmp_path):
    inputs = _stage_c_inputs(tmp_path)
    provenance = dict(inputs["selection_provenance"])
    provenance["selected_unique_id"] = "different-row"
    provenance.pop("canonical_sha256")
    provenance["canonical_sha256"] = sha256_json(provenance)
    inputs["selection_provenance"] = provenance
    with pytest.raises(Exception):
        verify_authorization_preconditions(**inputs)


def test_empty_partial_or_corrupt_claim_remains_consumed(tmp_path):
    """A pre-existing filesystem entry at the deterministic path -- even
    empty or corrupt -- means permanently consumed. No repair, no
    deletion, no retry."""
    payload = _stage_b_payload()
    claim_path = tmp_path / f"{payload['authorization_id']}.json"
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text("", encoding="utf-8")  # empty/corrupt claim, left by e.g. a crash

    with pytest.raises(AuthorizationAlreadyConsumed):
        create_authorization_claim(payload, claim_path=claim_path)


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
    claim_path = claims_root / f"{payload['authorization_id']}.json"

    barrier = threading.Barrier(2)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt() -> None:
        barrier.wait()
        try:
            create_authorization_claim(payload, claim_path=claim_path)
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
    assert claim_path.exists()
    # No overwrite: exactly one claim file, with genuine content (not empty).
    assert claim_path.stat().st_size > 0


def _multiprocess_claim_attempt(payload, claim_path, start_event, result_queue):
    """Top-level spawn target: must remain picklable on Windows."""
    from kvcot.discovery.b2a_r3_authorization import (
        AuthorizationAlreadyConsumed,
        _create_authorization_claim,
    )

    start_event.wait()
    try:
        _create_authorization_claim(payload, claim_path=claim_path)
        result_queue.put("success")
    except AuthorizationAlreadyConsumed:
        result_queue.put("AuthorizationAlreadyConsumed")


@pytest.mark.parametrize("trial", range(5))
def test_multiprocess_claim_exactly_one_winner_one_refusal(tmp_path, trial):
    ctx = multiprocessing.get_context("spawn")
    claims_root = tmp_path / f"process-trial-{trial}"
    authorization_id = f"process-race-{trial}"
    payload = _stage_b_payload(
        authorization_id=authorization_id,
        global_claim_path=global_claim_path(authorization_id),
    )
    claim_path = claims_root / f"{authorization_id}.json"
    start_event = ctx.Event()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_multiprocess_claim_attempt,
            args=(payload, str(claim_path), start_event, result_queue),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    outcomes = sorted(result_queue.get(timeout=5) for _ in range(2))
    assert outcomes == ["AuthorizationAlreadyConsumed", "success"]
    entries = list(claims_root.iterdir())
    assert len(entries) == 1
    assert entries[0].name == f"{authorization_id}.json"
    assert entries[0].stat().st_size > 0
