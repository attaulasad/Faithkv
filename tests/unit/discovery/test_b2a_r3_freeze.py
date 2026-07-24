"""B2A-R3 selected-row freezer tests. Synthetic-only fixtures -- no torch,
no CUDA, no real tokenizer, no real freeze against the actual repository
paths. Every write goes through `tmp_path`."""
from __future__ import annotations

import itertools
import json as _json
import subprocess
from pathlib import Path as _Path

import pytest

from kvcot.discovery.b2a_r3_artifacts import SELECTION_STATUS_NONE_QUALIFIED, SELECTION_STATUS_SELECTED
from kvcot.discovery.b2a_r3_candidates import build_candidate_manifest
from kvcot.discovery.b2a_r3_contract import PROMPT_SPECIAL_TOKENS_NOTE, SELECTED_MANIFEST_PATH
from kvcot.discovery.b2a_r3_freeze import (
    PromptRenderingResult,
    RowFreezeRefusedR3,
    construct_selected_manifest_and_provenance as _construct_selected_manifest_and_provenance,
    plan_freeze_dry_run as _plan_freeze_dry_run,
    verify_selection_provenance as _verify_selection_provenance,
    write_selected_manifest_and_provenance,
)
from kvcot.discovery.b2a_r3_qualification import build_qualification_outcome
from kvcot.discovery.b2a_r3_runtime import predict_runtime
from kvcot.discovery.manifest import B2AOneExampleManifest, ChatTemplateRenderingConfig
from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

from tests.unit.discovery.test_b2a_r3_qualification import _valid_evidence, _valid_placement, _valid_timing

DATASET_REPO = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"
BUDGET = 1024
CONFIG_SHA = "c" * 64


def _stage_b_context(candidate_manifest, maximum_candidates=8, phase_seconds=3600):
    from pathlib import Path
    from types import SimpleNamespace

    from kvcot.discovery.b2a_r3_authorization import (
        AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
        AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
        AuthorizationClaimR3,
        ConsumedAuthorizationContext,
        _CONSUMED_CONTEXT_TOKEN,
    )
    from kvcot.discovery.b2a_r3_contract import REQUIRED_REPOSITORY, global_claim_path

    auth_id = "stage-b-2026-08-01"
    payload = {
        "artifact_schema_version": AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
        "authorization_id": auth_id,
        "authorization_stage": AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
        "authorization_document_path": "docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-08-01.md",
        "authorization_document_sha256": "a" * 64,
        "authorized_repository": REQUIRED_REPOSITORY,
        "authorized_branch": "research/b2a-r3-runtime-qualified-calibration",
        "authorized_code_commit_sha": "b" * 40,
        "observed_repository": REQUIRED_REPOSITORY,
        "observed_branch": "research/b2a-r3-runtime-qualified-calibration",
        "observed_execution_commit_sha": "d" * 40,
        "required_ancestor_shas": ["c" * 40],
        "required_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "observed_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "candidate_manifest_canonical_sha256": candidate_manifest["canonical_sha256"],
        "qualification_artifact_canonical_sha256": None,
        "selected_manifest_sha256": None,
        "selected_manifest_hash_algorithm": None,
        "attempt_id": "deadbeef",
        "global_claim_path": global_claim_path(auth_id),
        "attempt_directory_path": "results/decisions/b2a_r3_attempt_20260801T000000000000Z_deadbeef",
        "claimed_at_utc": "2026-08-01T00:00:00+00:00",
    }
    payload["canonical_sha256"] = sha256_json(payload)
    claim = AuthorizationClaimR3.model_validate(payload)
    verified = SimpleNamespace(maximum_candidates=maximum_candidates, phase_wall_time_limit_seconds=phase_seconds)
    return ConsumedAuthorizationContext(
        claim=claim,
        verified_context=verified,
        claim_path=Path(claim.global_claim_path),
        authorization_claim_canonical_sha256=claim.canonical_sha256,
        _consumption_token=_CONSUMED_CONTEXT_TOKEN,
    )


def construct_selected_manifest_and_provenance(*, candidate_manifest, qualification_artifact, expected_config_sha256, tokenizer_renderer):
    return _construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=_stage_b_context(
            candidate_manifest,
            qualification_artifact["authorized_maximum_candidates"],
            qualification_artifact["authorized_phase_wall_time_limit_seconds"],
        ),
        tokenizer_renderer=tokenizer_renderer,
    )


def verify_selection_provenance(provenance, *, selected_manifest, candidate_manifest, qualification_artifact, expected_config_sha256):
    return _verify_selection_provenance(
        provenance,
        selected_manifest=selected_manifest,
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=_stage_b_context(
            candidate_manifest,
            qualification_artifact["authorized_maximum_candidates"],
            qualification_artifact["authorized_phase_wall_time_limit_seconds"],
        ),
    )


def plan_freeze_dry_run(*, candidate_manifest, qualification_artifact, expected_config_sha256):
    return _plan_freeze_dry_run(
        candidate_manifest=candidate_manifest,
        qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=_stage_b_context(
            candidate_manifest,
            qualification_artifact["authorized_maximum_candidates"],
            qualification_artifact["authorized_phase_wall_time_limit_seconds"],
        ),
    )


def _row(unique_id: str, level: str) -> dict:
    return {
        "problem": f"problem {unique_id}", "solution": "s", "answer": f"answer-{unique_id}",
        "subject": "Algebra", "level": level, "unique_id": unique_id,
    }


def _fake_renderer(row: dict) -> PromptRenderingResult:
    token_ids = tuple(range(10))
    return PromptRenderingResult(
        rendered_user_message_sha256=sha256_text(f"rendered:{row['problem']}"),
        chat_template_source_sha256="a" * 64,
        chat_message_payload_sha256=sha256_json([{"role": "user", "content": row["problem"]}]),
        prompt_token_ids=token_ids,
        prompt_token_ids_sha256=sha256_int_ids(list(token_ids)),
        prompt_token_count=len(token_ids),
        tokenizer_revision_used_for_prompt_hash=MODEL_REVISION,
        prompt_rendering_config=ChatTemplateRenderingConfig(
            message_roles=("user",), add_generation_prompt=True, tokenize=True,
            add_special_tokens_note=PROMPT_SPECIAL_TOKENS_NOTE,
        ),
    )


def _population(n_per_level=10):
    rows = []
    for i in range(n_per_level):
        rows.append(_row(f"level4-{i:03d}", "4"))
        rows.append(_row(f"level5-{i:03d}", "5"))
    return rows


def _candidate_manifest():
    return build_candidate_manifest(
        _population(),
        dataset_repo=DATASET_REPO, dataset_config="default", dataset_split="test",
        dataset_revision=DATASET_REVISION, model_name=MODEL_NAME, model_revision=MODEL_REVISION,
        tokenizer_name=MODEL_NAME, tokenizer_revision=MODEL_REVISION, budget=BUDGET,
        config_path="configs/discovery/llama8b_math500_b1024.yaml", config_sha256=CONFIG_SHA,
    )


def _outcome_for(candidate_manifest, ordinal: int, *, qualified: bool):
    candidate = next(c for c in candidate_manifest["candidates"] if c["candidate_ordinal"] == ordinal)
    row = candidate["row"]
    overrides = dict(
        candidate_ordinal=ordinal,
        source_example_index=candidate["source_example_index"],
        unique_id=candidate["unique_id"],
        row=row,
        raw_row_sha256=candidate["raw_row_sha256"],
        problem_sha256=candidate["problem_sha256"],
        gold_answer_sha256=candidate["gold_answer_sha256"],
        candidate_manifest_canonical_sha256=candidate_manifest["canonical_sha256"],
        config_sha256=CONFIG_SHA,
        expected_prompt_token_ids_sha256=sha256_int_ids(list(range(10))),
        observed_prompt_token_ids_sha256=sha256_int_ids(list(range(10))),
    )
    if not qualified:
        overrides["answer_verification_status"] = "incorrect"
    evidence = _valid_evidence(**overrides)
    return build_qualification_outcome(
        evidence, candidate_manifest=candidate_manifest, expected_config_sha256=CONFIG_SHA
    )


def _qualification_artifact(candidate_manifest, attempted, *, selection_status, first_ordinal, selected_unique_id):
    from kvcot.discovery.b2a_r3_artifacts import (
        STOPPED_REASON_ALL_CANDIDATES_EXHAUSTED,
        STOPPED_REASON_FIRST_PASS,
    )
    from kvcot.discovery.b2a_r3_contract import (
        BUDGET as C_BUDGET,
        CANDIDATE_MANIFEST_PATH,
        CANDIDATE_ORDER_PROTOCOL_VERSION,
        CONFIG_PATH,
        DATASET_CONFIG,
        DATASET_REPO as C_DATASET_REPO,
        DATASET_REVISION as C_DATASET_REVISION,
        DATASET_SPLIT,
        GENERATION_CONFIG_SHA256,
        MODEL_NAME as C_MODEL_NAME,
        MODEL_REVISION as C_MODEL_REVISION,
        QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
        QUALIFICATION_CANDIDATE_LIMIT,
        QUALIFICATION_PROTOCOL_VERSION,
        RUNTIME_PREDICTOR_VERSION,
        RUNTIME_SOURCE_ARTIFACT_PATH,
        RUNTIME_SOURCE_ARTIFACT_SHA256,
        TOKENIZER_NAME,
        TOKENIZER_REVISION,
    )

    authorized_maximum = QUALIFICATION_CANDIDATE_LIMIT if first_ordinal is not None else len(attempted)
    context = _stage_b_context(candidate_manifest, authorized_maximum)
    payload = {
        "artifact_schema_version": QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
        "candidate_order_protocol_version": CANDIDATE_ORDER_PROTOCOL_VERSION,
        "qualification_protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "runtime_predictor_version": RUNTIME_PREDICTOR_VERSION,
        "candidate_manifest_path": CANDIDATE_MANIFEST_PATH,
        "candidate_manifest_canonical_sha256": candidate_manifest["canonical_sha256"],
        "config_path": CONFIG_PATH,
        "config_sha256": CONFIG_SHA,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "dataset_repo": C_DATASET_REPO,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_revision": C_DATASET_REVISION,
        "model_name": C_MODEL_NAME,
        "model_revision": C_MODEL_REVISION,
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": TOKENIZER_REVISION,
        "budget": C_BUDGET,
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
        "authorized_maximum_candidates": authorized_maximum,
        "authorized_phase_wall_time_limit_seconds": context.verified_context.phase_wall_time_limit_seconds,
        "stage_b_authorization_id": context.claim.authorization_id,
        "authorization_document_sha256": context.claim.authorization_document_sha256,
        "authorization_claim_canonical_sha256": context.claim.canonical_sha256,
        "attempt_started_at_utc": "2026-07-23T00:00:00+00:00",
        "attempt_completed_at_utc": "2026-07-23T00:10:00+00:00",
    }
    payload["canonical_sha256"] = sha256_json(payload)
    return payload


def _valid_chain():
    """Returns (candidate_manifest, qualification_artifact) where ordinal 0
    (a real level-4 row) qualifies."""
    candidate_manifest = _candidate_manifest()
    attempted = [_outcome_for(candidate_manifest, 0, qualified=True)]
    selected_unique_id = attempted[0]["unique_id"]
    artifact = _qualification_artifact(
        candidate_manifest, attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=0,
        selected_unique_id=selected_unique_id,
    )
    return candidate_manifest, artifact


def _rehash(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("canonical_sha256", None)
    payload["canonical_sha256"] = sha256_json(payload)
    return payload


def test_construct_selected_manifest_and_provenance_succeeds():
    candidate_manifest, artifact = _valid_chain()
    new_manifest, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest, qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
    )
    assert isinstance(new_manifest, B2AOneExampleManifest)
    assert new_manifest.unique_id == artifact["selected_unique_id"]
    assert provenance["selected_manifest_sha256"] == new_manifest.manifest_hash()
    assert provenance["selected_manifest_path"] == SELECTED_MANIFEST_PATH


def test_no_candidate_qualified_refuses_freeze():
    candidate_manifest = _candidate_manifest()
    attempted = [_outcome_for(candidate_manifest, i, qualified=False) for i in range(3)]
    artifact = _qualification_artifact(
        candidate_manifest, attempted, selection_status=SELECTION_STATUS_NONE_QUALIFIED,
        first_ordinal=None, selected_unique_id=None,
    )
    with pytest.raises(RowFreezeRefusedR3):
        construct_selected_manifest_and_provenance(
            candidate_manifest=candidate_manifest, qualification_artifact=artifact,
            expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
        )


def test_reject_later_passing_row_over_earlier_one():
    """A qualification artifact claiming ordinal=1 selected while ordinal=0
    ALSO qualifies (violating first-pass-wins) must already fail schema
    validation before the freezer is ever reached."""
    candidate_manifest = _candidate_manifest()
    attempted = [
        _outcome_for(candidate_manifest, 0, qualified=True),
        _outcome_for(candidate_manifest, 1, qualified=True),
    ]
    ordinal1_unique_id = attempted[1]["unique_id"]
    artifact = _qualification_artifact(
        candidate_manifest, attempted, selection_status=SELECTION_STATUS_SELECTED, first_ordinal=1,
        selected_unique_id=ordinal1_unique_id,
    )
    with pytest.raises(Exception):
        construct_selected_manifest_and_provenance(
            candidate_manifest=candidate_manifest, qualification_artifact=artifact,
            expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
        )


def test_arbitrary_row_substitution_rejected():
    candidate_manifest, artifact = _valid_chain()
    # Tamper: point the qualification artifact's selection at a DIFFERENT
    # unique_id than the one its own outcome/ordinal actually recorded.
    tampered = dict(artifact)
    tampered["selected_unique_id"] = "level5-000"
    tampered["canonical_sha256"] = sha256_json({k: v for k, v in tampered.items() if k != "canonical_sha256"})
    with pytest.raises(Exception):
        construct_selected_manifest_and_provenance(
            candidate_manifest=candidate_manifest, qualification_artifact=tampered,
            expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
        )


def test_manual_hash_tampering_rejected():
    candidate_manifest, artifact = _valid_chain()
    tampered_manifest = dict(candidate_manifest)
    candidates = [dict(c) for c in tampered_manifest["candidates"]]
    candidates[0] = dict(candidates[0], raw_row_sha256="0" * 64)
    tampered_manifest["candidates"] = candidates
    # canonical_sha256 deliberately left stale -- this must fail on the
    # manifest's own canonical hash check before anything else.
    with pytest.raises(Exception):
        construct_selected_manifest_and_provenance(
            candidate_manifest=tampered_manifest, qualification_artifact=artifact,
            expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
        )


def test_write_and_verify_round_trip(tmp_path):
    candidate_manifest, artifact = _valid_chain()
    new_manifest, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest, qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
    )
    manifest_path = tmp_path / "b2a_one_example_manifest.json"
    provenance_path = tmp_path / "b2a_r3_selection_provenance.json"
    write_selected_manifest_and_provenance(
        new_manifest, provenance, manifest_path=manifest_path, provenance_path=provenance_path
    )
    assert manifest_path.exists()
    assert provenance_path.exists()

    import json

    reloaded_manifest = B2AOneExampleManifest.model_validate(json.loads(manifest_path.read_text(encoding="utf-8")))
    reloaded_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    typed = verify_selection_provenance(
        reloaded_provenance, selected_manifest=reloaded_manifest,
        candidate_manifest=candidate_manifest, qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA,
    )
    assert typed.selected_unique_id == new_manifest.unique_id


def test_verify_selection_provenance_rejects_wrong_external_hash(tmp_path):
    candidate_manifest, artifact = _valid_chain()
    new_manifest, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest, qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
    )
    tampered = dict(provenance)
    tampered["selected_manifest_sha256"] = "0" * 64
    tampered["canonical_sha256"] = sha256_json({k: v for k, v in tampered.items() if k != "canonical_sha256"})
    with pytest.raises(RowFreezeRefusedR3):
        verify_selection_provenance(
            tampered, selected_manifest=new_manifest, candidate_manifest=candidate_manifest,
            qualification_artifact=artifact, expected_config_sha256=CONFIG_SHA,
        )


def test_no_production_path_touched_by_construction():
    """Construction alone must never touch the filesystem at all -- the
    real SELECTED_MANIFEST_PATH must not exist purely as a side effect of
    calling the pure construction function."""
    import os

    candidate_manifest, artifact = _valid_chain()
    before = os.path.exists(SELECTED_MANIFEST_PATH)
    construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest, qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA, tokenizer_renderer=_fake_renderer,
    )
    after = os.path.exists(SELECTED_MANIFEST_PATH)
    # Whatever the pre-existing state was (the real manifest legitimately
    # exists from B2A-R1/R2), this call must not have CHANGED it.
    assert before == after


def test_plan_freeze_dry_run_reports_would_freeze_true():
    candidate_manifest, artifact = _valid_chain()
    plan = plan_freeze_dry_run(
        candidate_manifest=candidate_manifest, qualification_artifact=artifact, expected_config_sha256=CONFIG_SHA,
    )
    assert plan["would_freeze"] is True
    assert plan["would_load_tokenizer_for_execution"] is False
    assert plan["would_write_selected_manifest"] is False
    assert plan["would_write_selection_provenance"] is False


def test_plan_freeze_dry_run_reports_would_freeze_false_when_unqualified():
    candidate_manifest = _candidate_manifest()
    attempted = [_outcome_for(candidate_manifest, i, qualified=False) for i in range(3)]
    artifact = _qualification_artifact(
        candidate_manifest, attempted, selection_status=SELECTION_STATUS_NONE_QUALIFIED,
        first_ordinal=None, selected_unique_id=None,
    )
    plan = plan_freeze_dry_run(
        candidate_manifest=candidate_manifest, qualification_artifact=artifact, expected_config_sha256=CONFIG_SHA,
    )
    assert plan["would_freeze"] is False
    assert "refusal_reason" in plan


@pytest.mark.parametrize(
    "case",
    [
        "wrong_hash",
        "wrong_count",
        "changed_token",
        "bool_token",
        "wrong_tokenizer_revision",
        "wrong_roles",
        "no_generation_prompt",
        "no_tokenize",
        "wrong_special_token_note",
        "uppercase_hash",
    ],
)
def test_prompt_renderer_output_is_strictly_validated(case):
    candidate_manifest, artifact = _valid_chain()

    def malformed_renderer(row):
        values = _fake_renderer(row).model_dump(mode="python")
        if case == "wrong_hash":
            values["prompt_token_ids_sha256"] = "0" * 64
        elif case == "wrong_count":
            values["prompt_token_count"] += 1
        elif case == "changed_token":
            values["prompt_token_ids"] = (999, *values["prompt_token_ids"][1:])
        elif case == "bool_token":
            values["prompt_token_ids"] = (True, *values["prompt_token_ids"][1:])
        elif case == "wrong_tokenizer_revision":
            values["tokenizer_revision_used_for_prompt_hash"] = "0" * 40
        elif case == "uppercase_hash":
            values["rendered_user_message_sha256"] = "A" * 64
        else:
            config = dict(values["prompt_rendering_config"])
            if case == "wrong_roles":
                config["message_roles"] = ("system", "user")
            elif case == "no_generation_prompt":
                config["add_generation_prompt"] = False
            elif case == "no_tokenize":
                config["tokenize"] = False
            else:
                config["add_special_tokens_note"] = "different convention"
            values["prompt_rendering_config"] = config
        return values

    with pytest.raises(Exception):
        construct_selected_manifest_and_provenance(
            candidate_manifest=candidate_manifest,
            qualification_artifact=artifact,
            expected_config_sha256=CONFIG_SHA,
            tokenizer_renderer=malformed_renderer,
        )


@pytest.mark.parametrize(
    "case",
    [
        "provenance_ordinal",
        "provenance_unique_id",
        "selected_example_index",
        "selected_unique_id",
        "selected_gold_answer",
        "selected_raw_hash",
        "selected_prompt_hash",
        "selected_tokenizer_revision",
    ],
)
def test_canonically_rehashed_selection_chain_tampering_is_rejected(case):
    candidate_manifest, artifact = _valid_chain()
    selected, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA,
        tokenizer_renderer=_fake_renderer,
    )
    selected_values = selected.model_dump(mode="python")
    provenance = dict(provenance)
    if case == "provenance_ordinal":
        provenance["selected_ordinal"] = 1
    elif case == "provenance_unique_id":
        provenance["selected_unique_id"] = "different-row"
    elif case == "selected_example_index":
        selected_values["example_index"] += 1
    elif case == "selected_unique_id":
        selected_values["unique_id"] = "different-row"
    elif case == "selected_gold_answer":
        selected_values["gold_answer"] = "different answer"
    elif case == "selected_raw_hash":
        selected_values["raw_content_hash"] = "0" * 64
    elif case == "selected_prompt_hash":
        token_ids = (999, *selected_values["prompt_token_ids"][1:])
        selected_values["prompt_token_ids"] = token_ids
        selected_values["prompt_token_ids_sha256"] = sha256_int_ids(list(token_ids))
        provenance["prompt_token_ids_sha256"] = selected_values["prompt_token_ids_sha256"]
    else:
        selected_values["tokenizer_revision_used_for_prompt_hash"] = "0" * 40
        provenance["tokenizer_revision_used_for_prompt_hash"] = "0" * 40

    selected = B2AOneExampleManifest.model_validate(selected_values)
    provenance["selected_manifest_sha256"] = selected.manifest_hash()
    provenance = _rehash(provenance)
    with pytest.raises(Exception):
        verify_selection_provenance(
            provenance,
            selected_manifest=selected,
            candidate_manifest=candidate_manifest,
            qualification_artifact=artifact,
            expected_config_sha256=CONFIG_SHA,
        )


def test_canonically_rehashed_valid_qualification_replacement_is_rejected():
    candidate_manifest, artifact = _valid_chain()
    selected, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA,
        tokenizer_renderer=_fake_renderer,
    )
    replacement = _qualification_artifact(
        candidate_manifest,
        [
            _outcome_for(candidate_manifest, 0, qualified=False),
            _outcome_for(candidate_manifest, 1, qualified=True),
        ],
        selection_status=SELECTION_STATUS_SELECTED,
        first_ordinal=1,
        selected_unique_id=candidate_manifest["candidates"][1]["unique_id"],
    )
    provenance["qualification_artifact_canonical_sha256"] = replacement["canonical_sha256"]
    provenance = _rehash(provenance)
    with pytest.raises(Exception):
        verify_selection_provenance(
            provenance,
            selected_manifest=selected,
            candidate_manifest=candidate_manifest,
            qualification_artifact=replacement,
            expected_config_sha256=CONFIG_SHA,
        )


def test_canonically_rehashed_valid_candidate_chain_replacement_is_rejected():
    candidate_manifest, artifact = _valid_chain()
    selected, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA,
        tokenizer_renderer=_fake_renderer,
    )
    alternate_population = [
        dict(row, problem=f"alternate valid problem: {row['problem']}") for row in _population()
    ]
    replacement_manifest = build_candidate_manifest(
        alternate_population,
        dataset_repo=DATASET_REPO, dataset_config="default", dataset_split="test",
        dataset_revision=DATASET_REVISION, model_name=MODEL_NAME, model_revision=MODEL_REVISION,
        tokenizer_name=MODEL_NAME, tokenizer_revision=MODEL_REVISION, budget=BUDGET,
        config_path="configs/discovery/llama8b_math500_b1024.yaml", config_sha256=CONFIG_SHA,
    )
    replacement_artifact = _qualification_artifact(
        replacement_manifest,
        [_outcome_for(replacement_manifest, 0, qualified=True)],
        selection_status=SELECTION_STATUS_SELECTED,
        first_ordinal=0,
        selected_unique_id=replacement_manifest["candidates"][0]["unique_id"],
    )
    provenance["candidate_manifest_canonical_sha256"] = replacement_manifest["canonical_sha256"]
    provenance["qualification_artifact_canonical_sha256"] = replacement_artifact["canonical_sha256"]
    provenance = _rehash(provenance)
    with pytest.raises(Exception):
        verify_selection_provenance(
            provenance,
            selected_manifest=selected,
            candidate_manifest=replacement_manifest,
            qualification_artifact=replacement_artifact,
            expected_config_sha256=CONFIG_SHA,
        )


def test_canonically_rehashed_earlier_qualifying_candidate_is_rejected():
    candidate_manifest = _candidate_manifest()
    attempted = [
        _outcome_for(candidate_manifest, 0, qualified=False),
        _outcome_for(candidate_manifest, 1, qualified=True),
    ]
    artifact = _qualification_artifact(
        candidate_manifest,
        attempted,
        selection_status=SELECTION_STATUS_SELECTED,
        first_ordinal=1,
        selected_unique_id=attempted[1]["unique_id"],
    )
    selected, provenance = construct_selected_manifest_and_provenance(
        candidate_manifest=candidate_manifest,
        qualification_artifact=artifact,
        expected_config_sha256=CONFIG_SHA,
        tokenizer_renderer=_fake_renderer,
    )
    artifact["attempted"][0] = _outcome_for(candidate_manifest, 0, qualified=True)
    artifact = _rehash(artifact)
    provenance["qualification_artifact_canonical_sha256"] = artifact["canonical_sha256"]
    provenance = _rehash(provenance)
    with pytest.raises(Exception):
        verify_selection_provenance(
            provenance,
            selected_manifest=selected,
            candidate_manifest=candidate_manifest,
            qualification_artifact=artifact,
            expected_config_sha256=CONFIG_SHA,
        )


# ==========================================================================
# Phase 2 (freezer implementation authorization, dated 2026-07-24): the
# production freeze-plan / guarded publication state machine. Every write
# here goes through a real, temporary, throwaway git repository (never the
# actual repository) -- no torch, no CUDA, no real FullKV inference (the
# qualification coordinator is driven by a fake worker runner, the same
# idiom `test_b2a_r3_authorization.py`'s own end-to-end persisted-binding
# test already uses).
# ==========================================================================

from kvcot.discovery.b2a_r3_artifacts import write_qualification_artifact_atomic
from kvcot.discovery.b2a_r3_authorization import (
    AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
    claim_authorization,
    verify_authorization_preconditions,
)
from kvcot.discovery.b2a_r3_freeze import (
    ProductionPublicationRefused,
    PUBLICATION_STATE_A_INITIAL,
    PUBLICATION_STATE_B_COMPLETE,
    PUBLICATION_STATE_C_PROVENANCE_FIRST_PARTIAL,
    PUBLICATION_STATE_D_MANIFEST_FIRST_PARTIAL,
    PUBLICATION_STATE_E_INVALID,
    classify_publication_state,
    construct_production_freeze_plan,
    publish_production_freeze,
    verify_git_worktree_safety_for_freeze,
)
from kvcot.discovery.b2a_r3_contract import REQUIRED_REPOSITORY
from kvcot.discovery.b2a_r3_provenance import SubprocessGitStateProvider
from kvcot.discovery.b2a_r3_qualification_coordinator import run_b2a_r3_qualification_coordinator

from tests.unit.discovery.test_b2a_r3_authorization import _document_payload, _stage_b_payload, _write_document
from tests.unit.discovery.test_b2a_r3_worker_adapter import (
    CONFIG_SHA as REAL_CONFIG_SHA,
    _candidate_manifest as _real_candidate_manifest,
    _valid_worker_result,
)

REAL_CONFIG_PATH_REL = "configs/discovery/llama8b_math500_b1024.yaml"
_FAKE_PROMPT_HASH = sha256_int_ids(list(range(10)))


def _git(cwd, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args} failed in {cwd}: {r.stdout} {r.stderr}")
    return r.stdout.strip()


def _build_production_shaped_repo(tmp_path: _Path):
    """A real temporary git repository shaped exactly like the production
    repository at the fixed B2A-R3 paths: the REAL committed candidate
    manifest content and the REAL committed config file bytes (so
    `config_identity` needs no monkeypatching), a coordinator-produced
    qualification artifact, and a consumed Stage-B claim -- all built the
    same way `test_b2a_r3_authorization.py`'s own
    `test_persisted_stage_b_binding_survives_untracked_outputs_and_head_advance_in_real_git`
    builds its fixture."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test User")
    _git(tmp_path, "checkout", "-q", "-b", "research/b2a-r3-runtime-qualified-calibration")
    _git(tmp_path, "remote", "add", "origin", f"https://github.com/{REQUIRED_REPOSITORY}.git")

    (tmp_path / "README.md").write_text("root\n", encoding="utf-8")
    _git(tmp_path, "add", "README.md")
    _git(tmp_path, "commit", "-q", "-m", "root")
    ancestor_sha = _git(tmp_path, "rev-parse", "HEAD")

    rkv_dir = tmp_path / "third_party" / "R-KV"
    rkv_dir.mkdir(parents=True)
    _git(rkv_dir, "init", "-q")
    _git(rkv_dir, "config", "user.email", "test@example.invalid")
    _git(rkv_dir, "config", "user.name", "Test User")
    (rkv_dir / "README.md").write_text("rkv\n", encoding="utf-8")
    _git(rkv_dir, "add", "README.md")
    _git(rkv_dir, "commit", "-q", "-m", "rkv")
    rkv_sha = _git(rkv_dir, "rev-parse", "HEAD")
    _git(tmp_path, "update-index", "--add", "--cacheinfo", f"160000,{rkv_sha},third_party/R-KV")

    candidate_manifest = _real_candidate_manifest()
    (tmp_path / "configs/discovery").mkdir(parents=True)
    (tmp_path / "results/decisions").mkdir(parents=True)
    (tmp_path / REAL_CONFIG_PATH_REL).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / REAL_CONFIG_PATH_REL).write_bytes(_Path(REAL_CONFIG_PATH_REL).read_bytes())
    (tmp_path / "configs/discovery/b2a_r3_candidate_manifest.json").write_text(
        _json.dumps(candidate_manifest, indent=2) + "\n"
    )
    historical_manifest = {"dataset_repo": "x", "unique_id": "historical-row"}
    (tmp_path / "configs/discovery/b2a_one_example_manifest.json").write_text(
        _json.dumps(historical_manifest, indent=2) + "\n"
    )

    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "authorized historical tree")
    authorized_sha = _git(tmp_path, "rev-parse", "HEAD")

    stage_b_document_rel = "docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-08-01.md"
    _write_document(
        tmp_path / stage_b_document_rel,
        _document_payload(
            authorization_id="stage-b-2026-08-01",
            authorization_stage=AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
            authorized_branch="research/b2a-r3-runtime-qualified-calibration",
            authorized_code_commit_sha=authorized_sha,
            required_ancestor_shas=(ancestor_sha,),
            required_rkv_sha=rkv_sha,
            candidate_manifest_canonical_sha256=candidate_manifest["canonical_sha256"],
            maximum_candidates=8,
            phase_wall_time_limit_seconds=3600,
        ),
    )
    _git(tmp_path, "add", stage_b_document_rel)
    _git(tmp_path, "commit", "-q", "-m", "stage b authorization")
    stage_b_execution_sha = _git(tmp_path, "rev-parse", "HEAD")
    git_state = SubprocessGitStateProvider(str(tmp_path))

    claim_payload = _stage_b_payload(
        authorization_document_sha256=git_state.file_sha256_at_commit(stage_b_document_rel, stage_b_execution_sha),
        authorized_code_commit_sha=authorized_sha,
        observed_execution_commit_sha=stage_b_execution_sha,
        required_ancestor_shas=[ancestor_sha],
        required_rkv_sha=rkv_sha,
        observed_rkv_sha=rkv_sha,
        candidate_manifest_canonical_sha256=candidate_manifest["canonical_sha256"],
    )
    verified_context = verify_authorization_preconditions(
        claim_payload,
        git_state=git_state,
        authorization_document_path=tmp_path / stage_b_document_rel,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=REAL_CONFIG_SHA,
        repository_root=tmp_path,
    )
    consumed_context = claim_authorization(
        claim_payload, repository_root=tmp_path, verified_context=verified_context, git_state=git_state,
    )

    ticks = (1_800_000_000.0 + i for i in itertools.count())
    qualification_artifact = run_b2a_r3_qualification_coordinator(
        candidate_manifest=candidate_manifest,
        expected_config_sha256=REAL_CONFIG_SHA,
        consumed_authorization_context=consumed_context,
        fullkv_worker_runner=lambda ordinal, _timeout: _valid_worker_result(
            ordinal,
            expected_prompt_token_ids_sha256=_FAKE_PROMPT_HASH,
            observed_prompt_token_ids_sha256=_FAKE_PROMPT_HASH,
        ),
        clock=lambda: next(ticks),
        per_candidate_timeout_seconds=7200,
    )
    qualification_path = tmp_path / "results/decisions/b2a_r3_qualification.json"
    write_qualification_artifact_atomic(
        qualification_artifact,
        output_path=qualification_path,
        candidate_manifest=candidate_manifest,
        expected_config_sha256=REAL_CONFIG_SHA,
        stage_b_authorization_context=consumed_context,
    )
    _git(tmp_path, "add", claim_payload["global_claim_path"], "results/decisions/b2a_r3_qualification.json")
    _git(tmp_path, "commit", "-q", "-m", "stage b outputs")

    return candidate_manifest, qualification_artifact


def _build_prod_plan(tmp_path: _Path):
    return construct_production_freeze_plan(
        repository_root=str(tmp_path),
        config_path=REAL_CONFIG_PATH_REL,
        tokenizer_renderer=_fake_renderer,
        tokenizer_repository=MODEL_NAME,
        tokenizer_requested_revision=MODEL_REVISION,
        tokenizer_resolved_revision=MODEL_REVISION,
        tokenizer_local_path="/fake/local/tokenizer/path",
    )


@pytest.fixture
def production_repo(tmp_path):
    candidate_manifest, qualification_artifact = _build_production_shaped_repo(tmp_path)
    return tmp_path, candidate_manifest, qualification_artifact


def test_construct_production_freeze_plan_state_a_initial(production_repo):
    tmp_path, candidate_manifest, qualification_artifact = production_repo
    plan = _build_prod_plan(tmp_path)
    assert plan.publication_state_before == PUBLICATION_STATE_A_INITIAL
    assert plan.selected_unique_id == qualification_artifact["selected_unique_id"]
    assert plan.candidate_manifest_canonical_sha256 == candidate_manifest["canonical_sha256"]
    assert plan.qualification_artifact_canonical_sha256 == qualification_artifact["canonical_sha256"]
    assert plan.tokenizer_resolved_revision == MODEL_REVISION


def test_publish_production_freeze_state_a_completes(production_repo):
    tmp_path, _candidate_manifest, _qualification_artifact = production_repo
    plan = _build_prod_plan(tmp_path)
    result = publish_production_freeze(plan)
    assert result["publication_state_before"] == PUBLICATION_STATE_A_INITIAL
    assert result["publication_state_after"] == PUBLICATION_STATE_B_COMPLETE
    assert result["already_frozen"] is False
    assert result["verification_passed"] is True
    manifest_path = tmp_path / "configs/discovery/b2a_one_example_manifest.json"
    provenance_path = tmp_path / "results/decisions/b2a_r3_selection_provenance.json"
    assert manifest_path.exists() and provenance_path.exists()
    assert _json.loads(manifest_path.read_text())["unique_id"] == result["selected_unique_id"]


def test_publish_production_freeze_state_b_is_idempotent_no_rewrite(production_repo):
    tmp_path, _cm, _qa = production_repo
    publish_production_freeze(_build_prod_plan(tmp_path))
    manifest_path = tmp_path / "configs/discovery/b2a_one_example_manifest.json"
    provenance_path = tmp_path / "results/decisions/b2a_r3_selection_provenance.json"
    manifest_before = manifest_path.read_bytes()
    provenance_before = provenance_path.read_bytes()
    manifest_mtime_before = manifest_path.stat().st_mtime_ns
    provenance_mtime_before = provenance_path.stat().st_mtime_ns

    plan2 = _build_prod_plan(tmp_path)
    assert plan2.publication_state_before == PUBLICATION_STATE_B_COMPLETE
    result2 = publish_production_freeze(plan2)
    assert result2["already_frozen"] is True
    assert result2["publication_state_before"] == PUBLICATION_STATE_B_COMPLETE
    assert result2["publication_state_after"] == PUBLICATION_STATE_B_COMPLETE
    assert manifest_path.read_bytes() == manifest_before
    assert provenance_path.read_bytes() == provenance_before
    assert manifest_path.stat().st_mtime_ns == manifest_mtime_before
    assert provenance_path.stat().st_mtime_ns == provenance_mtime_before


def test_publish_production_freeze_state_d_recovers(production_repo):
    """Manifest-first partial: the new manifest is on disk, provenance is
    absent -- publication must write ONLY the provenance."""
    tmp_path, _cm, _qa = production_repo
    publish_production_freeze(_build_prod_plan(tmp_path))
    manifest_path = tmp_path / "configs/discovery/b2a_one_example_manifest.json"
    provenance_path = tmp_path / "results/decisions/b2a_r3_selection_provenance.json"
    provenance_bytes = provenance_path.read_bytes()
    provenance_path.unlink()
    manifest_bytes_before = manifest_path.read_bytes()

    plan = _build_prod_plan(tmp_path)
    assert plan.publication_state_before == PUBLICATION_STATE_D_MANIFEST_FIRST_PARTIAL
    result = publish_production_freeze(plan)
    assert result["publication_state_before"] == PUBLICATION_STATE_D_MANIFEST_FIRST_PARTIAL
    assert result["publication_state_after"] == PUBLICATION_STATE_B_COMPLETE
    assert manifest_path.read_bytes() == manifest_bytes_before  # manifest untouched
    assert provenance_path.read_bytes() == provenance_bytes


def test_publish_production_freeze_state_c_recovers(production_repo):
    """Provenance-first partial: the expected provenance is already on
    disk, but the manifest is still the historical one -- publication must
    write ONLY the manifest, never re-publish provenance."""
    tmp_path, _cm, _qa = production_repo
    publish_production_freeze(_build_prod_plan(tmp_path))
    manifest_path = tmp_path / "configs/discovery/b2a_one_example_manifest.json"
    provenance_path = tmp_path / "results/decisions/b2a_r3_selection_provenance.json"
    new_manifest_bytes = manifest_path.read_bytes()
    provenance_bytes_before = provenance_path.read_bytes()
    provenance_mtime_before = provenance_path.stat().st_mtime_ns

    # Roll the manifest back to the historical committed bytes (simulating
    # a crash between provenance publication and manifest publication --
    # this repo's construction order publishes manifest first, so State C
    # is the RECOVERABLE-but-non-normal ordering, exercised here directly).
    historical_bytes = SubprocessGitStateProvider(str(tmp_path)).file_text_at_commit(
        "configs/discovery/b2a_one_example_manifest.json",
        SubprocessGitStateProvider(str(tmp_path)).current_commit_sha(),
    ).encode("utf-8")
    manifest_path.write_bytes(historical_bytes)

    plan = _build_prod_plan(tmp_path)
    assert plan.publication_state_before == PUBLICATION_STATE_C_PROVENANCE_FIRST_PARTIAL
    result = publish_production_freeze(plan)
    assert result["publication_state_before"] == PUBLICATION_STATE_C_PROVENANCE_FIRST_PARTIAL
    assert result["publication_state_after"] == PUBLICATION_STATE_B_COMPLETE
    assert manifest_path.read_bytes() == new_manifest_bytes
    assert provenance_path.read_bytes() == provenance_bytes_before
    assert provenance_path.stat().st_mtime_ns == provenance_mtime_before  # never rewritten


def test_publish_production_freeze_state_e_refuses_on_corrupted_manifest(production_repo):
    tmp_path, _cm, _qa = production_repo
    publish_production_freeze(_build_prod_plan(tmp_path))
    manifest_path = tmp_path / "configs/discovery/b2a_one_example_manifest.json"
    provenance_path = tmp_path / "results/decisions/b2a_r3_selection_provenance.json"
    manifest_path.write_text('{"unexpected": "bytes"}')
    provenance_before = provenance_path.read_bytes()

    plan = _build_prod_plan(tmp_path)
    assert plan.publication_state_before == PUBLICATION_STATE_E_INVALID
    with pytest.raises(ProductionPublicationRefused):
        publish_production_freeze(plan)
    assert provenance_path.read_bytes() == provenance_before  # untouched


def test_publish_production_freeze_state_e_refuses_on_unknown_provenance(production_repo):
    """Manifest still historical (State A shape) but an UNKNOWN file
    already sits at the provenance path -- must refuse, never overwrite."""
    tmp_path, _cm, _qa = production_repo
    provenance_path = tmp_path / "results/decisions/b2a_r3_selection_provenance.json"
    provenance_path.write_text('{"unexpected": "provenance"}')
    manifest_path = tmp_path / "configs/discovery/b2a_one_example_manifest.json"
    manifest_before = manifest_path.read_bytes()

    plan = _build_prod_plan(tmp_path)
    assert plan.publication_state_before == PUBLICATION_STATE_E_INVALID
    with pytest.raises(ProductionPublicationRefused):
        publish_production_freeze(plan)
    assert manifest_path.read_bytes() == manifest_before  # untouched
    assert provenance_path.read_text() == '{"unexpected": "provenance"}'  # untouched, never clobbered


def test_no_clobber_provenance_publish_never_overwrites_unknown_file(tmp_path):
    from kvcot.discovery.b2a_r3_freeze import _exclusive_publish_no_clobber

    target = tmp_path / "provenance.json"
    target.write_bytes(b'{"pre-existing": true}')
    with pytest.raises(ProductionPublicationRefused):
        _exclusive_publish_no_clobber(target, b'{"new": true}')
    assert target.read_bytes() == b'{"pre-existing": true}'
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_atomic_replace_refuses_when_live_bytes_do_not_match_expected_historical(tmp_path):
    from kvcot.discovery.b2a_r3_freeze import _atomic_replace_verified_historical

    target = tmp_path / "manifest.json"
    target.write_bytes(b"unexpected live bytes")
    with pytest.raises(ProductionPublicationRefused):
        _atomic_replace_verified_historical(target, expected_current_bytes=b"expected historical", new_bytes=b"new")
    assert target.read_bytes() == b"unexpected live bytes"
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_temporary_write_failure_leaves_targets_unchanged(tmp_path, monkeypatch):
    from kvcot.discovery import b2a_r3_freeze as freeze_module

    target = tmp_path / "manifest.json"
    historical = b'{"historical": true}\n'
    target.write_bytes(historical)

    def _boom(*_args, **_kwargs):
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(freeze_module.os, "fsync", _boom)
    with pytest.raises(OSError):
        freeze_module._atomic_replace_verified_historical(
            target, expected_current_bytes=historical, new_bytes=b'{"new": true}\n'
        )
    assert target.read_bytes() == historical  # unchanged
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []  # temp file cleaned up after the synchronous exception


def test_classify_publication_state_requires_manifest_to_exist(production_repo):
    tmp_path, _cm, _qa = production_repo
    plan = _build_prod_plan(tmp_path)
    manifest_path = _Path(plan.selected_manifest_path)
    manifest_path.unlink()
    assert classify_publication_state(plan=plan) == PUBLICATION_STATE_E_INVALID


def test_verify_git_worktree_safety_rejects_unrelated_untracked_file(production_repo):
    tmp_path, _cm, _qa = production_repo
    plan = _build_prod_plan(tmp_path)
    (tmp_path / "unrelated_untracked_file.txt").write_text("surprise")
    with pytest.raises(ProductionPublicationRefused):
        verify_git_worktree_safety_for_freeze(plan)


def test_verify_git_worktree_safety_rejects_altered_candidate_manifest(production_repo):
    tmp_path, _cm, _qa = production_repo
    plan = _build_prod_plan(tmp_path)
    candidates_path = tmp_path / "configs/discovery/b2a_r3_candidate_manifest.json"
    candidates_path.write_text(candidates_path.read_text() + "\n")  # trivial tracked-file modification
    with pytest.raises(ProductionPublicationRefused):
        verify_git_worktree_safety_for_freeze(plan)


def test_verify_git_worktree_safety_allows_the_two_expected_output_paths(production_repo):
    tmp_path, _cm, _qa = production_repo
    plan = _build_prod_plan(tmp_path)
    # Simulate the exact State-A -> State-B transition's on-disk deltas
    # (manifest modified, provenance newly created) without actually
    # calling publish -- these must NOT trip the worktree guard.
    (tmp_path / "configs/discovery/b2a_one_example_manifest.json").write_bytes(plan.expected_new_manifest_bytes)
    (tmp_path / "results/decisions/b2a_r3_selection_provenance.json").write_bytes(plan.expected_provenance_bytes)
    verify_git_worktree_safety_for_freeze(plan)  # must not raise


def test_verify_git_worktree_safety_rejects_wrong_rkv_pin(production_repo):
    tmp_path, _cm, _qa = production_repo
    plan = _build_prod_plan(tmp_path)
    other_dir = tmp_path / "third_party" / "R-KV-other"
    other_dir.mkdir(parents=True)
    _git(other_dir, "init", "-q")
    _git(other_dir, "config", "user.email", "a@b.com")
    _git(other_dir, "config", "user.name", "t")
    (other_dir / "f.txt").write_text("x")
    _git(other_dir, "add", "f.txt")
    _git(other_dir, "commit", "-q", "-m", "x")
    other_sha = _git(other_dir, "rev-parse", "HEAD")
    import shutil as _shutil

    _shutil.rmtree(other_dir)
    _git(tmp_path, "rm", "--cached", "third_party/R-KV")
    _git(tmp_path, "update-index", "--add", "--cacheinfo", f"160000,{other_sha},third_party/R-KV")
    with pytest.raises(ProductionPublicationRefused):
        verify_git_worktree_safety_for_freeze(plan)


def test_construct_production_freeze_plan_rejects_wrong_authorization_id(production_repo):
    tmp_path, _cm, qualification_artifact = production_repo
    qualification_path = tmp_path / "results/decisions/b2a_r3_qualification.json"
    tampered = dict(qualification_artifact)
    tampered["stage_b_authorization_id"] = "stage-b-does-not-exist"
    tampered["canonical_sha256"] = sha256_json({k: v for k, v in tampered.items() if k != "canonical_sha256"})
    qualification_path.write_text(_json.dumps(tampered, indent=2) + "\n")
    with pytest.raises(Exception):
        _build_prod_plan(tmp_path)


def test_construct_production_freeze_plan_rejects_altered_candidate_manifest(production_repo):
    tmp_path, candidate_manifest, _qa = production_repo
    candidates_path = tmp_path / "configs/discovery/b2a_r3_candidate_manifest.json"
    tampered = dict(candidate_manifest)
    candidates = [dict(c) for c in tampered["candidates"]]
    candidates[0] = dict(candidates[0], raw_row_sha256="0" * 64)
    tampered["candidates"] = candidates
    candidates_path.write_text(_json.dumps(tampered, indent=2) + "\n")
    with pytest.raises(Exception):
        _build_prod_plan(tmp_path)


def test_construct_production_freeze_plan_rejects_altered_qualification_artifact(production_repo):
    tmp_path, _cm, qualification_artifact = production_repo
    qualification_path = tmp_path / "results/decisions/b2a_r3_qualification.json"
    tampered = dict(qualification_artifact)
    tampered["selected_unique_id"] = "some/other/row.json"
    # deliberately leave canonical_sha256 stale -- must fail the self-hash check
    qualification_path.write_text(_json.dumps(tampered, indent=2) + "\n")
    with pytest.raises(Exception):
        _build_prod_plan(tmp_path)


def test_full_chain_verification_is_mandatory_before_state_b(production_repo, monkeypatch):
    """If post-publication full-chain verification were skipped, a
    corrupted-but-byte-matching-by-construction scenario could slip
    through -- assert `publish_production_freeze` always calls the real
    verifier by making it raise and confirming the whole call fails."""
    tmp_path, _cm, _qa = production_repo
    from kvcot.discovery import b2a_r3_freeze as freeze_module

    plan = _build_prod_plan(tmp_path)

    def _explode(*_args, **_kwargs):
        raise RowFreezeRefusedR3("synthetic verification failure")

    monkeypatch.setattr(freeze_module, "verify_selection_provenance", _explode)
    with pytest.raises(RowFreezeRefusedR3):
        publish_production_freeze(plan)


# ==========================================================================
# Phase 2: production tokenizer renderer boundary
# (`kvcot.discovery.b2a_r3_production_tokenizer`). Tests that need the real
# local snapshot skip cleanly when it is not cached on this machine (CI
# runners have no model cache) -- everything else here needs no snapshot at
# all and always runs.
# ==========================================================================

from kvcot.discovery.b2a_r3_contract import TOKENIZER_NAME, TOKENIZER_REVISION
from kvcot.discovery.b2a_r3_production_tokenizer import (
    ProductionTokenizerResolutionRefused,
    build_production_tokenizer_renderer,
    render_production_prompt,
    resolve_production_tokenizer_snapshot,
)
from kvcot.discovery.snapshot_boundary import SnapshotBoundaryError


def _real_snapshot_or_skip():
    try:
        return resolve_production_tokenizer_snapshot()
    except ProductionTokenizerResolutionRefused:
        pytest.skip("exact local tokenizer snapshot is not cached on this machine")


def test_resolve_production_tokenizer_snapshot_exact_repository_and_revision():
    snapshot = _real_snapshot_or_skip()
    assert snapshot.repository_id == TOKENIZER_NAME
    assert snapshot.asset_type == "tokenizer"
    assert snapshot.requested_revision == TOKENIZER_REVISION
    assert snapshot.resolved_revision == TOKENIZER_REVISION
    assert snapshot.local_files_only is True
    assert snapshot.local_path  # non-empty exact local path


def test_render_production_prompt_reuses_frozen_convention_and_hashes_reproduce():
    snapshot = _real_snapshot_or_skip()
    result = render_production_prompt({"problem": "What is 2 + 2?"}, snapshot=snapshot)
    assert result.prompt_token_count == len(result.prompt_token_ids)
    assert result.prompt_token_ids_sha256 == sha256_int_ids(list(result.prompt_token_ids))
    assert result.tokenizer_revision_used_for_prompt_hash == TOKENIZER_REVISION
    assert result.prompt_rendering_config.message_roles == ("user",)
    assert result.prompt_rendering_config.add_generation_prompt is True
    assert result.prompt_rendering_config.tokenize is True
    # re-rendering the SAME problem text must reproduce every hash exactly
    result2 = render_production_prompt({"problem": "What is 2 + 2?"}, snapshot=snapshot)
    assert result2.prompt_token_ids == result.prompt_token_ids
    assert result2.rendered_user_message_sha256 == result.rendered_user_message_sha256
    assert result2.chat_template_source_sha256 == result.chat_template_source_sha256


def test_build_production_tokenizer_renderer_returns_a_valid_tokenizer_renderer():
    _real_snapshot_or_skip()
    renderer = build_production_tokenizer_renderer()
    result = renderer({"problem": "What is 3 + 3?"})
    assert result.prompt_token_count > 0


def test_resolve_production_tokenizer_snapshot_rejects_revision_mismatch(monkeypatch):
    import kvcot.discovery.b2a_r3_production_tokenizer as tok_module

    def _fake_resolve(**_kwargs):
        from kvcot.discovery.snapshot_boundary import VerifiedLocalSnapshot

        return VerifiedLocalSnapshot(
            repository_id=TOKENIZER_NAME, requested_revision=TOKENIZER_REVISION,
            resolved_revision="0" * 40, asset_type="tokenizer", local_path="/fake",
            files=(), total_bytes=0, required_free_bytes=0, free_bytes=0, local_files_only=True,
        )

    monkeypatch.setattr(tok_module, "resolve_local_snapshot", _fake_resolve)
    with pytest.raises(ProductionTokenizerResolutionRefused):
        resolve_production_tokenizer_snapshot()


def test_resolve_production_tokenizer_snapshot_never_falls_back_to_network(monkeypatch):
    import kvcot.discovery.b2a_r3_production_tokenizer as tok_module

    def _boom(**_kwargs):
        raise SnapshotBoundaryError("exact local snapshot unavailable; network/floating fallback is forbidden")

    monkeypatch.setattr(tok_module, "resolve_local_snapshot", _boom)
    with pytest.raises(ProductionTokenizerResolutionRefused):
        resolve_production_tokenizer_snapshot()


def test_render_production_prompt_rejects_missing_chat_template(monkeypatch):
    import kvcot.discovery.b2a_r3_production_tokenizer as tok_module
    from kvcot.discovery.snapshot_boundary import VerifiedLocalSnapshot

    class _FakeTokenizer:
        chat_template = None

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeTokenizer()

    fake_transformers = type("_m", (), {"AutoTokenizer": _FakeAutoTokenizer})
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    snapshot = VerifiedLocalSnapshot(
        repository_id=TOKENIZER_NAME, requested_revision=TOKENIZER_REVISION,
        resolved_revision=TOKENIZER_REVISION, asset_type="tokenizer", local_path="/fake",
        files=(), total_bytes=0, required_free_bytes=0, free_bytes=0, local_files_only=True,
    )
    with pytest.raises(RowFreezeRefusedR3):
        render_production_prompt({"problem": "x"}, snapshot=snapshot)


def test_render_production_prompt_rejects_empty_prompt(monkeypatch):
    import kvcot.discovery.b2a_r3_production_tokenizer as tok_module
    from kvcot.discovery.snapshot_boundary import VerifiedLocalSnapshot

    class _FakeTokenizer:
        chat_template = "not empty"

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeTokenizer()

    fake_transformers = type("_m", (), {"AutoTokenizer": _FakeAutoTokenizer})
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "kvcot.discovery.manifest_prepare.render_with_loaded_tokenizer",
        lambda _tokenizer, _row: ("user message", [{"role": "user", "content": "user message"}], []),
    )

    snapshot = VerifiedLocalSnapshot(
        repository_id=TOKENIZER_NAME, requested_revision=TOKENIZER_REVISION,
        resolved_revision=TOKENIZER_REVISION, asset_type="tokenizer", local_path="/fake",
        files=(), total_bytes=0, required_free_bytes=0, free_bytes=0, local_files_only=True,
    )
    with pytest.raises(RowFreezeRefusedR3):
        render_production_prompt({"problem": "x"}, snapshot=snapshot)


def test_production_tokenizer_module_never_imports_torch_or_dataset_fetch_helpers():
    """Static source scan: no direct `torch` import anywhere in the
    production tokenizer module, and no dataset-fetch helper
    (`_fetch_pinned_dataset_row`) is ever imported -- the row is always
    supplied by the caller, never refetched here."""
    import ast

    path = _Path("src/kvcot/discovery/b2a_r3_production_tokenizer.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found_names: set[str] = set()
    found_attrs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found_names.add(node.module)
            for alias in node.names:
                found_attrs.add(alias.name)
        elif isinstance(node, ast.Name) and node.id in ("AutoModel", "torch"):
            found_attrs.add(node.id)
    assert not any(n == "torch" or n.startswith("torch.") for n in found_names)
    assert "AutoModel" not in found_attrs
    assert "_fetch_pinned_dataset_row" not in found_attrs
    assert not any("cuda" in n.lower() for n in found_names)
