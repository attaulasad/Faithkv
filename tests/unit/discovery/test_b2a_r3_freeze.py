"""B2A-R3 selected-row freezer tests. Synthetic-only fixtures -- no torch,
no CUDA, no real tokenizer, no real freeze against the actual repository
paths. Every write goes through `tmp_path`."""
from __future__ import annotations

import pytest

from kvcot.discovery.b2a_r3_artifacts import SELECTION_STATUS_NONE_QUALIFIED, SELECTION_STATUS_SELECTED
from kvcot.discovery.b2a_r3_candidates import build_candidate_manifest
from kvcot.discovery.b2a_r3_contract import PROMPT_SPECIAL_TOKENS_NOTE, SELECTED_MANIFEST_PATH
from kvcot.discovery.b2a_r3_freeze import (
    PromptRenderingResult,
    RowFreezeRefusedR3,
    construct_selected_manifest_and_provenance,
    plan_freeze_dry_run,
    verify_selection_provenance,
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
    )
    if not qualified:
        overrides["answer_verification_status"] = "incorrect"
    evidence = _valid_evidence(**overrides)
    return build_qualification_outcome(
        evidence, candidate_manifest=candidate_manifest, expected_config_sha256=CONFIG_SHA
    )


def _qualification_artifact(candidate_manifest, attempted, *, selection_status, first_ordinal, selected_unique_id):
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
        QUALIFICATION_PROTOCOL_VERSION,
        RUNTIME_PREDICTOR_VERSION,
        RUNTIME_SOURCE_ARTIFACT_PATH,
        RUNTIME_SOURCE_ARTIFACT_SHA256,
        TOKENIZER_NAME,
        TOKENIZER_REVISION,
    )

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
        "qualification_stopped_reason": "first_pass" if first_ordinal is not None else "all_candidates_exhausted",
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
