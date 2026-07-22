"""B2A-R2 row-freezing tests (2026-07-22) -- proving every free-form
substitution path fails closed. `_render_and_tokenize` is monkeypatched (a
real tokenizer load is unnecessary and would require network/GPU-host
tokenizer files); every hash/identity check under test is otherwise real.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kvcot.discovery.b2a_r2_freeze import RowFreezeRefused, freeze_qualified_row, validate_freeze_request
from kvcot.discovery.discovery_config import load_discovery_config
from kvcot.utils.hashing import sha256_json

CONFIG_PATH = "configs/discovery/llama8b_math500_b1024.yaml"


@pytest.fixture
def config():
    return load_discovery_config(CONFIG_PATH)


def _row(unique_id="test/algebra/1.json"):
    return {
        "problem": "What is 2+2?", "solution": "4", "answer": "4",
        "subject": "Algebra", "level": "5", "unique_id": unique_id,
    }


def _candidate_manifest(config, *, ordinals_and_ids=((0, "test/algebra/1.json"),)):
    candidates = []
    for ordinal, unique_id in ordinals_and_ids:
        row = _row(unique_id)
        candidates.append({
            "candidate_ordinal": ordinal,
            "source_example_index": 100 + ordinal,
            "unique_id": unique_id,
            "subject": "Algebra",
            "level": 5,
            "row": row,
            "raw_row_sha256": sha256_json(row),
            "problem_sha256": "b" * 64,
            "gold_answer_sha256": "c" * 64,
            "ordering_hash": f"{ordinal:064x}",
            "dataset_revision": config.dataset.revision,
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "budget": config.rkv.budget,
            "protocol_version": "faithkv-b2a-r2-row-order-v1",
        })
    m = {
        "protocol_version": "faithkv-b2a-r2-row-order-v1",
        "dataset_repo": "HuggingFaceH4/MATH-500",
        "dataset_revision": config.dataset.revision,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "budget": config.rkv.budget,
        "level": 5,
        "candidate_count": len(candidates),
        "eligible_population_size": len(candidates),
        "candidates": candidates,
    }
    m["canonical_sha256"] = sha256_json(m)
    return m


def _qualification_artifact(config, candidate_manifest, *, selected_ordinal=0, all_conditions_true=True):
    selected_row = next(c for c in candidate_manifest["candidates"] if c["candidate_ordinal"] == selected_ordinal)
    conditions = {name: True for name in (
        "no_cap_hit", "fullkv_answer_verifiable", "fullkv_answer_correct",
        "predicted_schedule_has_at_least_three_events", "at_least_three_events_have_49_future_tokens",
        "identity_checks_pass", "batch_size_is_one", "all_parameters_on_cuda", "no_offload",
        "peak_memory_within_limit",
    )}
    if not all_conditions_true:
        conditions["fullkv_answer_correct"] = False
    attempted = [{
        "candidate_ordinal": selected_ordinal,
        "unique_id": selected_row["unique_id"],
        "qualified": all_conditions_true,
        "failed_conditions": [] if all_conditions_true else ["fullkv_answer_correct"],
        "conditions": conditions,
    }]
    return {
        "protocol_version": "faithkv-b2a-r2-qualification-v1",
        "candidate_manifest_path": "configs/discovery/b2a_r2_candidate_manifest.json",
        "candidate_manifest_hash": candidate_manifest["canonical_sha256"],
        "config_hash": "cfg" * 20 + "a",
        "dataset_revision": config.dataset.revision,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "budget": config.rkv.budget,
        "attempted": attempted,
        "selected_ordinal": selected_ordinal if all_conditions_true else None,
        "selected_unique_id": selected_row["unique_id"] if all_conditions_true else None,
    }


@pytest.fixture
def fake_render_and_tokenize(monkeypatch):
    fake_tokenizer = SimpleNamespace(chat_template="fake template")

    def _fake(row, tokenizer_name, tokenizer_revision, *, local_only_path=None):
        return fake_tokenizer, "rendered message", [{"role": "user", "content": "x"}], [1, 2, 3, 4, 5]

    monkeypatch.setattr("kvcot.discovery.manifest_prepare._render_and_tokenize", _fake)
    return _fake


def _run_freeze(config, candidate_manifest, qualification_artifact, tmp_path):
    return freeze_qualified_row(
        config=config,
        qualification_artifact=qualification_artifact,
        qualification_artifact_path="results/decisions/qual.json",
        qualification_artifact_hash="qualhash" * 8,
        candidate_manifest=candidate_manifest,
        candidate_manifest_path="configs/discovery/b2a_r2_candidate_manifest.json",
        manifest_path=tmp_path / "b2a_one_example_manifest.json",
    )


def test_successful_freeze_writes_manifest_and_provenance(config, fake_render_and_tokenize, tmp_path):
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)

    manifest, provenance = _run_freeze(config, candidate_manifest, qual, tmp_path)

    assert manifest.unique_id == "test/algebra/1.json"
    assert manifest.example_index == 100  # source_example_index, never a caller-supplied index
    assert manifest.gold_answer == "4"
    assert provenance.selected_ordinal == 0
    assert provenance.selected_unique_id == "test/algebra/1.json"
    assert provenance.candidate_manifest_hash == candidate_manifest["canonical_sha256"]
    assert (tmp_path / "b2a_one_example_manifest.json").is_file()


def test_rejects_when_candidate_manifest_hash_does_not_match_qualification_artifact(
    config, fake_render_and_tokenize, tmp_path
):
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    qual["candidate_manifest_hash"] = "tampered" * 8

    with pytest.raises(RowFreezeRefused, match="candidate_manifest_hash"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_rejects_dataset_revision_mismatch(config, fake_render_and_tokenize, tmp_path):
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    qual["dataset_revision"] = "wrong-revision"

    with pytest.raises(RowFreezeRefused, match="dataset_revision"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_rejects_budget_mismatch(config, fake_render_and_tokenize, tmp_path):
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    qual["budget"] = 1

    with pytest.raises(RowFreezeRefused, match="budget"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_rejects_when_no_row_qualified(config, fake_render_and_tokenize, tmp_path):
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest, all_conditions_true=False)

    with pytest.raises(RowFreezeRefused, match="no selected row"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_rejects_when_conditions_map_has_a_false_entry_despite_qualified_true(
    config, fake_render_and_tokenize, tmp_path
):
    """Defense in depth: even if `qualified`/`selected_ordinal` were
    tampered to look true, a false entry anywhere in the condition map
    itself must still be caught."""
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    qual["attempted"][0]["conditions"]["peak_memory_within_limit"] = False
    # qualified/selected_ordinal/selected_unique_id left as if it passed.

    with pytest.raises(RowFreezeRefused, match="condition map"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_rejects_arbitrary_ordinal_not_present_in_candidate_manifest(config, fake_render_and_tokenize, tmp_path):
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    qual["selected_ordinal"] = 7  # never attempted, never in the candidate manifest
    qual["attempted"][0]["candidate_ordinal"] = 7

    with pytest.raises(RowFreezeRefused, match="does not have exactly one candidate at ordinal"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_rejects_unique_id_mismatch_between_artifact_and_candidate_manifest(
    config, fake_render_and_tokenize, tmp_path
):
    """Simulates an attacker/bug substituting a DIFFERENT row at the same
    ordinal after qualification ran -- the frozen unique_id must still
    match."""
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    # Swap the candidate manifest's row content at ordinal 0 for a
    # different unique_id, without updating the qualification artifact.
    candidate_manifest["candidates"][0]["unique_id"] = "test/algebra/999.json"
    candidate_manifest["candidates"][0]["row"]["unique_id"] = "test/algebra/999.json"
    candidate_manifest["canonical_sha256"] = sha256_json(
        {k: v for k, v in candidate_manifest.items() if k != "canonical_sha256"}
    )
    qual["candidate_manifest_hash"] = candidate_manifest["canonical_sha256"]

    with pytest.raises(RowFreezeRefused, match="unique_id"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_rejects_tampered_row_content_that_does_not_reproduce_its_hash(config, fake_render_and_tokenize, tmp_path):
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    # Mutate the row AFTER computing raw_row_sha256 -- content no longer
    # reproduces its own recorded hash.
    candidate_manifest["candidates"][0]["row"]["problem"] = "TAMPERED PROBLEM TEXT"

    with pytest.raises(RowFreezeRefused, match="raw content hash"):
        _run_freeze(config, candidate_manifest, qual, tmp_path)


def test_validate_freeze_request_is_pure_and_side_effect_free(config):
    """`validate_freeze_request` alone must never touch a tokenizer/network
    -- it is safe to call with no `_render_and_tokenize` patch at all."""
    candidate_manifest = _candidate_manifest(config)
    qual = _qualification_artifact(config, candidate_manifest)
    row, ordinal, unique_id = validate_freeze_request(
        config=config, qualification_artifact=qual, candidate_manifest=candidate_manifest,
        candidate_manifest_path="path.json",
    )
    assert ordinal == 0
    assert unique_id == "test/algebra/1.json"
    assert row["unique_id"] == "test/algebra/1.json"
