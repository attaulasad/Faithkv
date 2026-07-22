"""B2A-R2 candidate manifest construction tests (2026-07-22).

Every test uses injected fake rows -- no network access, no dataset
download.
"""
from __future__ import annotations

import pytest

from kvcot.discovery.b2a_r2_candidates import (
    CANDIDATE_MANIFEST_PROTOCOL_VERSION,
    build_candidate_manifest,
)
from kvcot.discovery.manifest_prepare import ManifestPreparationError

DATASET_REPO = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MODEL_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"
TOKENIZER_REVISION = MODEL_REVISION
BUDGET = 1024


def _row(unique_id: str, level: str = "5", subject: str = "Precalculus") -> dict:
    return {
        "problem": f"problem for {unique_id}",
        "solution": f"solution for {unique_id}",
        "answer": f"answer-{unique_id}",
        "subject": subject,
        "level": level,
        "unique_id": unique_id,
    }


def _build(rows, **overrides):
    kwargs = dict(
        dataset_repo=DATASET_REPO, dataset_revision=DATASET_REVISION, model_revision=MODEL_REVISION,
        tokenizer_revision=TOKENIZER_REVISION, budget=BUDGET,
    )
    kwargs.update(overrides)
    return build_candidate_manifest(rows, **kwargs)


def test_filters_to_the_requested_level_only():
    rows = [_row("a", level="5"), _row("b", level="3"), _row("c", level="5"), _row("d", level="1")]
    manifest = _build(rows)
    levels = {c["level"] for c in manifest["candidates"]}
    assert levels == {5}
    assert manifest["eligible_population_size"] == 2


def test_duplicate_unique_id_is_rejected():
    rows = [_row("dup"), _row("dup")]
    with pytest.raises(ManifestPreparationError, match="duplicate unique_id"):
        _build(rows)


def test_malformed_row_schema_is_rejected():
    bad_row = {"problem": "p", "answer": "a"}  # missing required columns
    with pytest.raises(ManifestPreparationError, match="unexpected columns"):
        _build([bad_row])


def test_selects_at_most_candidate_count_rows():
    rows = [_row(f"id-{i}") for i in range(30)]
    manifest = _build(rows)
    assert manifest["candidate_count"] == 12
    assert len(manifest["candidates"]) == 12


def test_selects_fewer_than_twelve_when_population_is_smaller():
    rows = [_row(f"id-{i}") for i in range(5)]
    manifest = _build(rows)
    assert manifest["candidate_count"] == 5


def test_ordering_is_deterministic_across_independent_calls():
    rows = [_row(f"id-{i}") for i in range(20)]
    manifest_a = _build(list(rows))
    manifest_b = _build(list(reversed(rows)))  # different INPUT order
    ids_a = [c["unique_id"] for c in manifest_a["candidates"]]
    ids_b = [c["unique_id"] for c in manifest_b["candidates"]]
    assert ids_a == ids_b, "candidate order must depend only on content, never on input row order"


def test_ordering_is_never_based_on_row_content_beyond_identity_fields():
    """Two runs against the SAME rows must produce the SAME order even if
    unrelated fields (problem/solution text) differ, as long as identity
    fields (dataset/model revision, budget, unique_id) are unchanged --
    i.e. the ordering hash is a pure function of identity, not of problem
    length/content/generation outcome."""
    rows_a = [_row(f"id-{i}") for i in range(10)]
    rows_b = [dict(r, problem="COMPLETELY DIFFERENT TEXT " * 50) for r in rows_a]
    order_a = [c["unique_id"] for c in _build(rows_a)["candidates"]]
    order_b = [c["unique_id"] for c in _build(rows_b)["candidates"]]
    assert order_a == order_b


def test_ordering_changes_if_budget_changes():
    rows = [_row(f"id-{i}") for i in range(10)]
    order_b1024 = [c["unique_id"] for c in _build(rows, budget=1024)["candidates"]]
    order_b256 = [c["unique_id"] for c in _build(rows, budget=256)["candidates"]]
    assert order_b1024 != order_b256, "ordering hash must incorporate budget -- different budget, different order"


def test_canonical_hash_is_stable_and_present():
    rows = [_row(f"id-{i}") for i in range(15)]
    manifest_1 = _build(rows)
    manifest_2 = _build(rows)
    assert manifest_1["canonical_sha256"] == manifest_2["canonical_sha256"]
    assert len(manifest_1["canonical_sha256"]) == 64


def test_canonical_hash_excludes_itself():
    rows = [_row(f"id-{i}") for i in range(3)]
    manifest = _build(rows)
    stripped = dict(manifest)
    del stripped["canonical_sha256"]
    from kvcot.utils.hashing import sha256_json

    assert manifest["canonical_sha256"] == sha256_json(stripped)


def test_every_candidate_carries_full_identity_and_hash_fields():
    rows = [_row(f"id-{i}") for i in range(3)]
    manifest = _build(rows)
    required_fields = {
        "candidate_ordinal", "source_example_index", "unique_id", "subject", "level", "row",
        "raw_row_sha256", "problem_sha256", "gold_answer_sha256", "ordering_hash", "dataset_revision",
        "model_revision", "tokenizer_revision", "budget", "protocol_version",
    }
    for candidate in manifest["candidates"]:
        assert required_fields.issubset(candidate.keys())
        assert candidate["protocol_version"] == CANDIDATE_MANIFEST_PROTOCOL_VERSION
        assert candidate["dataset_revision"] == DATASET_REVISION
        assert candidate["model_revision"] == MODEL_REVISION
        assert candidate["budget"] == BUDGET


def test_candidate_ordinals_are_zero_indexed_and_sequential():
    rows = [_row(f"id-{i}") for i in range(6)]
    manifest = _build(rows)
    ordinals = [c["candidate_ordinal"] for c in manifest["candidates"]]
    assert ordinals == list(range(len(manifest["candidates"])))


def test_never_selects_by_observed_generation_length_no_such_input_exists():
    """The builder's signature accepts only raw dataset rows -- there is no
    generation-length/compaction-count parameter it could possibly sort by,
    structurally enforcing outcome-blind selection."""
    import inspect

    from kvcot.discovery.b2a_r2_candidates import build_candidate_manifest as fn

    params = set(inspect.signature(fn).parameters)
    assert not params & {"generated_length", "compaction_count", "eligible_events", "qualification_result"}
