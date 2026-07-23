"""B2A-R3 deterministic candidate-manifest construction tests. Every test
uses injected fake rows -- no network access, no dataset download."""
from __future__ import annotations

import pytest

from kvcot.discovery.b2a_r3_candidates import (
    CandidateManifestR3,
    atomic_write_json,
    build_candidate_manifest,
    verify_candidate_manifest_against_dataset,
    verify_candidate_manifest_structure,
)
from kvcot.discovery.b2a_r3_contract import (
    CANDIDATE_ORDER_PROTOCOL_VERSION,
    CANDIDATES_PER_LEVEL,
    EXCLUSION_SET,
)
from kvcot.discovery.manifest_prepare import ManifestPreparationError

DATASET_REPO = "HuggingFaceH4/MATH-500"
DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"
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
        dataset_repo=DATASET_REPO, dataset_config="default", dataset_split="test",
        dataset_revision=DATASET_REVISION, model_name=MODEL_NAME, model_revision=MODEL_REVISION,
        tokenizer_name=MODEL_NAME, tokenizer_revision=MODEL_REVISION, budget=BUDGET,
        config_path="configs/discovery/llama8b_math500_b1024.yaml", config_sha256="a" * 64,
    )
    kwargs.update(overrides)
    return build_candidate_manifest(rows, **kwargs)


def _population(n_per_level: int = 10, exclude: tuple[str, ...] = ()) -> list[dict]:
    rows = []
    for i in range(n_per_level):
        rows.append(_row(f"level4-{i:03d}", level="4"))
        rows.append(_row(f"level5-{i:03d}", level="5"))
    return [r for r in rows if r["unique_id"] not in exclude]


def test_filters_to_level_4_and_5_only():
    rows = _population(10) + [_row("x", level="3"), _row("y", level="1")]
    manifest = _build(rows)
    levels = {c["level"] for c in manifest["candidates"]}
    assert levels == {4, 5}


def test_duplicate_unique_id_rejected_across_whole_population():
    rows = _population(10)
    rows.append(_row("level4-000", level="4"))  # duplicate
    with pytest.raises(ManifestPreparationError, match="duplicate unique_id"):
        _build(rows)


def test_malformed_row_schema_rejected():
    bad_row = {"problem": "p", "answer": "a"}
    with pytest.raises(ManifestPreparationError, match="unexpected columns"):
        _build([bad_row])


def test_insufficient_population_rejected():
    rows = _population(3)  # only 3 per level, need 8
    with pytest.raises(ManifestPreparationError, match="insufficient eligible rows"):
        _build(rows)


def test_exclusion_set_rows_are_removed():
    excluded_id = next(iter(EXCLUSION_SET))
    rows = _population(10) + [_row(excluded_id, level="5")]
    manifest = _build(rows)
    ids = {c["unique_id"] for c in manifest["candidates"]}
    assert excluded_id not in ids


def test_exactly_16_candidates_8_and_8():
    rows = _population(10)
    manifest = _build(rows)
    assert manifest["candidate_count"] == 16
    assert manifest["level_mixture"] == {"level_4": 8, "level_5": 8}
    level4 = [c for c in manifest["candidates"] if c["level"] == 4]
    level5 = [c for c in manifest["candidates"] if c["level"] == 5]
    assert len(level4) == 8
    assert len(level5) == 8


def test_first_eight_ordinals_are_four_and_four():
    rows = _population(10)
    manifest = _build(rows)
    first_eight = manifest["candidates"][:8]
    levels = [c["level"] for c in first_eight]
    assert levels.count(4) == 4
    assert levels.count(5) == 4


def test_interleaving_is_exact_level4_first_alternating():
    rows = _population(10)
    manifest = _build(rows)
    levels_in_order = [c["level"] for c in manifest["candidates"]]
    assert levels_in_order == [4, 5] * 8


def test_ordinal_zero_is_level_4():
    rows = _population(10)
    manifest = _build(rows)
    assert manifest["candidates"][0]["level"] == 4
    assert manifest["candidates"][0]["candidate_ordinal"] == 0


def test_candidate_ordinals_assigned_only_after_interleaving():
    rows = _population(10)
    manifest = _build(rows)
    ordinals = [c["candidate_ordinal"] for c in manifest["candidates"]]
    assert ordinals == list(range(16))


def test_no_global_resort_after_interleave():
    """A level-5 candidate with a lexicographically smaller ordering_hash
    than a later level-4 candidate must NOT cause global re-sorting --
    interleave order is level-4-rank-i/level-5-rank-i alternating,
    regardless of the two subsets' relative hash values."""
    rows = _population(10)
    manifest = _build(rows)
    level4_hashes = [c["ordering_hash"] for c in manifest["candidates"] if c["level"] == 4]
    level5_hashes = [c["ordering_hash"] for c in manifest["candidates"] if c["level"] == 5]
    assert level4_hashes == sorted(level4_hashes)
    assert level5_hashes == sorted(level5_hashes)
    positions_level4 = [i for i, c in enumerate(manifest["candidates"]) if c["level"] == 4]
    positions_level5 = [i for i, c in enumerate(manifest["candidates"]) if c["level"] == 5]
    assert positions_level4 == [0, 2, 4, 6, 8, 10, 12, 14]
    assert positions_level5 == [1, 3, 5, 7, 9, 11, 13, 15]


def test_ordering_deterministic_regardless_of_input_row_order():
    rows = _population(10)
    manifest_a = _build(list(rows))
    manifest_b = _build(list(reversed(rows)))
    ids_a = [c["unique_id"] for c in manifest_a["candidates"]]
    ids_b = [c["unique_id"] for c in manifest_b["candidates"]]
    assert ids_a == ids_b


def test_ordering_ignores_problem_text_content():
    rows_a = _population(10)
    rows_b = [dict(r, problem="COMPLETELY DIFFERENT " * 20) for r in rows_a]
    ids_a = [c["unique_id"] for c in _build(rows_a)["candidates"]]
    ids_b = [c["unique_id"] for c in _build(rows_b)["candidates"]]
    assert ids_a == ids_b


def test_non_frozen_budget_is_rejected():
    rows = _population(10)
    _build(rows, budget=1024)
    with pytest.raises(Exception):
        _build(rows, budget=2048)


def test_canonical_hash_stable_and_excludes_itself():
    rows = _population(10)
    m1 = _build(rows)
    m2 = _build(rows)
    assert m1["canonical_sha256"] == m2["canonical_sha256"]

    from kvcot.utils.hashing import sha256_json

    stripped = dict(m1)
    del stripped["canonical_sha256"]
    assert m1["canonical_sha256"] == sha256_json(stripped)


def test_no_timestamp_or_model_outcome_fields():
    rows = _population(10)
    manifest = _build(rows)
    forbidden_top_level = {"created_at", "generated_at", "fetch_time", "random_id"}
    assert not (forbidden_top_level & set(manifest.keys()))
    for candidate in manifest["candidates"]:
        forbidden_candidate = {
            "generated_length", "model_outcome", "answer_correct", "compaction_count", "created_at",
        }
        assert not (forbidden_candidate & set(candidate.keys()))


def test_embedded_row_verification_formulas():
    rows = _population(10)
    manifest = _build(rows)
    from kvcot.utils.hashing import sha256_json, sha256_text

    for candidate in manifest["candidates"]:
        row = candidate["row"]
        assert sha256_json(row) == candidate["raw_row_sha256"]
        assert sha256_text(row["problem"]) == candidate["problem_sha256"]
        assert sha256_text(row["answer"]) == candidate["gold_answer_sha256"]
        assert row["unique_id"] == candidate["unique_id"]
        assert row["subject"] == candidate["subject"]
        assert int(row["level"]) == candidate["level"]


def test_verify_candidate_manifest_structure_accepts_valid_manifest():
    rows = _population(10)
    manifest = _build(rows)
    typed = verify_candidate_manifest_structure(manifest)
    assert isinstance(typed, CandidateManifestR3)


def test_verify_candidate_manifest_structure_rejects_unknown_field():
    rows = _population(10)
    manifest = dict(_build(rows))
    manifest["extra_field"] = "not allowed"
    with pytest.raises(Exception):
        verify_candidate_manifest_structure(manifest)


def test_verify_candidate_manifest_structure_rejects_tampered_hash():
    rows = _population(10)
    manifest = dict(_build(rows))
    manifest["canonical_sha256"] = "0" * 64
    with pytest.raises(Exception):
        verify_candidate_manifest_structure(manifest)


def test_verify_candidate_manifest_structure_rejects_broken_interleave():
    rows = _population(10)
    manifest = dict(_build(rows))
    candidates = list(manifest["candidates"])
    candidates[0], candidates[1] = candidates[1], candidates[0]  # swap ordinal 0/1 rows' content
    # Fix ordinals back so the swap is a genuine level-mismatch, not merely
    # an ordinal renumbering.
    fixed = []
    for i, c in enumerate(candidates):
        c = dict(c)
        c["candidate_ordinal"] = i
        fixed.append(c)
    manifest = dict(manifest)
    manifest["candidates"] = fixed
    # canonical_sha256 will now be stale too -- recompute so the interleave
    # check (not the hash check) is what's exercised.
    from kvcot.utils.hashing import sha256_json

    stripped = dict(manifest)
    del stripped["canonical_sha256"]
    manifest["canonical_sha256"] = sha256_json(stripped)
    with pytest.raises(Exception, match="interleaving is broken"):
        verify_candidate_manifest_structure(manifest)


def test_verify_against_dataset_matches_rebuild():
    rows = _population(10)
    manifest = _build(rows)
    verify_candidate_manifest_against_dataset(manifest, rows)  # must not raise


def test_verify_against_dataset_rejects_hand_edited_manifest():
    rows = _population(10)
    manifest = dict(_build(rows))
    # Hand-edit a candidate's subject (internally consistent alone since we
    # also patch the embedded row + hashes -- but it will not reproduce the
    # real deterministic selection over the full population).
    candidates = [dict(c) for c in manifest["candidates"]]
    candidates[0] = dict(candidates[0], subject="Tampered Subject")
    candidates[0]["row"] = dict(candidates[0]["row"], subject="Tampered Subject")
    from kvcot.utils.hashing import sha256_json

    candidates[0]["raw_row_sha256"] = sha256_json(candidates[0]["row"])
    manifest["candidates"] = candidates
    stripped = dict(manifest)
    del stripped["canonical_sha256"]
    manifest["canonical_sha256"] = sha256_json(stripped)
    with pytest.raises(ManifestPreparationError, match="does not reproduce from the pinned dataset"):
        verify_candidate_manifest_against_dataset(manifest, rows)


def test_atomic_write_json_round_trips(tmp_path):
    payload = {"a": 1, "b": [1, 2, 3], "z_last": "should not be reordered"}
    target = tmp_path / "sub" / "out.json"
    atomic_write_json(target, payload)
    import json

    with open(target, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded == payload
    # Insertion order preserved (never alphabetically re-sorted on disk).
    assert list(loaded.keys()) == ["a", "b", "z_last"]


def test_atomic_write_json_leaves_no_temp_file_on_success(tmp_path):
    target = tmp_path / "out.json"
    atomic_write_json(target, {"x": 1})
    leftovers = [p for p in tmp_path.iterdir() if p.name != "out.json"]
    assert leftovers == []


def test_byte_identical_repeated_generation():
    rows = _population(10)
    m1 = _build(rows)
    m2 = _build(rows)
    import json

    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)


def test_candidate_order_protocol_version_is_the_frozen_r3_string():
    rows = _population(10)
    manifest = _build(rows)
    assert manifest["candidate_order_protocol_version"] == CANDIDATE_ORDER_PROTOCOL_VERSION
    assert manifest["candidate_order_protocol_version"] != "faithkv-b2a-r2-row-order-v1"


# --------------------------------------------------------------------- committed manifest (no network)


def test_committed_candidate_manifest_passes_structural_verification():
    """The real, committed `configs/discovery/b2a_r3_candidate_manifest.json`
    (generated from the pinned MATH-500 dataset, protocol §8/§9) must pass
    every structural/internal-consistency check without needing network
    access -- this is the golden-file regression test for that artifact."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "configs" / "discovery" / "b2a_r3_candidate_manifest.json"
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    typed = verify_candidate_manifest_structure(manifest)
    assert typed.candidate_count == 16
    assert typed.level_mixture == {"level_4": 8, "level_5": 8}
    assert typed.dataset_revision == "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    assert typed.model_revision == "6a6f4aa4197940add57724a7707d069478df56b1"
    for unique_id in EXCLUSION_SET:
        assert unique_id not in {c.unique_id for c in typed.candidates}
