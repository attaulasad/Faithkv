"""Step 3R4 Finding 3: machine-readable authorization document tests.

Repairs the independent-audit finding that the dated Stage B/C
authorization Markdown document was treated as an opaque byte blob (only
its whole-file hash was checked, never its content), letting the claim
define its own enforced policy. No torch, no CUDA, no real authorization
document is ever committed by this test file -- every document lives
under `tmp_path`.
"""
from __future__ import annotations

import json

import pytest

from kvcot.discovery.b2a_r3_authorization_document import (
    AUTHORIZATION_DOCUMENT_BEGIN_MARKER,
    AUTHORIZATION_DOCUMENT_END_MARKER,
    AUTHORIZATION_DOCUMENT_SCHEMA_VERSION,
    STAGE_B2A_R3_EXECUTION,
    STAGE_FULLKV_QUALIFICATION,
    AuthorizationDocumentMalformed,
    AuthorizationDocumentR3,
    parse_authorization_document,
    parse_authorization_document_text,
    policy_from_authorization_document,
)
from kvcot.discovery.b2a_r3_authorization import (
    AUTHORIZATION_STAGE_B2A_R3_EXECUTION,
    AUTHORIZATION_STAGE_FULLKV_QUALIFICATION,
)
from kvcot.discovery.b2a_r3_contract import REQUIRED_REPOSITORY, SELECTED_MANIFEST_HASH_ALGORITHM


def test_stage_constants_match_authorization_module_verbatim():
    """The two stage-name literals are frozen in two modules (to avoid a
    circular import) -- this proves they can never silently drift apart."""
    assert STAGE_FULLKV_QUALIFICATION == AUTHORIZATION_STAGE_FULLKV_QUALIFICATION
    assert STAGE_B2A_R3_EXECUTION == AUTHORIZATION_STAGE_B2A_R3_EXECUTION


def _stage_b_payload(**overrides):
    payload = {
        "authorization_document_schema_version": AUTHORIZATION_DOCUMENT_SCHEMA_VERSION,
        "authorization_id": "stage-b-2026-08-01",
        "authorization_stage": STAGE_FULLKV_QUALIFICATION,
        "authorized_repository": REQUIRED_REPOSITORY,
        "authorized_branch": "research/b2a-r3-runtime-qualified-calibration",
        "authorized_code_commit_sha": "b" * 40,
        "required_ancestor_shas": ["c" * 40],
        "required_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
        "candidate_manifest_canonical_sha256": "d" * 64,
        "maximum_candidates": 8,
        "phase_wall_time_limit_seconds": 3600,
        "qualification_artifact_canonical_sha256": None,
        "selected_manifest_sha256": None,
        "selected_manifest_hash_algorithm": None,
        "created_at_utc": "2026-08-01T00:00:00+00:00",
    }
    payload.update(overrides)
    return payload


def _stage_c_payload(**overrides):
    base = _stage_b_payload(
        authorization_id="stage-c-2026-08-05",
        authorization_stage=STAGE_B2A_R3_EXECUTION,
        maximum_candidates=None,
        phase_wall_time_limit_seconds=None,
        qualification_artifact_canonical_sha256="e" * 64,
        selected_manifest_sha256="f" * 64,
        selected_manifest_hash_algorithm=SELECTED_MANIFEST_HASH_ALGORITHM,
    )
    base.update(overrides)
    return base


def _document_text(payload: dict, *, extra_text_in_block: str = "", fence_language: str = "json",
                    marker_count: int = 1) -> str:
    body = json.dumps(payload, indent=2)
    block = (
        f"{AUTHORIZATION_DOCUMENT_BEGIN_MARKER}\n"
        f"{extra_text_in_block}"
        f"```{fence_language}\n{body}\n```\n"
        f"{AUTHORIZATION_DOCUMENT_END_MARKER}\n"
    )
    return ("# doc\n\n" + block) * marker_count


def test_valid_stage_b_document_parses():
    doc = parse_authorization_document_text(_document_text(_stage_b_payload()))
    assert doc.authorization_stage == STAGE_FULLKV_QUALIFICATION
    assert doc.maximum_candidates == 8


def test_valid_stage_c_document_parses():
    doc = parse_authorization_document_text(_document_text(_stage_c_payload()))
    assert doc.authorization_stage == STAGE_B2A_R3_EXECUTION
    assert doc.selected_manifest_hash_algorithm == SELECTED_MANIFEST_HASH_ALGORITHM


def test_synthetic_placeholder_text_document_fails():
    """The exact historical defect: a document containing only
    "synthetic Stage B authorization" (no JSON block at all) must fail."""
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text("synthetic Stage B authorization\n")


def test_missing_markers_fails():
    body = json.dumps(_stage_b_payload())
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(f"```json\n{body}\n```\n")


def test_missing_begin_marker_only_fails():
    body = json.dumps(_stage_b_payload())
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(f"```json\n{body}\n```\n{AUTHORIZATION_DOCUMENT_END_MARKER}\n")


def test_multiple_marker_pairs_fails():
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(_document_text(_stage_b_payload(), marker_count=2))


def test_wrong_fence_language_fails():
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(_document_text(_stage_b_payload(), fence_language="yaml"))


def test_no_fence_language_fails():
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(_document_text(_stage_b_payload(), fence_language=""))


def test_text_outside_fence_but_inside_markers_fails():
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(
            _document_text(_stage_b_payload(), extra_text_in_block="some stray prose\n")
        )


def test_two_fenced_blocks_inside_markers_fails():
    body = json.dumps(_stage_b_payload())
    text = (
        f"{AUTHORIZATION_DOCUMENT_BEGIN_MARKER}\n"
        f"```json\n{body}\n```\n```json\n{body}\n```\n"
        f"{AUTHORIZATION_DOCUMENT_END_MARKER}\n"
    )
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(text)


def test_invalid_json_fails():
    text = (
        f"{AUTHORIZATION_DOCUMENT_BEGIN_MARKER}\n```json\n{{not valid json\n```\n"
        f"{AUTHORIZATION_DOCUMENT_END_MARKER}\n"
    )
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(text)


def test_duplicate_json_keys_fails():
    body = json.dumps(_stage_b_payload())
    # Inject a literal duplicate top-level key by string surgery.
    duplicated = body.rstrip("}") + ', "authorization_id": "duplicate-smuggled"}'
    text = (
        f"{AUTHORIZATION_DOCUMENT_BEGIN_MARKER}\n```json\n{duplicated}\n```\n"
        f"{AUTHORIZATION_DOCUMENT_END_MARKER}\n"
    )
    with pytest.raises(AuthorizationDocumentMalformed):
        parse_authorization_document_text(text)


def test_unknown_field_fails():
    payload = _stage_b_payload(unexpected_field="nope")
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


@pytest.mark.parametrize(
    "field",
    [
        "authorization_document_schema_version", "authorization_id", "authorization_stage",
        "authorized_repository", "authorized_branch", "authorized_code_commit_sha",
        "required_ancestor_shas", "required_rkv_sha", "candidate_manifest_canonical_sha256",
        "created_at_utc",
    ],
)
def test_missing_required_field_fails(field):
    payload = _stage_b_payload()
    del payload[field]
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_noncanonical_stage_fails():
    payload = _stage_b_payload(authorization_stage="not_a_real_stage")
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


@pytest.mark.parametrize("bad_id", ["", "../escape", "a/b", "a\\b", "a..b", "a" * 129])
def test_invalid_authorization_id_fails(bad_id):
    payload = _stage_b_payload(authorization_id=bad_id)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_b_missing_maximum_candidates_fails():
    payload = _stage_b_payload(maximum_candidates=None)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_b_maximum_candidates_out_of_range_fails():
    payload = _stage_b_payload(maximum_candidates=9)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_b_zero_maximum_candidates_fails():
    payload = _stage_b_payload(maximum_candidates=0)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_b_non_positive_wall_time_limit_fails():
    payload = _stage_b_payload(phase_wall_time_limit_seconds=0)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_b_boolean_maximum_candidates_fails():
    payload = _stage_b_payload(maximum_candidates=True)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_b_numeric_string_wall_time_fails():
    payload = _stage_b_payload(phase_wall_time_limit_seconds="3600")
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_b_rejects_stage_c_fields_present():
    payload = _stage_b_payload(qualification_artifact_canonical_sha256="e" * 64)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_c_missing_selected_manifest_sha_fails():
    payload = _stage_c_payload(selected_manifest_sha256=None)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_c_rejects_stage_b_fields_present():
    payload = _stage_c_payload(maximum_candidates=8)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_stage_c_wrong_hash_algorithm_name_fails():
    payload = _stage_c_payload(selected_manifest_hash_algorithm="wrong-v1")
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_wrong_repository_fails():
    payload = _stage_b_payload(authorized_repository="someone-else/Faithkv")
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_authorization_document_sha256_field_is_not_permitted_in_the_json_block():
    """The committed document cannot contain its own final byte hash."""
    payload = _stage_b_payload(authorization_document_sha256="a" * 64)
    with pytest.raises(Exception):
        parse_authorization_document_text(_document_text(payload))


def test_parse_authorization_document_reads_from_disk(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(_document_text(_stage_b_payload()), encoding="utf-8")
    doc = parse_authorization_document(path)
    assert doc.authorization_id == "stage-b-2026-08-01"


def test_policy_from_document_never_reads_a_claim():
    doc = parse_authorization_document_text(_document_text(_stage_b_payload()))
    policy = policy_from_authorization_document(doc, authorization_document_sha256="a" * 64)
    assert policy.required_branch == doc.authorized_branch
    assert policy.required_commit_sha == doc.authorized_code_commit_sha
    assert policy.required_ancestor_shas == tuple(doc.required_ancestor_shas)
    assert policy.required_rkv_sha == doc.required_rkv_sha
    assert policy.authorization_id == doc.authorization_id
    assert policy.authorization_document_sha256 == "a" * 64
