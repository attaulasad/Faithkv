"""CPU tests for the frozen B2A-R3 contract constants
(`kvcot.discovery.b2a_r3_contract`) -- exact constants/version strings,
canonical-hash rule, authorization-ID validation, strict scalar rejection,
the frozen exclusion set, and the frozen 27-name qualification-condition
tuple. No torch, no CUDA, no network."""
from __future__ import annotations

import pytest

from kvcot.discovery import b2a_r3_contract as c


def test_no_bare_protocol_version_name_anywhere_in_module():
    # R3-AUDIT-25: no bare, unqualified `protocol_version` constant.
    assert not hasattr(c, "PROTOCOL_VERSION")


def test_four_plus_one_identity_fields_are_distinct_strings():
    values = {
        c.CANDIDATE_ORDER_PROTOCOL_VERSION,
        c.QUALIFICATION_PROTOCOL_VERSION,
        c.RUNTIME_PREDICTOR_VERSION,
        c.SELECTION_PROTOCOL_VERSION,
        c.PROVENANCE_POLICY_VERSION,
    }
    assert len(values) == 5
    for v in values:
        assert v.startswith("faithkv-b2a-r3-")


def test_artifact_schema_versions_distinct_from_protocol_versions():
    schema_versions = {
        c.CANDIDATE_MANIFEST_ARTIFACT_SCHEMA_VERSION,
        c.QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
        c.SELECTION_PROVENANCE_ARTIFACT_SCHEMA_VERSION,
        c.AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION,
    }
    assert len(schema_versions) == 4


def test_dataset_model_identity_matches_frozen_protocol_values():
    assert c.DATASET_REPO == "HuggingFaceH4/MATH-500"
    assert c.DATASET_REVISION == "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    assert c.MODEL_NAME == "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    assert c.MODEL_REVISION == "6a6f4aa4197940add57724a7707d069478df56b1"
    assert c.BUDGET == 1024


def test_artifact_paths_match_frozen_protocol_table():
    assert c.CANDIDATE_MANIFEST_PATH == "configs/discovery/b2a_r3_candidate_manifest.json"
    assert c.QUALIFICATION_ARTIFACT_PATH == "results/decisions/b2a_r3_qualification.json"
    assert c.SELECTED_MANIFEST_PATH == "configs/discovery/b2a_one_example_manifest.json"
    assert c.SELECTION_PROVENANCE_PATH == "results/decisions/b2a_r3_selection_provenance.json"
    assert c.AUTHORIZATION_CLAIMS_DIR == "results/decisions/b2a_r3_authorization_claims"


def test_attempt_directory_name_uses_r3_prefix():
    name = c.attempt_directory_name("20260101T000000000000Z", "deadbeef")
    assert name == "b2a_r3_attempt_20260101T000000000000Z_deadbeef"


# --------------------------------------------------------------------- auth id


@pytest.mark.parametrize(
    "value",
    ["a", "A0", "abc-123._Z", "a" * 128],
)
def test_valid_authorization_ids_accepted(value):
    assert c.validate_authorization_id(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "",
        "/etc/passwd",
        "a/b",
        "a\\b",
        "..",
        "a..b",
        "../../x",
        "a b",
        "a\tb",
        "a\nb",
        "a" * 129,
        "-leading-dash",
        123,
        None,
        True,
    ],
)
def test_invalid_authorization_ids_rejected(value):
    with pytest.raises(ValueError):
        c.validate_authorization_id(value)


def test_global_claim_path_is_deterministic_and_under_claims_dir():
    path = c.global_claim_path("stage-b-2026-08-01")
    assert path == "results/decisions/b2a_r3_authorization_claims/stage-b-2026-08-01.json"


def test_global_claim_path_rejects_invalid_id():
    with pytest.raises(ValueError):
        c.global_claim_path("../escape")


# --------------------------------------------------------------------- hashing


def test_compute_and_verify_canonical_sha256_roundtrip():
    payload = {"a": 1, "b": [1, 2, 3], "c": {"x": "y"}}
    digest = c.compute_canonical_sha256(payload)
    full = dict(payload)
    full["canonical_sha256"] = digest
    c.verify_canonical_sha256(full)  # must not raise


def test_compute_canonical_sha256_rejects_payload_already_containing_field():
    with pytest.raises(ValueError):
        c.compute_canonical_sha256({"canonical_sha256": "x" * 64})


def test_verify_canonical_sha256_rejects_missing_field():
    with pytest.raises(ValueError):
        c.verify_canonical_sha256({"a": 1})


def test_verify_canonical_sha256_rejects_uppercase_hash():
    payload = {"a": 1}
    digest = c.compute_canonical_sha256(payload)
    full = dict(payload)
    full["canonical_sha256"] = digest.upper()
    with pytest.raises(ValueError):
        c.verify_canonical_sha256(full)


def test_verify_canonical_sha256_rejects_tampered_payload():
    payload = {"a": 1}
    digest = c.compute_canonical_sha256(payload)
    full = {"a": 2, "canonical_sha256": digest}
    with pytest.raises(ValueError):
        c.verify_canonical_sha256(full)


def test_hash_field_never_computed_including_itself():
    # Sanity check that omission actually changes the hash -- i.e. the
    # function is not accidentally hashing a payload that still contains
    # the field under a different key ordering.
    payload = {"a": 1}
    digest_without = c.compute_canonical_sha256(payload)
    from kvcot.utils.hashing import sha256_json

    digest_with_self = sha256_json({**payload, "canonical_sha256": "0" * 64})
    assert digest_without != digest_with_self


def test_reject_hash_aliases():
    with pytest.raises(ValueError):
        c.reject_hash_aliases({"manifest_hash": "x"})
    with pytest.raises(ValueError):
        c.reject_hash_aliases({"artifact_hash": "x"})
    c.reject_hash_aliases({"canonical_sha256": "x"})  # must not raise


# --------------------------------------------------------------------- strict scalars


def test_require_strict_int_rejects_bool_and_string():
    assert c.require_strict_int(5, "n") == 5
    with pytest.raises(ValueError):
        c.require_strict_int(True, "n")
    with pytest.raises(ValueError):
        c.require_strict_int(False, "n")
    with pytest.raises(ValueError):
        c.require_strict_int("5", "n")


def test_require_strict_bool_rejects_int_and_string():
    assert c.require_strict_bool(True, "b") is True
    with pytest.raises(ValueError):
        c.require_strict_bool(1, "b")
    with pytest.raises(ValueError):
        c.require_strict_bool(0, "b")
    with pytest.raises(ValueError):
        c.require_strict_bool("true", "b")


def test_require_strict_finite_float_rejects_nan_and_infinity():
    assert c.require_strict_finite_float(1.5, "f") == 1.5
    assert c.require_strict_finite_float(2, "f") == 2.0
    with pytest.raises(ValueError):
        c.require_strict_finite_float(float("nan"), "f")
    with pytest.raises(ValueError):
        c.require_strict_finite_float(float("inf"), "f")
    with pytest.raises(ValueError):
        c.require_strict_finite_float(float("-inf"), "f")
    with pytest.raises(ValueError):
        c.require_strict_finite_float(True, "f")
    with pytest.raises(ValueError):
        c.require_strict_finite_float("1.5", "f")


def test_require_lowercase_hex64():
    good = "a" * 64
    assert c.require_lowercase_hex64(good, "h") == good
    with pytest.raises(ValueError):
        c.require_lowercase_hex64("A" * 64, "h")
    with pytest.raises(ValueError):
        c.require_lowercase_hex64("a" * 63, "h")
    with pytest.raises(ValueError):
        c.require_lowercase_hex64("g" * 64, "h")


# --------------------------------------------------------------------- exclusion set


def test_exclusion_set_has_exactly_13_sorted_unique_rows():
    assert len(c.EXCLUSION_SET) == 13
    assert len(set(c.EXCLUSION_SET)) == 13
    assert list(c.EXCLUSION_SET) == sorted(c.EXCLUSION_SET)


def test_exclusion_set_reproduces_frozen_hash():
    from kvcot.utils.hashing import sha256_text

    assert sha256_text(c.exclusion_set_canonical_payload()) == c.EXCLUSION_SET_SHA256
    c.verify_exclusion_set()  # must not raise


def test_exclusion_set_contains_known_historical_rows():
    assert "test/precalculus/807.json" in c.EXCLUSION_SET  # B2A-R1's row
    assert "test/number_theory/820.json" in c.EXCLUSION_SET  # B2A-R2's executed row


# --------------------------------------------------------------------- generation config


def test_generation_config_hash_reproduces_frozen_value():
    from kvcot.utils.hashing import sha256_json

    assert sha256_json(c.FROZEN_GENERATION_CONFIG) == c.GENERATION_CONFIG_SHA256
    c.verify_generation_config_hash()  # must not raise


def test_generation_config_is_the_exact_frozen_object():
    assert c.FROZEN_GENERATION_CONFIG == {
        "generation_mode": "greedy",
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "batch_size": 1,
        "max_new_tokens": 6144,
        "framework_seed": 13,
        "attention_backend": "flash_attention_2",
        "cache_implementation": "DynamicCache",
        "no_offload_required": True,
    }


# --------------------------------------------------------------------- runtime constants


def test_runtime_constants_match_frozen_protocol_values():
    assert c.REFERENCE_TOTAL_TOKENS == 4931
    assert c.REFERENCE_EXAMPLE_SECONDS == 1378.3004406290129
    assert c.REFERENCE_PAIR_SECONDS == 10.247917714063078
    assert c.SAFETY_MULTIPLIER == 1.20
    assert c.QUALIFICATION_TARGET_HOURS == 3.60
    assert c.FINAL_RUNTIME_LIMIT_HOURS == 4.00
    assert c.B2B_EXAMPLE_COUNT == 12
    assert c.B2B_REAL_PAIR_COUNT == 144


def test_absolute_timeout_envelope_is_pure_arithmetic():
    assert c.PER_CANDIDATE_WORKER_TIMEOUT_SECONDS == 7200
    assert c.QUALIFICATION_CANDIDATE_LIMIT == 8
    assert c.ABSOLUTE_TIMEOUT_ENVELOPE_SECONDS == 8 * 7200 == 57600


# --------------------------------------------------------------------- qualification tuple


def test_qualification_conditions_tuple_exact_count_and_order():
    assert len(c.B2A_R3_QUALIFICATION_CONDITIONS) == 27
    assert len(set(c.B2A_R3_QUALIFICATION_CONDITIONS)) == 27
    assert c.B2A_R3_QUALIFICATION_CONDITIONS[0] == "no_cap_hit"
    assert c.B2A_R3_QUALIFICATION_CONDITIONS[-1] == "projected_runtime_within_qualification_target"
    assert "predicted_event_count_at_least_six" in c.B2A_R3_QUALIFICATION_CONDITIONS
    assert "at_least_three_events_have_49_future_tokens" in c.B2A_R3_QUALIFICATION_CONDITIONS


def test_qualification_conditions_never_collides_with_b2a_r2_tuple():
    from kvcot.discovery.b2a_qualification import QUALIFICATION_CONDITIONS as R2_CONDITIONS

    assert c.B2A_R3_QUALIFICATION_CONDITIONS != R2_CONDITIONS


def test_think_parse_success_statuses_matches_early_answering_module():
    from kvcot.probes.early_answering import find_think_span

    # The two known success statuses reachable from find_think_span.
    ok = find_think_span([1], [10, 11, 12], open_marker_ids=[10], close_marker_ids=[12])
    assert ok.think_parse_status in c.THINK_PARSE_SUCCESS_STATUSES
    failure = find_think_span([1], [], open_marker_ids=[10], close_marker_ids=[12])
    assert failure.think_parse_status not in c.THINK_PARSE_SUCCESS_STATUSES


def test_embedded_row_columns_match_manifest_prepare():
    from kvcot.discovery.manifest_prepare import EXPECTED_MATH500_COLUMNS

    assert c.EMBEDDED_ROW_COLUMNS == EXPECTED_MATH500_COLUMNS


def test_module_level_self_checks_do_not_raise_on_import():
    # Re-running the same self-checks the module already ran at import
    # time must remain a no-op -- a regression here would mean the module
    # mutated its own frozen constants somewhere.
    c.verify_exclusion_set()
    c.verify_generation_config_hash()
    c.verify_qualification_conditions_tuple()
