import copy

import pytest
from pydantic import ValidationError

from kvcot.discovery.schemas import DISCOVERY_SCHEMA_VERSION, SwapPairRecord

VALID_SHA = "a" * 64


def _base_kwargs() -> dict:
    return dict(
        example_id="ex-1",
        model_revision="modelrev",
        rkv_revision="45eaa7d69d20b7388321f077020a610d9afb65bd",
        compaction_event_id=3,
        chronological_event_ordinal=1,
        depth_stratum=2,
        layer_index=17,
        kv_head_index=2,
        event_token_absolute_position=100,
        bridge_token_absolute_position=101,
        first_affected_forward_input_absolute_position=101,
        first_affected_logit_target_absolute_position=102,
        first_scored_absolute_position=102,
        evicted_absolute_token_position=40,
        evicted_pre_storage_position=5,
        retained_absolute_token_position=60,
        retained_pre_storage_position=8,
        retained_post_storage_position=8,
        score_e=0.4,
        score_r=0.6,
        score_margin_e_minus_r=-0.2,
        attention_component_diff=0.01,
        similarity_component_diff=-0.02,
        recency_diff=5,
        key_norm_diff=0.1,
        value_norm_diff=-0.1,
        entropy_e=1.2,
        entropy_e_missing_reason=None,
        entropy_r=0.4,
        entropy_r_missing_reason=None,
        entropy_diff=0.8,
        logit_margin_e=3.0,
        logit_margin_e_missing_reason=None,
        logit_margin_r=5.5,
        logit_margin_r_missing_reason=None,
        logit_margin_diff=-2.5,
        parity_check_passed=True,
        parity_failure_reason=None,
        is_noop_control=False,
        net_physical_bytes_changed=0,
        cap_hit_flag=False,
        valid_flag=True,
        invalid_reason=None,
        reference_horizon_sha256=VALID_SHA,
        swap_gain=0.1,
        baseline_per_token_nll=[1.0] * 48,
        swapped_per_token_nll=[0.9] * 48,
    )


def test_valid_record_constructs():
    record = SwapPairRecord(**_base_kwargs())
    assert record.schema_version == DISCOVERY_SCHEMA_VERSION == "b0_5_r2_2.v1"


def test_chronological_ordinal_range_enforced():
    kwargs = _base_kwargs()
    kwargs["chronological_event_ordinal"] = 3
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_depth_stratum_range_enforced():
    kwargs = _base_kwargs()
    kwargs["depth_stratum"] = -1
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bridge_token_absolute_position", 999),
        ("first_affected_forward_input_absolute_position", 999),
        ("first_affected_logit_target_absolute_position", 999),
        ("first_scored_absolute_position", 999),
    ],
)
def test_timing_invariants_rejected_when_broken(field, value):
    kwargs = _base_kwargs()
    kwargs[field] = value
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_net_physical_bytes_must_be_zero_for_valid_record():
    kwargs = _base_kwargs()
    kwargs["net_physical_bytes_changed"] = 4
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_invalid_record_allows_nonzero_bytes_report_with_reason():
    kwargs = _base_kwargs()
    kwargs["valid_flag"] = False
    kwargs["invalid_reason"] = "capture_failed"
    kwargs["net_physical_bytes_changed"] = 4
    record = SwapPairRecord(**kwargs)
    assert record.valid_flag is False


def test_exactly_48_nll_values_required():
    kwargs = _base_kwargs()
    kwargs["baseline_per_token_nll"] = [1.0] * 47
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)

    kwargs = _base_kwargs()
    kwargs["swapped_per_token_nll"] = [1.0] * 49
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_all_valid_nlls_must_be_finite():
    kwargs = _base_kwargs()
    kwargs["baseline_per_token_nll"] = [1.0] * 47 + [float("nan")]
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)

    kwargs2 = _base_kwargs()
    kwargs2["swapped_per_token_nll"] = [float("inf")] + [1.0] * 47
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs2)


def test_invalid_record_may_contain_non_finite_nll():
    kwargs = _base_kwargs()
    kwargs["valid_flag"] = False
    kwargs["invalid_reason"] = "capture_failed"
    kwargs["baseline_per_token_nll"] = [float("nan")] * 48
    kwargs["swapped_per_token_nll"] = [float("nan")] * 48
    kwargs["swap_gain"] = float("nan")
    record = SwapPairRecord(**kwargs)
    assert record.valid_flag is False


def test_horizon_hash_must_be_64_char_lowercase_hex():
    kwargs = _base_kwargs()
    kwargs["reference_horizon_sha256"] = "A" * 64
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)

    kwargs2 = _base_kwargs()
    kwargs2["reference_horizon_sha256"] = "a" * 63
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs2)


def test_valid_flag_false_requires_invalid_reason():
    kwargs = _base_kwargs()
    kwargs["valid_flag"] = False
    kwargs["invalid_reason"] = None
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_valid_flag_true_requires_no_invalid_reason():
    kwargs = _base_kwargs()
    kwargs["valid_flag"] = True
    kwargs["invalid_reason"] = "should not be here"
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_noop_control_requires_matching_identity():
    kwargs = _base_kwargs()
    kwargs["is_noop_control"] = True
    # evicted != retained absolute position -- not a real no-op
    assert kwargs["evicted_absolute_token_position"] != kwargs["retained_absolute_token_position"]
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def _noop_ready_kwargs() -> dict:
    kwargs = _base_kwargs()
    kwargs["is_noop_control"] = True
    kwargs["evicted_absolute_token_position"] = kwargs["retained_absolute_token_position"]
    kwargs["score_e"] = kwargs["score_r"]
    kwargs["score_margin_e_minus_r"] = 0.0
    kwargs["baseline_per_token_nll"] = [1.0] * 48
    kwargs["swapped_per_token_nll"] = [1.0] * 48
    kwargs["swap_gain"] = 0.0
    return kwargs


def test_noop_control_accepted_when_identity_matches_and_invariants_hold():
    kwargs = _noop_ready_kwargs()
    record = SwapPairRecord(**kwargs)
    assert record.is_noop_control is True


def test_noop_with_unequal_nll_arrays_rejected():
    kwargs = _noop_ready_kwargs()
    kwargs["swapped_per_token_nll"] = [1.0] * 47 + [1.0000001]
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_noop_with_nonzero_gain_rejected():
    kwargs = _noop_ready_kwargs()
    # Arrays stay equal (mean(baseline) - mean(swapped) == 0.0), so a
    # nonzero declared swap_gain is inconsistent both with the derived-value
    # check AND the no-op-specific gate -- either validator rejecting it is
    # correct; this asserts the record is rejected, full stop.
    kwargs["swap_gain"] = 0.5
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_noop_with_different_candidate_donor_identity_rejected():
    kwargs = _noop_ready_kwargs()
    kwargs["evicted_absolute_token_position"] = kwargs["retained_absolute_token_position"] + 1
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_noop_requires_valid_flag_true():
    kwargs = _noop_ready_kwargs()
    kwargs["valid_flag"] = False
    kwargs["invalid_reason"] = "capture_failed"
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_noop_requires_parity_check_passed():
    kwargs = _noop_ready_kwargs()
    kwargs["parity_check_passed"] = False
    kwargs["parity_failure_reason"] = "parity_mismatch"
    kwargs["valid_flag"] = False
    kwargs["invalid_reason"] = "parity_mismatch"
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_entropy_diff_must_equal_e_minus_r_when_both_present():
    kwargs = _base_kwargs()
    kwargs["entropy_diff"] = 999.0
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_logit_margin_diff_must_equal_e_minus_r_when_both_present():
    kwargs = _base_kwargs()
    kwargs["logit_margin_diff"] = 999.0
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_entropy_diff_none_allowed_when_source_missing():
    kwargs = _base_kwargs()
    kwargs["entropy_e"] = None
    kwargs["entropy_e_missing_reason"] = "empty_logits_tensor"
    kwargs["entropy_diff"] = None
    record = SwapPairRecord(**kwargs)
    assert record.entropy_diff is None


def test_schema_version_is_frozen_literal():
    kwargs = _base_kwargs()
    kwargs["schema_version"] = "b0_5_r2_1.v1"
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_oracle_non_deployable_is_frozen_true():
    kwargs = _base_kwargs()
    kwargs["oracle_non_deployable"] = False
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_malformed_record_is_rejected_not_repaired():
    # Missing required field entirely -- must raise, never fill a default.
    kwargs = _base_kwargs()
    del kwargs["compaction_event_id"]
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


# --------------------------------------------------------------------------
# Blocker 3: missing-reason exclusivity for entropy/logit-margin sources
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value_field,reason_field",
    [
        ("entropy_e", "entropy_e_missing_reason"),
        ("entropy_r", "entropy_r_missing_reason"),
        ("logit_margin_e", "logit_margin_e_missing_reason"),
        ("logit_margin_r", "logit_margin_r_missing_reason"),
    ],
)
def test_source_missing_while_diff_nonnull_rejected(value_field, reason_field):
    kwargs = _base_kwargs()
    kwargs[value_field] = None
    kwargs[reason_field] = "some_reason"
    # entropy_diff/logit_margin_diff are still set from _base_kwargs() and
    # are now non-null while one source value is missing.
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


@pytest.mark.parametrize(
    "value_field,reason_field",
    [
        ("entropy_e", "entropy_e_missing_reason"),
        ("entropy_r", "entropy_r_missing_reason"),
        ("logit_margin_e", "logit_margin_e_missing_reason"),
        ("logit_margin_r", "logit_margin_r_missing_reason"),
    ],
)
def test_value_and_missing_reason_both_supplied_rejected(value_field, reason_field):
    kwargs = _base_kwargs()
    kwargs[reason_field] = "some_reason"  # value_field is still set (non-None) from _base_kwargs()
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


@pytest.mark.parametrize(
    "value_field,reason_field,diff_field",
    [
        ("entropy_e", "entropy_e_missing_reason", "entropy_diff"),
        ("entropy_r", "entropy_r_missing_reason", "entropy_diff"),
        ("logit_margin_e", "logit_margin_e_missing_reason", "logit_margin_diff"),
        ("logit_margin_r", "logit_margin_r_missing_reason", "logit_margin_diff"),
    ],
)
def test_value_and_missing_reason_both_absent_rejected(value_field, reason_field, diff_field):
    kwargs = _base_kwargs()
    kwargs[value_field] = None
    kwargs[reason_field] = None  # neither value nor reason supplied
    kwargs[diff_field] = None
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


# --------------------------------------------------------------------------
# Blocker 3: parity consistency
# --------------------------------------------------------------------------


def test_valid_record_with_failed_parity_rejected():
    kwargs = _base_kwargs()
    kwargs["parity_check_passed"] = False
    kwargs["parity_failure_reason"] = "gather_parity_failed"
    # valid_flag stays True -- exactly the contradiction that must be rejected.
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_failed_parity_with_no_failure_reason_rejected():
    kwargs = _base_kwargs()
    kwargs["parity_check_passed"] = False
    kwargs["parity_failure_reason"] = None
    kwargs["valid_flag"] = False
    kwargs["invalid_reason"] = "parity_mismatch"
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_passed_parity_with_a_failure_reason_rejected():
    kwargs = _base_kwargs()
    kwargs["parity_check_passed"] = True
    kwargs["parity_failure_reason"] = "gather_parity_failed"
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_failed_parity_invalid_reason_must_mention_parity():
    kwargs = _base_kwargs()
    kwargs["parity_check_passed"] = False
    kwargs["parity_failure_reason"] = "gather_parity_failed"
    kwargs["valid_flag"] = False
    kwargs["invalid_reason"] = "unrelated_failure"
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)

    kwargs["invalid_reason"] = "parity_mismatch"
    record = SwapPairRecord(**kwargs)
    assert record.valid_flag is False


# --------------------------------------------------------------------------
# Blocker 3: derived-value validation (score margin, swap gain)
# --------------------------------------------------------------------------


def test_incorrect_score_margin_rejected():
    kwargs = _base_kwargs()
    kwargs["score_margin_e_minus_r"] = kwargs["score_e"] - kwargs["score_r"] + 1.0
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_correct_score_margin_accepted():
    kwargs = _base_kwargs()
    kwargs["score_margin_e_minus_r"] = kwargs["score_e"] - kwargs["score_r"]
    record = SwapPairRecord(**kwargs)
    assert record.score_margin_e_minus_r == kwargs["score_margin_e_minus_r"]


def test_incorrect_swap_gain_rejected():
    kwargs = _base_kwargs()
    kwargs["baseline_per_token_nll"] = [1.0] * 48
    kwargs["swapped_per_token_nll"] = [0.5] * 48
    kwargs["swap_gain"] = 999.0  # should be 1.0 - 0.5 = 0.5
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_correct_swap_gain_accepted():
    kwargs = _base_kwargs()
    kwargs["baseline_per_token_nll"] = [1.0] * 48
    kwargs["swapped_per_token_nll"] = [0.5] * 48
    kwargs["swap_gain"] = 0.5
    record = SwapPairRecord(**kwargs)
    assert record.swap_gain == 0.5


# --------------------------------------------------------------------------
# Blocker 3: finiteness and horizon-length malformation
# --------------------------------------------------------------------------


def test_non_finite_valid_scalar_rejected():
    kwargs = _base_kwargs()
    kwargs["key_norm_diff"] = float("nan")
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)


def test_malformed_nll_horizon_length_rejected():
    kwargs = _base_kwargs()
    kwargs["baseline_per_token_nll"] = [1.0] * 10
    with pytest.raises(ValidationError):
        SwapPairRecord(**kwargs)
