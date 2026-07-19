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
        entropy_r=0.4,
        entropy_diff=0.8,
        logit_margin_e=3.0,
        logit_margin_r=5.5,
        logit_margin_diff=-2.5,
        parity_check_passed=True,
        parity_failure_reason=None,
        is_noop_control=False,
        net_physical_bytes_changed=0,
        cap_hit_flag=False,
        valid_flag=True,
        invalid_reason=None,
        reference_horizon_sha256=VALID_SHA,
        swap_gain=0.02,
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
    kwargs["invalid_reason"] = "parity_mismatch"
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


def test_noop_control_accepted_when_identity_matches():
    kwargs = _base_kwargs()
    kwargs["is_noop_control"] = True
    kwargs["evicted_absolute_token_position"] = kwargs["retained_absolute_token_position"]
    record = SwapPairRecord(**kwargs)
    assert record.is_noop_control is True


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
