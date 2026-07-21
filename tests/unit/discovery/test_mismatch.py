"""Independent-audit Gate H3 unit tests for
`kvcot.discovery.mismatch.build_mismatch_record`."""
from __future__ import annotations

from kvcot.discovery.mismatch import (
    MISMATCH_KIND_EXPECTED_ENDS_FIRST,
    MISMATCH_KIND_NONE,
    MISMATCH_KIND_OBSERVED_ENDS_FIRST,
    MISMATCH_KIND_VALUE_DIFFERS,
    build_mismatch_record,
)


def test_identical_sequences_report_matched():
    record = build_mismatch_record([1, 2, 3], [1, 2, 3])
    assert record.matched is True
    assert record.first_mismatch_index is None
    assert record.expected_value is None
    assert record.observed_value is None
    assert record.mismatch_kind == MISMATCH_KIND_NONE
    assert record.expected_length == record.observed_length == 3


def test_value_mismatch_reports_expected_and_observed_at_first_divergence():
    record = build_mismatch_record([1, 2, 3, 4], [1, 2, 99, 4])
    assert record.matched is False
    assert record.first_mismatch_index == 2
    assert record.expected_value == 3
    assert record.observed_value == 99
    assert record.mismatch_kind == MISMATCH_KIND_VALUE_DIFFERS
    assert record.expected_length == 4
    assert record.observed_length == 4


def test_expected_sequence_shorter_reports_observed_ends_first_kind_correctly():
    """`expected` ending first means `observed` has an extra trailing value
    the expected side never had -- the mismatch begins at the shorter
    (expected) length, with a real observed value and no expected value."""
    record = build_mismatch_record([1, 2], [1, 2, 3])
    assert record.matched is False
    assert record.first_mismatch_index == 2
    assert record.expected_value is None
    assert record.observed_value == 3
    assert record.expected_length == 2
    assert record.observed_length == 3
    assert record.mismatch_kind == MISMATCH_KIND_EXPECTED_ENDS_FIRST


def test_observed_sequence_shorter_reports_expected_ends_first_kind_correctly():
    record = build_mismatch_record([1, 2, 3], [1, 2])
    assert record.matched is False
    assert record.first_mismatch_index == 2
    assert record.expected_value == 3
    assert record.observed_value is None
    assert record.expected_length == 3
    assert record.observed_length == 2
    assert record.mismatch_kind == MISMATCH_KIND_OBSERVED_ENDS_FIRST


def test_never_indexes_beyond_either_sequence():
    """A pathological empty/empty or empty/non-empty pair must never raise
    an IndexError -- the record must still be constructed cleanly."""
    record_both_empty = build_mismatch_record([], [])
    assert record_both_empty.matched is True

    record_one_empty = build_mismatch_record([], [1, 2])
    assert record_one_empty.matched is False
    assert record_one_empty.first_mismatch_index == 0
    assert record_one_empty.expected_value is None
    assert record_one_empty.observed_value == 1


def test_dict_sequences_compare_by_value_for_logical_and_actual_call_records():
    """`build_mismatch_record` is used for logical-call and actual-call
    comparisons too, where elements are dicts (call-kind/token-shape
    records), not just plain token IDs -- equality must compare by dict
    value, never by identity."""
    left = [{"call_kind": "prefill", "token_ids": [1, 2]}, {"call_kind": "decode", "token_ids": [3]}]
    right = [{"call_kind": "prefill", "token_ids": [1, 2]}, {"call_kind": "decode", "token_ids": [4]}]
    record = build_mismatch_record(left, right)
    assert record.matched is False
    assert record.first_mismatch_index == 1
    assert record.expected_value == {"call_kind": "decode", "token_ids": [3]}
    assert record.observed_value == {"call_kind": "decode", "token_ids": [4]}


def test_export_round_trips_to_a_plain_json_serializable_dict():
    record = build_mismatch_record([1, 2], [1, 3])
    exported = record.export()
    assert exported == {
        "matched": False,
        "first_mismatch_index": 1,
        "expected_value": 2,
        "observed_value": 3,
        "expected_length": 2,
        "observed_length": 2,
        "mismatch_kind": MISMATCH_KIND_VALUE_DIFFERS,
    }
