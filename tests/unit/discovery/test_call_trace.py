"""B1B-R4 §5/§8 tests for `kvcot.discovery.call_trace.CallTraceRecorder`."""
from __future__ import annotations

from kvcot.discovery.call_trace import CallTraceRecorder, compare_call_boundary_traces


def _fake_prefill_fn(state, prompt_token_ids):
    return ("prefill-result", list(prompt_token_ids))


def _fake_decode_one_fn(state, token_id):
    return ("decode-result", token_id)


def test_records_exactly_one_prefill_call_with_full_token_count():
    rec = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    rec.prefill(None, [1, 2, 3])
    assert rec.prefill_call_count == 1
    assert rec.prefill_token_count == 3
    assert rec.decode_call_count == 0


def test_records_one_decode_call_per_invocation():
    rec = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    rec.prefill(None, [1, 2, 3])
    rec.decode_one(None, 4)
    rec.decode_one(None, 5)
    assert rec.decode_call_count == 2
    assert rec.prefill_call_count == 1


def test_pass_through_return_values_unchanged():
    rec = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    result = rec.prefill(None, [7, 8])
    assert result == ("prefill-result", [7, 8])
    result2 = rec.decode_one(None, 9)
    assert result2 == ("decode-result", 9)


def test_identical_call_sequences_produce_identical_hash():
    a = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    b = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    for rec in (a, b):
        rec.prefill(None, [1, 2, 3])
        rec.decode_one(None, 4)
        rec.decode_one(None, 5)
    assert a.ordered_call_kinds_and_tokens_hash() == b.ordered_call_kinds_and_tokens_hash()

    comparison = compare_call_boundary_traces(a, b)
    assert comparison.all_match is True


def test_different_token_content_changes_the_hash_even_with_same_counts():
    a = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    b = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    a.prefill(None, [1, 2, 3])
    a.decode_one(None, 4)
    b.prefill(None, [1, 2, 3])
    b.decode_one(None, 999)  # different token content, same call shape

    comparison = compare_call_boundary_traces(a, b)
    assert comparison.prefill_call_count_match is True
    assert comparison.decode_call_count_match is True
    assert comparison.ordered_trace_hash_match is False
    assert comparison.all_match is False


def test_different_decode_call_count_is_caught_independently():
    a = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    b = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    a.prefill(None, [1])
    a.decode_one(None, 2)
    a.decode_one(None, 3)
    b.prefill(None, [1])
    b.decode_one(None, 2)

    comparison = compare_call_boundary_traces(a, b)
    assert comparison.decode_call_count_match is False
    assert comparison.all_match is False


def test_different_prefill_token_count_is_caught_independently():
    a = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    b = CallTraceRecorder(_fake_prefill_fn, _fake_decode_one_fn)
    a.prefill(None, [1, 2, 3])
    b.prefill(None, [1, 2])

    comparison = compare_call_boundary_traces(a, b)
    assert comparison.prefill_call_count_match is True  # both exactly one prefill call
    assert comparison.prefill_token_count_match is False
    assert comparison.all_match is False
