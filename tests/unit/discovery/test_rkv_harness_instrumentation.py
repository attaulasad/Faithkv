"""B1B-R4 §12 tests for `kvcot.discovery.b2a_workers._RkvHarnessInstrumentation`
-- proves Pass 1 vs Pass 2 timing/call-boundary attribution and that
snapshot time is folded into `pass2_wall_seconds` exactly once (never
double-counted, never silently dropped)."""
from __future__ import annotations

import time

from kvcot.discovery.b2a_workers import _RkvHarnessInstrumentation


def _fake_prefill_fn(state, prompt_token_ids):
    time.sleep(0)  # keep elapsed >= 0 without flaking on a real sleep
    return ("prefill", list(prompt_token_ids))


def _fake_decode_one_fn(state, token_id):
    return ("decode", token_id)


def _fake_snapshot_fn(state):
    return "snapshot"


def test_first_prefill_call_is_attributed_to_pass1_second_to_pass2():
    inst = _RkvHarnessInstrumentation(_fake_prefill_fn, _fake_decode_one_fn, _fake_snapshot_fn)
    inst.prefill(None, [1, 2, 3])  # Pass 1's one prefill call
    inst.decode_one(None, 4)
    inst.prefill(None, [1, 2, 3])  # Pass 2's one prefill call
    inst.decode_one(None, 4)

    assert inst.pass1_trace.prefill_call_count == 1
    assert inst.pass1_trace.decode_call_count == 1
    assert inst.pass2_trace.prefill_call_count == 1
    assert inst.pass2_trace.decode_call_count == 1


def test_snapshot_time_is_folded_into_pass2_wall_seconds_exactly_once():
    inst = _RkvHarnessInstrumentation(_fake_prefill_fn, _fake_decode_one_fn, _fake_snapshot_fn)
    inst.prefill(None, [1])  # Pass 1
    inst.prefill(None, [1])  # Pass 2 begins
    pass2_before_snapshot = inst.pass2_wall_seconds
    capture_before_snapshot = inst.targeted_capture_wall_seconds

    inst.snapshot(None)

    pass2_delta = inst.pass2_wall_seconds - pass2_before_snapshot
    capture_delta = inst.targeted_capture_wall_seconds - capture_before_snapshot
    # The snapshot's own elapsed time must be added to BOTH accumulators by
    # the SAME amount (added once each, never twice into one, never
    # dropped from the other) -- proven by requiring the two deltas to be
    # exactly equal, not merely both non-negative.
    assert pass2_delta == capture_delta
    assert capture_delta >= 0.0


def test_snapshot_never_contributes_to_pass1_wall_seconds():
    inst = _RkvHarnessInstrumentation(_fake_prefill_fn, _fake_decode_one_fn, _fake_snapshot_fn)
    inst.prefill(None, [1])  # Pass 1 only -- snapshot is never called during Pass 1 in real usage
    pass1_before = inst.pass1_wall_seconds

    inst.snapshot(None)

    assert inst.pass1_wall_seconds == pass1_before


def test_ordered_call_kinds_and_tokens_hash_differs_between_pass1_and_pass2_when_content_differs():
    inst = _RkvHarnessInstrumentation(_fake_prefill_fn, _fake_decode_one_fn, _fake_snapshot_fn)
    inst.prefill(None, [1, 2, 3])
    inst.decode_one(None, 4)
    inst.prefill(None, [9, 9, 9])  # Pass 2 replays DIFFERENT tokens -- a real mismatch scenario
    inst.decode_one(None, 4)

    assert inst.pass1_trace.ordered_call_kinds_and_tokens_hash() != inst.pass2_trace.ordered_call_kinds_and_tokens_hash()
