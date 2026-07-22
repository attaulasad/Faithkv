"""Direct unit tests for `kvcot.discovery.pass1.eligible_event_positions` --
the position-based eligibility rule extracted (B2A-R2 qualification,
2026-07-22) so FullKV-only row qualification
(`kvcot.discovery.b2a_qualification`) and the CPU-harness's
`eligible_event_ids` share one rule instead of two independently
maintained copies of it.
"""
from __future__ import annotations

from kvcot.discovery.constants import MINIMUM_FUTURE_TOKENS_AFTER_EVENT
from kvcot.discovery.pass1 import eligible_event_positions


def test_fewer_than_three_events_is_never_eligible():
    assert eligible_event_positions([10, 20], prompt_length=5, total_len=1000) == []
    assert eligible_event_positions([], prompt_length=5, total_len=1000) == []


def test_first_and_last_event_are_never_eligible():
    total_len = 200
    positions = [10, 20, 30, 40, 50]
    eligible = eligible_event_positions(positions, prompt_length=5, total_len=total_len)
    assert 0 not in eligible  # index of first event (position 10)
    assert 4 not in eligible  # index of last event (position 50)


def test_prefill_phase_event_is_never_eligible():
    # Event at index 1 (position 3) is BEFORE prompt_length=5 -- excluded
    # even though it is neither first nor last.
    positions = [1, 3, 20, 30, 40]
    eligible = eligible_event_positions(positions, prompt_length=5, total_len=200)
    assert 1 not in eligible


def test_requires_at_least_49_future_tokens():
    total_len = 100
    boundary = total_len - 1 - MINIMUM_FUTURE_TOKENS_AFTER_EVENT  # exactly 49 future tokens here
    positions = [10, boundary, boundary + 1, 90]
    eligible = eligible_event_positions(positions, prompt_length=5, total_len=total_len)
    # index 1 (exactly 49 future tokens) is eligible; index 2 (48 future
    # tokens) is not.
    assert 1 in eligible
    assert 2 not in eligible


def test_middle_events_satisfying_every_condition_are_eligible():
    total_len = 500
    positions = [10, 100, 200, 300, 490]
    eligible = eligible_event_positions(positions, prompt_length=5, total_len=total_len)
    assert eligible == [1, 2, 3]


def test_returned_values_are_indices_into_the_input_sequence():
    positions = [10, 100, 200, 300, 400]
    eligible = eligible_event_positions(positions, prompt_length=5, total_len=1000)
    for i in eligible:
        assert 0 <= i < len(positions)
