from kvcot.discovery.constants import (
    B2A_NOOP_CALIBRATION_COUNT,
    B2B_PILOT_EXAMPLE_COUNT,
    B2B_PILOT_TOTAL_REAL_BRANCHES,
    EVENTS_SELECTED_PER_EXAMPLE,
    PAIR_BRANCHES_PER_EVENT,
    NoOpMode,
)


def test_b2b_total_is_144_and_excludes_noop():
    assert B2B_PILOT_TOTAL_REAL_BRANCHES == 144
    assert B2B_PILOT_EXAMPLE_COUNT * EVENTS_SELECTED_PER_EXAMPLE * PAIR_BRANCHES_PER_EVENT == 144


def test_b2a_noop_calibration_count_is_exactly_one():
    assert B2A_NOOP_CALIBRATION_COUNT == 1


def test_noop_mode_has_three_distinct_members():
    assert {NoOpMode.CPU_REQUIRED, NoOpMode.B2A_SINGLE_CALIBRATION, NoOpMode.DISABLED} == set(NoOpMode)
