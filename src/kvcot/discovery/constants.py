"""Shared frozen numeric constants for the B1B CPU harness
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9). Pure Python, no torch
import — this is the one place `kvcot.cli`'s `--dry-run` planning path
(which must stay torch-free, matching every other dry-run command in this
CLI) and the torch-importing runtime modules
(`kvcot.discovery.branch_eval`, `kvcot.discovery.schemas`,
`kvcot.discovery.pass1`) both read these numbers from, so they can never
silently drift apart.
"""
from __future__ import annotations

import enum

SCORED_HORIZON = 48
MINIMUM_FUTURE_TOKENS_AFTER_EVENT = 49
BRIDGE_TOKEN_COUNT = 1
EVENTS_SELECTED_PER_EXAMPLE = 3
CANDIDATES_PER_EVENT = 2
DONORS_PER_EVENT = 2
PAIR_BRANCHES_PER_EVENT = CANDIDATES_PER_EVENT * DONORS_PER_EVENT
B2B_PILOT_EXAMPLE_COUNT = 12
# 3 events x 4 real cross-product swaps x 12 examples = 144 -- the no-op
# control is mandatory but NEVER counted in this total (B1B-R2 §9).
B2B_PILOT_TOTAL_REAL_BRANCHES = B2B_PILOT_EXAMPLE_COUNT * EVENTS_SELECTED_PER_EXAMPLE * PAIR_BRANCHES_PER_EVENT

# B2A no-op numerical calibration count -- exactly ONE, drawn from the
# shared orchestrator's structurally-always-3 (one-per-event) mandatory
# no-op pairs, never one full additional no-op branch per event (B1B-R3
# §13). `kvcot.discovery.b2a_execute` selects the FIRST valid one; the
# other (structurally-produced, still CPU-mandatory) no-ops are extra
# confirmatory data, never double-counted into any total.
B2A_NOOP_CALIBRATION_COUNT = 1


class NoOpMode(enum.Enum):
    """B1B-R3 §13: an explicit policy for how the mandatory no-op control
    (Part IX.20 -- `evicted_absolute_position == donor_absolute_position`)
    is interpreted at each protocol stage, so the shared orchestrator
    (`kvcot.discovery.orchestrator.run_example`, which always builds one
    no-op pair per selected event, `4 cross-product + 1 no-op = 5` attempts,
    unchanged by this enum) is never silently read as producing MORE real
    branches than it does."""

    # CPU tests: every example's no-op pair(s) are mandatory, structural,
    # and reported/asserted individually (test_b1b_integration.py etc.) --
    # this is the orchestrator's unmodified default behavior.
    CPU_REQUIRED = "cpu_required"
    # B2A: exactly ONE no-op numerical calibration is reported as B2A
    # evidence (`no_op_numerical_parity`), drawn from the orchestrator's
    # first produced no-op pair for the one B2A example -- never one
    # additional full no-op branch per event.
    B2A_SINGLE_CALIBRATION = "b2a_single_calibration"
    # Reserved for a future mode that suppresses no-op pair construction
    # entirely -- not used by any code path in this repository yet; never
    # apply this to a CPU test path, which always requires the no-op.
    DISABLED = "disabled"
