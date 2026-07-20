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

# B1B-R4 §4: "pair evaluation" is the frozen, unambiguous vocabulary -- one
# candidate-donor pair evaluation contains one baseline continuation AND one
# swapped continuation (never counted as two). `PAIR_BRANCHES_PER_EVENT` is
# kept as a deprecated compatibility alias (old name, same value) so no
# import site breaks; new code must use `REAL_PAIR_EVALUATIONS_PER_EVENT`.
REAL_PAIR_EVALUATIONS_PER_EVENT = CANDIDATES_PER_EVENT * DONORS_PER_EVENT
PAIR_BRANCHES_PER_EVENT = REAL_PAIR_EVALUATIONS_PER_EVENT  # deprecated alias -- see docstring above

B2B_PILOT_EXAMPLE_COUNT = 12
# 12 examples x 3 events x 4 real cross-product pair evaluations = 144 --
# the no-op control is mandatory in the CPU harness but is NEVER counted in
# this total, and B2B itself runs ZERO GPU no-op evaluations (B1B-R4 §4:
# "B2B contains no GPU no-op evaluations").
B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS = (
    B2B_PILOT_EXAMPLE_COUNT * EVENTS_SELECTED_PER_EXAMPLE * REAL_PAIR_EVALUATIONS_PER_EVENT
)
B2B_PILOT_TOTAL_REAL_BRANCHES = B2B_PILOT_TOTAL_REAL_PAIR_EVALUATIONS  # deprecated alias

# B2A: exactly 3 selected events x 4 real pair evaluations = 12 real pair
# evaluations, PLUS exactly 1 no-op pair evaluation (separate from, never
# folded into, the 12) -- B1B-R4 §4's frozen execution-count semantics.
B2A_SELECTED_EVENTS = EVENTS_SELECTED_PER_EXAMPLE  # 3
B2A_REAL_PAIR_EVALUATIONS_TOTAL = B2A_SELECTED_EVENTS * REAL_PAIR_EVALUATIONS_PER_EVENT  # 12

# B2A no-op numerical calibration count -- exactly ONE, for the whole B2A
# example (never one per event). B1B-R4 §7 makes `NoOpMode.B2A_SINGLE_
# CALIBRATION` an ACTUAL execution-count control on
# `kvcot.discovery.orchestrator.run_example` (via `PairExecutionPolicy`),
# not just documentation: exactly one no-op pair is built, for the FIRST
# selected event in the frozen plan, deterministically.
B2A_NOOP_CALIBRATION_COUNT = 1
B2A_NOOP_PAIR_EVALUATIONS_TOTAL = B2A_NOOP_CALIBRATION_COUNT  # 1

# B1B-R4 §16: frozen subprocess timeout for each B2A worker (FullKV, R-KV),
# in seconds -- 2 hours. `kvcot.discovery.b2a_workers.run_both_workers_via_
# subprocess` passes this to its `subprocess_runner`; a worker that runs
# longer is a hard `subprocess.TimeoutExpired` failure, never silently
# waited on indefinitely.
B2A_WORKER_TIMEOUT_SECONDS = 7200


class NoOpMode(enum.Enum):
    """B1B-R4 §7: an explicit policy that ACTUALLY CONTROLS how many no-op
    pair evaluations `kvcot.discovery.orchestrator.run_example` builds per
    example, via `PairExecutionPolicy` -- not merely documentation layered
    on top of a fixed, unconditional "1 no-op per event" orchestrator
    behavior (B1B-R3's version of this enum documented three named
    interpretations while the orchestrator itself always built exactly one
    no-op pair per selected event, regardless of which mode name a caller
    cited -- this repaired version makes the enum value the single source
    of truth `run_example` actually branches on)."""

    # CPU tests: every example's no-op pair(s) are mandatory, structural,
    # and reported/asserted individually -- one no-op pair PER SELECTED
    # EVENT (`3 events x (4 real + 1 no-op) = 15` pair evaluations for a
    # valid 3-event example). This is `PairExecutionPolicy`'s default.
    CPU_REQUIRED = "cpu_required"
    # B2A: exactly ONE no-op pair evaluation for the WHOLE example, built
    # only for the first selected event in the frozen plan (deterministic,
    # never re-selected per run) -- `12 real + 1 no-op = 13` pair
    # evaluations total, never one additional no-op branch per event.
    B2A_SINGLE_CALIBRATION = "b2a_single_calibration"
    # B2B planning and execution: no no-op pair evaluations at all --
    # `3 events x 4 real = 12` real pair evaluations per example, zero GPU
    # no-op evaluations, matching B1B-R4 §4's frozen B2B accounting.
    DISABLED = "disabled"
