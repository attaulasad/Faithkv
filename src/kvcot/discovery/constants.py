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
