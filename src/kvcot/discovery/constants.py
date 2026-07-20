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
