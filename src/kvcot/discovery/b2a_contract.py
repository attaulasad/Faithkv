"""Future one-example B2A contract — definition and validation ONLY
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §11). This module is
DOCUMENTATION AND VALIDATION CODE. It defines what a future B2A one-example
GPU calibration run must measure and the hard stop conditions that must
gate any further B2A/B2B activity — it never executes a GPU run, never
loads a model, and is not itself authorized to run anything (CLAUDE.md
§1a/§1b: B2A remains unauthorized, requiring its own separate, future,
dated authorization).

Because no dataset manifest has been downloaded in this pass (`CLAUDE.md`
§1a/§4a: MATH-500 is not downloaded), actual B2A execution ALSO remains
blocked until the one-example manifest identity and dataset revision are
independently frozen (`kvcot.discovery.discovery_config.DiscoveryDatasetLock
.revision_is_frozen`) — this contract does not and cannot resolve that gap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

# What a future B2A one-example run must measure, separately, per the task
# brief -- this tuple is the canonical checklist; a future implementation
# must report every one of these, not a subset.
B2A_ONE_EXAMPLE_REQUIRED_MEASUREMENTS: tuple[str, ...] = (
    "fullkv_natural_generation",
    "rkv_pass1",
    "token_identical_pass2",
    "score_recomputation",
    "targeted_capture",
    "cache_clone_restore",
    "one_fixed_shape_swap",
    "one_bridge_plus_48_scored_teacher_forced_tokens",
    "peak_cuda_allocated_memory",
    "peak_cuda_reserved_memory",
    "parameter_placement_assertion",
    "observed_retention",
    "event_count",
    "projected_complete_pilot_runtime",
)

# Hard future stop conditions (§11). ANY one of these being true means B2A/
# B2B/the discovery pilot must stop -- this module only defines the
# thresholds and a pure-Python evaluator; it never runs a GPU measurement
# itself.
MAX_PROJECTED_PILOT_GPU_HOURS = 4.00
MAX_PEAK_ALLOCATED_MEMORY_GIB = 22.0


class B2AOneExampleMeasurement(BaseModel):
    """What a future (not-yet-authorized) B2A one-example run must report.
    Constructing this model does not run anything -- it is a schema a
    future GPU-side script would populate and this module would then
    evaluate against the hard stop conditions below."""

    fullkv_natural_generation_wall_seconds: float = Field(ge=0.0)
    rkv_pass1_wall_seconds: float = Field(ge=0.0)
    token_identical_pass2_wall_seconds: float = Field(ge=0.0)
    score_recomputation_wall_seconds: float = Field(ge=0.0)
    targeted_capture_wall_seconds: float = Field(ge=0.0)
    cache_clone_restore_wall_seconds: float = Field(ge=0.0)
    one_fixed_shape_swap_wall_seconds: float = Field(ge=0.0)
    bridge_plus_48_scored_wall_seconds: float = Field(ge=0.0)

    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)
    every_parameter_on_cuda: bool
    observed_retention_ratio: float = Field(ge=0.0, le=1.0)
    event_count: int = Field(ge=0)
    projected_complete_pilot_gpu_hours: float = Field(ge=0.0)


@dataclass(frozen=True)
class B2AGateResult:
    passed: bool
    failed_conditions: tuple[str, ...]


HardStopReason = Literal[
    "projected_pilot_exceeds_4_gpu_hours",
    "peak_allocated_memory_exceeds_22_gib",
    "a_parameter_is_not_on_cuda",
    "no_meaningful_compression_observed",
    "insufficient_eligible_events",
]


def evaluate_b2a_hard_stop_gate(
    measurement: B2AOneExampleMeasurement,
    *,
    meaningful_compression_observed: bool,
    sufficient_eligible_events: bool,
) -> B2AGateResult:
    """Pure evaluation of the frozen hard-stop conditions against an
    ALREADY-COLLECTED `B2AOneExampleMeasurement` -- never collects one
    itself, never touches a GPU, never imports torch. `trajectory mismatch`
    and `capture/gather/absolute-position parity failure` (the other two
    hard stops named in the task brief) are structural pass/fail outcomes
    already produced by `kvcot.discovery.pass2.run_pass2_capture` --
    reused, never re-evaluated independently here."""
    failed: list[HardStopReason] = []
    if measurement.projected_complete_pilot_gpu_hours > MAX_PROJECTED_PILOT_GPU_HOURS:
        failed.append("projected_pilot_exceeds_4_gpu_hours")
    if measurement.peak_cuda_allocated_bytes / (1024**3) > MAX_PEAK_ALLOCATED_MEMORY_GIB:
        failed.append("peak_allocated_memory_exceeds_22_gib")
    if not measurement.every_parameter_on_cuda:
        failed.append("a_parameter_is_not_on_cuda")
    if not meaningful_compression_observed:
        failed.append("no_meaningful_compression_observed")
    if not sufficient_eligible_events:
        failed.append("insufficient_eligible_events")
    return B2AGateResult(passed=not failed, failed_conditions=tuple(failed))
