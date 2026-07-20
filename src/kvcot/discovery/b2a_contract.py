"""Future one-example B2A contract — definition and validation ONLY
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §11, strengthened by
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §10). This module is
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

## Strengthened gate (B1B-R2, review defect: "a `B2AGateResult` must not be
constructible as passing without explicit evidence")

The previous version of this module's `B2AGateResult` was a bare
`(passed, failed_conditions)` dataclass — trivially constructible as
`B2AGateResult(passed=True, failed_conditions=())` with no evidence
whatsoever backing that claim. `B2AGateEvidence` (a pydantic model with
every field REQUIRED, never `Optional`/defaulted) makes omitting any one of
the mandatory trajectory/parity/identity/environment conditions a
`ValidationError` at construction time, not a silent default `True`,
and `B2AGateResult.__post_init__` re-derives `passed`/`failed_conditions`
from the mandatory fields itself — a caller cannot hand-construct a
passing result while any mandatory field is `False`. `evaluate_b2a_gate`
is the only intended way to obtain a `B2AGateResult`.
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

# Hard future stop conditions. ANY one of these being true means B2A/B2B/the
# discovery pilot must stop -- this module only defines the thresholds and a
# pure-Python evaluator; it never runs a GPU measurement itself.
MAX_PROJECTED_PILOT_GPU_HOURS = 4.00
# B1B-R4 §14: renamed from allocated-only to total-peak-CUDA-tracked-memory
# terminology -- the gate now compares against `max(peak_allocated,
# peak_reserved)`, never allocated alone. `MAX_PEAK_ALLOCATED_MEMORY_GIB` is
# kept as a deprecated compatibility alias (same value, same name import
# sites already use).
MAX_PEAK_TRACKED_MEMORY_GIB = 22.0
MAX_PEAK_ALLOCATED_MEMORY_GIB = MAX_PEAK_TRACKED_MEMORY_GIB  # deprecated alias -- see comment above


class B2AOneExampleMeasurement(BaseModel):
    """What a B2A one-example run reports (B1B-R4 §12, superseding B1B-R3's
    version of this schema, which conflated restore/bridge-forward timing
    into aggregate buckets and then multiplied one of them by 144 in the
    projection). `real_pair_wall_seconds` holds one entry PER completed real
    pair evaluation (up to 12) -- never an aggregate bucket."""

    fullkv_natural_generation_wall_seconds: float = Field(ge=0.0)
    rkv_pass1_wall_seconds: float = Field(ge=0.0)
    rkv_pass2_wall_seconds: float = Field(ge=0.0)
    # Diagnostic submeasurement only -- ALREADY CONTAINED inside
    # `rkv_pass2_wall_seconds` (Pass 2 performs the targeted capture inline),
    # never added again on top of it in any projection.
    targeted_capture_wall_seconds: float = Field(ge=0.0)
    # `fullkv_natural_generation_wall_seconds + rkv_pass1_wall_seconds +
    # rkv_pass2_wall_seconds` -- the frozen once-per-example projection
    # component (B1B-R4 §12's "Recommended frozen projection semantics").
    per_example_total_wall_seconds: float = Field(ge=0.0)
    real_pair_wall_seconds: list[float] = Field(default_factory=list)
    no_op_pair_wall_seconds: list[float] = Field(default_factory=list)
    # The conservative per-pair statistic actually used by the projection:
    # `max(real_pair_wall_seconds)`, never their sum and never a separately
    # invented number -- `kvcot.discovery.b2a_evidence
    # .per_real_pair_projection_seconds`'s output.
    per_real_pair_seconds: float = Field(ge=0.0)

    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)
    every_parameter_on_cuda: bool
    observed_retention_ratio: float = Field(ge=0.0, le=1.0)
    event_count: int = Field(ge=0)
    projected_complete_pilot_gpu_hours: float = Field(ge=0.0)

    @property
    def peak_vram_gib(self) -> float:
        """B1B-R4 §14: gates on the MAXIMUM of allocated and reserved --
        never allocated alone (a run can be reserved-heavy/allocated-light
        or vice versa; both must be tracked)."""
        return max(self.peak_cuda_allocated_bytes, self.peak_cuda_reserved_bytes) / (1024**3)


# Every one of these must be `True` on `B2AGateEvidence`/`B2AGateResult` for
# `passed` to be `True` -- a missing value is impossible (pydantic requires
# every `B2AGateEvidence` field), and a `False` value on even one is a hard
# stop. `runtime_within_limit`/`peak_vram_within_limit` are the two
# threshold-derived conditions; every other name here is a direct identity/
# parity/environment condition.
MANDATORY_GATE_CONDITIONS: tuple[str, ...] = (
    "token_identical_replay",
    "prefill_decode_boundary_parity",
    "compaction_position_equality",
    "capture_gather_parity",
    "absolute_position_parity",
    "no_op_numerical_parity",
    # B1B-R4.1 §18/§30: named separately from the five trajectory/parity
    # conditions above -- those are all Pass-2-level (capture correctness,
    # evaluated before any pair is ever attempted); this one is swap-level
    # (did the semantic swap actually update provenance/kept-index
    # bookkeeping for every completed real pair,
    # `kvcot.discovery.pipeline.build_swap_pair_record`'s own parity
    # derivation). A worker reporting a pair-level `semantic_swap_parity_
    # failure` (`kvcot.discovery.attrition.STAGE_SEMANTIC_SWAP_PARITY_
    # FAILURE`) must fail this condition specifically, never only the
    # coarser `all_required_pair_evaluations_completed`.
    "semantic_swap_parity",
    "dataset_revision_match",
    "dataset_row_identity_match",
    "manifest_hash_match",
    "prompt_token_hash_match",
    "model_revision_match",
    "tokenizer_revision_match",
    "generation_config_hash_match",
    "rkv_config_hash_match",
    "no_offload_verified",
    "batch_size_verified",
    "runtime_within_limit",
    "peak_vram_within_limit",
    "one_example_only",
    # Retained from the pre-B1B-R2 hard-stop gate (CLAUDE.md-referenced
    # discovery protocol docs) -- not renamed away, only joined by the
    # identity/parity conditions above, never replaced by them.
    "meaningful_compression_observed",
    "sufficient_eligible_events",
    # B1B-R4 §21: exact selected-event and pair-completion counts as their
    # own mandatory gate evidence -- a worker that reports only an umbrella
    # validity boolean can no longer pass; every one of these four must be
    # independently demonstrated.
    "selected_event_count_exact",
    "real_pair_count_exact",
    "no_op_count_exact",
    "all_required_pair_evaluations_completed",
    # B1 execution-boundary closure §13: exact, DUPLICATE-DETECTING pair
    # identity accounting (`kvcot.discovery.b2a_evidence.PairIdentityEvidence`)
    # -- `real_pair_count_exact` above only ever compared a COUNT
    # (`count >= 4` per event in the prior derivation) and could not tell
    # four genuinely distinct pairs apart from the same pair recorded four
    # times, or twelve total real records that were not actually twelve
    # distinct (event, layer, head, candidate, donor) identities.
    "unique_real_pair_count_exact",
    "events_with_four_unique_pairs_exact",
    "no_duplicate_pair_identity",
)


class B2AGateEvidence(BaseModel):
    """Every field is REQUIRED (no `Optional`, no default) -- pydantic
    itself makes omitting any one of them a `ValidationError` at
    construction time, never a silently-assumed `True`. `runtime_gpu_hours`/
    `peak_vram_gib` are raw measurements; the corresponding
    `*_within_limit` booleans on `B2AGateResult` are always DERIVED from
    them against the frozen module-level thresholds by `evaluate_b2a_gate`
    -- never accepted here as an independent caller-supplied claim, so
    evidence can never assert "within limit" while also reporting an
    over-threshold raw number.
    """

    token_identical_replay: bool
    prefill_decode_boundary_parity: bool
    compaction_position_equality: bool
    capture_gather_parity: bool
    absolute_position_parity: bool
    no_op_numerical_parity: bool
    semantic_swap_parity: bool
    dataset_revision_match: bool
    dataset_row_identity_match: bool
    manifest_hash_match: bool
    prompt_token_hash_match: bool
    model_revision_match: bool
    tokenizer_revision_match: bool
    generation_config_hash_match: bool
    rkv_config_hash_match: bool
    no_offload_verified: bool
    batch_size_verified: bool
    one_example_only: bool
    meaningful_compression_observed: bool
    sufficient_eligible_events: bool
    selected_event_count_exact: bool
    real_pair_count_exact: bool
    no_op_count_exact: bool
    all_required_pair_evaluations_completed: bool
    unique_real_pair_count_exact: bool
    events_with_four_unique_pairs_exact: bool
    no_duplicate_pair_identity: bool

    runtime_gpu_hours: float = Field(ge=0.0)
    peak_vram_gib: float = Field(ge=0.0)


def build_gate_evidence_from_measurement(
    measurement: B2AOneExampleMeasurement,
    *,
    token_identical_replay: bool,
    prefill_decode_boundary_parity: bool,
    compaction_position_equality: bool,
    capture_gather_parity: bool,
    absolute_position_parity: bool,
    no_op_numerical_parity: bool,
    semantic_swap_parity: bool,
    unique_real_pair_count_exact: bool,
    events_with_four_unique_pairs_exact: bool,
    no_duplicate_pair_identity: bool,
    dataset_revision_match: bool,
    dataset_row_identity_match: bool,
    manifest_hash_match: bool,
    prompt_token_hash_match: bool,
    model_revision_match: bool,
    tokenizer_revision_match: bool,
    generation_config_hash_match: bool,
    rkv_config_hash_match: bool,
    batch_size_verified: bool,
    one_example_only: bool,
    meaningful_compression_observed: bool,
    sufficient_eligible_events: bool,
    selected_event_count_exact: bool,
    real_pair_count_exact: bool,
    no_op_count_exact: bool,
    all_required_pair_evaluations_completed: bool,
) -> B2AGateEvidence:
    """Build gate evidence from an already-collected `B2AOneExampleMeasurement`
    plus every identity/parity/environment condition the caller must supply
    explicitly. `no_offload_verified` and the two threshold-derived raw
    numbers (`runtime_gpu_hours`/`peak_vram_gib`) come FROM the measurement
    itself (never a second, independently-asserted copy) so the two can
    never disagree with what was actually measured."""
    return B2AGateEvidence(
        token_identical_replay=token_identical_replay,
        prefill_decode_boundary_parity=prefill_decode_boundary_parity,
        compaction_position_equality=compaction_position_equality,
        capture_gather_parity=capture_gather_parity,
        absolute_position_parity=absolute_position_parity,
        no_op_numerical_parity=no_op_numerical_parity,
        semantic_swap_parity=semantic_swap_parity,
        unique_real_pair_count_exact=unique_real_pair_count_exact,
        events_with_four_unique_pairs_exact=events_with_four_unique_pairs_exact,
        no_duplicate_pair_identity=no_duplicate_pair_identity,
        dataset_revision_match=dataset_revision_match,
        dataset_row_identity_match=dataset_row_identity_match,
        manifest_hash_match=manifest_hash_match,
        prompt_token_hash_match=prompt_token_hash_match,
        model_revision_match=model_revision_match,
        tokenizer_revision_match=tokenizer_revision_match,
        generation_config_hash_match=generation_config_hash_match,
        rkv_config_hash_match=rkv_config_hash_match,
        no_offload_verified=measurement.every_parameter_on_cuda,
        batch_size_verified=batch_size_verified,
        one_example_only=one_example_only,
        meaningful_compression_observed=meaningful_compression_observed,
        sufficient_eligible_events=sufficient_eligible_events,
        selected_event_count_exact=selected_event_count_exact,
        real_pair_count_exact=real_pair_count_exact,
        no_op_count_exact=no_op_count_exact,
        all_required_pair_evaluations_completed=all_required_pair_evaluations_completed,
        runtime_gpu_hours=measurement.projected_complete_pilot_gpu_hours,
        peak_vram_gib=measurement.peak_vram_gib,
    )


@dataclass(frozen=True)
class B2AGateResult:
    """Every mandatory condition is its own explicit field (never collapsed
    into `passed` alone) so a caller/report can see exactly which condition
    failed without re-deriving anything. `__post_init__` re-derives both
    `passed` and `failed_conditions` from the mandatory fields themselves --
    constructing a `B2AGateResult` directly with an internally-inconsistent
    combination (e.g. `passed=True` while a mandatory field is `False`)
    raises immediately. `evaluate_b2a_gate` is the only intended
    constructor."""

    passed: bool
    token_identical_replay: bool
    prefill_decode_boundary_parity: bool
    compaction_position_equality: bool
    capture_gather_parity: bool
    absolute_position_parity: bool
    no_op_numerical_parity: bool
    semantic_swap_parity: bool
    dataset_revision_match: bool
    dataset_row_identity_match: bool
    manifest_hash_match: bool
    prompt_token_hash_match: bool
    model_revision_match: bool
    tokenizer_revision_match: bool
    generation_config_hash_match: bool
    rkv_config_hash_match: bool
    no_offload_verified: bool
    batch_size_verified: bool
    runtime_within_limit: bool
    peak_vram_within_limit: bool
    one_example_only: bool
    meaningful_compression_observed: bool
    sufficient_eligible_events: bool
    selected_event_count_exact: bool
    real_pair_count_exact: bool
    no_op_count_exact: bool
    all_required_pair_evaluations_completed: bool
    unique_real_pair_count_exact: bool
    events_with_four_unique_pairs_exact: bool
    no_duplicate_pair_identity: bool
    failed_conditions: tuple[str, ...]

    def __post_init__(self) -> None:
        values = {name: getattr(self, name) for name in MANDATORY_GATE_CONDITIONS}
        expected_failed = tuple(name for name in MANDATORY_GATE_CONDITIONS if not values[name])
        if tuple(sorted(self.failed_conditions)) != tuple(sorted(expected_failed)):
            raise ValueError(
                f"failed_conditions {self.failed_conditions!r} disagrees with the mandatory field values "
                f"{values!r} -- expected failed_conditions={expected_failed!r}. B2AGateResult must always "
                "be constructed via evaluate_b2a_gate(), never hand-assembled."
            )
        expected_passed = not expected_failed
        if self.passed != expected_passed:
            raise ValueError(
                f"passed={self.passed!r} disagrees with the derived value {expected_passed!r} given the "
                "mandatory field values -- it is impossible for this result to PASS while any mandatory "
                "condition is False, and impossible to FAIL while every condition is True."
            )


def evaluate_b2a_gate(evidence: B2AGateEvidence) -> B2AGateResult:
    """The only intended way to obtain a `B2AGateResult`. Pure evaluation of
    an ALREADY-COLLECTED `B2AGateEvidence` against the frozen thresholds and
    every mandatory condition -- never collects evidence itself, never
    touches a GPU, never imports torch. Missing evidence is structurally
    impossible (`B2AGateEvidence` requires every field); any single `False`
    condition or any measurement over threshold is an independent hard
    stop, and every hard stop is reported (never short-circuited on the
    first failure), so a real run gets the complete failure list at once."""
    runtime_within_limit = evidence.runtime_gpu_hours <= MAX_PROJECTED_PILOT_GPU_HOURS
    peak_vram_within_limit = evidence.peak_vram_gib <= MAX_PEAK_ALLOCATED_MEMORY_GIB

    values: dict[str, bool] = {
        "token_identical_replay": evidence.token_identical_replay,
        "prefill_decode_boundary_parity": evidence.prefill_decode_boundary_parity,
        "compaction_position_equality": evidence.compaction_position_equality,
        "capture_gather_parity": evidence.capture_gather_parity,
        "absolute_position_parity": evidence.absolute_position_parity,
        "no_op_numerical_parity": evidence.no_op_numerical_parity,
        "semantic_swap_parity": evidence.semantic_swap_parity,
        "dataset_revision_match": evidence.dataset_revision_match,
        "dataset_row_identity_match": evidence.dataset_row_identity_match,
        "manifest_hash_match": evidence.manifest_hash_match,
        "prompt_token_hash_match": evidence.prompt_token_hash_match,
        "model_revision_match": evidence.model_revision_match,
        "tokenizer_revision_match": evidence.tokenizer_revision_match,
        "generation_config_hash_match": evidence.generation_config_hash_match,
        "rkv_config_hash_match": evidence.rkv_config_hash_match,
        "no_offload_verified": evidence.no_offload_verified,
        "batch_size_verified": evidence.batch_size_verified,
        "runtime_within_limit": runtime_within_limit,
        "peak_vram_within_limit": peak_vram_within_limit,
        "one_example_only": evidence.one_example_only,
        "meaningful_compression_observed": evidence.meaningful_compression_observed,
        "sufficient_eligible_events": evidence.sufficient_eligible_events,
        "selected_event_count_exact": evidence.selected_event_count_exact,
        "real_pair_count_exact": evidence.real_pair_count_exact,
        "no_op_count_exact": evidence.no_op_count_exact,
        "all_required_pair_evaluations_completed": evidence.all_required_pair_evaluations_completed,
        "unique_real_pair_count_exact": evidence.unique_real_pair_count_exact,
        "events_with_four_unique_pairs_exact": evidence.events_with_four_unique_pairs_exact,
        "no_duplicate_pair_identity": evidence.no_duplicate_pair_identity,
    }
    failed = tuple(name for name in MANDATORY_GATE_CONDITIONS if not values[name])
    return B2AGateResult(passed=not failed, failed_conditions=failed, **values)
