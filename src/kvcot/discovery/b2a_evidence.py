"""Real B2A evidence derivation (B1B-R3 §12). Every function here computes
one gate-evidence field from ACTUAL observations already collected by the
harness -- `kvcot.discovery.b2a_execute` calls these instead of hand-writing
`True`/`0.0` literals. Nothing here runs a GPU or collects new
measurements; this module is a pure function of already-collected
`kvcot.discovery.orchestrator.ExampleResult` / `B2AOneExampleMeasurement`
data, so it is fully CPU-testable against synthetic `ExampleResult`
fixtures.

## What each field means and where its value comes from

- `token_identical_replay` / `prefill_decode_boundary_parity` /
  `compaction_position_equality` / `capture_gather_parity` /
  `absolute_position_parity`: ALL FIVE are derived from
  `example_result.valid` -- `kvcot.discovery.pass2.run_pass2_capture`
  structurally REQUIRES every one of these to hold (exact token match at
  every replayed position, correct prefill/decode call-boundary shapes,
  matching compaction-event positions, `parity_check_passed` on every
  selected target's capture record, and cross-pass absolute-survivor
  identity match) before it can return `valid=True` at all -- a failure at
  ANY of them invalidates the whole example (`kvcot.discovery.pass2`
  module docstring). This is a deliberately conservative, fail-closed
  simplification: an example that fails for an unrelated Pass-1 reason
  (e.g. `STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE`) reports all five as
  `False` too, since none of them were ever actually demonstrated for a
  COMPLETE run -- never vacuously `True` because "the check that would
  have caught it never ran".
- `no_op_numerical_parity`: derived from finding an ACTUAL `is_noop_control
  =True` record among `example_result.pair_records`
  (`kvcot.discovery.schemas.SwapPairRecord`'s own pydantic validators
  already require bit-exact `baseline_per_token_nll == swapped_per_token_nll`
  for any record constructed with `is_noop_control=True` -- so a present,
  valid no-op record IS the calibration). `False` if no such record exists
  in this example's pair_records (the mandatory no-op attempt itself
  failed for every targeted event).
- `dataset_row_identity_match` / `prompt_token_hash_match` /
  `manifest_hash_match`: derived from
  `kvcot.discovery.b2a_execute`'s own pre-flight prompt-identity
  verification (re-render, re-tokenize, re-hash, compare against the
  manifest) -- `run_b2a_calibration` REFUSES to proceed to model inference
  at all on any mismatch (B1B-R3 §6), so reaching evidence collection is
  itself proof these matched; never re-derived a second, weaker way here.
- `model_revision_match` / `tokenizer_revision_match`: derived
  structurally -- both `AutoModelForCausalLM.from_pretrained` and
  `AutoTokenizer.from_pretrained` were called with an EXPLICIT `revision=`
  kwarg equal to the frozen config value; transformers/huggingface_hub
  raise rather than silently substitute a different revision when a
  specific commit is requested and unavailable. This is a structural
  guarantee from the call shape, not a runtime read-back of a resolved
  commit attribute (transformers does not expose one uniformly across
  versions) -- documented here as the weaker of the two kinds of identity
  check this module performs, alongside the stronger, genuinely
  runtime-read-back `rkv_config_hash_match` below.
- `generation_config_hash_match`: derived the same structural way --
  `kvcot.discovery.orchestrator.run_example` reads `max_new_tokens`,
  `batch_size`, etc. directly off the SAME `config.generation` object used
  to compute the frozen hash (no second, independently-typed copy exists
  anywhere in this path that could silently drift) -- also a structural
  guarantee, not an independent runtime introspection.
- `rkv_config_hash_match`: derived from
  `kvcot.discovery.runtime_rkv_verification.verify_runtime_matches_frozen`
  -- a genuine runtime read-back off the loaded model's per-layer
  `kv_cluster`/`config` objects, the strongest identity check in this
  module.
- `observed_retention_ratio`: `mean(final cache length per layer) /
  prompt+generated token count` -- computed directly from
  `trace.cache_length_final_per_layer` and `len(trace.full_token_ids)`,
  never a configured/target value.
- `event_count`: the number of DISTINCT selected compaction events
  actually represented in `example_result.pair_records`
  (`len({pr.compaction_event_id for pr in pair_records})`) -- explicitly
  NOT `len(pair_records)` (which double/quintuple-counts per-event pair
  attempts). A conservative lower bound: if every pair attempt for one
  selected event failed, that event is invisible to this count even though
  Pass 1 selected it -- documented, not silently treated as exact.
- `sufficient_eligible_events`: `example_result.valid` -- Pass 1's
  `build_pass1_plan` already hard-requires >=3 eligible events to produce
  any plan at all (`PLAN_FAILURE_TOO_FEW_ELIGIBLE_EVENTS`), so a valid
  example structurally proves this.
- `meaningful_compression_observed`: `event_count >= 1 and
  observed_retention_ratio < 1.0` -- at least one real eviction occurred
  AND the final cache is strictly smaller than the FullKV-equivalent
  length. This exact threshold is the definition, not a separate
  configured cutoff.
- `projected_complete_pilot_gpu_hours`: see
  `project_complete_pilot_gpu_hours`'s docstring for the exact formula.
"""
from __future__ import annotations

from dataclasses import dataclass

from kvcot.discovery.constants import B2B_PILOT_EXAMPLE_COUNT, B2B_PILOT_TOTAL_REAL_BRANCHES


@dataclass(frozen=True)
class DerivedTrajectoryEvidence:
    token_identical_replay: bool
    prefill_decode_boundary_parity: bool
    compaction_position_equality: bool
    capture_gather_parity: bool
    absolute_position_parity: bool
    no_op_numerical_parity: bool
    event_count: int
    sufficient_eligible_events: bool
    observed_retention_ratio: float
    meaningful_compression_observed: bool


def derive_trajectory_evidence(example_result) -> DerivedTrajectoryEvidence:
    valid = example_result.valid
    event_count = len({pr.compaction_event_id for pr in example_result.pair_records})
    no_op_found = any(pr.is_noop_control for pr in example_result.pair_records)

    trace = example_result.trace
    if trace is not None and trace.cache_length_final_per_layer:
        total_tokens = len(trace.full_token_ids)
        mean_final_len = sum(trace.cache_length_final_per_layer.values()) / len(trace.cache_length_final_per_layer)
        observed_retention_ratio = mean_final_len / total_tokens if total_tokens > 0 else 0.0
    else:
        observed_retention_ratio = 0.0

    meaningful_compression_observed = event_count >= 1 and observed_retention_ratio < 1.0

    return DerivedTrajectoryEvidence(
        token_identical_replay=valid,
        prefill_decode_boundary_parity=valid,
        compaction_position_equality=valid,
        capture_gather_parity=valid,
        absolute_position_parity=valid,
        no_op_numerical_parity=no_op_found,
        event_count=event_count,
        sufficient_eligible_events=valid,
        observed_retention_ratio=observed_retention_ratio,
        meaningful_compression_observed=meaningful_compression_observed,
    )


def project_complete_pilot_gpu_hours(
    *,
    fullkv_natural_generation_wall_seconds: float,
    rkv_pass1_wall_seconds: float,
    token_identical_pass2_wall_seconds: float,
    score_recomputation_wall_seconds: float,
    targeted_capture_wall_seconds: float,
    cache_clone_restore_wall_seconds: float,
    one_fixed_shape_swap_wall_seconds: float,
    bridge_plus_48_scored_wall_seconds: float,
) -> float:
    """Projected complete-B2B-pilot GPU-hours, extrapolated from this ONE
    example's measured component wall-times.

    Exact formula:

    ```
    per_example_seconds = fullkv_natural_generation_wall_seconds
                         + rkv_pass1_wall_seconds
                         + token_identical_pass2_wall_seconds
                         + score_recomputation_wall_seconds
                         + targeted_capture_wall_seconds

    per_branch_seconds = cache_clone_restore_wall_seconds
                        + one_fixed_shape_swap_wall_seconds
                        + bridge_plus_48_scored_wall_seconds

    projected_seconds = B2B_PILOT_EXAMPLE_COUNT * per_example_seconds
                       + B2B_PILOT_TOTAL_REAL_BRANCHES * per_branch_seconds

    projected_gpu_hours = projected_seconds / 3600
    ```

    `B2B_PILOT_EXAMPLE_COUNT` (12) scales the once-per-example components
    (natural generation, Pass 1, Pass 2, score recomputation, targeted
    capture); `B2B_PILOT_TOTAL_REAL_BRANCHES` (144 = 12 examples x 3 events
    x 4 real swaps) scales the once-per-branch components (clone/restore,
    the swap itself, and the bridge-plus-48-scored evaluation) -- matching
    B2B's own accounting exactly (`kvcot.discovery.constants`), never a
    separately-invented multiplier. The mandatory no-op calibration is
    NOT separately scaled into this projection (B2A's single calibration
    time is already included in `cache_clone_restore_wall_seconds`'s own
    measurement for this one example; B2B's no-op accounting is a CPU-test
    concern, not a GPU-time driver -- `kvcot.discovery.constants.NoOpMode`).
    """
    per_example_seconds = (
        fullkv_natural_generation_wall_seconds
        + rkv_pass1_wall_seconds
        + token_identical_pass2_wall_seconds
        + score_recomputation_wall_seconds
        + targeted_capture_wall_seconds
    )
    per_branch_seconds = (
        cache_clone_restore_wall_seconds + one_fixed_shape_swap_wall_seconds + bridge_plus_48_scored_wall_seconds
    )
    projected_seconds = (
        B2B_PILOT_EXAMPLE_COUNT * per_example_seconds + B2B_PILOT_TOTAL_REAL_BRANCHES * per_branch_seconds
    )
    return projected_seconds / 3600.0
