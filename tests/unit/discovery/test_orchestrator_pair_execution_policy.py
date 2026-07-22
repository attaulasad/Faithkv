"""B1B-R4 §7 regression tests: `NoOpMode` must ACTUALLY control how many
no-op pair evaluations `run_example` builds, not merely document an
intended interpretation while always building one no-op pair per event
regardless of the mode named. Reuses the exact synthetic CPU harness
`test_b1b_integration.py` already exercises -- never a second,
independently-written orchestration fixture.
"""
from __future__ import annotations

from _synthetic_harness import (
    EOS_TOKEN_ID,
    NUM_HEADS,
    NUM_LAYERS,
    HarnessState,
    branch_step_fn,
    build_snapshot_from_state,
    fresh_state_factory,
    install_fake_rkv_compression_module,
    make_step_fns,
)

from kvcot.discovery.attrition import AttritionCounters
from kvcot.discovery.constants import NoOpMode
from kvcot.discovery.orchestrator import PairExecutionPolicy, run_example
from kvcot.discovery.pass1 import NaturalRunProvenance
from kvcot.discovery.sampling import IdentitySeedParts

PROMPT_LENGTH = 10
DESIRED_GENERATED_LENGTH = 290
STOP_AT = PROMPT_LENGTH + DESIRED_GENERATED_LENGTH
MAX_NEW_TOKENS = 295
PROMPT_TOKEN_IDS = list(range(1, PROMPT_LENGTH + 1))
IDENTITY = IdentitySeedParts(
    global_seed=13, dataset_name="synthetic", problem_index=0, model_revision="rev-a", rkv_revision="rkv-rev"
)
PROVENANCE = NaturalRunProvenance(
    model_name="synthetic-model", model_revision="rev-a", tokenizer_name="synthetic-tokenizer",
    tokenizer_revision="rev-a", rkv_revision="rkv-rev", config_sha256="deadbeef",
    dataset_name="synthetic", example_id="ex-1",
)


def _always_correct(generated_ids: list[int]) -> tuple[str, str]:
    return "42", "correct"


def _run(monkeypatch, policy: PairExecutionPolicy | None, clock_fn=None, capture_timer_fn=None):
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    example_attrition, pair_attrition = AttritionCounters(), AttritionCounters()
    kwargs = dict(
        example_id="ex-policy",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        provenance=PROVENANCE,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        pass1_initial_state=HarnessState(),
        pass2_initial_state_factory=fresh_state_factory(),
        prefill_fn=prefill_fn,
        decode_one_fn=decode_one_fn,
        snapshot_fn=build_snapshot_from_state,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=EOS_TOKEN_ID,
        answer_fn=_always_correct,
        num_hidden_layers=NUM_LAYERS,
        num_key_value_heads=NUM_HEADS,
        identity=IDENTITY,
        branch_step_fn=branch_step_fn,
        example_attrition=example_attrition,
        pair_attrition=pair_attrition,
    )
    if policy is not None:
        kwargs["pair_execution_policy"] = policy
    if clock_fn is not None:
        kwargs["clock_fn"] = clock_fn
    if capture_timer_fn is not None:
        kwargs["capture_timer_fn"] = capture_timer_fn
    result = run_example(**kwargs)
    return result, example_attrition, pair_attrition


def test_default_policy_is_cpu_required_and_unchanged_from_prior_behavior(monkeypatch):
    result, _, pair_attrition = _run(monkeypatch, None)
    assert result.valid is True
    assert len(result.pair_records) == 15  # 3 events x (4 real + 1 no-op)
    assert result.attempted_real_pair_count == 12
    assert result.attempted_no_op_pair_count == 3
    assert result.completed_real_pair_count == 12
    assert result.completed_no_op_pair_count == 3
    assert pair_attrition.total_entered == 15


def test_cpu_required_explicit_matches_default(monkeypatch):
    result, _, _ = _run(monkeypatch, PairExecutionPolicy(no_op_mode=NoOpMode.CPU_REQUIRED))
    assert len(result.pair_records) == 15
    assert result.attempted_no_op_pair_count == 3


def test_b2a_single_calibration_builds_exactly_one_noop_for_first_event_only(monkeypatch):
    result, _, pair_attrition = _run(monkeypatch, PairExecutionPolicy(no_op_mode=NoOpMode.B2A_SINGLE_CALIBRATION))
    assert result.valid is True
    assert result.attempted_real_pair_count == 12
    assert result.attempted_no_op_pair_count == 1
    assert result.completed_real_pair_count == 12
    assert result.completed_no_op_pair_count == 1
    assert len(result.pair_records) == 13  # 12 real + 1 no-op
    assert pair_attrition.total_entered == 13

    no_op_records = [r for r in result.pair_records if r.is_noop_control]
    assert len(no_op_records) == 1
    real_records = [r for r in result.pair_records if not r.is_noop_control]
    assert len(real_records) == 12

    # The single no-op must come from the FIRST selected event (chronological
    # ordinal 0 in the frozen plan), never a later one.
    assert no_op_records[0].chronological_event_ordinal == 0


def test_disabled_builds_zero_noop_pairs(monkeypatch):
    result, _, pair_attrition = _run(monkeypatch, PairExecutionPolicy(no_op_mode=NoOpMode.DISABLED))
    assert result.valid is True
    assert result.attempted_real_pair_count == 12
    assert result.attempted_no_op_pair_count == 0
    assert result.completed_no_op_pair_count == 0
    assert len(result.pair_records) == 12  # 3 events x 4 real pairs only
    assert pair_attrition.total_entered == 12
    assert all(not r.is_noop_control for r in result.pair_records)


def test_per_pair_wall_seconds_uses_injected_deterministic_clock(monkeypatch):
    """B1B-R4 §12: `run_example` records one wall-clock duration PER
    completed pair evaluation, via an injectable clock -- proven here with
    a fake clock that advances by a fixed amount on every call, so the
    resulting durations are exactly predictable rather than depending on
    real timing noise."""
    ticks = iter(float(i) for i in range(0, 100_000))

    def fake_clock():
        return next(ticks)

    result, _, _ = _run(monkeypatch, PairExecutionPolicy(no_op_mode=NoOpMode.B2A_SINGLE_CALIBRATION), clock_fn=fake_clock)

    assert len(result.real_pair_wall_seconds) == 12
    assert len(result.no_op_pair_wall_seconds) == 1
    # Every duration is start-to-end on a strictly-increasing fake clock, so
    # every recorded duration must be positive and finite.
    assert all(d > 0 for d in result.real_pair_wall_seconds)
    assert all(d > 0 for d in result.no_op_pair_wall_seconds)


def test_disabled_mode_records_zero_no_op_timings(monkeypatch):
    result, _, _ = _run(monkeypatch, PairExecutionPolicy(no_op_mode=NoOpMode.DISABLED))
    assert result.no_op_pair_wall_seconds == ()
    assert len(result.real_pair_wall_seconds) == 12


def test_capture_timer_fn_threads_through_to_the_real_pass2_capture_path(monkeypatch):
    """Independent-audit Gate H2.2: `run_example`'s `capture_timer_fn`
    parameter must reach `kvcot.discovery.capture.capture_update_kv`'s real
    wrapped call (via `pass2.run_pass2_capture`) -- firing exactly once per
    selected target (3, matching the frozen `EVENTS_SELECTED_PER_EXAMPLE`),
    each timing the genuine gather/parity computation, never a no-op stub
    that silently never gets invoked."""
    calls = []

    def fake_capture_timer(phase, operation):
        calls.append(phase)
        return operation()

    result, _, _ = _run(monkeypatch, None, capture_timer_fn=fake_capture_timer)
    assert result.valid is True
    assert calls == ["capture_gather_and_parity"] * 3


def test_unrecognized_no_op_mode_raises(monkeypatch):
    import pytest

    # Bypass the enum to simulate a future unrecognized mode value --
    # `run_example` must fail closed rather than silently defaulting to
    # CPU_REQUIRED-like behavior.
    bad_policy = PairExecutionPolicy()
    object.__setattr__(bad_policy, "no_op_mode", "not-a-real-mode")

    with pytest.raises(ValueError, match="unrecognized NoOpMode"):
        _run(monkeypatch, bad_policy)
