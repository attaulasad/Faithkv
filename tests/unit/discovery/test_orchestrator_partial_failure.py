"""Independent-audit Gate H1 regression tests: `run_example`
(`kvcot.discovery.orchestrator`) must never let an unexpected exception
mid-Pass-2 or mid-pair-loop propagate bare and lose everything accumulated
so far -- it must return an `ExampleResult` with `aborted=True` and every
already-completed pair/attrition/mutation record preserved, exactly like a
worker body failing after real work has begun (`docs
/B1_INDEPENDENT_AUDIT_REPAIR.md` Gate H1). Reuses the exact synthetic CPU
harness `test_b1b_integration.py`/`test_orchestrator_pair_execution_policy.py`
already exercise -- never a second, independently-written fixture.
"""
from __future__ import annotations

import pytest
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

from kvcot.discovery.attrition import (
    STAGE_PASS2_EXECUTION_EXCEPTION,
    STAGE_UNEXPECTED_PAIR_EXCEPTION,
    AttritionCounters,
)
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


def _base_kwargs(**overrides):
    kwargs = dict(
        example_id="ex-partial-failure",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        provenance=PROVENANCE,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        pass1_initial_state=HarnessState(),
        pass2_initial_state_factory=fresh_state_factory(),
        snapshot_fn=build_snapshot_from_state,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=EOS_TOKEN_ID,
        answer_fn=_always_correct,
        num_hidden_layers=NUM_LAYERS,
        num_key_value_heads=NUM_HEADS,
        identity=IDENTITY,
        branch_step_fn=branch_step_fn,
        example_attrition=AttritionCounters(),
        pair_attrition=AttritionCounters(),
        pair_execution_policy=PairExecutionPolicy(no_op_mode=NoOpMode.CPU_REQUIRED),
    )
    kwargs.update(overrides)
    return kwargs


def test_unexpected_exception_during_pre_branch_guard_aborts_with_partial_evidence(monkeypatch):
    """A `pre_branch_guard` that raises (e.g. simulating a real CUDA OOM
    while estimating pre-branch memory) partway through the per-pair loop
    must not lose the pairs already completed before it -- `run_example`
    must catch it, mark `aborted=True`, and return every already-built
    `pair_records`/`attempted_pair_identities`/`completed_pair_identities`
    entry rather than propagating the bare exception."""
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    call_count = {"n": 0}

    class _AcceptedGuardEvidence:
        accepted = True
        rejection_reason = None

    def failing_after_two(target, kind):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise RuntimeError("simulated CUDA out of memory: allocation failed")
        return _AcceptedGuardEvidence()

    example_attrition, pair_attrition = AttritionCounters(), AttritionCounters()
    result = run_example(**_base_kwargs(
        prefill_fn=prefill_fn, decode_one_fn=decode_one_fn,
        example_attrition=example_attrition, pair_attrition=pair_attrition,
        pre_branch_guard=failing_after_two,
    ))

    assert result.aborted is True
    assert result.abort_failure_type == "RuntimeError"
    assert "out of memory" in result.abort_failure_message.lower()
    assert result.abort_is_oom is True
    # The first two pair attempts (guard calls 1 and 2) completed fully
    # before the third call raised -- their evidence must survive.
    assert len(result.attempted_pair_identities) == 3  # 2 completed + 1 aborted attempt
    assert len(result.completed_pair_identities) == 2
    assert len(result.pair_records) == 2
    assert len(result.semantic_mutation_reports) == 2
    # The aborted pair's attrition entry is accounted for -- the invariant
    # `total_entered == passed_all + sum(dropped_at)` must still hold.
    pair_attrition.assert_consistent()
    assert pair_attrition.dropped_at[STAGE_UNEXPECTED_PAIR_EXCEPTION] == 1
    assert any(
        detail.stage == STAGE_UNEXPECTED_PAIR_EXCEPTION for detail in result.pair_failure_details
    )
    # Example-level attrition still records the example as having PASSED
    # Pass 1/Pass 2 -- the abort happened at the pair level, not earlier.
    example_attrition.assert_consistent()
    assert example_attrition.passed_all == 1


def test_unexpected_exception_during_pass2_capture_aborts_with_trace_preserved(monkeypatch):
    """Pass 2 raising (e.g. a real CUDA OOM during targeted capture) must
    not lose Pass 1's already-valid `trace` -- `run_example` must catch it
    and return an `ExampleResult` with `aborted=True`,
    `invalid_stage=STAGE_PASS2_EXECUTION_EXCEPTION`, and `trace` populated
    from the real Pass 1 run rather than `None`."""
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    def failing_snapshot_fn(state):
        raise RuntimeError("simulated CUDA out of memory during targeted capture")

    example_attrition, pair_attrition = AttritionCounters(), AttritionCounters()
    result = run_example(**_base_kwargs(
        prefill_fn=prefill_fn, decode_one_fn=decode_one_fn,
        example_attrition=example_attrition, pair_attrition=pair_attrition,
        snapshot_fn=failing_snapshot_fn,
    ))

    assert result.valid is False
    assert result.aborted is True
    assert result.invalid_stage == STAGE_PASS2_EXECUTION_EXCEPTION
    assert result.abort_failure_type == "RuntimeError"
    assert result.abort_is_oom is True
    assert result.trace is not None
    assert result.trace.natural_answer_status == "correct"
    example_attrition.assert_consistent()
    assert example_attrition.dropped_at[STAGE_PASS2_EXECUTION_EXCEPTION] == 1


def test_pass2_token_mismatch_preserves_the_actual_replayed_tokens(monkeypatch):
    """Independent-audit Gate H3.7: a detected Pass-2 token mismatch (a
    NORMAL `Pass2Result(valid=False, ...)` return, never an exception) must
    not leave only the bare `pass2_token_mismatch` stage name --
    `pass2_result.replayed_token_ids` (real diagnostic evidence: exactly
    what WAS fed) must survive onto the returned `ExampleResult`, where
    `kvcot.discovery.mismatch.build_mismatch_record` can compare it against
    `trace.full_token_ids` without re-running the model."""
    from kvcot.discovery import orchestrator
    from kvcot.discovery.pass2 import INVALID_TOKEN_MISMATCH, Pass2Result

    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    fake_replayed = (999, 998, 997, 996)
    monkeypatch.setattr(
        orchestrator, "run_pass2_capture",
        lambda *args, **kwargs: Pass2Result(False, INVALID_TOKEN_MISMATCH, fake_replayed, ()),
    )

    example_attrition, pair_attrition = AttritionCounters(), AttritionCounters()
    result = run_example(**_base_kwargs(
        prefill_fn=prefill_fn, decode_one_fn=decode_one_fn,
        example_attrition=example_attrition, pair_attrition=pair_attrition,
    ))

    assert result.valid is False
    assert result.invalid_stage == INVALID_TOKEN_MISMATCH
    assert result.pass2_invalid_reason == INVALID_TOKEN_MISMATCH
    assert result.pass2_replayed_token_ids == fake_replayed
    assert result.trace is not None  # Pass 1's trace still present


def test_unexpected_exception_during_natural_pass1_run_is_marked_aborted(monkeypatch):
    """Hostile-audit follow-up to Gate H1: `run_natural_pass1` only ever
    raises or returns a valid trace (never an "invalid" sentinel) -- any
    exception here is a genuine, unexpected failure (e.g. a real CUDA OOM),
    not a normal answer-incorrect/cap-hit case. The funnel stage name
    (`STAGE_NATURAL_RUN_INVALID`) is unchanged, but `aborted`/
    `abort_failure_type`/`abort_is_oom` must now be populated so a caller
    (`kvcot.discovery.b2a_workers.run_rkv_worker`) can tell a crash apart
    from a legitimate structural attrition drop."""
    from kvcot.discovery import orchestrator
    from kvcot.discovery.attrition import STAGE_NATURAL_RUN_INVALID

    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)

    def failing_run_natural_pass1(*args, **kwargs):
        raise RuntimeError("CUDA out of memory during natural generation")

    monkeypatch.setattr(orchestrator, "run_natural_pass1", failing_run_natural_pass1)

    example_attrition, pair_attrition = AttritionCounters(), AttritionCounters()
    result = run_example(**_base_kwargs(
        prefill_fn=prefill_fn, decode_one_fn=decode_one_fn,
        example_attrition=example_attrition, pair_attrition=pair_attrition,
    ))

    assert result.valid is False
    assert result.invalid_stage == STAGE_NATURAL_RUN_INVALID
    assert result.aborted is True
    assert result.abort_failure_type == "RuntimeError"
    assert result.abort_is_oom is True
    example_attrition.assert_consistent()
    assert example_attrition.dropped_at[STAGE_NATURAL_RUN_INVALID] == 1


def test_normal_completion_is_never_marked_aborted(monkeypatch):
    """Regression guard: a fully successful run (no injected failure) must
    have `aborted=False` and no abort metadata -- Gate H1's new fields must
    never appear on an ordinary success."""
    install_fake_rkv_compression_module(monkeypatch)
    prefill_fn, decode_one_fn = make_step_fns(stop_at_predicted_position=STOP_AT)
    result = run_example(**_base_kwargs(prefill_fn=prefill_fn, decode_one_fn=decode_one_fn))
    assert result.valid is True
    assert result.aborted is False
    assert result.abort_failure_type is None
    assert result.abort_failure_message is None
    assert result.abort_is_oom is False
