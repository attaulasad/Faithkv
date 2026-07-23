"""B2A-R3 pure qualification evaluator tests (protocol §10). Every test
uses injected synthetic evidence -- no torch, no CUDA, no R-KV import, no
real FullKV inference."""
from __future__ import annotations

import copy

import pytest

from kvcot.discovery.b2a_r3_contract import (
    B2A_R3_QUALIFICATION_CONDITIONS,
    GENERATION_CONFIG_SHA256,
    QUALIFICATION_CANDIDATE_LIMIT,
)
from kvcot.discovery.b2a_r3_qualification import (
    B2AR3FullKVQualificationEvidence,
    CandidateQualificationOutcomeR3,
    QualificationRefused,
    build_qualification_outcome,
    evaluate_b2a_r3_qualification_conditions,
    select_first_qualified_r3,
)
from kvcot.discovery.b2a_r3_runtime import predict_runtime
from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text


def _row(unique_id="test/algebra/1.json"):
    return {
        "problem": "p", "solution": "s", "answer": "42", "subject": "Algebra", "level": 5,
        "unique_id": unique_id,
    }


def _valid_placement():
    return {
        "requested_device": "cuda:0",
        "every_parameter_on_cuda": True,
        "no_offload_verified": True,
        "parameter_count": 100,
        "unique_device_types": ["cuda"],
        "unique_devices": ["cuda:0"],
        "hf_device_map": None,
    }


def _valid_timing():
    return [
        {
            "phase": "generation", "started_at": 0.0, "ended_at": 1.0, "duration_seconds": 1.0,
            "synchronize_before_start": True, "synchronize_before_end": True, "completed": True,
            "failure_type": None, "failure_message": None,
        }
    ]


CANDIDATE_MANIFEST_HASH = "b" * 64
CONFIG_SHA = "c" * 64
PROMPT_HASH = "d" * 64


def _candidate_manifest():
    payload = {"dataset_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "x": 1}
    payload["canonical_sha256"] = sha256_json(payload)
    return payload


def _valid_evidence(**overrides) -> B2AR3FullKVQualificationEvidence:
    row = _row()
    generated_ids = [1, 2, 3, 4, 5]
    manifest = _candidate_manifest()
    fields = dict(
        candidate_ordinal=0,
        source_example_index=7,
        unique_id=row["unique_id"],
        row=row,
        raw_row_sha256=sha256_json(row),
        problem_sha256=sha256_text(row["problem"]),
        gold_answer_sha256=sha256_text(row["answer"]),
        worker_dataset_repo="HuggingFaceH4/MATH-500",
        worker_dataset_config="default",
        worker_dataset_split="test",
        worker_dataset_revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        worker_model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        worker_model_revision="6a6f4aa4197940add57724a7707d069478df56b1",
        worker_tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        worker_tokenizer_revision="6a6f4aa4197940add57724a7707d069478df56b1",
        expected_prompt_token_ids_sha256=PROMPT_HASH,
        observed_prompt_token_ids_sha256=PROMPT_HASH,
        prompt_token_count=1995,  # + 5 generated = 2000 > BUDGET(1024)
        natural_generated_token_ids=generated_ids,
        generated_token_count=len(generated_ids),
        generated_token_ids_sha256=sha256_int_ids(generated_ids),
        cap_hit=False,
        extracted_answer="42",
        answer_verification_status="correct",
        think_parse_status="ok",
        think_start_index=0,
        think_end_index=3,
        generation_prompt_preopened_think=False,
        fullkv_wall_seconds=12.5,
        fullkv_timing_evidence=_valid_timing(),
        requested_device="cuda:0",
        parameter_placement_evidence=_valid_placement(),
        actual_batch_size=1,
        peak_cuda_allocated_bytes=1000,
        peak_cuda_reserved_bytes=2000,
        predicted_compaction_event_positions=list(range(6)),
        predicted_event_count=6,
        eligible_event_indices=[1, 2, 3],
        eligible_event_count=3,
        generation_config_sha256=GENERATION_CONFIG_SHA256,
        runtime_prediction=predict_runtime(2000).to_json(),
        candidate_manifest_canonical_sha256=manifest["canonical_sha256"],
        config_sha256=CONFIG_SHA,
    )
    fields.update(overrides)
    return B2AR3FullKVQualificationEvidence.model_validate(fields)


def _evaluate(**overrides):
    evidence = _valid_evidence(**overrides)
    return evaluate_b2a_r3_qualification_conditions(
        evidence, candidate_manifest=_candidate_manifest(), expected_config_sha256=CONFIG_SHA
    )


def test_all_conditions_pass_for_genuinely_valid_evidence():
    conditions = _evaluate()
    assert set(conditions) == set(B2A_R3_QUALIFICATION_CONDITIONS)
    failed = [name for name, ok in conditions.items() if not ok]
    assert failed == [], f"unexpected failures: {failed}"
    assert all(isinstance(v, bool) for v in conditions.values())


@pytest.mark.parametrize(
    "status,expected_no_cap,expected_verifiable,expected_correct",
    [
        ("correct", True, True, True),
        ("incorrect", True, True, False),
        ("unverifiable", True, False, False),
    ],
)
def test_every_answer_status(status, expected_no_cap, expected_verifiable, expected_correct):
    conditions = _evaluate(answer_verification_status=status)
    assert conditions["answer_verifiable"] == expected_verifiable
    assert conditions["fullkv_answer_correct"] == expected_correct


def test_cap_hit_fails_no_cap_hit_and_trace_complete():
    conditions = _evaluate(cap_hit=True)
    assert conditions["no_cap_hit"] is False
    assert conditions["trace_complete"] is False


@pytest.mark.parametrize(
    "status", ["no_open_marker", "no_close_marker"],
)
def test_think_parse_failure_statuses_fail_thinking_span_valid(status):
    conditions = _evaluate(think_parse_status=status, think_start_index=None, think_end_index=None)
    assert conditions["thinking_span_valid"] is False
    assert conditions["trace_complete"] is False


def test_generation_prompt_preopened_ok_is_a_success_status():
    conditions = _evaluate(think_parse_status="generation_prompt_preopened_ok", think_start_index=0)
    assert conditions["thinking_span_valid"] is True


def test_think_end_index_exceeding_generated_count_fails():
    conditions = _evaluate(think_end_index=999)
    assert conditions["thinking_span_valid"] is False


def test_think_end_before_start_fails():
    conditions = _evaluate(think_start_index=3, think_end_index=1)
    assert conditions["thinking_span_valid"] is False


def test_prompt_hash_mismatch_fails_prompt_identity_match():
    conditions = _evaluate(observed_prompt_token_ids_sha256="e" * 64)
    assert conditions["prompt_identity_match"] is False


def test_dataset_mismatch_fails():
    conditions = _evaluate(worker_dataset_revision="0" * 40)
    assert conditions["dataset_identity_match"] is False


def test_model_mismatch_fails():
    conditions = _evaluate(worker_model_revision="0" * 40)
    assert conditions["model_identity_match"] is False


def test_tokenizer_mismatch_fails():
    conditions = _evaluate(worker_tokenizer_revision="0" * 40)
    assert conditions["tokenizer_identity_match"] is False


def test_runtime_generation_hash_mismatch_fails():
    conditions = _evaluate(generation_config_sha256="0" * 64)
    assert conditions["generation_config_hash_match"] is False


def test_batch_size_mismatch_fails():
    conditions = _evaluate(actual_batch_size=2)
    assert conditions["batch_size_is_one"] is False


def test_placement_mismatch_fails():
    bad_placement = dict(_valid_placement(), every_parameter_on_cuda=False)
    conditions = _evaluate(parameter_placement_evidence=bad_placement)
    assert conditions["all_parameters_on_requested_cuda"] is False


def test_offload_evidence_fails():
    bad_placement = dict(_valid_placement(), hf_device_map={"layer.0": "cpu"})
    conditions = _evaluate(parameter_placement_evidence=bad_placement)
    assert conditions["no_offload_verified"] is False


def test_memory_boundary_exactly_at_limit_passes():
    from kvcot.discovery.b2a_r3_contract import QUALIFICATION_MEMORY_LIMIT_BYTES

    conditions = _evaluate(
        peak_cuda_allocated_bytes=QUALIFICATION_MEMORY_LIMIT_BYTES, peak_cuda_reserved_bytes=0,
    )
    assert conditions["peak_memory_within_limit"] is True


def test_memory_boundary_one_byte_over_fails():
    from kvcot.discovery.b2a_r3_contract import QUALIFICATION_MEMORY_LIMIT_BYTES

    conditions = _evaluate(
        peak_cuda_allocated_bytes=QUALIFICATION_MEMORY_LIMIT_BYTES + 1, peak_cuda_reserved_bytes=0,
    )
    assert conditions["peak_memory_within_limit"] is False


def test_sequence_length_1024_vs_1025():
    conditions_1024 = _evaluate(prompt_token_count=1024, generated_token_count=0, natural_generated_token_ids=[])
    conditions_1025 = _evaluate(prompt_token_count=1025, generated_token_count=0, natural_generated_token_ids=[])
    assert conditions_1024["sequence_exceeds_budget"] is False
    assert conditions_1025["sequence_exceeds_budget"] is True


def test_zero_vs_present_predicted_compaction():
    conditions_zero = _evaluate(predicted_compaction_event_positions=[], predicted_event_count=0)
    conditions_present = _evaluate(predicted_compaction_event_positions=[10], predicted_event_count=6)
    assert conditions_zero["predicted_compaction_present"] is False
    assert conditions_present["predicted_compaction_present"] is True


def test_five_vs_six_predicted_events():
    conditions_five = _evaluate(predicted_compaction_event_positions=list(range(5)), predicted_event_count=5)
    conditions_six = _evaluate(predicted_compaction_event_positions=list(range(6)), predicted_event_count=6)
    assert conditions_five["predicted_event_count_at_least_six"] is False
    assert conditions_six["predicted_event_count_at_least_six"] is True


def test_two_vs_three_eligible_events():
    conditions_two = _evaluate(eligible_event_indices=[1, 2], eligible_event_count=2)
    conditions_three = _evaluate(eligible_event_indices=[1, 2, 3], eligible_event_count=3)
    assert conditions_two["at_least_three_events_have_49_future_tokens"] is False
    assert conditions_three["at_least_three_events_have_49_future_tokens"] is True


def test_runtime_boundary_via_predictor():
    passing = predict_runtime(2775).to_json()
    failing = predict_runtime(2776).to_json()
    conditions_pass = _evaluate(runtime_prediction=passing)
    conditions_fail = _evaluate(runtime_prediction=failing)
    assert conditions_pass["projected_runtime_within_qualification_target"] is True
    assert conditions_fail["projected_runtime_within_qualification_target"] is False


def test_safety_multiplier_exact_rejects_non_frozen_value():
    tampered = predict_runtime(2000).to_json()
    tampered["safety_multiplier"] = 1.0
    conditions = _evaluate(runtime_prediction=tampered)
    assert conditions["safety_multiplier_exact"] is False
    # tampering breaks the whole recomputation too
    assert conditions["runtime_inputs_complete"] is False


def test_candidate_manifest_hash_mismatch_fails():
    conditions = _evaluate(candidate_manifest_canonical_sha256="0" * 64)
    assert conditions["candidate_manifest_hash_match"] is False


def test_config_hash_mismatch_fails():
    conditions = _evaluate(config_sha256="0" * 64)
    assert conditions["config_hash_match"] is False


def test_exact_27_condition_tuple_enforced():
    conditions = _evaluate()
    assert len(conditions) == 27
    assert tuple(sorted(conditions)) == tuple(sorted(B2A_R3_QUALIFICATION_CONDITIONS))


def test_evidence_rejects_unknown_field():
    with pytest.raises(Exception):
        _valid_evidence(unknown_field="x")


def test_evidence_rejects_bad_answer_status():
    with pytest.raises(Exception):
        _valid_evidence(answer_verification_status="maybe")


def test_generated_count_mismatching_array_length_fails_condition():
    # Evidence construction itself must still succeed (a genuinely
    # malformed worker observation must remain representable so it can be
    # attempted and recorded as a FAILED candidate, never crash the run) --
    # but the derived condition must be False.
    conditions = _evaluate(generated_token_count=999)
    assert conditions["generated_token_count_present"] is False


# --------------------------------------------------------------------- outcome


def _valid_outcome_dict(**overrides):
    evidence = _valid_evidence(**overrides)
    return build_qualification_outcome(
        evidence, candidate_manifest=_candidate_manifest(), expected_config_sha256=CONFIG_SHA
    )


def test_build_qualification_outcome_round_trips_through_strict_schema():
    outcome = _valid_outcome_dict()
    typed = CandidateQualificationOutcomeR3.model_validate(outcome)
    assert typed.qualified is True
    assert typed.failed_conditions == []


def test_qualified_candidate_ordinal_zero_through_seven_accepted():
    for ordinal in range(QUALIFICATION_CANDIDATE_LIMIT):
        outcome = _valid_outcome_dict(candidate_ordinal=ordinal)
        assert outcome["candidate_ordinal"] == ordinal


def test_outcome_rejects_fabricated_qualified_true():
    outcome = _valid_outcome_dict(answer_verification_status="incorrect")
    assert outcome["qualified"] is False
    tampered = dict(outcome)
    tampered["qualified"] = True
    with pytest.raises(Exception):
        CandidateQualificationOutcomeR3.model_validate(tampered)


def test_outcome_rejects_wrong_failed_conditions_order():
    outcome = _valid_outcome_dict(answer_verification_status="incorrect", cap_hit=True)
    tampered = dict(outcome)
    tampered["failed_conditions"] = list(reversed(tampered["failed_conditions"]))
    with pytest.raises(Exception):
        CandidateQualificationOutcomeR3.model_validate(tampered)


def test_outcome_rejects_missing_condition_name():
    outcome = _valid_outcome_dict()
    tampered = dict(outcome)
    tampered["conditions"] = {k: v for k, v in tampered["conditions"].items() if k != "no_cap_hit"}
    with pytest.raises(Exception):
        CandidateQualificationOutcomeR3.model_validate(tampered)


def test_outcome_rejects_extra_condition_name():
    outcome = _valid_outcome_dict()
    tampered = dict(outcome)
    tampered["conditions"] = dict(tampered["conditions"], not_a_real_condition=True)
    with pytest.raises(Exception):
        CandidateQualificationOutcomeR3.model_validate(tampered)


def test_outcome_rejects_non_boolean_condition():
    outcome = _valid_outcome_dict()
    tampered = dict(outcome)
    tampered["conditions"] = dict(tampered["conditions"], no_cap_hit=1)
    with pytest.raises(Exception):
        CandidateQualificationOutcomeR3.model_validate(tampered)


# --------------------------------------------------------------------- selection


def test_first_pass_stops_selection():
    attempted = [
        _valid_outcome_dict(candidate_ordinal=0, answer_verification_status="incorrect"),
        _valid_outcome_dict(candidate_ordinal=1),  # first pass
    ]
    selected = select_first_qualified_r3(attempted)
    assert selected["candidate_ordinal"] == 1


def test_no_candidate_qualifies_returns_none():
    attempted = [_valid_outcome_dict(candidate_ordinal=i, answer_verification_status="incorrect") for i in range(3)]
    assert select_first_qualified_r3(attempted) is None


def test_candidate_nine_rejected_by_ordinal_range():
    attempted = [_valid_outcome_dict(candidate_ordinal=i, answer_verification_status="incorrect") for i in range(8)]
    attempted.append(_valid_outcome_dict(candidate_ordinal=8))
    with pytest.raises(QualificationRefused):
        select_first_qualified_r3(attempted)


def test_evidence_after_pass_rejected():
    attempted = [
        _valid_outcome_dict(candidate_ordinal=0),  # passes immediately
        _valid_outcome_dict(candidate_ordinal=1, answer_verification_status="incorrect"),
    ]
    with pytest.raises(QualificationRefused):
        select_first_qualified_r3(attempted)


def test_duplicate_ordinal_rejected():
    attempted = [
        _valid_outcome_dict(candidate_ordinal=0, answer_verification_status="incorrect"),
        _valid_outcome_dict(candidate_ordinal=0, answer_verification_status="incorrect"),
    ]
    with pytest.raises(QualificationRefused):
        select_first_qualified_r3(attempted)


def test_non_contiguous_ordinals_rejected():
    attempted = [
        _valid_outcome_dict(candidate_ordinal=0, answer_verification_status="incorrect"),
        _valid_outcome_dict(candidate_ordinal=2, answer_verification_status="incorrect"),
    ]
    with pytest.raises(QualificationRefused):
        select_first_qualified_r3(attempted)


def test_never_ranks_by_best_score_first_pass_wins_even_if_later_also_qualifies():
    attempted = [
        _valid_outcome_dict(candidate_ordinal=0),  # qualifies
        _valid_outcome_dict(candidate_ordinal=1),  # would also qualify, never reached
    ]
    with pytest.raises(QualificationRefused):
        # Evidence for ordinal=1 must never have been attempted after ordinal=0
        # already passed -- this is a structural rejection, not a ranking
        # decision.
        select_first_qualified_r3(attempted)
