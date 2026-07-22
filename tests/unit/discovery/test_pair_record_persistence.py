"""B2A-R2 forensic pair-record persistence repair tests
(`docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md`).

Covers: RKVWorkerResultV1/V2 serialization and structural version dispatch,
the pure-Python scientific summary (including tie-aware Spearman), the
coordinator's durable `rkv/pair_records.json`/`rkv/scientific_summary.json`
artifacts, the dedicated `verify_pair_record_artifacts` verifier, and the
partial-failure/historical paths. Never touches a real model, GPU, or
subprocess -- every worker body is faked exactly like
`tests/unit/discovery/test_b2a_workers.py`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kvcot.discovery.attempt_artifacts import create_attempt_directory, semantic_role_for
from kvcot.discovery.attempt_verification import verify_pair_record_artifacts
from kvcot.discovery.b2a_workers import (
    RKVWorkerResult,
    RKVWorkerResultV1,
    RKVWorkerResultV2,
    classify_pair_record_availability,
    parse_rkv_worker_result,
    run_both_workers_via_subprocess,
)
from kvcot.discovery.constants import B2A_NOOP_PAIR_EVALUATIONS_TOTAL, B2A_REAL_PAIR_EVALUATIONS_TOTAL
from kvcot.discovery.scientific_summary import build_scientific_summary, tie_aware_spearman
from kvcot.discovery.schemas import SwapPairRecord
from kvcot.utils.hashing import sha256_json

VALID_SHA = "a" * 64
MANIFEST_HASH = "m" * 64
PROMPT_HASH = "p" * 64

_DETERMINISM_POLICY = dict(
    framework_seed=13, python_random_seeded=True, torch_cpu_seeded=True, torch_cuda_seeded=True,
    cudnn_deterministic_requested=True, attention_backend="flash_attention_2",
    bitwise_determinism_guaranteed=False, tolerance_note="note",
)
_RUNTIME_GENERATION = dict(
    generation_mode="greedy", do_sample=False, temperature=None, top_p=None, batch_size=1, max_new_tokens=48,
    eos_token_id=99, eos_append_feed_policy="p", one_prefill_policy="p", single_token_decode_policy="p",
    attention_backend="flash_attention_2", cache_implementation="DynamicCache", framework_seed=13,
    prompt_token_count=200,
)
_PARAMETER_PLACEMENT = dict(
    unique_device_types=["cuda"], every_parameter_on_cuda=True, hf_device_map=None, no_offload_verified=True,
    parameter_count=100,
)
_RUNTIME_IDENTITY = dict(
    requested_model_revision="modelrev", resolved_model_revision="modelrev", model_revision_match=True,
    requested_tokenizer_revision="tokrev", resolved_tokenizer_revision="tokrev", tokenizer_revision_match=True,
)
_MEMORY = dict(
    allocated_before_reset_bytes=100, reserved_before_reset_bytes=200, peak_allocated_bytes=1000,
    peak_reserved_bytes=2000, reset_point="after_model_and_tokenizer_load_before_measured_inference",
)


def _fullkv_payload(**overrides) -> dict:
    payload = dict(
        role="fullkv", model_revision="modelrev", tokenizer_revision="tokrev",
        dataset_repo="HuggingFaceH4/MATH-500", dataset_revision="d" * 40,
        manifest_hash=MANIFEST_HASH, prompt_token_ids_sha256=PROMPT_HASH, prompt_token_count=200,
        natural_generated_token_ids=[1, 2, 3], natural_answer="42", natural_answer_status="correct",
        cap_hit=False, prefill_call_count=1, decode_call_count=3, call_boundary_trace_hash="t" * 64,
        wall_seconds=1.5,
        determinism_policy=_DETERMINISM_POLICY, runtime_generation=_RUNTIME_GENERATION,
        runtime_generation_config_hash="g" * 64, parameter_placement=_PARAMETER_PLACEMENT,
        runtime_identity=_RUNTIME_IDENTITY, memory=_MEMORY,
        peak_cuda_allocated_bytes=1000, peak_cuda_reserved_bytes=2000,
        every_parameter_on_cuda=True, batch_size=1, software_versions={"torch": "2.0"},
    )
    payload.update(overrides)
    return payload


def _rkv_payload(**overrides) -> dict:
    payload = dict(
        role="rkv", model_revision="modelrev", tokenizer_revision="tokrev",
        dataset_repo="HuggingFaceH4/MATH-500", dataset_revision="d" * 40,
        manifest_hash=MANIFEST_HASH, prompt_token_ids_sha256=PROMPT_HASH, prompt_token_count=200,
        rkv_upstream_revision="r" * 40, runtime_rkv_config_hash="h" * 64, frozen_rkv_config_hash="h" * 64,
        rkv_config_hash_match=True,
        example_valid=True, natural_answer_status="correct",
        token_identical_replay=True, prefill_decode_boundary_parity=True, compaction_position_equality=True,
        capture_gather_parity=True, absolute_position_parity=True, no_op_numerical_parity=True,
        pass1_call_boundary={"prefill_call_count": 1, "prefill_token_count": 200, "decode_call_count": 300, "ordered_trace_hash": "a" * 64},
        pass2_call_boundary={"prefill_call_count": 1, "prefill_token_count": 200, "decode_call_count": 300, "ordered_trace_hash": "a" * 64},
        observed_total_compaction_events=5, eligible_compaction_events=3, selected_compaction_events=3,
        events_with_at_least_one_completed_real_pair=3,
        events_with_all_four_real_pairs_completed=3, attempted_real_pair_count=12, completed_real_pair_count=12,
        failed_real_pair_count=0, attempted_no_op_pair_count=1, completed_no_op_pair_count=1,
        pair_failure_details=[],
        semantic_swap_checks_required=12, semantic_swap_checks_attempted=12, semantic_swap_checks_passed=12,
        semantic_swap_checks_failed=0,
        unique_completed_real_pair_count=12, events_with_exactly_four_unique_real_pairs=3,
        has_duplicate_real_pair_identity=False, has_duplicate_no_op_pair_identity=False,
        selected_event_count_exact=True, real_pair_count_exact=True, no_op_count_exact=True,
        all_required_pair_evaluations_completed=True,
        observed_retention_ratio=0.5,
        wall_seconds_pass1=1.0, wall_seconds_pass2=1.0, wall_seconds_targeted_capture=0.1,
        real_pair_wall_seconds=[1.0] * 12, no_op_pair_wall_seconds=[0.5],
        determinism_policy=_DETERMINISM_POLICY, runtime_generation=_RUNTIME_GENERATION,
        runtime_generation_config_hash="g" * 64, parameter_placement=_PARAMETER_PLACEMENT,
        runtime_identity=_RUNTIME_IDENTITY, memory=_MEMORY, minimized_target_evidence=[],
        peak_cuda_allocated_bytes=1000, peak_cuda_reserved_bytes=2000, every_parameter_on_cuda=True,
        batch_size=1, software_versions={"torch": "2.0"},
        pair_records=[],
    )
    payload.update(overrides)
    return payload


def _make_fake_runner(fullkv_payload: dict | None, rkv_payload: dict | None, *, fail_role: str | None = None):
    """Writes both `result.json` AND its mandatory atomic
    `result.json.envelope.json` sibling
    (`kvcot.discovery.worker_envelope.write_worker_envelope`'s exact naming
    convention) -- a successful worker result is invalid without one
    (`kvcot.discovery.b2a_workers._validate_atomic_worker_envelope`)."""

    def runner(argv, **kwargs):
        role = argv[argv.index("--role") + 1]
        output_path = Path(argv[argv.index("--output") + 1])
        if role == fail_role:
            return SimpleNamespace(returncode=1, stdout="", stderr="simulated worker failure")
        payload = fullkv_payload if role == "fullkv" else rkv_payload

        # The coordinator recomputes the envelope hash from the TYPED,
        # round-tripped model it reads back
        # (`FullKVWorkerResult`/`RKVWorkerResult.model_validate_json(...)
        # .model_dump(mode="json")`) -- the envelope here must be built
        # from that SAME canonical form, never the raw payload dict, or
        # pydantic-filled defaults make the hashes disagree.
        from kvcot.discovery.b2a_workers import FullKVWorkerResult, RKVWorkerResult

        model = FullKVWorkerResult if role == "fullkv" else RKVWorkerResult
        canonical_payload = model.model_validate(payload).model_dump(mode="json")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(canonical_payload), encoding="utf-8")

        from kvcot.discovery.worker_envelope import build_success_envelope, write_worker_envelope

        envelope = build_success_envelope(
            role=role, attempt_id="pair-record-test", started_at="2026-01-01T00:00:00+00:00",
            requested_identities={}, resolved_identities={}, result_payload=canonical_payload,
            determinism_policy=None, software_versions={"torch": "0.0.0-stub"}, hardware_metadata={},
        )
        write_worker_envelope(envelope, output_path)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return runner


def _record(
    *, event: int = 0, layer: int = 0, candidate: int = 10, donor: int = 20,
    is_noop: bool = False, swap_gain: float | None = None, score_margin: float = -0.2,
    valid: bool = True,
) -> SwapPairRecord:
    t = 100 + event
    baseline = [1.0] * 48
    if is_noop:
        swapped = list(baseline)
        gain = 0.0
    else:
        swapped = [1.0 - (0.1 if swap_gain is None else swap_gain)] * 48
        gain = (sum(baseline) / 48) - (sum(swapped) / 48)
    return SwapPairRecord(
        example_id="ex-1", model_revision="modelrev", rkv_revision="r" * 40,
        compaction_event_id=event, chronological_event_ordinal=event % 3, depth_stratum=event % 3,
        layer_index=layer, kv_head_index=0,
        event_token_absolute_position=t, bridge_token_absolute_position=t + 1,
        first_affected_forward_input_absolute_position=t + 1,
        first_affected_logit_target_absolute_position=t + 2, first_scored_absolute_position=t + 2,
        evicted_absolute_token_position=candidate, evicted_pre_storage_position=5,
        retained_absolute_token_position=donor, retained_pre_storage_position=8,
        retained_post_storage_position=8,
        score_e=0.4, score_r=0.4 - score_margin, score_margin_e_minus_r=score_margin,
        attention_component_diff=0.01, similarity_component_diff=-0.02, recency_diff=candidate - donor,
        key_norm_diff=0.1, value_norm_diff=-0.1,
        entropy_e=1.2, entropy_e_missing_reason=None, entropy_r=0.4, entropy_r_missing_reason=None,
        entropy_diff=0.8,
        logit_margin_e=3.0, logit_margin_e_missing_reason=None, logit_margin_r=5.5,
        logit_margin_r_missing_reason=None, logit_margin_diff=-2.5,
        parity_check_passed=True, parity_failure_reason=None,
        is_noop_control=is_noop, net_physical_bytes_changed=0, cap_hit_flag=False,
        valid_flag=valid, invalid_reason=None if valid else "capture_failed",
        reference_horizon_sha256=VALID_SHA,
        swap_gain=gain, baseline_per_token_nll=baseline, swapped_per_token_nll=swapped,
    )


def _full_population() -> list[SwapPairRecord]:
    real = [
        _record(event=event, layer=event, candidate=candidate, donor=donor, score_margin=-0.1 * (candidate - donor))
        for event in range(3) for candidate in (10, 11) for donor in (20, 21)
    ]
    no_op = _record(event=0, layer=0, candidate=30, donor=30, is_noop=True)
    return real + [no_op]


# --------------------------------------------------------------------------
# Serialization: RKVWorkerResultV1/V2 structural versioning
# --------------------------------------------------------------------------


def test_pair_record_survives_worker_result_json_round_trip():
    records = _full_population()
    payload = _rkv_payload(pair_records=[r.model_dump(mode="json") for r in records])
    result = RKVWorkerResultV2.model_validate(payload)
    round_tripped = RKVWorkerResultV2.model_validate_json(result.model_dump_json())
    assert len(round_tripped.pair_records) == 13
    real = [r for r in round_tripped.pair_records if not r.is_noop_control][0]
    assert real.baseline_per_token_nll == records[0].baseline_per_token_nll
    assert real.swapped_per_token_nll == records[0].swapped_per_token_nll
    assert len(real.baseline_per_token_nll) == 48
    assert len(real.swapped_per_token_nll) == 48
    assert real.swap_gain == records[0].swap_gain
    assert real.score_margin_e_minus_r == records[0].score_margin_e_minus_r
    assert real.compaction_event_id == records[0].compaction_event_id
    assert real.evicted_absolute_token_position == records[0].evicted_absolute_token_position
    assert real.retained_absolute_token_position == records[0].retained_absolute_token_position


def test_v2_rejects_missing_pair_records():
    payload = _rkv_payload()
    del payload["pair_records"]
    with pytest.raises(ValidationError):
        RKVWorkerResultV2.model_validate(payload)


def test_rkv_worker_result_alias_is_v2():
    assert RKVWorkerResult is RKVWorkerResultV2


def test_v1_legacy_result_remains_parseable_without_fabricated_records():
    legacy_payload = _rkv_payload()
    del legacy_payload["pair_records"]
    parsed = parse_rkv_worker_result(legacy_payload)
    assert isinstance(parsed, RKVWorkerResultV1)
    assert not isinstance(parsed, RKVWorkerResultV2)
    assert not hasattr(parsed, "pair_records")


def test_parse_dispatches_to_v2_when_pair_records_key_present():
    payload = _rkv_payload(pair_records=[r.model_dump(mode="json") for r in _full_population()])
    parsed = parse_rkv_worker_result(payload)
    assert isinstance(parsed, RKVWorkerResultV2)
    assert len(parsed.pair_records) == 13


def test_classify_pair_record_availability_legacy_vs_v2():
    legacy_payload = _rkv_payload()
    del legacy_payload["pair_records"]
    legacy = parse_rkv_worker_result(legacy_payload)
    legacy_availability = classify_pair_record_availability(legacy)
    assert legacy_availability.scientific_pair_records_available is False
    assert legacy_availability.scientific_pair_artifacts_verified is False
    assert legacy_availability.legacy_pair_record_schema is True

    v2_payload = _rkv_payload(pair_records=[r.model_dump(mode="json") for r in _full_population()])
    v2 = parse_rkv_worker_result(v2_payload)
    v2_availability = classify_pair_record_availability(v2, artifacts_verified=True)
    assert v2_availability.scientific_pair_records_available is True
    assert v2_availability.scientific_pair_artifacts_verified is True
    assert v2_availability.legacy_pair_record_schema is False


# --------------------------------------------------------------------------
# Scientific summary / tie-aware Spearman
# --------------------------------------------------------------------------


def test_spearman_perfect_positive_correlation():
    assert tie_aware_spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]) == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    assert tie_aware_spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0]) == pytest.approx(-1.0)


def test_spearman_tie_aware_average_ranks():
    # x has a tie at rank (2+3)/2=2.5 for the two 5.0 values.
    x = [5.0, 5.0, 1.0, 9.0]
    y = [1.0, 2.0, 3.0, 4.0]
    result = tie_aware_spearman(x, y)
    assert result is not None
    assert -1.0 <= result <= 1.0


def test_spearman_none_below_two_pairs():
    assert tie_aware_spearman([1.0], [2.0]) is None
    assert tie_aware_spearman([], []) is None


def test_spearman_none_for_zero_variance():
    assert tie_aware_spearman([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]) is None
    assert tie_aware_spearman([1.0, 2.0, 3.0], [3.0, 3.0, 3.0]) is None


def test_spearman_none_for_non_finite_input():
    assert tie_aware_spearman([1.0, math.nan, 3.0], [1.0, 2.0, 3.0]) is None
    assert tie_aware_spearman([1.0, 2.0, 3.0], [1.0, math.inf, 3.0]) is None


def test_spearman_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        tie_aware_spearman([1.0, 2.0], [1.0])


def test_build_scientific_summary_on_full_population():
    records = _full_population()
    summary = build_scientific_summary(records)
    assert summary["real_pair_count"] == 12
    assert summary["no_op_pair_count"] == 1
    assert summary["pair_records_sha256"] == sha256_json([r.model_dump(mode="json") for r in records])
    real_gains = [r.swap_gain for r in records if not r.is_noop_control]
    assert summary["positive_gain_count"] == sum(1 for g in real_gains if g > 0.0)
    assert summary["gain_above_0_01_count"] == sum(1 for g in real_gains if g > 0.01)
    assert summary["mean_swap_gain"] == pytest.approx(sum(real_gains) / len(real_gains))
    assert summary["minimum_swap_gain"] == pytest.approx(min(real_gains))
    assert summary["maximum_swap_gain"] == pytest.approx(max(real_gains))


def test_build_scientific_summary_on_empty_population_never_fabricates_zero():
    summary = build_scientific_summary([])
    assert summary["real_pair_count"] == 0
    assert summary["no_op_pair_count"] == 0
    assert summary["positive_gain_count"] == 0
    assert summary["median_swap_gain"] is None
    assert summary["mean_swap_gain"] is None
    assert summary["minimum_swap_gain"] is None
    assert summary["maximum_swap_gain"] is None
    assert summary["spearman_score_margin_vs_swap_gain"] is None


def test_build_scientific_summary_excludes_invalid_records_from_statistics():
    valid = _record(event=0, candidate=10, donor=20, valid=True)
    invalid = _record(event=1, candidate=11, donor=21, valid=False)
    summary = build_scientific_summary([valid, invalid])
    assert summary["real_pair_count"] == 2  # execution count includes the invalid attempt
    assert summary["mean_swap_gain"] == pytest.approx(valid.swap_gain)  # statistics exclude it


# --------------------------------------------------------------------------
# Coordinator persistence -- real coordinator, faked subprocess runner
# --------------------------------------------------------------------------


def test_coordinator_writes_pair_records_and_scientific_summary(tmp_path):
    records = _full_population()
    rkv_payload = _rkv_payload(pair_records=[r.model_dump(mode="json") for r in records])
    runner = _make_fake_runner(_fullkv_payload(), rkv_payload)
    attempt = create_attempt_directory(root=tmp_path, attempt_id="pair-record-test")

    run_both_workers_via_subprocess(
        "cfg.yaml", "manifest.json", python_executable="fake-python", subprocess_runner=runner,
        attempt_directory=attempt.path,
    )

    pair_records_path = attempt.path / "rkv" / "pair_records.json"
    summary_path = attempt.path / "rkv" / "scientific_summary.json"
    assert pair_records_path.is_file()
    assert summary_path.is_file()

    on_disk_records = json.loads(pair_records_path.read_text(encoding="utf-8"))
    assert len(on_disk_records) == B2A_REAL_PAIR_EVALUATIONS_TOTAL + B2A_NOOP_PAIR_EVALUATIONS_TOTAL
    assert sum(1 for r in on_disk_records if not r["is_noop_control"]) == B2A_REAL_PAIR_EVALUATIONS_TOTAL
    assert sum(1 for r in on_disk_records if r["is_noop_control"]) == B2A_NOOP_PAIR_EVALUATIONS_TOTAL

    rkv_result = json.loads((attempt.path / "rkv" / "result.json").read_text(encoding="utf-8"))
    assert on_disk_records == rkv_result["pair_records"]

    on_disk_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk_summary == build_scientific_summary([SwapPairRecord.model_validate(r) for r in on_disk_records])


def test_pair_records_and_scientific_summary_have_known_semantic_roles():
    assert semantic_role_for("rkv/pair_records.json") == "pair_records"
    assert semantic_role_for("rkv/scientific_summary.json") == "scientific_summary"


# --------------------------------------------------------------------------
# verify_pair_record_artifacts -- dedicated verification failures
# --------------------------------------------------------------------------


def _write_attempt_with_pair_records(tmp_path, records: list[SwapPairRecord]) -> tuple[Path, dict]:
    attempt = create_attempt_directory(root=tmp_path, attempt_id="verify-test")
    pair_records_payload = [r.model_dump(mode="json") for r in records]
    (attempt.path / "rkv" / "pair_records.json").write_text(
        json.dumps(pair_records_payload), encoding="utf-8"
    )
    (attempt.path / "rkv" / "scientific_summary.json").write_text(
        json.dumps(build_scientific_summary(records)), encoding="utf-8"
    )
    rkv_result = {
        "pair_records": pair_records_payload,
        "completed_pair_identities": [
            {
                "compaction_event_id": r.compaction_event_id, "layer_index": r.layer_index,
                "kv_head_index": r.kv_head_index, "candidate_absolute_position": r.evicted_absolute_token_position,
                "donor_absolute_position": r.retained_absolute_token_position,
                "pair_kind": "no_op" if r.is_noop_control else "real",
            }
            for r in records
        ],
        "failed_pair_identities": [],
    }
    return attempt.path, rkv_result


def test_verify_pair_record_artifacts_accepts_a_genuinely_consistent_attempt(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is True, reasons
    assert reasons == ()


def test_verify_pair_record_artifacts_legacy_result_short_circuits_to_true(tmp_path):
    attempt = create_attempt_directory(root=tmp_path, attempt_id="legacy-test")
    verified, reasons = verify_pair_record_artifacts(attempt.path, rkv_result={"role": "rkv"})
    assert verified is True
    assert reasons == ()


def test_verify_pair_record_artifacts_fails_on_missing_pair_records_file(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    (attempt_path / "rkv" / "pair_records.json").unlink()
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("pair_records.json is missing" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_missing_summary_file(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    (attempt_path / "rkv" / "scientific_summary.json").unlink()
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("scientific_summary.json is missing" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_one_missing_real_record(tmp_path):
    records = _full_population()[:-2] + _full_population()[-1:]  # drop one real record, keep the no-op
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("11 real records, expected 12" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_extra_record(tmp_path):
    records = _full_population() + [_record(event=2, candidate=99, donor=98)]
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("13 real records, expected 12" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_duplicate_identity(tmp_path):
    records = _full_population()
    duplicate = records[0].model_copy()
    records = records[:-1] + [duplicate]  # replace the no-op with a duplicate real record
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("duplicate pair identities" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_identity_mismatch_with_completed_pair_identities(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    rkv_result["completed_pair_identities"][0]["compaction_event_id"] = 999
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("do not exactly match" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_failed_identity_represented_as_completed(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    real = records[0]
    rkv_result["failed_pair_identities"] = [{
        "compaction_event_id": real.compaction_event_id, "layer_index": real.layer_index,
        "kv_head_index": real.kv_head_index, "candidate_absolute_position": real.evicted_absolute_token_position,
        "donor_absolute_position": real.retained_absolute_token_position, "pair_kind": "real",
    }]
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("represents a failed pair identity as completed" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_changed_swap_gain(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    tampered = json.loads((attempt_path / "rkv" / "pair_records.json").read_text(encoding="utf-8"))
    tampered[0]["swap_gain"] = 999.0
    (attempt_path / "rkv" / "pair_records.json").write_text(json.dumps(tampered), encoding="utf-8")
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    # Either the worker-result/dedicated-file equality check or SwapPairRecord's
    # own swap_gain-consistency validator catches this -- both are acceptable,
    # never a silent pass.
    assert any(
        "does not match rkv/result.json's pair_records" in r or "does not validate as a typed SwapPairRecord" in r
        for r in reasons
    )


def test_verify_pair_record_artifacts_fails_on_changed_baseline_nll(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    rkv_result["pair_records"][0]["baseline_per_token_nll"] = [5.0] * 48  # only mutate rkv_result, not the file
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("does not match rkv/result.json's pair_records" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_changed_summary_statistic(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    summary = json.loads((attempt_path / "rkv" / "scientific_summary.json").read_text(encoding="utf-8"))
    summary["mean_swap_gain"] = -999.0
    (attempt_path / "rkv" / "scientific_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("does not recompute exactly" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_changed_pair_records_hash(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    summary = json.loads((attempt_path / "rkv" / "scientific_summary.json").read_text(encoding="utf-8"))
    summary["pair_records_sha256"] = "0" * 64
    (attempt_path / "rkv" / "scientific_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("does not recompute exactly" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_worker_result_dedicated_file_mismatch(tmp_path):
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    rkv_result["pair_records"] = [r.model_dump(mode="json") for r in records[:-1]]  # drop the last record
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("does not match rkv/result.json's pair_records" in r for r in reasons)


def test_verify_pair_record_artifacts_fails_on_no_op_inequality():
    """A no-op record whose baseline/swapped NLL differ is rejected by
    SwapPairRecord's own validator -- never reaches the persistence layer at
    all, proving the invariant is enforced at construction time."""
    with pytest.raises(ValidationError):
        SwapPairRecord(
            example_id="ex-1", model_revision="modelrev", rkv_revision="r" * 40,
            compaction_event_id=0, chronological_event_ordinal=0, depth_stratum=0,
            layer_index=0, kv_head_index=0,
            event_token_absolute_position=100, bridge_token_absolute_position=101,
            first_affected_forward_input_absolute_position=101,
            first_affected_logit_target_absolute_position=102, first_scored_absolute_position=102,
            evicted_absolute_token_position=30, evicted_pre_storage_position=5,
            retained_absolute_token_position=30, retained_pre_storage_position=8,
            retained_post_storage_position=8,
            score_e=0.4, score_r=0.6, score_margin_e_minus_r=-0.2,
            attention_component_diff=0.01, similarity_component_diff=-0.02, recency_diff=0,
            key_norm_diff=0.1, value_norm_diff=-0.1,
            parity_check_passed=True, parity_failure_reason=None,
            is_noop_control=True, net_physical_bytes_changed=0, cap_hit_flag=False,
            valid_flag=True, invalid_reason=None,
            reference_horizon_sha256=VALID_SHA,
            swap_gain=0.0, baseline_per_token_nll=[1.0] * 48, swapped_per_token_nll=[0.9] * 48,
        )


def test_verify_pair_record_artifacts_fails_on_non_finite_real_value():
    """A non-finite swap_gain on a record claiming valid_flag=True is
    rejected by SwapPairRecord's own finiteness validator."""
    with pytest.raises(ValidationError):
        SwapPairRecord(
            example_id="ex-1", model_revision="modelrev", rkv_revision="r" * 40,
            compaction_event_id=0, chronological_event_ordinal=0, depth_stratum=0,
            layer_index=0, kv_head_index=0,
            event_token_absolute_position=100, bridge_token_absolute_position=101,
            first_affected_forward_input_absolute_position=101,
            first_affected_logit_target_absolute_position=102, first_scored_absolute_position=102,
            evicted_absolute_token_position=10, evicted_pre_storage_position=5,
            retained_absolute_token_position=20, retained_pre_storage_position=8,
            retained_post_storage_position=8,
            score_e=0.4, score_r=0.6, score_margin_e_minus_r=-0.2,
            attention_component_diff=0.01, similarity_component_diff=-0.02, recency_diff=-10,
            key_norm_diff=0.1, value_norm_diff=-0.1,
            parity_check_passed=True, parity_failure_reason=None,
            is_noop_control=False, net_physical_bytes_changed=0, cap_hit_flag=False,
            valid_flag=True, invalid_reason=None,
            reference_horizon_sha256=VALID_SHA,
            swap_gain=math.nan, baseline_per_token_nll=[1.0] * 48, swapped_per_token_nll=[0.9] * 48,
        )


def test_verify_pair_record_artifacts_fails_on_reference_manifest_hash_mutation(tmp_path):
    """Byte-level tampering of the on-disk file (never touching rkv_result)
    is caught by the worker-result-vs-dedicated-file equality check."""
    records = _full_population()
    attempt_path, rkv_result = _write_attempt_with_pair_records(tmp_path, records)
    (attempt_path / "rkv" / "pair_records.json").write_text("[]", encoding="utf-8")
    verified, reasons = verify_pair_record_artifacts(attempt_path, rkv_result=rkv_result)
    assert verified is False
    assert any("does not match rkv/result.json's pair_records" in r for r in reasons)


# --------------------------------------------------------------------------
# Partial-failure and historical paths
# --------------------------------------------------------------------------


def test_partial_worker_evidence_preserves_exactly_completed_records():
    from types import SimpleNamespace

    from kvcot.discovery.worker_partial_evidence import capture_partial_evidence

    completed = [_record(event=0, candidate=10, donor=20), _record(event=0, candidate=11, donor=20)]
    example_result = SimpleNamespace(
        selected_event_evidence=(), attempted_pair_identities=(), completed_pair_identities=(),
        semantic_mutation_reports=(), minimized_target_evidence=(), pre_branch_memory_evidence=(),
        pair_failure_details=(), aborted=False, abort_failure_type=None, abort_failure_message=None,
        abort_is_oom=False, trace=None, pass2_replayed_token_ids=(), pair_records=tuple(completed),
    )
    evidence = capture_partial_evidence(
        role="rkv", failing_stage="real_pair:0:12:21", exc=RuntimeError("boom"),
        scope={"example_result": example_result},
    )
    assert len(evidence.pair_records) == 2
    assert evidence.pair_records[0]["evicted_absolute_token_position"] == 10
    assert evidence.pair_records[1]["evicted_absolute_token_position"] == 11


def test_partial_worker_evidence_never_pads_to_target_count():
    from types import SimpleNamespace

    from kvcot.discovery.worker_partial_evidence import capture_partial_evidence

    example_result = SimpleNamespace(
        selected_event_evidence=(), attempted_pair_identities=(), completed_pair_identities=(),
        semantic_mutation_reports=(), minimized_target_evidence=(), pre_branch_memory_evidence=(),
        pair_failure_details=(), aborted=False, abort_failure_type=None, abort_failure_message=None,
        abort_is_oom=False, trace=None, pass2_replayed_token_ids=(), pair_records=(_record(),),
    )
    evidence = capture_partial_evidence(
        role="rkv", failing_stage="real_pair:0:12:21", exc=RuntimeError("boom"),
        scope={"example_result": example_result},
    )
    assert len(evidence.pair_records) == 1  # never padded to 12 or 13


def test_partial_worker_evidence_preserves_zero_records_before_any_pair_evaluation():
    evidence_scope = {}  # no example_result bound at all -- failure before Pass 1/2 even started
    from kvcot.discovery.worker_partial_evidence import capture_partial_evidence

    evidence = capture_partial_evidence(
        role="rkv", failing_stage="model-load", exc=RuntimeError("boom"), scope=evidence_scope,
    )
    assert evidence.pair_records == []
