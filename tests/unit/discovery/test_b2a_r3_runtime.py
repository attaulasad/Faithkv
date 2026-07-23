"""B2A-R3 exact runtime predictor tests (protocol §7). No torch, no CUDA."""
from __future__ import annotations

import math

import pytest

from kvcot.discovery.b2a_r3_contract import QUALIFICATION_TARGET_HOURS, SAFETY_MULTIPLIER
from kvcot.discovery.b2a_r3_runtime import (
    RuntimePredictionRefused,
    continuous_token_ceiling,
    predict_runtime,
    verify_runtime_prediction,
)


def test_frozen_integer_boundary_2775_passes_2776_fails():
    p2775 = predict_runtime(2775)
    p2776 = predict_runtime(2776)
    assert p2775.projected_gpu_hours == pytest.approx(3.5999041059674, abs=1e-10)
    assert p2775.projected_runtime_within_qualification_target is True
    assert p2776.projected_gpu_hours == pytest.approx(3.6010221756819636, abs=1e-10)
    assert p2776.projected_runtime_within_qualification_target is False


def test_exact_equality_at_3_60_boundary_passes_above_fails():
    # Construct an evidence-level check: exactly at the gate value.
    p = predict_runtime(2775)
    assert p.projected_gpu_hours <= 3.60
    p_over = predict_runtime(2776)
    assert p_over.projected_gpu_hours > 3.60


def test_no_intermediate_rounding_full_precision_preserved():
    p = predict_runtime(1000)
    # Recompute independently, bit for bit, and require EXACT float equality
    # (not approx) -- proving no rounding was introduced anywhere.
    ref_sec_per_token = 1378.3004406290129 / 4931
    predicted_example_seconds = ref_sec_per_token * 1000 * 1.20
    predicted_pair_seconds = 10.247917714063078 * 1.20
    projected_total = 19.298151996218968 + 12 * predicted_example_seconds + 144 * predicted_pair_seconds
    assert p.reference_seconds_per_token == ref_sec_per_token
    assert p.predicted_example_seconds == predicted_example_seconds
    assert p.predicted_pair_seconds == predicted_pair_seconds
    assert p.projected_total_seconds == projected_total
    assert p.projected_gpu_hours == projected_total / 3600.0


def test_continuous_ceiling_is_derived_not_hardcoded():
    ceiling = continuous_token_ceiling()
    assert ceiling == pytest.approx(2775.0857674895859, abs=1e-6)
    # 2775 (floor of ceiling) passes, 2776 (above ceiling) fails -- the
    # derived ceiling and the integer boundary must agree.
    assert 2775 < ceiling < 2776


def test_missing_input_rejected():
    with pytest.raises(RuntimePredictionRefused):
        predict_runtime(None)


def test_negative_token_count_rejected():
    with pytest.raises(RuntimePredictionRefused):
        predict_runtime(-5)


def test_zero_token_count_rejected():
    with pytest.raises(RuntimePredictionRefused):
        predict_runtime(0)


def test_boolean_substituted_for_number_rejected():
    with pytest.raises(RuntimePredictionRefused):
        predict_runtime(True)


def test_non_finite_rejected_via_float_input():
    with pytest.raises(RuntimePredictionRefused):
        predict_runtime(float("nan"))
    with pytest.raises(RuntimePredictionRefused):
        predict_runtime(float("inf"))


def test_predictor_version_matches_frozen_contract():
    from kvcot.discovery.b2a_r3_contract import RUNTIME_PREDICTOR_VERSION

    p = predict_runtime(2775)
    assert p.runtime_predictor_version == RUNTIME_PREDICTOR_VERSION


def test_predictor_preserves_source_artifact_hash():
    from kvcot.discovery.b2a_r3_contract import RUNTIME_SOURCE_ARTIFACT_SHA256

    p = predict_runtime(2775)
    assert p.runtime_source_artifact_sha256 == RUNTIME_SOURCE_ARTIFACT_SHA256


# --------------------------------------------------------------------- verifier


def test_verify_runtime_prediction_accepts_genuine_evidence():
    p = predict_runtime(2775).to_json()
    verify_runtime_prediction(p)  # must not raise


def test_verify_runtime_prediction_rejects_missing_field():
    p = predict_runtime(2775).to_json()
    del p["predicted_pair_seconds"]
    with pytest.raises(RuntimePredictionRefused):
        verify_runtime_prediction(p)


def test_verify_runtime_prediction_rejects_altered_intermediate():
    p = predict_runtime(2775).to_json()
    p["predicted_example_seconds"] += 1.0
    with pytest.raises(RuntimePredictionRefused):
        verify_runtime_prediction(p)


def test_verify_runtime_prediction_rejects_modified_reference_constant():
    p = predict_runtime(2775).to_json()
    p["reference_pair_seconds"] = 5.0
    with pytest.raises(RuntimePredictionRefused):
        verify_runtime_prediction(p)


def test_verify_runtime_prediction_rejects_operator_supplied_multiplier_change():
    p = predict_runtime(2775).to_json()
    p["safety_multiplier"] = 1.0
    with pytest.raises(RuntimePredictionRefused):
        verify_runtime_prediction(p)


def test_verify_runtime_prediction_rejects_rounded_output():
    p = predict_runtime(2775).to_json()
    p["projected_gpu_hours"] = round(p["projected_gpu_hours"], 4)
    with pytest.raises(RuntimePredictionRefused):
        verify_runtime_prediction(p)


def test_verify_runtime_prediction_rejects_nan_field():
    p = predict_runtime(2775).to_json()
    p["projected_total_seconds"] = float("nan")
    with pytest.raises(RuntimePredictionRefused):
        verify_runtime_prediction(p)


def test_verify_runtime_prediction_rejects_boolean_field():
    p = predict_runtime(2775).to_json()
    p["safety_multiplier"] = True
    with pytest.raises(RuntimePredictionRefused):
        verify_runtime_prediction(p)
