"""B2A-R3 exact runtime predictor (Step 3 Stage-A, protocol §7).

Pure arithmetic over the frozen §7.3 constants -- no rounding of any
intermediate value, no operator-suppliable override of any constant (the
public `predict_runtime` function accepts exactly one input,
`candidate_total_tokens`; every other value comes from
`kvcot.discovery.b2a_r3_contract`'s frozen module constants). No torch, no
CUDA, no network.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from kvcot.discovery.b2a_r3_contract import (
    B2B_EXAMPLE_COUNT,
    B2B_REAL_PAIR_COUNT,
    QUALIFICATION_TARGET_HOURS,
    REFERENCE_EXAMPLE_SECONDS,
    REFERENCE_PAIR_SECONDS,
    REFERENCE_SETUP_SECONDS,
    REFERENCE_TOTAL_TOKENS,
    RUNTIME_PREDICTOR_VERSION,
    RUNTIME_SOURCE_ARTIFACT_PATH,
    RUNTIME_SOURCE_ARTIFACT_SHA256,
    SAFETY_MULTIPLIER,
    require_strict_int,
)

__all__ = [
    "RuntimePredictionRefused",
    "RuntimePredictionEvidence",
    "predict_runtime",
    "continuous_token_ceiling",
    "verify_runtime_prediction",
]

_REQUIRED_EVIDENCE_FIELDS: tuple[str, ...] = (
    "candidate_total_tokens",
    "reference_total_tokens",
    "reference_example_seconds",
    "reference_pair_seconds",
    "reference_setup_seconds",
    "safety_multiplier",
    "b2b_example_count",
    "b2b_real_pair_count",
    "qualification_target_hours",
    "runtime_predictor_version",
    "runtime_source_artifact_path",
    "runtime_source_artifact_sha256",
    "reference_seconds_per_token",
    "predicted_example_seconds",
    "predicted_pair_seconds",
    "projected_total_seconds",
    "projected_gpu_hours",
    "projected_runtime_within_qualification_target",
)


class RuntimePredictionRefused(ValueError):
    """Raised for any missing, malformed, or non-reproducing runtime
    prediction input/evidence -- a candidate missing a required predictor
    input is rejected outright, never silently defaulted (protocol §7.5,
    §10.3)."""


@dataclass(frozen=True)
class RuntimePredictionEvidence:
    """Every raw input, every §7.3 constant, every intermediate, every
    output, the predictor version, and the source timing-artifact hash --
    preserved together so the prediction is independently re-derivable
    from this record alone (protocol §7.5's closing paragraph)."""

    candidate_total_tokens: int
    reference_total_tokens: int
    reference_example_seconds: float
    reference_pair_seconds: float
    reference_setup_seconds: float
    safety_multiplier: float
    b2b_example_count: int
    b2b_real_pair_count: int
    qualification_target_hours: float
    runtime_predictor_version: str
    runtime_source_artifact_path: str
    runtime_source_artifact_sha256: str
    reference_seconds_per_token: float
    predicted_example_seconds: float
    predicted_pair_seconds: float
    projected_total_seconds: float
    projected_gpu_hours: float
    projected_runtime_within_qualification_target: bool

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def predict_runtime(candidate_total_tokens: Any) -> RuntimePredictionEvidence:
    """The frozen §7.4 formula, exact operation order, no intermediate
    rounding:

        reference_seconds_per_token = REFERENCE_EXAMPLE_SECONDS / REFERENCE_TOTAL_TOKENS
        predicted_example_seconds   = reference_seconds_per_token * candidate_total_tokens * SAFETY_MULTIPLIER
        predicted_pair_seconds      = REFERENCE_PAIR_SECONDS * SAFETY_MULTIPLIER
        projected_total_seconds     = REFERENCE_SETUP_SECONDS
                                       + B2B_EXAMPLE_COUNT * predicted_example_seconds
                                       + B2B_REAL_PAIR_COUNT * predicted_pair_seconds
        projected_gpu_hours         = projected_total_seconds / 3600.0

    Rejects a missing, boolean, non-finite, negative, or zero
    `candidate_total_tokens` -- never silently coerced or defaulted.
    """
    try:
        require_strict_int(candidate_total_tokens, "candidate_total_tokens")
    except ValueError as exc:
        raise RuntimePredictionRefused(str(exc)) from exc
    if candidate_total_tokens <= 0:
        raise RuntimePredictionRefused(
            f"candidate_total_tokens must be a positive int, got {candidate_total_tokens}"
        )

    reference_seconds_per_token = REFERENCE_EXAMPLE_SECONDS / REFERENCE_TOTAL_TOKENS
    predicted_example_seconds = reference_seconds_per_token * candidate_total_tokens * SAFETY_MULTIPLIER
    predicted_pair_seconds = REFERENCE_PAIR_SECONDS * SAFETY_MULTIPLIER
    projected_total_seconds = (
        REFERENCE_SETUP_SECONDS
        + B2B_EXAMPLE_COUNT * predicted_example_seconds
        + B2B_REAL_PAIR_COUNT * predicted_pair_seconds
    )
    projected_gpu_hours = projected_total_seconds / 3600.0

    return RuntimePredictionEvidence(
        candidate_total_tokens=candidate_total_tokens,
        reference_total_tokens=REFERENCE_TOTAL_TOKENS,
        reference_example_seconds=REFERENCE_EXAMPLE_SECONDS,
        reference_pair_seconds=REFERENCE_PAIR_SECONDS,
        reference_setup_seconds=REFERENCE_SETUP_SECONDS,
        safety_multiplier=SAFETY_MULTIPLIER,
        b2b_example_count=B2B_EXAMPLE_COUNT,
        b2b_real_pair_count=B2B_REAL_PAIR_COUNT,
        qualification_target_hours=QUALIFICATION_TARGET_HOURS,
        runtime_predictor_version=RUNTIME_PREDICTOR_VERSION,
        runtime_source_artifact_path=RUNTIME_SOURCE_ARTIFACT_PATH,
        runtime_source_artifact_sha256=RUNTIME_SOURCE_ARTIFACT_SHA256,
        reference_seconds_per_token=reference_seconds_per_token,
        predicted_example_seconds=predicted_example_seconds,
        predicted_pair_seconds=predicted_pair_seconds,
        projected_total_seconds=projected_total_seconds,
        projected_gpu_hours=projected_gpu_hours,
        # Exact protocol equality (safety_multiplier == 1.20 compared
        # against the SAME frozen Python float literal SAFETY_MULTIPLIER)
        # is a separate, dedicated condition (`safety_multiplier_exact`,
        # kvcot.discovery.b2a_r3_qualification) -- this field is only the
        # frozen gate `projected_gpu_hours <= QUALIFICATION_TARGET_HOURS`.
        projected_runtime_within_qualification_target=(projected_gpu_hours <= QUALIFICATION_TARGET_HOURS),
    )


def continuous_token_ceiling() -> float:
    """Solves §7.4's formula for `candidate_total_tokens` at
    `projected_gpu_hours == QUALIFICATION_TARGET_HOURS`, derived from the
    frozen §7.3 constants at call time -- never a hard-coded decimal
    (protocol §7.5: "Step 3 must not hard-code this decimal... so that if
    any reference constant were ever amended by a future dated protocol
    revision, the ceiling recomputes correctly")."""
    reference_seconds_per_token = REFERENCE_EXAMPLE_SECONDS / REFERENCE_TOTAL_TOKENS
    predicted_pair_seconds = REFERENCE_PAIR_SECONDS * SAFETY_MULTIPLIER
    target_total_seconds = QUALIFICATION_TARGET_HOURS * 3600.0
    remaining = target_total_seconds - REFERENCE_SETUP_SECONDS - B2B_REAL_PAIR_COUNT * predicted_pair_seconds
    denominator = B2B_EXAMPLE_COUNT * reference_seconds_per_token * SAFETY_MULTIPLIER
    return remaining / denominator


def verify_runtime_prediction(evidence: dict[str, Any]) -> None:
    """Recompute `predict_runtime(evidence['candidate_total_tokens'])` and
    require EXACT field-by-field agreement with the stored evidence --
    catches a hand-edited/rounded field, a modified reference constant, or
    an operator-supplied safety-multiplier override. Fails closed on any
    missing field, any NaN/Infinity, or any boolean substituted for a
    number."""
    missing = [f for f in _REQUIRED_EVIDENCE_FIELDS if f not in evidence]
    if missing:
        raise RuntimePredictionRefused(f"runtime prediction evidence missing required field(s): {missing}")

    for field in (
        "reference_example_seconds", "reference_pair_seconds", "reference_setup_seconds",
        "safety_multiplier", "qualification_target_hours", "reference_seconds_per_token",
        "predicted_example_seconds", "predicted_pair_seconds", "projected_total_seconds",
        "projected_gpu_hours",
    ):
        value = evidence[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimePredictionRefused(f"{field} must be a strict number, got {value!r}")
        if not math.isfinite(float(value)):
            raise RuntimePredictionRefused(f"{field} must be finite, got {value!r}")

    recomputed = predict_runtime(evidence["candidate_total_tokens"]).to_json()
    for field in _REQUIRED_EVIDENCE_FIELDS:
        if evidence[field] != recomputed[field]:
            raise RuntimePredictionRefused(
                f"runtime prediction field {field!r} does not reproduce: "
                f"stored={evidence[field]!r} recomputed={recomputed[field]!r}"
            )
