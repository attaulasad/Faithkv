"""B2A-R3 pure qualification evaluator (Step 3 Stage-A, protocol §10,
§10.1-§10.5).

Qualification is a PURE transformation from typed raw FullKV evidence,
plus candidate/config/manifest identity, into a qualification outcome. No
line in this module ever initializes CUDA, loads a model, loads a
tokenizer for execution, or imports R-KV / `kvcot.discovery.b2a_workers
.run_rkv_worker` / `kvcot.discovery.schemas.SwapPairRecord` /
`kvcot.discovery.scientific_summary` -- Stage-A tests inject typed
synthetic evidence; real evidence must originate from the canonical
`kvcot.probes.early_answering.find_think_span` FullKV worker path in a
future Stage B, never a second, independently-written `<think>` parser
here.

The evaluator NEVER accepts a caller-authored gate boolean as
authoritative -- every one of the 27 conditions in
`kvcot.discovery.b2a_r3_contract.B2A_R3_QUALIFICATION_CONDITIONS` is
derived here, from raw evidence, every time.
"""
from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kvcot.discovery.b2a_r3_contract import (
    B2A_R3_QUALIFICATION_CONDITIONS,
    EMBEDDED_ROW_COLUMNS,
    GENERATION_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    QUALIFICATION_ARTIFACT_SCHEMA_VERSION,
    QUALIFICATION_CANDIDATE_LIMIT,
    QUALIFICATION_MEMORY_LIMIT_BYTES,
    QUALIFICATION_MINIMUM_ELIGIBLE_EVENTS,
    QUALIFICATION_MINIMUM_PREDICTED_EVENTS,
    QUALIFICATION_PROTOCOL_VERSION,
    RUNTIME_PREDICTOR_VERSION,
    SAFETY_MULTIPLIER,
    THINK_PARSE_SUCCESS_STATUSES,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    require_lowercase_hex64,
    verify_canonical_sha256,
)
from kvcot.discovery.b2a_r3_runtime import RuntimePredictionRefused, verify_runtime_prediction
from kvcot.utils.hashing import sha256_json, sha256_text

__all__ = [
    "QualificationRefused",
    "B2AR3FullKVQualificationEvidence",
    "evaluate_b2a_r3_qualification_conditions",
    "build_qualification_outcome",
    "select_first_qualified_r3",
]


class QualificationRefused(ValueError):
    """Any hard rejection in this module -- missing evidence, a malformed
    condition map, or a violated first-pass selection rule."""


BUDGET = 1024  # local mirror of the frozen R-KV budget, matches b2a_r3_contract.BUDGET
MEMORY_LIMIT_BYTES = QUALIFICATION_MEMORY_LIMIT_BYTES


class B2AR3FullKVQualificationEvidence(BaseModel):
    """Everything needed to independently re-derive all 27 qualification
    conditions (protocol §12.5), consumed by the pure evaluator below.
    Strict, extra fields forbidden."""

    model_config = ConfigDict(strict=True, extra="forbid")

    candidate_ordinal: int = Field(ge=0, le=15)
    source_example_index: int = Field(ge=0)
    unique_id: str
    row: dict[str, Any]
    raw_row_sha256: str
    problem_sha256: str
    gold_answer_sha256: str

    worker_dataset_repo: str
    worker_dataset_config: str
    worker_dataset_split: str
    worker_dataset_revision: str
    worker_model_name: str
    worker_model_revision: str
    worker_tokenizer_name: str
    worker_tokenizer_revision: str

    expected_prompt_token_ids_sha256: str
    observed_prompt_token_ids_sha256: str
    prompt_token_count: int = Field(ge=0)

    natural_generated_token_ids: list[int]
    generated_token_count: int = Field(ge=0)
    generated_token_ids_sha256: str

    cap_hit: bool
    extracted_answer: str | None
    answer_verification_status: str

    think_parse_status: str
    think_start_index: int | None = Field(default=None, ge=0)
    think_end_index: int | None = Field(default=None, ge=0)
    generation_prompt_preopened_think: bool

    fullkv_wall_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    fullkv_timing_evidence: list[dict[str, Any]]

    requested_device: str
    parameter_placement_evidence: dict[str, Any]
    actual_batch_size: int = Field(ge=0)
    peak_cuda_allocated_bytes: int = Field(ge=0)
    peak_cuda_reserved_bytes: int = Field(ge=0)

    predicted_compaction_event_positions: list[int]
    predicted_event_count: int = Field(ge=0)
    eligible_event_indices: list[int]
    eligible_event_count: int = Field(ge=0)

    generation_config_sha256: str
    runtime_prediction: dict[str, Any]

    candidate_manifest_canonical_sha256: str
    config_sha256: str

    @field_validator(
        "raw_row_sha256", "problem_sha256", "gold_answer_sha256", "expected_prompt_token_ids_sha256",
        "observed_prompt_token_ids_sha256", "generated_token_ids_sha256", "generation_config_sha256",
        "candidate_manifest_canonical_sha256", "config_sha256",
    )
    @classmethod
    def _hex64(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @field_validator("answer_verification_status")
    @classmethod
    def _valid_answer_status(cls, v: str) -> str:
        if v not in ("correct", "incorrect", "unverifiable"):
            raise ValueError(f"answer_verification_status must be correct/incorrect/unverifiable, got {v!r}")
        return v

    @model_validator(mode="after")
    def _row_matches_declared_hashes(self) -> "B2AR3FullKVQualificationEvidence":
        if tuple(self.row.keys()) != EMBEDDED_ROW_COLUMNS:
            raise ValueError(f"embedded row has unexpected columns: {tuple(self.row.keys())}")
        if sha256_json(self.row) != self.raw_row_sha256:
            raise ValueError("raw_row_sha256 does not reproduce sha256_json(row)")
        if sha256_text(self.row["problem"]) != self.problem_sha256:
            raise ValueError("problem_sha256 does not reproduce sha256_text(row['problem'])")
        if sha256_text(self.row["answer"]) != self.gold_answer_sha256:
            raise ValueError("gold_answer_sha256 does not reproduce sha256_text(row['answer'])")
        if self.row["unique_id"] != self.unique_id:
            raise ValueError("row['unique_id'] does not match unique_id")
        return self


def _think_span_valid(evidence: B2AR3FullKVQualificationEvidence) -> bool:
    if evidence.think_parse_status not in THINK_PARSE_SUCCESS_STATUSES:
        return False
    if evidence.think_start_index is None or evidence.think_end_index is None:
        return False
    if evidence.think_start_index < 0:
        return False
    if evidence.think_end_index < evidence.think_start_index:
        return False
    if evidence.think_end_index > evidence.generated_token_count:
        return False
    return True


def _fullkv_timing_complete(evidence: B2AR3FullKVQualificationEvidence) -> bool:
    if not math.isfinite(evidence.fullkv_wall_seconds) or evidence.fullkv_wall_seconds < 0.0:
        return False
    timing = evidence.fullkv_timing_evidence
    if not isinstance(timing, list) or len(timing) == 0:
        return False
    required_keys = {
        "phase", "started_at", "ended_at", "duration_seconds", "synchronize_before_start",
        "synchronize_before_end", "completed", "failure_type", "failure_message",
    }
    for entry in timing:
        if not isinstance(entry, dict) or set(entry) != required_keys:
            return False
        if entry.get("completed") is not True:
            return False
    return True


def _placement_conditions(evidence: B2AR3FullKVQualificationEvidence) -> tuple[bool, bool]:
    """Reuses `kvcot.discovery.strict_device`'s per-worker placement
    predicate rather than inventing a second placement definition --
    factored below into `_single_worker_placement_ok`, called both here and
    (unchanged) by `strict_device.verify_placement_from_raw_evidence`'s own
    two-worker wrapper."""
    from kvcot.discovery.strict_device import _single_worker_placement_ok

    placement = evidence.parameter_placement_evidence
    ok = _single_worker_placement_ok(placement, requested_device=evidence.requested_device)
    if not ok:
        return False, False

    all_on_requested = (
        placement.get("requested_device") == "cuda:0"
        and placement.get("every_parameter_on_cuda") is True
        and isinstance(placement.get("parameter_count"), int)
        and not isinstance(placement.get("parameter_count"), bool)
        and placement.get("parameter_count") > 0
        and list(placement.get("unique_device_types") or []) == ["cuda"]
        and list(placement.get("unique_devices") or []) == ["cuda:0"]
    )
    no_offload = bool(placement.get("no_offload_verified") is True)
    return all_on_requested, no_offload


def evaluate_b2a_r3_qualification_conditions(
    evidence: B2AR3FullKVQualificationEvidence,
    *,
    candidate_manifest: dict[str, Any],
    expected_config_sha256: str,
) -> dict[str, bool]:
    """Derives every one of the 27 frozen conditions (protocol §10.5) from
    raw evidence -- never accepts a caller-supplied boolean for any of
    them."""
    try:
        verify_canonical_sha256(candidate_manifest)
        candidate_manifest_ok = True
    except Exception:
        candidate_manifest_ok = False

    candidate_manifest_hash_match = (
        candidate_manifest_ok
        and evidence.candidate_manifest_canonical_sha256 == candidate_manifest.get("canonical_sha256")
    )
    config_hash_match = evidence.config_sha256 == expected_config_sha256

    dataset_identity_match = (
        evidence.worker_dataset_repo == "HuggingFaceH4/MATH-500"
        and evidence.worker_dataset_config == "default"
        and evidence.worker_dataset_split == "test"
        and evidence.worker_dataset_revision == candidate_manifest.get("dataset_revision")
        and evidence.worker_dataset_revision == "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    )

    model_identity_match = (
        evidence.worker_model_name == MODEL_NAME and evidence.worker_model_revision == MODEL_REVISION
    )
    tokenizer_identity_match = (
        evidence.worker_tokenizer_name == TOKENIZER_NAME
        and evidence.worker_tokenizer_revision == TOKENIZER_REVISION
    )
    generation_config_hash_match = evidence.generation_config_sha256 == GENERATION_CONFIG_SHA256
    prompt_identity_match = evidence.observed_prompt_token_ids_sha256 == evidence.expected_prompt_token_ids_sha256

    all_on_requested_cuda, no_offload_verified = _placement_conditions(evidence)
    peak_tracked = max(evidence.peak_cuda_allocated_bytes, evidence.peak_cuda_reserved_bytes)

    try:
        verify_runtime_prediction(evidence.runtime_prediction)
        runtime_inputs_complete = True
    except RuntimePredictionRefused:
        runtime_inputs_complete = False

    runtime_predictor_version_match = evidence.runtime_prediction.get("runtime_predictor_version") == (
        RUNTIME_PREDICTOR_VERSION
    )
    safety_multiplier = evidence.runtime_prediction.get("safety_multiplier")
    safety_multiplier_exact = (
        not isinstance(safety_multiplier, bool)
        and isinstance(safety_multiplier, float)
        and safety_multiplier == SAFETY_MULTIPLIER
    )
    projected_gpu_hours = evidence.runtime_prediction.get("projected_gpu_hours")
    projected_runtime_within_qualification_target = (
        isinstance(projected_gpu_hours, (int, float))
        and not isinstance(projected_gpu_hours, bool)
        and projected_gpu_hours <= 3.60
    )

    conditions: dict[str, bool] = {
        "no_cap_hit": evidence.cap_hit is False,
        "answer_verifiable": evidence.answer_verification_status != "unverifiable",
        "fullkv_answer_correct": evidence.answer_verification_status == "correct",
        "thinking_span_valid": _think_span_valid(evidence),
        "trace_complete": (
            evidence.cap_hit is False
            and _think_span_valid(evidence)
            and evidence.answer_verification_status != "unverifiable"
        ),
        "prompt_token_count_present": evidence.prompt_token_count > 0,
        "generated_token_count_present": (
            evidence.generated_token_count > 0
            and evidence.generated_token_count == len(evidence.natural_generated_token_ids)
        ),
        "fullkv_timing_complete": _fullkv_timing_complete(evidence),
        "candidate_manifest_hash_match": candidate_manifest_hash_match,
        "config_hash_match": config_hash_match,
        "dataset_identity_match": dataset_identity_match,
        "model_identity_match": model_identity_match,
        "tokenizer_identity_match": tokenizer_identity_match,
        "generation_config_hash_match": generation_config_hash_match,
        "prompt_identity_match": prompt_identity_match,
        "batch_size_is_one": evidence.actual_batch_size == 1,
        "all_parameters_on_requested_cuda": all_on_requested_cuda,
        "no_offload_verified": no_offload_verified,
        "peak_memory_within_limit": peak_tracked <= MEMORY_LIMIT_BYTES,
        "sequence_exceeds_budget": (evidence.prompt_token_count + evidence.generated_token_count) > BUDGET,
        "predicted_compaction_present": len(evidence.predicted_compaction_event_positions) >= 1,
        "predicted_event_count_at_least_six": (
            evidence.predicted_event_count == len(evidence.predicted_compaction_event_positions)
            and evidence.predicted_event_count >= QUALIFICATION_MINIMUM_PREDICTED_EVENTS
        ),
        "at_least_three_events_have_49_future_tokens": (
            evidence.eligible_event_count == len(evidence.eligible_event_indices)
            and evidence.eligible_event_count >= QUALIFICATION_MINIMUM_ELIGIBLE_EVENTS
        ),
        "runtime_inputs_complete": runtime_inputs_complete,
        "runtime_predictor_version_match": runtime_predictor_version_match,
        "safety_multiplier_exact": safety_multiplier_exact,
        "projected_runtime_within_qualification_target": projected_runtime_within_qualification_target,
    }

    if set(conditions) != set(B2A_R3_QUALIFICATION_CONDITIONS):
        raise QualificationRefused(
            f"internal defect: derived condition names {sorted(conditions)} do not exactly match "
            f"the frozen tuple {sorted(B2A_R3_QUALIFICATION_CONDITIONS)}"
        )
    for name, value in conditions.items():
        if not isinstance(value, bool):
            raise QualificationRefused(f"condition {name!r} derived a non-bool value {value!r}")
    return conditions


class CandidateQualificationOutcomeR3(BaseModel):
    """Protocol §12.5. The persisted, per-candidate outcome record."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    candidate_ordinal: int
    source_example_index: int
    unique_id: str
    raw_row_sha256: str
    problem_sha256: str
    gold_answer_sha256: str
    worker_dataset_repo: str
    worker_dataset_config: str
    worker_dataset_split: str
    worker_dataset_revision: str
    worker_model_name: str
    worker_model_revision: str
    worker_tokenizer_name: str
    worker_tokenizer_revision: str
    expected_prompt_token_ids_sha256: str
    observed_prompt_token_ids_sha256: str
    prompt_token_count: int
    generated_token_count: int
    generated_token_ids_sha256: str
    total_processed_tokens: int
    cap_hit: bool
    extracted_answer: str | None
    answer_verification_status: str
    think_parse_status: str
    think_start_index: int | None
    think_end_index: int | None
    generation_prompt_preopened_think: bool
    thinking_span_valid: bool
    trace_complete: bool
    fullkv_wall_seconds: float
    fullkv_timing_evidence: list[dict[str, Any]]
    requested_device: str
    parameter_placement_evidence: dict[str, Any]
    actual_batch_size: int
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    peak_cuda_tracked_bytes: int
    predicted_compaction_event_positions: list[int]
    predicted_event_count: int
    eligible_event_indices: list[int]
    eligible_event_count: int
    reference_seconds_per_token: float
    predicted_example_seconds: float
    predicted_pair_seconds: float
    projected_total_seconds: float
    projected_gpu_hours: float
    safety_multiplier: float
    runtime_predictor_version: str
    conditions: dict[str, bool]
    qualified: bool
    failed_conditions: list[str]

    @model_validator(mode="after")
    def _conditions_are_internally_consistent(self) -> "CandidateQualificationOutcomeR3":
        if set(self.conditions) != set(B2A_R3_QUALIFICATION_CONDITIONS):
            raise ValueError("conditions map does not exactly match the frozen 27-name tuple")
        for name, value in self.conditions.items():
            if not isinstance(value, bool):
                raise ValueError(f"condition {name!r} is not a concrete bool")
        expected_qualified = all(self.conditions[name] for name in B2A_R3_QUALIFICATION_CONDITIONS)
        if self.qualified != expected_qualified:
            raise ValueError("qualified disagrees with all(conditions.values())")
        expected_failed = [name for name in B2A_R3_QUALIFICATION_CONDITIONS if not self.conditions[name]]
        if self.failed_conditions != expected_failed:
            raise ValueError("failed_conditions does not match the frozen tuple-ordered derivation")
        # Named fields must exactly mirror the conditions map -- no
        # condition "stored" under a different field name than the tuple.
        for name, field_name in (
            ("thinking_span_valid", "thinking_span_valid"),
            ("trace_complete", "trace_complete"),
        ):
            if self.conditions[name] != getattr(self, field_name):
                raise ValueError(f"named field {field_name!r} disagrees with conditions[{name!r}]")
        if self.total_processed_tokens != self.prompt_token_count + self.generated_token_count:
            raise ValueError("total_processed_tokens does not equal prompt_token_count + generated_token_count")
        if self.peak_cuda_tracked_bytes != max(self.peak_cuda_allocated_bytes, self.peak_cuda_reserved_bytes):
            raise ValueError("peak_cuda_tracked_bytes does not equal max(allocated, reserved)")
        return self


def build_qualification_outcome(
    evidence: B2AR3FullKVQualificationEvidence,
    *,
    candidate_manifest: dict[str, Any],
    expected_config_sha256: str,
) -> dict[str, Any]:
    """Combines a fresh condition derivation with the evidence's own echoed
    fields into the full protocol §12.5 outcome record, validated through
    the strict `CandidateQualificationOutcomeR3` schema before being
    returned as a plain JSON-serializable dict."""
    conditions = evaluate_b2a_r3_qualification_conditions(
        evidence, candidate_manifest=candidate_manifest, expected_config_sha256=expected_config_sha256
    )
    qualified = all(conditions[name] for name in B2A_R3_QUALIFICATION_CONDITIONS)
    failed_conditions = [name for name in B2A_R3_QUALIFICATION_CONDITIONS if not conditions[name]]

    runtime = evidence.runtime_prediction
    outcome = CandidateQualificationOutcomeR3(
        candidate_ordinal=evidence.candidate_ordinal,
        source_example_index=evidence.source_example_index,
        unique_id=evidence.unique_id,
        raw_row_sha256=evidence.raw_row_sha256,
        problem_sha256=evidence.problem_sha256,
        gold_answer_sha256=evidence.gold_answer_sha256,
        worker_dataset_repo=evidence.worker_dataset_repo,
        worker_dataset_config=evidence.worker_dataset_config,
        worker_dataset_split=evidence.worker_dataset_split,
        worker_dataset_revision=evidence.worker_dataset_revision,
        worker_model_name=evidence.worker_model_name,
        worker_model_revision=evidence.worker_model_revision,
        worker_tokenizer_name=evidence.worker_tokenizer_name,
        worker_tokenizer_revision=evidence.worker_tokenizer_revision,
        expected_prompt_token_ids_sha256=evidence.expected_prompt_token_ids_sha256,
        observed_prompt_token_ids_sha256=evidence.observed_prompt_token_ids_sha256,
        prompt_token_count=evidence.prompt_token_count,
        generated_token_count=evidence.generated_token_count,
        generated_token_ids_sha256=evidence.generated_token_ids_sha256,
        total_processed_tokens=evidence.prompt_token_count + evidence.generated_token_count,
        cap_hit=evidence.cap_hit,
        extracted_answer=evidence.extracted_answer,
        answer_verification_status=evidence.answer_verification_status,
        think_parse_status=evidence.think_parse_status,
        think_start_index=evidence.think_start_index,
        think_end_index=evidence.think_end_index,
        generation_prompt_preopened_think=evidence.generation_prompt_preopened_think,
        thinking_span_valid=conditions["thinking_span_valid"],
        trace_complete=conditions["trace_complete"],
        fullkv_wall_seconds=evidence.fullkv_wall_seconds,
        fullkv_timing_evidence=evidence.fullkv_timing_evidence,
        requested_device=evidence.requested_device,
        parameter_placement_evidence=evidence.parameter_placement_evidence,
        actual_batch_size=evidence.actual_batch_size,
        peak_cuda_allocated_bytes=evidence.peak_cuda_allocated_bytes,
        peak_cuda_reserved_bytes=evidence.peak_cuda_reserved_bytes,
        peak_cuda_tracked_bytes=max(evidence.peak_cuda_allocated_bytes, evidence.peak_cuda_reserved_bytes),
        predicted_compaction_event_positions=list(evidence.predicted_compaction_event_positions),
        predicted_event_count=evidence.predicted_event_count,
        eligible_event_indices=list(evidence.eligible_event_indices),
        eligible_event_count=evidence.eligible_event_count,
        reference_seconds_per_token=runtime["reference_seconds_per_token"],
        predicted_example_seconds=runtime["predicted_example_seconds"],
        predicted_pair_seconds=runtime["predicted_pair_seconds"],
        projected_total_seconds=runtime["projected_total_seconds"],
        projected_gpu_hours=runtime["projected_gpu_hours"],
        safety_multiplier=runtime["safety_multiplier"],
        runtime_predictor_version=runtime["runtime_predictor_version"],
        conditions=conditions,
        qualified=qualified,
        failed_conditions=failed_conditions,
    )
    return outcome.model_dump(mode="json")


def select_first_qualified_r3(attempted: list[dict[str, Any]]) -> dict[str, Any] | None:
    """First-pass selection (protocol §10.4): attempted evidence must be in
    exact ordinal order, ordinals 0-7 only, no duplicates, no gaps, stop at
    the first qualified candidate -- reject evidence for any row after a
    passing row, never rank by "best" score, never evaluate a ninth
    candidate."""
    seen_ordinals: list[int] = []
    passed_index: int | None = None
    for index, outcome in enumerate(attempted):
        ordinal = outcome["candidate_ordinal"]
        if not (0 <= ordinal <= QUALIFICATION_CANDIDATE_LIMIT - 1):
            raise QualificationRefused(f"candidate_ordinal {ordinal} is outside the 0-7 qualification window")
        if ordinal in seen_ordinals:
            raise QualificationRefused(f"duplicate candidate_ordinal {ordinal} in attempted list")
        if seen_ordinals and ordinal != seen_ordinals[-1] + 1:
            raise QualificationRefused(
                f"attempted candidates are not in contiguous ordinal order: {seen_ordinals} then {ordinal}"
            )
        seen_ordinals.append(ordinal)
        if passed_index is not None:
            raise QualificationRefused(
                f"candidate ordinal={ordinal} was evaluated after an earlier passing candidate "
                f"(ordinal={attempted[passed_index]['candidate_ordinal']}) -- refusing"
            )
        if outcome["qualified"]:
            passed_index = index
    if len(attempted) > QUALIFICATION_CANDIDATE_LIMIT:
        raise QualificationRefused(
            f"attempted {len(attempted)} candidates, exceeding the {QUALIFICATION_CANDIDATE_LIMIT}-candidate limit"
        )
    return attempted[passed_index] if passed_index is not None else None
