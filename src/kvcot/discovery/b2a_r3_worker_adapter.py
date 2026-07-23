"""B2A-R3 canonical FullKV worker-result adapter (Step 3R4 Finding 5,
docs/B2A_R3_STAGE_A_PROTOCOL_ALIGNMENT_AMENDMENT_2026-07-23.md §7).

Before this module, no canonical, schema-validated path existed from a
real (or realistically synthetic) FullKV worker result into
`kvcot.discovery.b2a_r3_qualification.B2AR3FullKVQualificationEvidence` --
a future Stage B integration would have had no choice but to hand-
assemble evidence fields the same way this repository's own audit-fixture
test helpers do today.

`FullKVWorkerResultR3` is a NEW, versioned, backward-compatible schema --
it does not modify, subclass, or reinterpret the historical
`kvcot.discovery.b2a_workers.FullKVWorkerResult` (the B2A-R1/R2 shape,
still used unchanged). `adapt_fullkv_worker_result_to_r3_evidence` is the
ONE authoritative conversion function: it strictly validates the worker
result, binds it to an exact candidate in a strictly-verified candidate
manifest, recomputes the static R-KV schedule and runtime prediction
fresh (never trusting a worker-supplied schedule/runtime), and validates
canonical timing/memory evidence separately (Step 3R4 Finding 2) before
extracting peak CUDA byte counts from memory evidence alone.

No line in this module imports R-KV, initializes CUDA, or loads a model
or tokenizer. `find_think_span` (`kvcot.probes.early_answering`) is
called only by the real FullKV worker itself (future Stage B, using its
own already-loaded tokenizer) -- this adapter, being CPU-only and
tokenizer-free by design, never re-implements or re-invokes it; it trusts
the worker-reported `think_parse_status`/`think_start_index`/
`think_end_index`/`generation_prompt_preopened_think` fields as the
canonical worker's own output, exactly as it trusts the worker's
`expected_prompt_token_ids_sha256` (itself computed by the worker via the
canonical `kvcot.discovery.manifest_prepare._render_and_tokenize` path,
never re-derived here).
"""
from __future__ import annotations

from typing import Any, Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kvcot.analysis.rkv_schedule import predicted_compaction_event_positions
from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure
from kvcot.discovery.b2a_r3_contract import (
    BUDGET,
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SPLIT,
    DIVIDE_LENGTH,
    MODEL_NAME,
    MODEL_REVISION,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    require_lowercase_hex64,
)
from kvcot.discovery.b2a_r3_qualification import B2AR3FullKVQualificationEvidence
from kvcot.discovery.b2a_r3_runtime import predict_runtime
from kvcot.discovery.final_contract import (
    fullkv_qualification_memory_complete,
    fullkv_qualification_timing_complete,
    peak_cuda_bytes_from_qualification_memory_evidence,
)
from kvcot.discovery.pass1 import eligible_event_positions
from kvcot.utils.hashing import sha256_int_ids, sha256_json

__all__ = [
    "FULLKV_WORKER_RESULT_R3_SCHEMA_VERSION",
    "WorkerAdapterRefused",
    "FullKVWorkerResultR3",
    "adapt_fullkv_worker_result_to_r3_evidence",
]

FULLKV_WORKER_RESULT_R3_SCHEMA_VERSION: Final[str] = "faithkv-b2a-r3-fullkv-worker-result-v1"


class WorkerAdapterRefused(ValueError):
    """Any hard rejection while binding a worker result to a candidate --
    wrong type, wrong candidate identity, wrong frozen contract identity,
    or evidence that fails canonical timing/memory validation."""


class FullKVWorkerResultR3(BaseModel):
    """The canonical, versioned FullKV worker-result shape a real (future)
    Stage B worker would emit for exactly one B2A-R3 qualification
    candidate. Strict, extra fields forbidden. Never confused with, and
    never replacing, the historical `FullKVWorkerResult`
    (`kvcot.discovery.b2a_workers`) B2A-R1/R2 already use -- that schema
    is untouched by this module."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    worker_result_schema_version: str
    role: str

    model_name: str
    model_revision: str
    requested_model_revision: str
    model_revision_match: bool
    tokenizer_name: str
    tokenizer_revision: str
    requested_tokenizer_revision: str
    tokenizer_revision_match: bool

    dataset_repo: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str

    source_example_index: int
    unique_id: str
    raw_row_sha256: str
    problem_sha256: str
    gold_answer_sha256: str

    expected_prompt_token_ids_sha256: str
    observed_prompt_token_ids_sha256: str
    prompt_token_count: int

    natural_generated_token_ids: list[int]
    generated_token_ids_sha256: str
    natural_answer: str | None
    natural_answer_status: str
    cap_hit: bool

    think_parse_status: str
    think_start_index: int | None = None
    think_end_index: int | None = None
    generation_prompt_preopened_think: bool

    requested_device: str
    parameter_placement_evidence: dict[str, Any]
    actual_batch_size: int

    timing_evidence: list[dict[str, Any]]
    memory_phase_evidence: list[dict[str, Any]]
    wall_seconds: float

    runtime_generation_config: dict[str, Any]
    worker_generation_config_sha256: str
    worker_config_sha256: str

    software_versions: dict[str, str]

    @field_validator(
        "raw_row_sha256", "problem_sha256", "gold_answer_sha256",
        "expected_prompt_token_ids_sha256", "observed_prompt_token_ids_sha256",
        "generated_token_ids_sha256", "worker_generation_config_sha256", "worker_config_sha256",
    )
    @classmethod
    def _hex64(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @field_validator("natural_answer_status")
    @classmethod
    def _valid_answer_status(cls, v: str) -> str:
        if v not in ("correct", "incorrect", "unverifiable"):
            raise ValueError(f"natural_answer_status must be correct/incorrect/unverifiable, got {v!r}")
        return v

    @model_validator(mode="after")
    def _internally_consistent(self) -> "FullKVWorkerResultR3":
        if self.role != "fullkv":
            raise ValueError(f"role must be 'fullkv', got {self.role!r}")
        if self.worker_result_schema_version != FULLKV_WORKER_RESULT_R3_SCHEMA_VERSION:
            raise ValueError("worker_result_schema_version does not match the frozen value")
        if any(type(token_id) is not int for token_id in self.natural_generated_token_ids):
            raise ValueError("natural_generated_token_ids must contain strict integers, never bool")
        if sha256_int_ids(self.natural_generated_token_ids) != self.generated_token_ids_sha256:
            raise ValueError("generated_token_ids_sha256 does not reproduce from natural_generated_token_ids")
        if sha256_json(self.runtime_generation_config) != self.worker_generation_config_sha256:
            raise ValueError("worker_generation_config_sha256 does not reproduce sha256_json(runtime_generation_config)")
        if self.model_revision_match is not True:
            raise ValueError("model_revision_match must be true for the resolved runtime model revision")
        if self.tokenizer_revision_match is not True:
            raise ValueError("tokenizer_revision_match must be true for the resolved runtime tokenizer revision")
        if self.model_revision != self.requested_model_revision:
            raise ValueError("resolved model_revision must equal requested_model_revision")
        if self.tokenizer_revision != self.requested_tokenizer_revision:
            raise ValueError("resolved tokenizer_revision must equal requested_tokenizer_revision")
        if self.think_start_index is not None and self.think_start_index < 0:
            raise ValueError("think_start_index must be >= 0 when present")
        if self.think_end_index is not None and self.think_end_index < 0:
            raise ValueError("think_end_index must be >= 0 when present")
        return self


def adapt_fullkv_worker_result_to_r3_evidence(
    *,
    worker_result: Any,
    candidate_manifest: dict[str, Any],
    candidate_ordinal: int,
    expected_config_sha256: str,
) -> B2AR3FullKVQualificationEvidence:
    """The ONE authoritative conversion from a canonical FullKV worker
    result into qualification evidence (Step 3R4 Finding 5).

    Never accepts a legacy `FullKVWorkerResult`, a plain dict, or any
    caller-authored gate boolean -- every one of the 27 qualification
    conditions is still derived downstream, from raw evidence, by
    `kvcot.discovery.b2a_r3_qualification.evaluate_b2a_r3_qualification_conditions`,
    exactly as before. Never imports R-KV; never initializes CUDA itself
    (the worker result it consumes may describe a real Stage B CUDA run,
    but this function's own execution is pure CPU Python).
    """
    if not isinstance(worker_result, FullKVWorkerResultR3):
        raise WorkerAdapterRefused(
            "worker_result must be a strict FullKVWorkerResultR3 instance -- a legacy FullKVWorkerResult "
            "or an arbitrary dict that bypasses worker-schema validation is never accepted"
        )

    manifest = verify_candidate_manifest_structure(
        candidate_manifest, expected_config_sha256=expected_config_sha256
    )
    if not (0 <= candidate_ordinal < len(manifest.candidates)):
        raise WorkerAdapterRefused(f"candidate_ordinal {candidate_ordinal} is not present in the candidate manifest")
    candidate = manifest.candidates[candidate_ordinal]

    for field_name in ("source_example_index", "unique_id", "raw_row_sha256", "problem_sha256", "gold_answer_sha256"):
        if getattr(worker_result, field_name) != getattr(candidate, field_name):
            raise WorkerAdapterRefused(
                f"worker_result {field_name} does not match candidate ordinal {candidate_ordinal}"
            )

    if (
        worker_result.dataset_repo, worker_result.dataset_config,
        worker_result.dataset_split, worker_result.dataset_revision,
    ) != (DATASET_REPO, DATASET_CONFIG, DATASET_SPLIT, DATASET_REVISION):
        raise WorkerAdapterRefused("worker_result dataset identity does not match the frozen contract")
    if (worker_result.model_name, worker_result.model_revision) != (MODEL_NAME, MODEL_REVISION):
        raise WorkerAdapterRefused("worker_result model identity does not match the frozen contract")
    if (worker_result.tokenizer_name, worker_result.tokenizer_revision) != (TOKENIZER_NAME, TOKENIZER_REVISION):
        raise WorkerAdapterRefused("worker_result tokenizer identity does not match the frozen contract")

    if not fullkv_qualification_timing_complete(worker_result.timing_evidence, fullkv_wall_seconds=worker_result.wall_seconds):
        raise WorkerAdapterRefused("worker_result timing evidence failed the canonical FullKV timing contract")
    if not fullkv_qualification_memory_complete(worker_result.memory_phase_evidence):
        raise WorkerAdapterRefused("worker_result memory-phase evidence failed the canonical FullKV memory contract")
    peak_allocated, peak_reserved = peak_cuda_bytes_from_qualification_memory_evidence(
        worker_result.memory_phase_evidence
    )

    generated_token_count = len(worker_result.natural_generated_token_ids)
    total_len = worker_result.prompt_token_count + generated_token_count
    max_position = total_len - 1 if total_len > worker_result.prompt_token_count else worker_result.prompt_token_count
    positions = predicted_compaction_event_positions(
        prompt_length=worker_result.prompt_token_count, max_position=max_position, budget=BUDGET,
        divide_length=DIVIDE_LENGTH,
    )
    eligible = eligible_event_positions(
        positions, prompt_length=worker_result.prompt_token_count, total_len=total_len
    )
    runtime = predict_runtime(total_len)

    return B2AR3FullKVQualificationEvidence(
        candidate_ordinal=candidate_ordinal,
        source_example_index=worker_result.source_example_index,
        unique_id=worker_result.unique_id,
        row=candidate.row,
        raw_row_sha256=worker_result.raw_row_sha256,
        problem_sha256=worker_result.problem_sha256,
        gold_answer_sha256=worker_result.gold_answer_sha256,
        worker_dataset_repo=worker_result.dataset_repo,
        worker_dataset_config=worker_result.dataset_config,
        worker_dataset_split=worker_result.dataset_split,
        worker_dataset_revision=worker_result.dataset_revision,
        worker_model_name=worker_result.model_name,
        worker_model_revision=worker_result.model_revision,
        worker_tokenizer_name=worker_result.tokenizer_name,
        worker_tokenizer_revision=worker_result.tokenizer_revision,
        expected_prompt_token_ids_sha256=worker_result.expected_prompt_token_ids_sha256,
        observed_prompt_token_ids_sha256=worker_result.observed_prompt_token_ids_sha256,
        prompt_token_count=worker_result.prompt_token_count,
        natural_generated_token_ids=list(worker_result.natural_generated_token_ids),
        generated_token_count=generated_token_count,
        generated_token_ids_sha256=worker_result.generated_token_ids_sha256,
        cap_hit=worker_result.cap_hit,
        extracted_answer=worker_result.natural_answer,
        answer_verification_status=worker_result.natural_answer_status,
        think_parse_status=worker_result.think_parse_status,
        think_start_index=worker_result.think_start_index,
        think_end_index=worker_result.think_end_index,
        generation_prompt_preopened_think=worker_result.generation_prompt_preopened_think,
        fullkv_wall_seconds=worker_result.wall_seconds,
        fullkv_timing_evidence=worker_result.timing_evidence,
        requested_device=worker_result.requested_device,
        parameter_placement_evidence=worker_result.parameter_placement_evidence,
        actual_batch_size=worker_result.actual_batch_size,
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        predicted_compaction_event_positions=positions,
        predicted_event_count=len(positions),
        eligible_event_indices=eligible,
        eligible_event_count=len(eligible),
        worker_generation_config_sha256=worker_result.worker_generation_config_sha256,
        runtime_prediction=runtime.to_json(),
        candidate_manifest_canonical_sha256=manifest.canonical_sha256,
        config_sha256=worker_result.worker_config_sha256,
    )
