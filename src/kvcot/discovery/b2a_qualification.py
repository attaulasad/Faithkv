"""B2A-R2 FullKV-only row qualification (2026-07-22).

Pre-registered by `docs/B2A_R1_FAILURE_AND_B2A_R2_PROTOCOL_2026-07-22.md`
BEFORE any qualification inference runs (deterministic, outcome-blind
candidate order already frozen by `kvcot.discovery.b2a_r2_candidates`).

B2A-R1's frozen row (`example_index=0`) produced zero R-KV compaction
events -- a genuinely ineligible calibration that tested no eviction at
all. This module attempts candidates from the committed manifest, IN THEIR
COMMITTED ORDER, using FullKV natural generation ONLY -- R-KV is never
imported, patched, or loaded here, matching CLAUDE.md's discipline that no
GPU/inference activity beyond what is explicitly authorized may occur. It
stops at the FIRST candidate that satisfies every frozen qualification
condition (`QUALIFICATION_CONDITIONS`) and never inspects a later candidate
once one qualifies.

Reuses, never duplicates:
  - `kvcot.discovery.b2a_workers.run_fullkv_worker` for the actual FullKV
    natural-generation loop, model/tokenizer loading, answer verification,
    memory/timing/placement evidence (the SAME code the real B2A execution
    itself uses for its FullKV worker).
  - `kvcot.analysis.rkv_schedule.predicted_compaction_event_positions` for
    the R-KV schedule/trigger simulation.
  - `kvcot.discovery.pass1.eligible_event_positions` for the event
    eligibility rule.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from kvcot.discovery.constants import MINIMUM_FUTURE_TOKENS_AFTER_EVENT
from kvcot.discovery.manifest_prepare import ManifestPreparationError

QUALIFICATION_PROTOCOL_VERSION = "faithkv-b2a-r2-qualification-v1"
MAX_CANDIDATES_ATTEMPTED = 12
QUALIFICATION_MEMORY_LIMIT_BYTES = 22 * 1024**3
QUALIFICATION_MINIMUM_EVENTS = 3

QUALIFICATION_CONDITIONS: tuple[str, ...] = (
    "no_cap_hit",
    "fullkv_answer_verifiable",
    "fullkv_answer_correct",
    "predicted_schedule_has_at_least_three_events",
    "at_least_three_events_have_49_future_tokens",
    "identity_checks_pass",
    "batch_size_is_one",
    "all_parameters_on_cuda",
    "no_offload",
    "peak_memory_within_limit",
)


@dataclass(frozen=True)
class CandidateQualificationOutcome:
    candidate_ordinal: int
    source_example_index: int
    unique_id: str
    prompt_token_count: int
    prompt_token_ids_sha256: str
    generated_token_count: int
    generated_token_ids_sha256: str
    total_processed_tokens: int
    cap_hit: bool
    extracted_answer: str | None
    answer_verification_status: str
    fullkv_wall_seconds: float
    peak_cuda_allocated_bytes: int
    peak_cuda_reserved_bytes: int
    peak_cuda_tracked_bytes: int
    predicted_compaction_event_positions: list[int]
    predicted_event_count: int
    eligible_event_indices: list[int]
    eligible_event_count: int
    conditions: dict[str, bool]
    qualified: bool
    failed_conditions: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QualificationArtifact:
    protocol_version: str
    candidate_manifest_path: str
    candidate_manifest_hash: str
    config_hash: str
    dataset_revision: str
    model_revision: str
    tokenizer_revision: str
    budget: int
    attempted: list[CandidateQualificationOutcome]
    selected_ordinal: int | None
    selected_unique_id: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "candidate_manifest_path": self.candidate_manifest_path,
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "config_hash": self.config_hash,
            "dataset_revision": self.dataset_revision,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "budget": self.budget,
            "attempted": [outcome.to_json() for outcome in self.attempted],
            "selected_ordinal": self.selected_ordinal,
            "selected_unique_id": self.selected_unique_id,
        }


def evaluate_candidate_qualification(
    *,
    cap_hit: bool,
    answer_status: str,
    predicted_event_count: int,
    eligible_event_count: int,
    identity_ok: bool,
    batch_size: int,
    every_parameter_on_cuda: bool,
    no_offload_verified: bool,
    peak_cuda_allocated_bytes: int,
    peak_cuda_reserved_bytes: int,
    memory_limit_bytes: int = QUALIFICATION_MEMORY_LIMIT_BYTES,
    minimum_events: int = QUALIFICATION_MINIMUM_EVENTS,
) -> dict[str, bool]:
    """Pure evaluation of the 10 frozen mandatory conditions against
    already-collected raw evidence -- never touches CUDA, never loads a
    model, mirrors `kvcot.discovery.b2a_contract.evaluate_b2a_gate`'s
    separation of evidence collection from evidence evaluation."""
    peak_tracked = max(peak_cuda_allocated_bytes, peak_cuda_reserved_bytes)
    if answer_status not in ("correct", "incorrect", "unverifiable"):
        raise ValueError(f"answer_status must be correct/incorrect/unverifiable, got {answer_status!r}")
    return {
        "no_cap_hit": not cap_hit,
        "fullkv_answer_verifiable": answer_status != "unverifiable",
        "fullkv_answer_correct": answer_status == "correct",
        "predicted_schedule_has_at_least_three_events": predicted_event_count >= minimum_events,
        "at_least_three_events_have_49_future_tokens": eligible_event_count >= minimum_events,
        "identity_checks_pass": bool(identity_ok),
        "batch_size_is_one": batch_size == 1,
        "all_parameters_on_cuda": bool(every_parameter_on_cuda),
        "no_offload": bool(no_offload_verified),
        "peak_memory_within_limit": peak_tracked <= memory_limit_bytes,
    }


def build_candidate_outcome(
    *,
    candidate_ordinal: int,
    source_example_index: int,
    unique_id: str,
    prompt_token_count: int,
    prompt_token_ids_sha256: str,
    generated_token_ids: list[int],
    generated_token_ids_sha256: str,
    cap_hit: bool,
    extracted_answer: str | None,
    answer_status: str,
    fullkv_wall_seconds: float,
    peak_cuda_allocated_bytes: int,
    peak_cuda_reserved_bytes: int,
    predicted_event_positions: list[int],
    identity_ok: bool,
    batch_size: int,
    every_parameter_on_cuda: bool,
    no_offload_verified: bool,
) -> CandidateQualificationOutcome:
    from kvcot.discovery.pass1 import eligible_event_positions

    total_len = prompt_token_count + len(generated_token_ids)
    eligible_indices = eligible_event_positions(
        predicted_event_positions, prompt_length=prompt_token_count, total_len=total_len
    )
    conditions = evaluate_candidate_qualification(
        cap_hit=cap_hit,
        answer_status=answer_status,
        predicted_event_count=len(predicted_event_positions),
        eligible_event_count=len(eligible_indices),
        identity_ok=identity_ok,
        batch_size=batch_size,
        every_parameter_on_cuda=every_parameter_on_cuda,
        no_offload_verified=no_offload_verified,
        peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
        peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
    )
    qualified = all(conditions.values())
    failed = [name for name in QUALIFICATION_CONDITIONS if not conditions[name]]
    return CandidateQualificationOutcome(
        candidate_ordinal=candidate_ordinal,
        source_example_index=source_example_index,
        unique_id=unique_id,
        prompt_token_count=prompt_token_count,
        prompt_token_ids_sha256=prompt_token_ids_sha256,
        generated_token_count=len(generated_token_ids),
        generated_token_ids_sha256=generated_token_ids_sha256,
        total_processed_tokens=total_len,
        cap_hit=cap_hit,
        extracted_answer=extracted_answer,
        answer_verification_status=answer_status,
        fullkv_wall_seconds=fullkv_wall_seconds,
        peak_cuda_allocated_bytes=peak_cuda_allocated_bytes,
        peak_cuda_reserved_bytes=peak_cuda_reserved_bytes,
        peak_cuda_tracked_bytes=max(peak_cuda_allocated_bytes, peak_cuda_reserved_bytes),
        predicted_compaction_event_positions=list(predicted_event_positions),
        predicted_event_count=len(predicted_event_positions),
        eligible_event_indices=list(eligible_indices),
        eligible_event_count=len(eligible_indices),
        conditions=conditions,
        qualified=qualified,
        failed_conditions=failed,
    )


def select_first_qualified(attempted: list[CandidateQualificationOutcome]) -> CandidateQualificationOutcome | None:
    """First candidate (in ATTEMPT order, which is already the committed
    deterministic candidate order) with every condition true -- `None` if
    none qualified. Never re-orders, never picks by "best" score."""
    for outcome in attempted:
        if outcome.qualified:
            return outcome
    return None


def run_qualification_dry_run(config: Any, candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    """No CUDA access, no model load, no inference -- structural/identity
    validation only, plus the plan `--execute` would follow."""
    validate_candidate_manifest_identity(candidate_manifest, config=config)
    candidates = candidate_manifest["candidates"][:MAX_CANDIDATES_ATTEMPTED]
    return {
        "protocol_version": QUALIFICATION_PROTOCOL_VERSION,
        "candidate_manifest_hash": candidate_manifest["canonical_sha256"],
        "candidates_to_attempt_in_order": [c["unique_id"] for c in candidates],
        "max_candidates": MAX_CANDIDATES_ATTEMPTED,
        "qualification_conditions": list(QUALIFICATION_CONDITIONS),
        "would_touch_cuda": False,
        "would_load_model": False,
        "would_import_rkv": False,
    }


def _default_fullkv_runner(config: Any):
    """Real production per-candidate FullKV runner: loads the model and
    tokenizer exactly ONCE (outside the returned closure's per-call body)
    and reuses them for every candidate via `run_fullkv_worker`'s own
    dependency-injection seam -- `reset_patched_state` inside that function
    already resets all per-example mutable state, so reusing one loaded
    model/tokenizer across candidates is the same safe pattern the real B2A
    execution relies on for its own single example. R-KV is never imported
    anywhere in this closure."""
    import torch

    from kvcot.discovery.b2a_workers import run_fullkv_worker
    from kvcot.discovery.snapshot_boundary import resolve_local_snapshot
    from kvcot.discovery.strict_device import load_fullkv_discovery_model, verify_single_rtx3090
    from transformers import AutoTokenizer

    verify_single_rtx3090(torch.cuda, torch_module=torch)
    model_snapshot = resolve_local_snapshot(config.model.name, config.model.revision, "model")
    tokenizer_snapshot = resolve_local_snapshot(config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot.local_path, local_files_only=True, use_fast=True)
    model = load_fullkv_discovery_model(config, model_snapshot.local_path, "cuda:0")

    def run(candidate_row: dict[str, Any]) -> dict[str, Any]:
        from kvcot.discovery.discovery_config import MATH500_DATASET_REPO
        from kvcot.discovery.manifest import B2AOneExampleManifest, ChatTemplateRenderingConfig
        from kvcot.discovery.manifest_prepare import _render_and_tokenize
        from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

        row = candidate_row["row"]
        tokenizer_ret, user_message, messages, token_ids = _render_and_tokenize(
            row, config.model.tokenizer_name, config.model.tokenizer_revision,
            local_only_path=tokenizer_snapshot.local_path,
        )
        manifest = B2AOneExampleManifest(
            dataset_repo=MATH500_DATASET_REPO,
            dataset_config=config.dataset.config,
            dataset_split=config.dataset.split,
            dataset_revision=candidate_row["dataset_revision"],
            example_index=candidate_row["source_example_index"],
            unique_id=candidate_row["unique_id"],
            raw_content_hash=candidate_row["raw_row_sha256"],
            gold_answer=row["answer"],
            prompt_token_ids_sha256=sha256_int_ids(token_ids),
            tokenizer_revision_used_for_prompt_hash=config.model.tokenizer_revision,
            rendered_user_message_sha256=sha256_text(user_message),
            chat_template_source_sha256=sha256_text(tokenizer_ret.chat_template),
            chat_message_payload_sha256=sha256_json(messages),
            prompt_rendering_config=ChatTemplateRenderingConfig(
                message_roles=("user",), add_generation_prompt=True, tokenize=True,
                add_special_tokens_note="delegated to chat_template",
            ),
            prompt_token_count=len(token_ids),
            prompt_token_ids=tuple(token_ids),
        )
        # `run_fullkv_worker` already returns a JSON-serializable dict (its
        # own final `.model_dump(mode="json")` call, not a pydantic object)
        # -- calling `.model_dump()` again here was the actual bug this
        # comment now guards against regressing.
        return run_fullkv_worker(config, manifest, _load_model=lambda: model, _load_tokenizer=lambda: tokenizer)

    return run


def run_qualification_execute(
    config: Any,
    candidate_manifest: dict[str, Any],
    candidate_manifest_path: str,
    *,
    config_hash: str,
    fullkv_runner=None,
) -> QualificationArtifact:
    """The real qualification loop. `fullkv_runner`, when supplied (CPU
    tests only -- production always uses `_default_fullkv_runner`), is a
    `Callable[[dict], dict]` taking one candidate row dict and returning a
    `FullKVWorkerResult`-shaped dict; this keeps the orchestration logic
    (order, stop-at-first-qualified, max 12, schedule prediction,
    eligibility, condition evaluation, immutable artifact construction)
    fully exercisable without CUDA, a model, or a network call."""
    from kvcot.analysis.rkv_schedule import predicted_compaction_event_positions
    from kvcot.utils.hashing import sha256_int_ids

    validate_candidate_manifest_identity(candidate_manifest, config=config)
    runner = fullkv_runner or _default_fullkv_runner(config)

    attempted: list[CandidateQualificationOutcome] = []
    selected: CandidateQualificationOutcome | None = None
    for candidate in candidate_manifest["candidates"][:MAX_CANDIDATES_ATTEMPTED]:
        raw_result = runner(candidate)
        prompt_token_count = raw_result["prompt_token_count"]
        generated_token_ids = raw_result["natural_generated_token_ids"]
        total_len = prompt_token_count + len(generated_token_ids)
        max_position = total_len - 1 if total_len > prompt_token_count else prompt_token_count
        predicted_events = predicted_compaction_event_positions(
            prompt_length=prompt_token_count, max_position=max_position,
            budget=config.rkv.budget, divide_length=config.rkv.divide_length,
        )
        parameter_placement = raw_result.get("parameter_placement") or {}
        # Identity check: the worker's OWN reported identity fields must
        # agree with the config it was launched under AND with the
        # candidate it was asked to run -- never assumed true merely
        # because the same candidate dict was passed in.
        identity_ok = (
            raw_result["dataset_revision"] == candidate["dataset_revision"]
            == config.dataset.revision
            and raw_result["model_revision"] == config.model.revision
            and raw_result["tokenizer_revision"] == config.model.tokenizer_revision
        )
        outcome = build_candidate_outcome(
            candidate_ordinal=candidate["candidate_ordinal"],
            source_example_index=candidate["source_example_index"],
            unique_id=candidate["unique_id"],
            prompt_token_count=prompt_token_count,
            prompt_token_ids_sha256=raw_result["prompt_token_ids_sha256"],
            generated_token_ids=generated_token_ids,
            generated_token_ids_sha256=sha256_int_ids(generated_token_ids),
            cap_hit=raw_result["cap_hit"],
            extracted_answer=raw_result.get("natural_answer"),
            answer_status=raw_result["natural_answer_status"],
            fullkv_wall_seconds=raw_result["wall_seconds"],
            peak_cuda_allocated_bytes=raw_result["peak_cuda_allocated_bytes"],
            peak_cuda_reserved_bytes=raw_result["peak_cuda_reserved_bytes"],
            predicted_event_positions=predicted_events,
            identity_ok=identity_ok,
            batch_size=raw_result["batch_size"],
            every_parameter_on_cuda=raw_result["every_parameter_on_cuda"],
            no_offload_verified=bool(parameter_placement.get("no_offload_verified", raw_result["every_parameter_on_cuda"])),
        )
        attempted.append(outcome)
        if outcome.qualified:
            selected = outcome
            break  # stop immediately at the first qualified row -- never attempt a later candidate

    return QualificationArtifact(
        protocol_version=QUALIFICATION_PROTOCOL_VERSION,
        candidate_manifest_path=candidate_manifest_path,
        candidate_manifest_hash=candidate_manifest["canonical_sha256"],
        config_hash=config_hash,
        dataset_revision=candidate_manifest["dataset_revision"],
        model_revision=candidate_manifest["model_revision"],
        tokenizer_revision=candidate_manifest["tokenizer_revision"],
        budget=candidate_manifest["budget"],
        attempted=attempted,
        selected_ordinal=None if selected is None else selected.candidate_ordinal,
        selected_unique_id=None if selected is None else selected.unique_id,
    )


def validate_candidate_manifest_identity(
    candidate_manifest: dict[str, Any], *, config: Any,
) -> None:
    """Refuses a candidate manifest that does not match the discovery
    config it is being qualified against -- dataset revision, model
    revision, tokenizer revision, and budget must all agree. This is a
    CPU-only, no-CUDA check, safe for `--dry-run`."""
    checks = (
        ("dataset_revision", candidate_manifest["dataset_revision"], config.dataset.revision),
        ("model_revision", candidate_manifest["model_revision"], config.model.revision),
        ("tokenizer_revision", candidate_manifest["tokenizer_revision"], config.model.tokenizer_revision),
        ("budget", candidate_manifest["budget"], config.rkv.budget),
    )
    for name, candidate_value, config_value in checks:
        if candidate_value != config_value:
            raise ManifestPreparationError(
                f"candidate manifest {name}={candidate_value!r} does not match config {name}={config_value!r}"
            )
    candidates = candidate_manifest["candidates"]
    if len(candidates) > MAX_CANDIDATES_ATTEMPTED:
        raise ManifestPreparationError(
            f"candidate manifest has {len(candidates)} candidates, exceeding the "
            f"{MAX_CANDIDATES_ATTEMPTED}-candidate qualification limit"
        )
    seen_ordinals = set()
    for candidate in candidates:
        ordinal = candidate["candidate_ordinal"]
        if ordinal in seen_ordinals:
            raise ManifestPreparationError(f"duplicate candidate_ordinal {ordinal} in candidate manifest")
        seen_ordinals.add(ordinal)
