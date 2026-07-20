"""Final B1 execution-boundary contract.

This module is the canonical cross-layer vocabulary for execute-mode timing,
memory, evidence, artifact, and gate validation.  It is intentionally pure
Python: importing it cannot inspect CUDA, load a model, or touch the network.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


FINAL_MANDATORY_GATE_CONDITIONS: tuple[str, ...] = (
    "git_clean_verified",
    "rkv_submodule_match",
    "single_rtx3090_verified",
    "local_model_snapshot_verified",
    "local_tokenizer_snapshot_verified",
    "dataset_row_identity_verified",
    "prompt_identity_verified",
    "fullkv_generation_matches_expected",
    "rkv_generation_matches_expected",
    "workers_generation_match",
    "actual_batch_size_verified",
    "complete_token_trace_match",
    "complete_call_trace_match",
    "complete_compaction_trace_match",
    "capture_gather_parity",
    "absolute_position_parity",
    "selected_event_ids_exact",
    "unique_real_pair_count_exact",
    "events_with_four_unique_pairs_exact",
    "no_duplicate_pair_identity",
    "authorized_no_op_identity_exact",
    "positive_semantic_swap_parity",
    "no_op_exact_parity",
    "all_required_timings_present",
    "all_required_memory_phases_present",
    "runtime_within_limit",
    "peak_vram_within_limit",
    "worker_envelopes_verified",
    "attempt_artifacts_verified",
)

FULLKV_REQUIRED_TIMING_PHASES: tuple[str, ...] = (
    "fullkv_worker_startup",
    "snapshot_tokenizer_resolution",
    "tokenizer_load",
    "model_load",
    "post_load_validation",
    "fullkv_prefill",
    "fullkv_decode",
    "fullkv_complete_natural_generation",
    "answer_verification",
    "fullkv_complete_worker",
)

RKV_REQUIRED_TIMING_PHASES: tuple[str, ...] = (
    "rkv_worker_startup",
    "snapshot_tokenizer_resolution",
    "tokenizer_load",
    "model_load",
    "post_load_validation",
    "rkv_pass1_prefill",
    "rkv_pass1_decode",
    "rkv_complete_pass1",
    "rkv_pass2_prefill",
    "rkv_pass2_decode",
    "capture_and_parity",
    "compact_target_conversion",
    "snapshot_creation",
    "rkv_complete_pass2",
    "rkv_complete_worker",
)

PAIR_REQUIRED_TIMING_SUBPHASES: tuple[str, ...] = (
    "baseline_clone",
    "baseline_restore",
    "baseline_bridge_plus_48_scored_tokens",
    "baseline_release",
    "swapped_clone",
    "semantic_mutation",
    "swapped_restore",
    "swapped_bridge_plus_48_scored_tokens",
    "swapped_release",
    "record_construction",
)

FULLKV_REQUIRED_MEMORY_PHASES: tuple[str, ...] = (
    "before_model_load",
    "model_load",
    "post_load_baseline",
    "fullkv_complete_natural_generation",
    "fullkv_complete_worker",
)

RKV_REQUIRED_MEMORY_PHASES: tuple[str, ...] = (
    "before_model_load",
    "model_load",
    "post_load_baseline",
    "rkv_complete_pass1",
    "rkv_complete_pass2",
    "compact_target_conversion",
    "rkv_complete_worker",
)


@dataclass(frozen=True)
class FinalGateResult:
    passed: bool
    conditions: dict[str, bool]
    failed_conditions: tuple[str, ...]


def evaluate_final_gates(conditions: Mapping[str, Any]) -> FinalGateResult:
    """Reject missing, extra, or non-boolean mandatory evidence."""
    expected = set(FINAL_MANDATORY_GATE_CONDITIONS)
    observed = set(conditions)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"final gate map disagrees with contract: missing={missing}, extra={extra}")
    if any(type(conditions[name]) is not bool for name in FINAL_MANDATORY_GATE_CONDITIONS):
        raise TypeError("every final mandatory gate must be a concrete bool")
    normalized = {name: conditions[name] for name in FINAL_MANDATORY_GATE_CONDITIONS}
    failed = tuple(name for name, passed in normalized.items() if not passed)
    return FinalGateResult(passed=not failed, conditions=normalized, failed_conditions=failed)


def timing_contract_satisfied(fullkv_records: Sequence[Mapping[str, Any]], rkv_records: Sequence[Mapping[str, Any]]) -> bool:
    def valid(record: Mapping[str, Any]) -> bool:
        value = record.get("duration_seconds")
        return (
            record.get("completed") is True
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    full_valid = [record for record in fullkv_records if valid(record)]
    rkv_valid = [record for record in rkv_records if valid(record)]
    full_names = {str(record.get("phase")) for record in full_valid}
    rkv_names = {str(record.get("phase")) for record in rkv_valid}
    if not set(FULLKV_REQUIRED_TIMING_PHASES).issubset(full_names):
        return False
    if not set(RKV_REQUIRED_TIMING_PHASES).issubset(rkv_names):
        return False

    complete_real = [name for name in rkv_names if name.startswith("real_pair:") and name.count(":") == 3]
    complete_noop = [name for name in rkv_names if name.startswith("no_op_pair:") and name.count(":") == 3]
    if len(complete_real) != 12 or len(complete_noop) != 1:
        return False
    for complete_name in complete_real + complete_noop:
        if any(f"{complete_name}:{subphase}" not in rkv_names for subphase in PAIR_REQUIRED_TIMING_SUBPHASES):
            return False
    return True


def memory_contract_satisfied(fullkv_records: Sequence[Mapping[str, Any]], rkv_records: Sequence[Mapping[str, Any]]) -> bool:
    required_fields = {
        "allocated_before", "reserved_before", "peak_allocated", "peak_reserved",
        "allocated_after", "reserved_after", "reset_point", "synchronized_before",
        "synchronized_after", "completed",
    }

    def valid(record: Mapping[str, Any]) -> bool:
        return (
            required_fields.issubset(record)
            and record.get("completed") is True
            and record.get("synchronized_before") is True
            and record.get("synchronized_after") is True
            and all(isinstance(record.get(name), int) and record[name] >= 0 for name in (
                "allocated_before", "reserved_before", "peak_allocated", "peak_reserved",
                "allocated_after", "reserved_after",
            ))
        )

    full_valid = [record for record in fullkv_records if valid(record)]
    rkv_valid = [record for record in rkv_records if valid(record)]
    full_names = {str(record.get("phase")) for record in full_valid}
    rkv_names = {str(record.get("phase")) for record in rkv_valid}
    if not set(FULLKV_REQUIRED_MEMORY_PHASES).issubset(full_names):
        return False
    if not set(RKV_REQUIRED_MEMORY_PHASES).issubset(rkv_names):
        return False
    return (
        len([name for name in rkv_names if name.startswith("real_pair:") and name.count(":") == 3]) == 12
        and len([name for name in rkv_names if name.startswith("no_op_pair:") and name.count(":") == 3]) == 1
    )


def expected_generation_record(config: Any, manifest: Any, eos_token_id: int | None) -> dict[str, Any]:
    """Canonical expected record, reconstructed without worker aggregates."""
    return {
        "generation_mode": config.generation.generation_mode,
        "do_sample": config.generation.do_sample,
        "temperature": config.generation.temperature,
        "top_p": config.generation.top_p,
        "batch_size": config.generation.batch_size,
        "max_new_tokens": config.generation.max_new_tokens,
        "eos_token_id": eos_token_id,
        "eos_append_feed_policy": "eos_never_appended_to_full_token_ids_or_fed_to_the_next_forward_call",
        "one_prefill_policy": "exactly_one_prefill_call_for_the_complete_prompt",
        "single_token_decode_policy": "exactly_one_decode_call_per_generated_non_eos_token",
        "attention_backend": config.generation.attention_backend,
        "cache_implementation": config.generation.cache_implementation,
        "framework_seed": config.generation.framework_seed,
        "prompt_token_count": manifest.prompt_token_count,
    }
