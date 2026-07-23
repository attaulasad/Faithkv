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
    # F7 (final independent-audit repair): a dedicated mandatory placement
    # condition -- explicit requested-device identity plus complete
    # CPU/disk/meta/offload rejection from BOTH workers' raw parameter-
    # placement evidence (`kvcot.discovery.strict_device
    # .verify_placement_from_raw_evidence`), never only the legacy
    # measurement-level `every_parameter_on_cuda` field.
    "no_offload_and_placement_verified",
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
    # Independent-audit Gate H2 repair: this phase used to be named
    # `capture_and_parity` despite only ever timing the POST-Pass-2 call-
    # boundary trace comparison (`compare_call_boundary_traces`), never the
    # actual target capture/gather/parity work. Renamed to match what it
    # actually measures.
    "call_trace_comparison",
    # Independent-audit Gate H2.2 repair: the REAL target capture gather +
    # gather-parity + absolute-position-parity computation
    # (`kvcot.discovery.capture.capture_update_kv`'s wrapped
    # `_build_capture_record` call, threaded via `capture_timer_fn` through
    # `run_pass2_capture`/`run_example`) now has its own accurately-named,
    # synchronized timing boundary -- fires once per selected target (3
    # times per example), nested inside whichever `rkv_pass2_prefill`/
    # `rkv_pass2_decode` call it occurs within (never double-counted in
    # projection, since neither of those two phases' recorded durations is
    # itself summed into `runtime_projection` -- only `rkv_complete_pass2`
    # is).
    "capture_gather_and_parity",
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


# F9 (final independent-audit repair): exact multiplicity rules -- a
# phase-name SET check tolerated duplicated singleton phases (two
# `model_load` records satisfied "model_load present"). Every entry below
# is the exact number of valid completed records that phase must have.
FULLKV_TIMING_EXACT_MULTIPLICITY: dict[str, int] = {
    "fullkv_worker_startup": 1,
    "snapshot_tokenizer_resolution": 1,
    "tokenizer_load": 1,
    "model_load": 1,
    "post_load_validation": 1,
    "fullkv_prefill": 1,
    "fullkv_complete_natural_generation": 1,
    "fullkv_complete_worker": 1,
}

_FULLKV_QUALIFICATION_SINGLETON_PHASES: tuple[str, ...] = (
    "before_model_load",
    "fullkv_worker_startup",
    "snapshot_tokenizer_resolution",
    "tokenizer_load",
    "model_load",
    "post_load_validation",
    "post_load_baseline",
    "fullkv_prefill",
    "answer_verification",
    "fullkv_complete_natural_generation",
    "fullkv_complete_worker",
)
_FULLKV_QUALIFICATION_PHASE_ORDER: tuple[str, ...] = (
    "before_model_load",
    "fullkv_worker_startup",
    "snapshot_tokenizer_resolution",
    "tokenizer_load",
    "model_load",
    "post_load_validation",
    "post_load_baseline",
    "fullkv_prefill",
    "fullkv_decode",
    "answer_verification",
    "fullkv_complete_natural_generation",
    "fullkv_complete_worker",
)


def fullkv_qualification_timing_complete(
    records: Sequence[Mapping[str, Any]], *, fullkv_wall_seconds: Any
) -> bool:
    """Strictly validate the timing list emitted by ``run_fullkv_worker``.

    This is a qualification-only semantic check over the existing B1
    vocabulary.  It does not alter ``timing_contract_satisfied`` and thus
    leaves historical B2A-R1/R2 verification behavior unchanged.
    """
    if type(fullkv_wall_seconds) not in (int, float) or not math.isfinite(float(fullkv_wall_seconds)):
        return False
    if fullkv_wall_seconds < 0:
        return False
    required_fields = {
        "phase",
        "started_at",
        "ended_at",
        "duration_seconds",
        "synchronize_before_start",
        "synchronize_before_end",
        "completed",
        "failure_type",
        "failure_message",
    }
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)) or not records:
        return False

    phase_names: list[str] = []
    counts: dict[str, int] = {}
    allowed = set(_FULLKV_QUALIFICATION_PHASE_ORDER)
    natural_duration: float | None = None
    for record in records:
        if not isinstance(record, Mapping) or set(record) != required_fields:
            return False
        phase = record.get("phase")
        if not isinstance(phase, str) or phase not in allowed:
            return False
        for name in ("started_at", "ended_at", "duration_seconds"):
            value = record.get(name)
            if type(value) not in (int, float) or not math.isfinite(float(value)) or value < 0:
                return False
        started = record["started_at"]
        ended = record["ended_at"]
        duration = record["duration_seconds"]
        if ended < started or duration != ended - started:
            return False
        if record.get("synchronize_before_start") is not True:
            return False
        if record.get("synchronize_before_end") is not True:
            return False
        if record.get("completed") is not True:
            return False
        if record.get("failure_type") is not None or record.get("failure_message") is not None:
            return False
        phase_names.append(phase)
        counts[phase] = counts.get(phase, 0) + 1
        if phase == "fullkv_complete_natural_generation":
            natural_duration = float(duration)

    if any(counts.get(phase, 0) != 1 for phase in _FULLKV_QUALIFICATION_SINGLETON_PHASES):
        return False
    if counts.get("fullkv_decode", 0) < 1:
        return False
    expected_order = list(_FULLKV_QUALIFICATION_PHASE_ORDER[:8])
    expected_order.extend(["fullkv_decode"] * counts["fullkv_decode"])
    expected_order.extend(_FULLKV_QUALIFICATION_PHASE_ORDER[9:])
    if phase_names != expected_order:
        return False
    return natural_duration == float(fullkv_wall_seconds)
RKV_TIMING_EXACT_MULTIPLICITY: dict[str, int] = {
    "rkv_worker_startup": 1,
    "snapshot_tokenizer_resolution": 1,
    "tokenizer_load": 1,
    "model_load": 1,
    "post_load_validation": 1,
    "rkv_pass1_prefill": 1,
    "rkv_pass2_prefill": 1,
    "rkv_complete_pass1": 1,
    "rkv_complete_pass2": 1,
    "call_trace_comparison": 1,
    "capture_gather_and_parity": 3,
    "snapshot_creation": 3,
    "compact_target_conversion": 1,
    "rkv_complete_worker": 1,
}
FULLKV_MEMORY_EXACT_MULTIPLICITY: dict[str, int] = {
    "before_model_load": 1,
    "model_load": 1,
    "post_load_baseline": 1,
    "fullkv_complete_natural_generation": 1,
    "fullkv_complete_worker": 1,
}
RKV_MEMORY_EXACT_MULTIPLICITY: dict[str, int] = {
    "before_model_load": 1,
    "model_load": 1,
    "post_load_baseline": 1,
    "rkv_complete_pass1": 1,
    "rkv_complete_pass2": 1,
    "compact_target_conversion": 1,
    "rkv_complete_worker": 1,
}


def _valid_timing(record: Mapping[str, Any]) -> bool:
    value = record.get("duration_seconds")
    return (
        record.get("completed") is True
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _phase_counts(records: Sequence[Mapping[str, Any]], valid) -> dict[str, int] | None:
    """Counts of VALID records per phase -- `None` (reject) when any record
    for a contract-relevant phase is present but invalid (a failed record
    must never silently vanish and let the name-set look complete)."""
    counts: dict[str, int] = {}
    for record in records:
        name = str(record.get("phase"))
        if not valid(record):
            return None
        counts[name] = counts.get(name, 0) + 1
    return counts


def _pair_phase_multiplicities_ok(counts: dict[str, int]) -> bool:
    complete_real = [name for name in counts if name.startswith("real_pair:") and name.count(":") == 3]
    complete_noop = [name for name in counts if name.startswith("no_op_pair:") and name.count(":") == 3]
    if len(complete_real) != 12 or len(complete_noop) != 1:
        return False
    for complete_name in complete_real + complete_noop:
        if counts[complete_name] != 1:
            return False
        for subphase in PAIR_REQUIRED_TIMING_SUBPHASES:
            if counts.get(f"{complete_name}:{subphase}") != 1:
                return False
    return True


def timing_contract_satisfied(
    fullkv_records: Sequence[Mapping[str, Any]],
    rkv_records: Sequence[Mapping[str, Any]],
    *,
    fullkv_actual_calls: Sequence[Mapping[str, Any]] | None = None,
    rkv_actual_calls: Sequence[Mapping[str, Any]] | None = None,
) -> bool:
    """Exact-multiplicity timing contract (F9). Rejects a missing phase, a
    duplicate singleton, a duplicate pair identity, a wrong capture count,
    a non-finite/zero/negative duration, and a failed record masquerading
    as completed. When actual model-call evidence is supplied, per-call
    decode/prefill record counts must agree with it rather than any
    hard-coded number (R-KV branch steps also record `decode` actual-call
    events outside the pass loops, so the R-KV decode bound is a floor)."""
    full_counts = _phase_counts(fullkv_records, _valid_timing)
    rkv_counts = _phase_counts(rkv_records, _valid_timing)
    if full_counts is None or rkv_counts is None:
        return False
    for phase, expected in FULLKV_TIMING_EXACT_MULTIPLICITY.items():
        if full_counts.get(phase, 0) != expected:
            return False
    for phase, expected in RKV_TIMING_EXACT_MULTIPLICITY.items():
        if rkv_counts.get(phase, 0) != expected:
            return False
    if full_counts.get("fullkv_decode", 0) < 1 or full_counts.get("answer_verification", 0) < 1:
        return False
    if rkv_counts.get("rkv_pass1_decode", 0) < 1 or rkv_counts.get("rkv_pass2_decode", 0) < 1:
        return False
    if rkv_counts.get("answer_verification", 0) < 1:
        return False
    if not _pair_phase_multiplicities_ok(rkv_counts):
        return False

    if fullkv_actual_calls is not None:
        actual_prefill = sum(1 for event in fullkv_actual_calls if event.get("call_kind") == "prefill")
        actual_decode = sum(1 for event in fullkv_actual_calls if event.get("call_kind") == "decode")
        if full_counts.get("fullkv_prefill", 0) != actual_prefill:
            return False
        if full_counts.get("fullkv_decode", 0) != actual_decode:
            return False
    if rkv_actual_calls is not None:
        actual_prefill = sum(1 for event in rkv_actual_calls if event.get("call_kind") == "prefill")
        actual_decode = sum(1 for event in rkv_actual_calls if event.get("call_kind") == "decode")
        if rkv_counts.get("rkv_pass1_prefill", 0) + rkv_counts.get("rkv_pass2_prefill", 0) != actual_prefill:
            return False
        # Branch-step decodes appear in actual-call evidence but not as
        # per-call pass-loop timing records -- the pass-loop decode records
        # can never exceed the raw actual decode count.
        if rkv_counts.get("rkv_pass1_decode", 0) + rkv_counts.get("rkv_pass2_decode", 0) > actual_decode:
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

    full_counts = _phase_counts(fullkv_records, valid)
    rkv_counts = _phase_counts(rkv_records, valid)
    if full_counts is None or rkv_counts is None:
        return False
    for phase, expected in FULLKV_MEMORY_EXACT_MULTIPLICITY.items():
        if full_counts.get(phase, 0) != expected:
            return False
    for phase, expected in RKV_MEMORY_EXACT_MULTIPLICITY.items():
        if rkv_counts.get(phase, 0) != expected:
            return False
    real_pairs = {name: count for name, count in rkv_counts.items() if name.startswith("real_pair:") and name.count(":") == 3}
    noop_pairs = {name: count for name, count in rkv_counts.items() if name.startswith("no_op_pair:") and name.count(":") == 3}
    return (
        len(real_pairs) == 12
        and all(count == 1 for count in real_pairs.values())
        and len(noop_pairs) == 1
        and all(count == 1 for count in noop_pairs.values())
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
