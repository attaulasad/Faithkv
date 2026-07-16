"""Pydantic record schemas (§12 of the build brief). Every JSONL record
written by this repository is validated against one of these models before
it is appended — see kvcot.utils.io.JsonlWriter. See docs/SCHEMA.md for the
field-by-field rationale; this module is the executable source of truth,
that document is the human-readable explanation of it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.2.0"

Condition = str  # "full" | "patched_noop" | f"rkv_b{budget}" — validated by callers against configs, not hardcoded here
ThinkParseStatus = Literal[
    "ok", "no_open_marker", "no_close_marker", "generation_prompt_preopened_ok", "malformed"
]
ExtractionMethod = Literal["boxed", "final_answer_marker", "final_number_fallback", "none"]
ProbeStopReason = Literal["eos", "boxed_answer_complete", "max_new_tokens"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class VersionInfo(BaseModel):
    python: str
    torch: str | None = None
    cuda: str | None = None
    transformers: str | None = None
    flash_attn: str | None = None


class ProvenanceState(BaseModel):
    """Per-record book-keeping about the pinned upstream R-KV, needed to
    tell records apart across code/upstream revisions on resume (§13)."""

    upstream_rkv_commit: str
    git_commit: str
    git_dirty: bool


class DatasetProvenance(BaseModel):
    dataset_name: str
    dataset_config: str | None = None
    dataset_revision: str | None = None
    dataset_fingerprint: str | None = None
    source_row_index: int
    question_hash: str
    normalized_gold: str


class MethodConfig(BaseModel):
    """Configured (not measured) R-KV/FullKV parameters. Realized retention
    is a separate, measured concept — see RetentionSummary. Never conflate
    the two (§9)."""

    method: Literal["fullkv", "patched_noop", "rkv"]
    budget: int | None = None  # None for method == "fullkv"
    window_size: int | None = None
    mix_lambda: float | None = None
    retain_ratio: float | None = None
    retain_direction: Literal["last", "first", "last_percent", "first_percent"] | None = None
    divide_method: Literal["newline", "step_length"] | None = None
    divide_length: int | None = None
    compression_content: Literal["think", "all"] | None = None

    @field_validator("method")
    @classmethod
    def _no_ten_percent_style_naming(cls, v: str) -> str:
        # This validator exists purely as a structural backstop; the real
        # enforcement is the repo-wide grep in
        # tests/unit/test_no_ten_percent_naming.py, since a percent string
        # could appear anywhere (condition name, docs, configs), not just here.
        return v


class RetentionSummary(BaseModel):
    """Measured, not configured (§9). `fullkv_equivalent_slots` is the
    absolute count of tokens processed so far; `physical_cache_slots` is
    what's actually resident post-compaction; their ratio is the realized
    retention at this snapshot — never the static `budget` divided by
    prompt length."""

    fullkv_equivalent_slots: int
    physical_cache_slots_per_layer: list[int]
    instantaneous_retention_ratio: float
    post_compaction_budget_tokens: int | None = None
    tokens_since_last_compaction: int


class ProvenanceRetentionSummary(BaseModel):
    prompt_tokens_total: int
    prompt_tokens_surviving_mean: float
    think_tokens_total: int
    think_tokens_surviving_mean: float
    aggregation_method: str = "mean_over_layers_and_kv_heads"


class ThinkSpanInfo(BaseModel):
    think_start_index: int | None  # index into generated_token_ids, inclusive
    think_end_index: int | None  # index into generated_token_ids, exclusive of the close-marker sequence
    think_parse_status: ThinkParseStatus
    generation_prompt_preopened_think: bool


class BaseRunRecord(BaseModel):
    schema_version: Literal["1.2.0"] = SCHEMA_VERSION
    record_id: str
    parent_record_id: str | None = None
    record_type: Literal["base_generation"] = "base_generation"
    timestamp_utc: str = Field(default_factory=utc_now_iso)

    config_path: str
    config_sha256: str
    provenance: ProvenanceState
    versions: VersionInfo
    gpu_model: str | None = None

    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str

    dataset: DatasetProvenance

    condition: Condition
    method_config: MethodConfig

    global_seed: int
    derived_seed: int

    prompt_text: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    decoded_output: str

    think_span: ThinkSpanInfo

    extracted_answer: str | None
    extraction_method: ExtractionMethod
    extraction_failure_reason: str | None = None
    is_correct: bool | None  # None iff extraction failed (never coerced to False)
    cap_hit: bool

    wall_time_seconds: float
    generated_token_count: int
    peak_vram_bytes: int | None = None

    compaction_count: int
    compaction_event_steps: list[int]
    cache_length_final_per_layer: list[int]
    retention: RetentionSummary | None = None  # None for condition == "full"
    provenance_retention: ProvenanceRetentionSummary | None = None

    replay_state_hash: str | None = None  # filled in once replay validates this base run


class ProbeRunRecord(BaseModel):
    schema_version: Literal["1.2.0"] = SCHEMA_VERSION
    record_id: str
    parent_record_id: str  # the base_record_id this probe branched from
    record_type: Literal["probe"] = "probe"
    timestamp_utc: str = Field(default_factory=utc_now_iso)

    config_path: str
    config_sha256: str
    provenance: ProvenanceState
    versions: VersionInfo

    base_record_id: str
    condition: Condition
    fraction: float  # one of the frozen probe fractions, §4
    think_span_length: int = Field(description="L: number of tokens strictly inside the think span")
    cut_index: int = Field(description="floor(fraction * L)")

    control_suffix_token_ids: list[int]
    probe_decoding_max_new_tokens: int

    probe_output_token_ids: list[int]
    probe_output_text: str
    normalized_probe_answer: str | None
    probe_extraction_status: ExtractionMethod

    matches_own_condition_base_answer: bool | None  # None iff either side failed extraction
    is_f1_stability_probe: bool

    snapshot_cache_hash: str
    snapshot_provenance_hash: str
    snapshot_state_hash: str


class FixedTraceProbeRecord(BaseModel):
    """Secondary, additive diagnostic — NOT the frozen primary record type.

    `ProbeRunRecord` has one `condition` field because its probe always
    branches from that same condition's own generated trace. This schema
    exists because the fixed-trace design intentionally decouples those two
    things: `trace_source_condition` is always the condition whose tokens
    were teacher-forced (always "full" in practice — see
    kvcot.cli.cmd_replay_fixed_trace's critical rule), `replay_policy_condition`
    is the cache policy actually applied while replaying them. Matching is
    against `normalized_f1_anchor_answer` (this replay policy's own greedy
    f=1 answer), never `source_base_answer` (the trace source's SAMPLED
    natural answer) — see kvcot.analysis.fixed_trace's module docstring for
    why that distinction is the entire point of this record type.
    """

    schema_version: Literal["1.2.0"] = SCHEMA_VERSION
    record_id: str
    parent_record_id: str  # the base_record_id this probe branched from
    record_type: Literal["fixed_trace_probe"] = "fixed_trace_probe"
    timestamp_utc: str = Field(default_factory=utc_now_iso)

    config_path: str
    config_sha256: str
    provenance: ProvenanceState
    versions: VersionInfo

    # Canonical trace identity
    base_record_id: str
    trace_source_condition: Condition
    replay_policy_condition: Condition
    source_row_index: int
    global_seed: int

    # Problem references
    normalized_gold: str
    source_base_answer: str | None  # the trace source's own SAMPLED answer — diagnostic only, never the match target
    source_base_is_correct: bool | None

    # Probe position
    fraction: float
    think_span_length: int = Field(description="L: number of tokens strictly inside the think span")
    cut_index: int = Field(description="floor(fraction * L)")

    # Probe generation
    close_marker_token_ids: list[int]
    # Protocol v2 (2026-07-16): NO LONGER always [] — the fixed-trace suffix
    # is now a teacher-forced boxed-answer format prefix (§ kvcot.probes.
    # templates.FIXED_TRACE_SUFFIX_TEXT), not the empty string. Historic
    # protocol-v1 records (schema_version < 1.2.0) had this always empty.
    control_suffix_token_ids: list[int]
    probe_decoding_max_new_tokens: int
    probe_output_token_ids: list[int]
    probe_output_text: str  # decoded from probe_output_token_ids alone (no prefix)
    # Decoded from control_suffix_token_ids + probe_output_token_ids — the
    # text actually passed to kvcot.utils.answers.extract_answer, since the
    # boxed-answer prefix supplies the opening `\boxed{` that the generated
    # tokens alone do not contain (kvcot.cli.cmd_replay_fixed_trace).
    probe_extraction_text: str
    normalized_probe_answer: str | None
    probe_extraction_status: ExtractionMethod

    # Why greedy answer decoding actually stopped for this probe
    # (kvcot.generation.replay.ProbeResult.stop_reason). A fixed-trace probe
    # must never rely on "max_new_tokens" to have produced a complete boxed
    # answer — "boxed_answer_complete" (via has_complete_boxed_answer as the
    # stop_predicate) is the expected normal case; "max_new_tokens" without a
    # complete box is `probe_cap_hit`.
    probe_stop_reason: ProbeStopReason
    probe_cap_hit: bool  # True iff probe_stop_reason == "max_new_tokens"

    # Realized (measured, never configured — §9) retention AT THE SNAPSHOT
    # this probe branched from, under replay_policy_condition. A recorded
    # replay_compaction_count_at_cut > 0 is NOT sufficient evidence of actual
    # compression: at the exact budget boundary R-KV can record a compaction
    # event that evicts zero tokens (kvcot.generation.replay._sync_layer_
    # after_call's documented boundary case). actual_compression_at_cut is
    # the physical-cache-shrank check that event counts alone cannot give.
    replay_retention_at_cut: RetentionSummary
    actual_compression_at_cut: bool

    # Cache state AFTER this probe's own greedy answer decoding (not just at
    # the snapshot it branched from) — needed to detect a further eviction
    # that happened WHILE writing the answer, which the reasoning-cut
    # snapshot above cannot show.
    probe_cache_length_final_per_layer: list[int]
    probe_actual_eviction_during_answer: bool

    # f=1 anchored measurement (this replay policy's own greedy f=1 answer —
    # the metric target every fraction, including f=1 itself, is matched
    # against; NOT the trace source's sampled natural answer)
    anchor_fraction: float = 1.0
    normalized_f1_anchor_answer: str | None
    matches_f1_anchor_answer: bool | None  # None iff either side failed extraction
    f1_anchor_matches_source_base_answer: bool | None  # diagnostic only — never used for eligibility or the metric
    f1_anchor_is_correct: bool | None

    # Compression observed during THIS replay (under replay_policy_condition),
    # never the trace source's own generation-time compaction count — the
    # trace source is always FullKV in practice, which never compacts.
    replay_compaction_count_at_cut: int
    replay_compaction_event_steps_at_cut: list[int]

    snapshot_cache_hash: str
    snapshot_provenance_hash: str
    snapshot_state_hash: str


class RunManifest(BaseModel):
    schema_version: Literal["1.2.0"] = SCHEMA_VERSION
    command: str
    config_path: str
    config_sha256: str
    git_commit: str
    git_dirty: bool
    versions: VersionInfo

    start_time_utc: str
    end_time_utc: str | None = None

    n_attempted: int = 0
    n_completed: int = 0
    n_skipped_resumed: int = 0
    n_failed: int = 0

    total_generated_tokens: int = 0
    n_cap_hits: int = 0
    n_think_parse_failures: int = 0
    n_extraction_failures: int = 0
    total_compaction_events: int = 0

    wall_time_seconds: float | None = None
    peak_vram_bytes: int | None = None
