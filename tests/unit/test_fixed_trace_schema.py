"""Schema tests for kvcot.schemas.FixedTraceProbeRecord — the secondary,
additive record type for the fixed-trace prefix-sufficiency screen. Mirrors
tests/unit/test_schema.py's style/coverage for the existing record types.
"""
import pytest
from pydantic import ValidationError

from kvcot.schemas import (
    FixedTraceProbeRecord,
    ProvenanceState,
    RetentionSummary,
    SCHEMA_VERSION,
    VersionInfo,
)


def _provenance():
    return ProvenanceState(upstream_rkv_commit="45eaa7d69d20b7388321f077020a610d9afb65bd", git_commit="deadbeef", git_dirty=False)


def _versions():
    return VersionInfo(python="3.10.0")


def _retention(*, ratio: float = 0.6) -> RetentionSummary:
    return RetentionSummary(
        fullkv_equivalent_slots=200,
        physical_cache_slots_per_layer=[int(200 * ratio)] * 2,
        instantaneous_retention_ratio=ratio,
        post_compaction_budget_tokens=512,
        tokens_since_last_compaction=10,
    )


def _valid_kwargs(**overrides) -> dict:
    defaults = dict(
        record_id="fixed-probe-rkv_b512-on-full-base-full-ds-0-seed42-f0.5",
        parent_record_id="base-full-ds-0-seed42",
        config_path="configs/early_gap_b512.yaml",
        config_sha256="b" * 64,
        provenance=_provenance(),
        versions=_versions(),
        base_record_id="base-full-ds-0-seed42",
        trace_source_condition="full",
        replay_policy_condition="rkv_b512",
        source_row_index=0,
        global_seed=42,
        normalized_gold="42",
        source_base_answer="42",
        source_base_is_correct=True,
        fraction=0.5,
        think_span_length=100,
        cut_index=50,
        close_marker_token_ids=[151649],
        control_suffix_token_ids=[123, 456],
        probe_decoding_max_new_tokens=64,
        probe_output_token_ids=[9, 9],
        probe_output_text="42}",
        probe_extraction_text="Final answer: \\boxed{42}",
        normalized_probe_answer="42",
        probe_extraction_status="boxed",
        probe_stop_reason="boxed_answer_complete",
        probe_cap_hit=False,
        replay_retention_at_cut=_retention(),
        actual_compression_at_cut=True,
        probe_cache_length_final_per_layer=[130, 130],
        probe_actual_eviction_during_answer=False,
        normalized_f1_anchor_answer="42",
        matches_f1_anchor_answer=True,
        f1_anchor_matches_source_base_answer=True,
        f1_anchor_is_correct=True,
        replay_compaction_count_at_cut=2,
        replay_compaction_event_steps_at_cut=[130, 260],
        snapshot_cache_hash="c" * 64,
        snapshot_provenance_hash="d" * 64,
        snapshot_state_hash="e" * 64,
    )
    defaults.update(overrides)
    return defaults


def test_fixed_trace_probe_record_valid_construction():
    rec = FixedTraceProbeRecord(**_valid_kwargs())
    assert rec.schema_version == SCHEMA_VERSION == "1.2.0"
    assert rec.record_type == "fixed_trace_probe"
    assert rec.anchor_fraction == 1.0
    assert rec.trace_source_condition == "full"
    assert rec.replay_policy_condition == "rkv_b512"


def test_fixed_trace_probe_record_rejects_stale_schema_version():
    with pytest.raises(ValidationError):
        FixedTraceProbeRecord(**_valid_kwargs(schema_version="1.1.0"))


def test_fixed_trace_probe_record_rejects_missing_required_field():
    kwargs = _valid_kwargs()
    del kwargs["snapshot_cache_hash"]
    with pytest.raises(ValidationError):
        FixedTraceProbeRecord(**kwargs)


def test_fixed_trace_probe_record_allows_none_on_extraction_failure():
    rec = FixedTraceProbeRecord(
        **_valid_kwargs(
            normalized_probe_answer=None,
            probe_extraction_status="none",
            matches_f1_anchor_answer=None,
        )
    )
    assert rec.normalized_probe_answer is None
    assert rec.matches_f1_anchor_answer is None


def test_fixed_trace_probe_record_type_literal_cannot_be_overridden_to_another_value():
    with pytest.raises(ValidationError):
        FixedTraceProbeRecord(**_valid_kwargs(record_type="probe"))


def test_trace_source_and_replay_policy_conditions_are_independent_fields():
    # The entire point of this schema vs. ProbeRunRecord: these two may differ.
    rec = FixedTraceProbeRecord(**_valid_kwargs(trace_source_condition="full", replay_policy_condition="full"))
    assert rec.trace_source_condition == rec.replay_policy_condition == "full"
    rec2 = FixedTraceProbeRecord(**_valid_kwargs(trace_source_condition="full", replay_policy_condition="rkv_b256"))
    assert rec2.trace_source_condition != rec2.replay_policy_condition


def test_control_suffix_token_ids_may_be_empty():
    # The fixed-trace suffix is deliberately "" (kvcot.probes.templates.
    # render_fixed_trace_suffix) -- must encode to an empty token list without
    # tripping any non-empty validation ProbeRunRecord never needed.
    rec = FixedTraceProbeRecord(**_valid_kwargs(control_suffix_token_ids=[]))
    assert rec.control_suffix_token_ids == []
