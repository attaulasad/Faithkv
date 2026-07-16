"""CPU-only tests for the fixed-trace CLI surface: the pure/IO-only helpers
(`_resolve_condition_name`, `_filter_base_records_for_run`) plus `--dry-run`
smoke tests for `replay-fixed-trace`/`analyze-fixed-trace` against the real
`configs/early_gap_b512.yaml`. Mirrors tests/unit/test_cli_helpers.py's style
for the existing `generate`/`replay-probe` helpers.

The dry-run tests never import torch/transformers and never touch a GPU —
that is the whole point of `--dry-run` (kvcot.cli's module docstring). On
this build machine torch is not installed at all, so if the dry-run code
path ever accidentally imported it, these tests would fail with
ModuleNotFoundError rather than silently passing.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from kvcot.cli import (
    _filter_base_records_for_run,
    _resolve_condition_name,
    cmd_analyze_fixed_trace,
    cmd_replay_fixed_trace,
)
from kvcot.config import FixedTraceSettings, StageConfig
from kvcot.utils.io import JsonlWriter

EARLY_GAP_CONFIG = "configs/early_gap_b512.yaml"


def _stage(conditions, **overrides):
    defaults = dict(
        stage_name="early_gap_b512",
        dataset_manifest="data/manifests/gsm8k_calibration_50.jsonl",
        conditions=conditions,
        output_dir="results/raw/early_gap_b512",
        fixed_trace=FixedTraceSettings(),
    )
    defaults.update(overrides)
    return StageConfig(**defaults)


def _write_manifest(path, rows) -> None:
    w = JsonlWriter(path, validator=None)
    for r in rows:
        w.append(r)


# --- _resolve_condition_name ---

def test_resolve_condition_name_passthrough_for_concrete_condition():
    stage = _stage(["full", "rkv_b512"])
    assert _resolve_condition_name(stage, "rkv_b512") == "rkv_b512"
    assert _resolve_condition_name(stage, "full") == "full"


def test_resolve_condition_name_rejects_condition_not_declared_by_stage():
    stage = _stage(["full", "rkv_b512"])
    with pytest.raises(SystemExit):
        _resolve_condition_name(stage, "rkv_b999")


def test_resolve_condition_name_resolves_rkv_selected_placeholder(monkeypatch):
    stage = _stage(["full", "rkv_selected"], stage_name="stage2_main")
    monkeypatch.setattr("kvcot.cli.resolve_conditions", lambda s: ["full", "rkv_b256"])
    assert _resolve_condition_name(stage, "rkv_selected") == "rkv_b256"


def test_resolve_condition_name_used_independently_for_two_args(monkeypatch):
    # replay-fixed-trace resolves --trace-condition and --replay-condition
    # separately through the same function -- exercise both in one call.
    stage = _stage(["full", "rkv_b512"])
    trace_condition = _resolve_condition_name(stage, "full")
    replay_condition = _resolve_condition_name(stage, "rkv_b512")
    assert (trace_condition, replay_condition) == ("full", "rkv_b512")


# --- _filter_base_records_for_run ---

def test_filter_base_records_for_run_applies_limit_and_seed(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            {"source_row_index": 0, "question": "q0", "question_hash": "h0", "normalized_gold": "1"},
            {"source_row_index": 1, "question": "q1", "question_hash": "h1", "normalized_gold": "2"},
            {"source_row_index": 2, "question": "q2", "question_hash": "h2", "normalized_gold": "3"},
        ],
    )
    stage = _stage(["full"], dataset_manifest=str(manifest_path))
    args = SimpleNamespace(problem_index=None, limit=2, seed=42)
    base_records = [
        {"record_id": "b0", "dataset": {"source_row_index": 0}, "global_seed": 42},
        {"record_id": "b1", "dataset": {"source_row_index": 1}, "global_seed": 13},  # wrong seed
        {"record_id": "b2", "dataset": {"source_row_index": 2}, "global_seed": 42},  # excluded by limit=2
    ]
    filtered = _filter_base_records_for_run(stage, args, base_records)
    assert [r["record_id"] for r in filtered] == ["b0"]


def test_filter_base_records_for_run_no_seed_filter_keeps_all_seeds(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest_path,
        [{"source_row_index": 0, "question": "q0", "question_hash": "h0", "normalized_gold": "1"}],
    )
    stage = _stage(["full"], dataset_manifest=str(manifest_path))
    args = SimpleNamespace(problem_index=None, limit=None, seed=None)
    base_records = [
        {"record_id": "b0", "dataset": {"source_row_index": 0}, "global_seed": 13},
        {"record_id": "b1", "dataset": {"source_row_index": 0}, "global_seed": 42},
    ]
    filtered = _filter_base_records_for_run(stage, args, base_records)
    assert {r["record_id"] for r in filtered} == {"b0", "b1"}


def test_filter_base_records_for_run_respects_problem_index(tmp_path):
    manifest_path = tmp_path / "manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            {"source_row_index": 0, "question": "q0", "question_hash": "h0", "normalized_gold": "1"},
            {"source_row_index": 5, "question": "q5", "question_hash": "h5", "normalized_gold": "2"},
        ],
    )
    stage = _stage(["full"], dataset_manifest=str(manifest_path))
    args = SimpleNamespace(problem_index=5, limit=None, seed=None)
    base_records = [
        {"record_id": "b0", "dataset": {"source_row_index": 0}, "global_seed": 42},
        {"record_id": "b5", "dataset": {"source_row_index": 5}, "global_seed": 42},
    ]
    filtered = _filter_base_records_for_run(stage, args, base_records)
    assert [r["record_id"] for r in filtered] == ["b5"]


# --- replay-fixed-trace --dry-run (real config, matches the CPU acceptance checklist) ---

def _fixed_trace_args(**overrides):
    defaults = dict(
        config=EARLY_GAP_CONFIG,
        trace_condition="full",
        replay_condition="full",
        limit=10,
        problem_index=None,
        seed=42,
        resume=False,
        dry_run=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_replay_fixed_trace_dry_run_reports_expected_plan(capsys):
    rc = cmd_replay_fixed_trace(_fixed_trace_args())
    assert rc == 0
    out = capsys.readouterr().out
    assert "trace_condition=full" in out
    assert "replay_condition=full" in out
    assert "planned examples: 10" in out
    assert "fixed-trace fractions per example: 9" in out
    assert "planned probe records: 90" in out
    assert "early_gap_b512" in out  # output path reported, cross-platform-safe substring check


def test_replay_fixed_trace_dry_run_works_for_rkv_replay_condition(capsys):
    rc = cmd_replay_fixed_trace(_fixed_trace_args(replay_condition="rkv_b512"))
    assert rc == 0
    out = capsys.readouterr().out
    assert "replay_condition=rkv_b512" in out
    # Critical rule: still reads the FullKV trace source, never rkv_b512's own base file.
    assert "full.jsonl" in out


def test_replay_fixed_trace_dry_run_rejects_unknown_replay_condition():
    with pytest.raises(SystemExit):
        cmd_replay_fixed_trace(_fixed_trace_args(replay_condition="rkv_b999"))


def test_replay_fixed_trace_dry_run_uses_configs_own_limit_without_cli_limit(capsys):
    # § external review 2026-07-16: configs/early_gap_v2_b128.yaml declares
    # `limit: 10` against a 50-row manifest. Without an explicit `--limit`
    # on the command line, the dry-run plan must reflect the CONFIG's own
    # limit (10 examples, 90 fixed-trace probe records) -- not silently run
    # against all 50 rows the way `stage.limit` being ignored used to.
    rc = cmd_replay_fixed_trace(
        _fixed_trace_args(
            config="configs/early_gap_v2_b128.yaml", replay_condition="rkv_b128", limit=None,
        )
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "planned examples: 10" in out
    assert "planned probe records: 90" in out


def test_replay_fixed_trace_requires_fixed_trace_settings(monkeypatch, capsys):
    # §ステップ2/4: a stage config missing `fixed_trace:` must refuse to run —
    # falling back to the frozen primary probes.* settings would silently let
    # a fixed-trace-motivated change alter the frozen EAS experiment.
    from kvcot.config import load_stage_config as real_load_stage_config

    def _load_without_fixed_trace(path):
        stage, lock = real_load_stage_config(path)
        stage.fixed_trace = None
        return stage, lock

    monkeypatch.setattr("kvcot.cli.load_stage_config", _load_without_fixed_trace)
    with pytest.raises(SystemExit):
        cmd_replay_fixed_trace(_fixed_trace_args())


def test_analyze_fixed_trace_requires_fixed_trace_settings(monkeypatch):
    from kvcot.config import load_stage_config as real_load_stage_config

    def _load_without_fixed_trace(path):
        stage, lock = real_load_stage_config(path)
        stage.fixed_trace = None
        return stage, lock

    monkeypatch.setattr("kvcot.cli.load_stage_config", _load_without_fixed_trace)
    args = SimpleNamespace(
        config=EARLY_GAP_CONFIG, trace_condition="full", replay_condition="rkv_b512", dry_run=True
    )
    with pytest.raises(SystemExit):
        cmd_analyze_fixed_trace(args)


# --- analyze-fixed-trace --dry-run ---

def test_analyze_fixed_trace_dry_run_reports_stage_scoped_output_path(capsys):
    args = SimpleNamespace(
        config=EARLY_GAP_CONFIG, trace_condition="full", replay_condition="rkv_b512", dry_run=True
    )
    rc = cmd_analyze_fixed_trace(args)
    assert rc == 0
    out = capsys.readouterr().out
    # Stage-scoped filename -- must not collide with early_gap_b256/b1024's own decisions.
    assert "early_gap_b512_fixed_trace.json" in out


# --- replay-fixed-trace --resume identity check must cover model/tokenizer
# revision (§ external review 2026-07-16) ---
#
# cmd_replay_fixed_trace's expected_identity dict previously only carried
# config_sha256/upstream_rkv_commit -- a --resume into a fixed-trace probe
# file recorded under a stale model/tokenizer revision was accepted here,
# even though FixedTraceProbeRecord has carried its own model_revision/
# tokenizer_revision fields since schema 1.3.0 specifically so this could be
# checked. The final analysis (kvcot.analysis.fixed_trace) would eventually
# reject the resulting mixed-identity file, but only after wasting GPU time
# generating it.

def _stale_fixed_trace_probe_record(config_sha256: str, upstream_commit: str, **overrides) -> dict:
    from kvcot.schemas import FixedTraceProbeRecord, ProvenanceState, RetentionSummary, VersionInfo

    defaults = dict(
        record_id="fixed-probe-rkv_b512-on-full-base-full-x-0-seed42-f1.0",
        parent_record_id="base-full-x-0-seed42",
        config_path=EARLY_GAP_CONFIG,
        config_sha256=config_sha256,
        provenance=ProvenanceState(upstream_rkv_commit=upstream_commit, git_commit="deadbeef", git_dirty=False),
        versions=VersionInfo(python="3.10.0"),
        model_revision="stale-model-revision",
        tokenizer_revision="stale-tokenizer-revision",
        base_record_id="base-full-x-0-seed42",
        trace_source_condition="full",
        replay_policy_condition="rkv_b512",
        source_row_index=0,
        global_seed=42,
        normalized_gold="42",
        source_base_answer="42",
        source_base_is_correct=True,
        fraction=1.0,
        think_span_length=10,
        cut_index=10,
        close_marker_token_ids=[1],
        control_suffix_token_ids=[2],
        probe_decoding_max_new_tokens=64,
        probe_output_token_ids=[3],
        probe_output_text="42}",
        probe_extraction_text="Final answer: \\boxed{42}",
        normalized_probe_answer="42",
        probe_extraction_status="boxed",
        probe_stop_reason="boxed_answer_complete",
        probe_cap_hit=False,
        replay_retention_at_cut=RetentionSummary(
            fullkv_equivalent_slots=200, physical_cache_slots_per_layer=[100, 100],
            instantaneous_retention_ratio=0.5, post_compaction_budget_tokens=512,
            tokens_since_last_compaction=5,
        ),
        actual_compression_at_cut=True,
        probe_cache_length_final_per_layer=[100, 100],
        probe_actual_eviction_during_answer=False,
        normalized_f1_anchor_answer="42",
        matches_f1_anchor_answer=True,
        f1_anchor_matches_source_base_answer=True,
        f1_anchor_is_correct=True,
        replay_compaction_count_at_cut=1,
        replay_compaction_event_steps_at_cut=[64],
        snapshot_cache_hash="a" * 64,
        snapshot_provenance_hash="b" * 64,
        snapshot_state_hash="c" * 64,
    )
    defaults.update(overrides)
    return FixedTraceProbeRecord(**defaults).model_dump(mode="json")


def _patch_output_dir(monkeypatch, tmp_path):
    from kvcot.config import load_stage_config as real_load_stage_config

    def _load_with_tmp_output_dir(path):
        stage, lock = real_load_stage_config(path)
        stage.output_dir = str(tmp_path)
        return stage, lock

    monkeypatch.setattr("kvcot.cli.load_stage_config", _load_with_tmp_output_dir)


def test_replay_fixed_trace_resume_rejects_model_revision_mismatch(tmp_path, monkeypatch):
    from kvcot.cli import ResumeIdentityMismatchError
    from kvcot.config import config_identity, load_stage_config
    from kvcot.runtime import upstream_submodule_commit
    from kvcot.utils.io import JsonlWriter

    _, lock = load_stage_config(EARLY_GAP_CONFIG)
    config_sha256 = config_identity(EARLY_GAP_CONFIG)
    upstream_commit = upstream_submodule_commit(lock)
    _patch_output_dir(monkeypatch, tmp_path)

    stale_rec = _stale_fixed_trace_probe_record(config_sha256, upstream_commit)
    JsonlWriter(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", validator=None).append(stale_rec)

    args = _fixed_trace_args(replay_condition="rkv_b512", resume=True)
    with pytest.raises(ResumeIdentityMismatchError, match="model_revision"):
        cmd_replay_fixed_trace(args)


def test_replay_fixed_trace_resume_rejects_tokenizer_revision_mismatch(tmp_path, monkeypatch):
    from kvcot.cli import ResumeIdentityMismatchError
    from kvcot.config import config_identity, load_stage_config
    from kvcot.runtime import upstream_submodule_commit
    from kvcot.utils.io import JsonlWriter

    _, lock = load_stage_config(EARLY_GAP_CONFIG)
    config_sha256 = config_identity(EARLY_GAP_CONFIG)
    upstream_commit = upstream_submodule_commit(lock)
    _patch_output_dir(monkeypatch, tmp_path)

    # model_revision matches the current lock -- only tokenizer_revision is stale.
    stale_rec = _stale_fixed_trace_probe_record(
        config_sha256, upstream_commit, model_revision=lock.model.revision,
    )
    JsonlWriter(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", validator=None).append(stale_rec)

    args = _fixed_trace_args(replay_condition="rkv_b512", resume=True)
    with pytest.raises(ResumeIdentityMismatchError, match="tokenizer_revision"):
        cmd_replay_fixed_trace(args)


def test_replay_fixed_trace_resume_accepts_matching_model_and_tokenizer_revision(tmp_path, monkeypatch):
    from kvcot.config import config_identity, load_stage_config
    from kvcot.runtime import upstream_submodule_commit
    from kvcot.utils.io import JsonlWriter

    _, lock = load_stage_config(EARLY_GAP_CONFIG)
    config_sha256 = config_identity(EARLY_GAP_CONFIG)
    upstream_commit = upstream_submodule_commit(lock)
    _patch_output_dir(monkeypatch, tmp_path)

    matching_rec = _stale_fixed_trace_probe_record(
        config_sha256, upstream_commit,
        model_revision=lock.model.revision, tokenizer_revision=lock.model.tokenizer_revision,
    )
    JsonlWriter(tmp_path / "rkv_b512_on_full_fixed_trace_probes.jsonl", validator=None).append(matching_rec)

    args = _fixed_trace_args(replay_condition="rkv_b512", resume=True)
    rc = cmd_replay_fixed_trace(args)
    assert rc == 0
