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
