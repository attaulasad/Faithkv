"""CPU-only tests for `kvcot inspect-fixed-trace` — the trace-length
preflight (§ Step 16, strengthened 2026-07-16 per external review). Never
imports torch/transformers; reads plain JSONL fixtures the same way the
command itself does.
"""
from __future__ import annotations

from types import SimpleNamespace

from kvcot.cli import cmd_inspect_fixed_trace
from kvcot.config import FixedTraceSettings, StageConfig
from kvcot.utils.io import JsonlWriter


def _base_row(prompt_len: int, think_len: int, *, idx: int = 0, is_correct: bool = True, cap_hit: bool = False) -> dict:
    return {
        "record_id": f"base-full-x-{prompt_len}-{think_len}-{idx}",
        "is_correct": is_correct,
        "cap_hit": cap_hit,
        "prompt_token_ids": list(range(prompt_len)),
        "think_span": {
            "think_parse_status": "generation_prompt_preopened_ok",
            "think_start_index": 0,
            "think_end_index": think_len,
        },
    }


def _write_base_file(path, rows: list[dict]) -> None:
    w = JsonlWriter(path, validator=None)
    for i, r in enumerate(rows):
        r = dict(r)
        r["record_id"] = f"{r['record_id']}-row{i}"
        w.append(r)


def _args(config="unused.yaml", trace_condition="full", dry_run=False):
    return SimpleNamespace(config=config, trace_condition=trace_condition, dry_run=dry_run)


def _stage(tmp_path, budget: int, **fixed_trace_overrides):
    settings = FixedTraceSettings(
        min_actual_compression_rate=fixed_trace_overrides.pop("min_actual_compression_rate", 0.7),
        max_mean_f1_retention_ratio=fixed_trace_overrides.pop("max_mean_f1_retention_ratio", 0.7),
    )
    return StageConfig(
        stage_name="test_stage",
        dataset_manifest="data/manifests/gsm8k_calibration_50.jsonl",
        conditions=["full", f"rkv_b{budget}"],
        rkv_budgets=[budget],
        output_dir=str(tmp_path),
        fixed_trace=settings,
    )


def test_stops_when_no_trace_exceeds_budget(tmp_path, monkeypatch, capsys):
    # 10 traces, all prompt+think ~= 300 tokens, budget=512 -- none exceed.
    rows = [_base_row(prompt_len=100, think_len=200) for _ in range(10)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=512)
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, None))

    rc = cmd_inspect_fixed_trace(_args())
    out = capsys.readouterr().out
    assert rc == 1
    assert "no trace in this file is longer than the configured" in out


def test_stops_when_compression_rate_upper_bound_below_threshold(tmp_path, monkeypatch, capsys):
    # 10 traces at ~300 tokens; budget=256 -- only some exceed (upper bound
    # on achievable actual_compression_rate is well below the 0.7 default).
    rows = [_base_row(prompt_len=100, think_len=200) for _ in range(4)]  # 300 tokens, exceeds 256
    rows += [_base_row(prompt_len=50, think_len=100) for _ in range(6)]  # 150 tokens, does not exceed 256
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=256, min_actual_compression_rate=0.7)
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, None))

    rc = cmd_inspect_fixed_trace(_args())
    out = capsys.readouterr().out
    assert rc == 1
    assert "mathematically unreachable" in out


def test_stops_when_optimistic_retention_cannot_clear_threshold(tmp_path, monkeypatch, capsys):
    # All 10 traces exceed the budget (upper bound on compression rate is
    # 1.0, clears min_actual_compression_rate easily), but the budget is
    # still so large relative to trace length that even a maximally
    # aggressive compaction (budget / length) cannot bring mean retention
    # under the configured ceiling.
    rows = [_base_row(prompt_len=50, think_len=250) for _ in range(10)]  # 300 tokens each
    _write_base_file(tmp_path / "full.jsonl", rows)
    # budget=256: 256/300 ~= 0.853, well above max_mean_f1_retention_ratio=0.7,
    # and every trace exceeds the budget so the compression-rate check passes.
    stage = _stage(tmp_path, budget=256, min_actual_compression_rate=0.5, max_mean_f1_retention_ratio=0.7)
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, None))

    rc = cmd_inspect_fixed_trace(_args())
    out = capsys.readouterr().out
    assert rc == 1
    assert "even the best-case" in out


def test_passes_when_budget_can_plausibly_clear_both_thresholds(tmp_path, monkeypatch, capsys):
    # All 10 traces exceed the budget by a wide margin (budget/length small
    # enough that even the optimistic bound clears the retention ceiling).
    rows = [_base_row(prompt_len=50, think_len=250) for _ in range(10)]  # 300 tokens each
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128, min_actual_compression_rate=0.7, max_mean_f1_retention_ratio=0.7)
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, None))

    rc = cmd_inspect_fixed_trace(_args())
    assert rc == 0


def test_no_fixed_trace_settings_only_runs_the_basic_check(tmp_path, monkeypatch, capsys):
    # Without stage.fixed_trace, the threshold-aware checks are skipped
    # entirely -- only the absolute "nothing exceeds the budget" check runs.
    rows = [_base_row(prompt_len=100, think_len=200) for _ in range(4)]
    rows += [_base_row(prompt_len=50, think_len=100) for _ in range(6)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = StageConfig(
        stage_name="test_stage",
        dataset_manifest="data/manifests/gsm8k_calibration_50.jsonl",
        conditions=["full", "rkv_b256"],
        rkv_budgets=[256],
        output_dir=str(tmp_path),
    )
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, None))

    rc = cmd_inspect_fixed_trace(_args())
    # 4/10 traces exceed budget=256, so the basic "zero traces exceed" gate
    # does not fire -- and with no fixed_trace settings there is nothing
    # else to check against.
    assert rc == 0
