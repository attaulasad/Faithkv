"""CPU-only tests for `kvcot inspect-fixed-trace --write-selection`
(protocol v3, CHANGELOG.md 2026-07-17): deterministic, outcome-blind trace
selection using kvcot.analysis.rkv_schedule's predicted retention. Never
imports torch/transformers.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from kvcot.cli import cmd_inspect_fixed_trace
from kvcot.config import FixedTraceSettings, StageConfig, load_lock_config
from kvcot.utils.io import JsonlWriter, read_json

_LOCK_PATH = str(Path(__file__).resolve().parents[2] / "configs" / "lock.yaml")


def _base_row(prompt_len: int, think_len: int, *, idx: int, seed: int = 42, is_correct: bool = True, cap_hit: bool = False, parse_ok: bool = True) -> dict:
    return {
        "record_id": f"base-full-x-{idx}",
        "dataset": {"source_row_index": idx},
        "global_seed": seed,
        "is_correct": is_correct,
        "cap_hit": cap_hit,
        "prompt_token_ids": list(range(prompt_len)),
        "think_span": {
            "think_parse_status": "generation_prompt_preopened_ok" if parse_ok else "malformed",
            "think_start_index": 0,
            "think_end_index": think_len,
            "generation_prompt_preopened_think": True,
        },
    }


def _write_base_file(path, rows: list[dict]) -> None:
    w = JsonlWriter(path, validator=None)
    for r in rows:
        w.append(r)


def _args(config, tmp_path):
    return SimpleNamespace(
        config=config, trace_condition="full", dry_run=False,
        write_selection=True, max_selected=None,
    )


def _stage(tmp_path, budget: int, **overrides):
    settings = FixedTraceSettings(
        min_actual_compression_rate=0.0, max_mean_f1_retention_ratio=1.0,
        meaningful_retention_ceiling=overrides.pop("meaningful_retention_ceiling", 0.7),
        min_meaningfully_compressed_scored_fractions=overrides.pop("min_meaningfully_compressed_scored_fractions", 0),
    )
    return StageConfig(
        stage_name="test_v3_selection",
        dataset_manifest="data/manifests/gsm8k_calibration_50.jsonl",
        conditions=["full", f"rkv_b{budget}"],
        rkv_budgets=[budget],
        output_dir=str(tmp_path),
        fixed_trace=settings,
    )


def test_write_selection_writes_json_with_predicted_retention(tmp_path, monkeypatch):
    # 300-token trace, budget=128 -- well above budget, should predict low
    # f=1 retention and be selected.
    rows = [_base_row(prompt_len=50, think_len=250, idx=0)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    rc = cmd_inspect_fixed_trace(_args(_LOCK_PATH, tmp_path))
    assert rc == 0

    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert selection["n_ranked"] == 1
    assert selection["n_selected"] == 1
    assert selection["selected_source_row_indices"] == [0]
    assert selection["candidates"][0]["predicted_retention_by_fraction"]["1.0"] < 1.0


def test_write_selection_rejects_incorrect_capped_or_unparsed_base_records(tmp_path, monkeypatch):
    rows = [
        _base_row(prompt_len=50, think_len=250, idx=0, is_correct=False),
        _base_row(prompt_len=50, think_len=250, idx=1, cap_hit=True),
        _base_row(prompt_len=50, think_len=250, idx=2, parse_ok=False),
        _base_row(prompt_len=50, think_len=250, idx=3),  # the only valid one
    ]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    cmd_inspect_fixed_trace(_args(_LOCK_PATH, tmp_path))
    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert selection["n_rejected_invalid_base"] == 3
    assert selection["n_ranked"] == 1
    assert selection["selected_source_row_indices"] == [3]


def test_write_selection_rejects_traces_that_never_reach_meaningful_retention(tmp_path, monkeypatch):
    # 100-token trace, budget=90 (exceeds budget so the basic preflight gate
    # does not itself stop the command) but divide_length=128 (lock.yaml) is
    # never reached within a 100-token trace, so the compression schedule
    # never fires even once -- predicted retention stays 1.0 at every
    # fraction, never clears the ceiling.
    rows = [_base_row(prompt_len=50, think_len=50, idx=0)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=90)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    cmd_inspect_fixed_trace(_args(_LOCK_PATH, tmp_path))
    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert selection["n_selected"] == 0
    assert selection["candidates"][0]["predicted_eligible"] is False


def test_write_selection_truncation_uses_only_source_row_index_never_predicted_values(tmp_path, monkeypatch):
    rows = [_base_row(prompt_len=50, think_len=250, idx=i) for i in (5, 3, 1, 4, 2, 0)]  # out of order
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    args = _args(_LOCK_PATH, tmp_path)
    args.max_selected = 3
    cmd_inspect_fixed_trace(args)
    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    # Sorted by source_row_index, then truncated to the first 3 -- 0, 1, 2 --
    # regardless of the order rows appeared in the file.
    assert [c["source_row_index"] for c in selection["candidates"]] == [0, 1, 2]


def test_write_selection_is_deterministic_across_repeated_calls(tmp_path, monkeypatch):
    rows = [_base_row(prompt_len=50, think_len=250, idx=i) for i in range(5)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    cmd_inspect_fixed_trace(_args(_LOCK_PATH, tmp_path))
    first = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    cmd_inspect_fixed_trace(_args(_LOCK_PATH, tmp_path))
    second = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert first["selected_source_row_indices"] == second["selected_source_row_indices"]
    assert first["candidates"] == second["candidates"]


def test_write_selection_requires_fixed_trace_settings(tmp_path, monkeypatch):
    rows = [_base_row(prompt_len=50, think_len=250, idx=0)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = StageConfig(
        stage_name="test_stage", dataset_manifest="data/manifests/gsm8k_calibration_50.jsonl",
        conditions=["full", "rkv_b128"], rkv_budgets=[128], output_dir=str(tmp_path),
    )
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))

    try:
        cmd_inspect_fixed_trace(_args(_LOCK_PATH, tmp_path))
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "fixed_trace" in str(e)
