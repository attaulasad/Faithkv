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
        min_eligible_examples=overrides.pop("min_eligible_examples", 1),
        max_selected_examples=overrides.pop("max_selected_examples", None),
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
    # `candidates` holds every ranked candidate, uncapped (2026-07-18 fix) --
    # only the SELECTED subset is capped, sorted by source_row_index -- 0, 1,
    # 2 -- regardless of the order rows appeared in the file.
    assert [c["source_row_index"] for c in selection["candidates"]] == [0, 1, 2, 3, 4, 5]
    assert selection["selected_source_row_indices"] == [0, 1, 2]
    assert selection["n_ranked"] == 6
    assert selection["n_selected"] == 3


def test_write_selection_caps_eligible_set_not_the_full_ranked_list(tmp_path, monkeypatch):
    # Regression for the exact bug the 2026-07-18 review found: rows 0-2 are
    # short (never exceed budget=128, predicted_eligible=False); rows 3-5 are
    # long (well above budget, predicted_eligible=True). With max_selected=2,
    # the BUGGY behavior (cap the full source_row_index-sorted list BEFORE
    # filtering to eligible) would keep only rows [0, 1] -- both ineligible,
    # n_selected=0 -- even though two perfectly good eligible candidates (3,
    # 4) exist. The FIXED behavior must select [3, 4].
    rows = [_base_row(prompt_len=10, think_len=10, idx=i) for i in (0, 1, 2)]
    rows += [_base_row(prompt_len=50, think_len=250, idx=i) for i in (3, 4, 5)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    args = _args(_LOCK_PATH, tmp_path)
    args.max_selected = 2
    cmd_inspect_fixed_trace(args)
    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert selection["n_ranked"] == 6
    assert selection["n_predicted_eligible"] == 3  # rows 3, 4, 5
    assert selection["n_selected"] == 2
    assert selection["selected_source_row_indices"] == [3, 4]


def test_write_selection_returns_exit_1_when_below_min_eligible_examples(tmp_path, monkeypatch):
    rows = [_base_row(prompt_len=50, think_len=250, idx=0)]  # only 1 candidate
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128, min_eligible_examples=5)  # need 5, only 1 exists
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    rc = cmd_inspect_fixed_trace(_args(_LOCK_PATH, tmp_path))
    assert rc == 1


def test_write_selection_config_max_selected_examples_used_when_cli_flag_omitted(tmp_path, monkeypatch):
    rows = [_base_row(prompt_len=50, think_len=250, idx=i) for i in range(5)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128, max_selected_examples=2)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    args = _args(_LOCK_PATH, tmp_path)  # args.max_selected stays None
    cmd_inspect_fixed_trace(args)
    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert selection["n_selected"] == 2
    assert selection["max_selected"] == 2


def test_write_selection_cli_flag_overrides_config_max_selected_examples(tmp_path, monkeypatch):
    rows = [_base_row(prompt_len=50, think_len=250, idx=i) for i in range(5)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128, max_selected_examples=2)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    args = _args(_LOCK_PATH, tmp_path)
    args.max_selected = 4  # explicit CLI override wins over the config's 2
    cmd_inspect_fixed_trace(args)
    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert selection["n_selected"] == 4


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


def test_write_selection_warns_when_cli_max_selected_overrides_config(tmp_path, monkeypatch, capsys):
    # 2026-07-19 review: silently letting --max-selected override the
    # pre-registered fixed_trace.max_selected_examples defeats
    # pre-registration -- must at least warn loudly (the resulting file is
    # then rejected at replay time by _load_fixed_trace_selection unless
    # the config is updated to match).
    rows = [_base_row(prompt_len=50, think_len=250, idx=i) for i in range(5)]
    _write_base_file(tmp_path / "full.jsonl", rows)
    stage = _stage(tmp_path, budget=128, max_selected_examples=20)
    lock = load_lock_config("configs/lock.yaml")
    monkeypatch.setattr("kvcot.cli.load_stage_config", lambda path: (stage, lock))
    monkeypatch.chdir(tmp_path)

    args = _args(_LOCK_PATH, tmp_path)
    args.max_selected = 2  # overrides the config's pre-registered 20
    cmd_inspect_fixed_trace(args)
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "max_selected_examples" in out
    selection = read_json(tmp_path / "results" / "selections" / "test_v3_selection.json")
    assert selection["max_selected"] == 2


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
