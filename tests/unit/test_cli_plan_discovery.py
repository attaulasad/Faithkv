"""Structural + smoke tests for `kvcot plan-discovery --dry-run`
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §10, CLAUDE.md §1b/§4b).

Proves the command prints the required planning numbers, validates config/
revisions, creates no result files, and -- structurally -- can never reach
`from_pretrained`, `load_dataset`, a CUDA API, or the real R-KV patcher,
by intercepting `builtins.__import__` for the disallowed module names for
the duration of the call.
"""
from __future__ import annotations

import builtins
from argparse import Namespace
from pathlib import Path

import pytest

from kvcot.cli import build_parser, cmd_plan_discovery

DISCOVERY_CONFIG = "configs/discovery/llama8b_math500_b1024.yaml"

_DISALLOWED_MODULES = {"torch", "transformers", "datasets", "huggingface_hub", "rkv", "rkv.monkeypatch"}


@pytest.fixture
def block_disallowed_imports(monkeypatch):
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in _DISALLOWED_MODULES or name.split(".")[0] in _DISALLOWED_MODULES:
            raise AssertionError(f"plan-discovery must never import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def test_plan_discovery_prints_required_numbers(capsys, block_disallowed_imports):
    exit_code = cmd_plan_discovery(Namespace(config=DISCOVERY_CONFIG, dry_run=True))
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "bridge_tokens=1" in out
    assert "scored_horizon=48" in out
    assert "minimum_future_tokens_after_event=49" in out
    assert "events=3 candidates=2 donors=2 pair_branches_per_event=4" in out
    assert "144" in out
    assert "BLOCKED" in out  # B2A/B2B/Vast.ai remain blocked regardless of config freeze state
    assert "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" in out
    assert "6a6f4aa4197940add57724a7707d069478df56b1" in out
    assert "no result files created" in out


def test_plan_discovery_reports_frozen_dataset_revision_for_the_real_config(capsys, block_disallowed_imports):
    """B1B-R2 §8: the real discovery config's dataset revision is now
    independently verified and frozen -- the real config file must NOT
    print the "dataset.revision is not frozen" blocker any more."""
    cmd_plan_discovery(Namespace(config=DISCOVERY_CONFIG, dry_run=True))
    out = capsys.readouterr().out
    assert "revision_frozen=True" in out
    assert "dataset.revision is not frozen" not in out


def test_plan_discovery_flags_unfrozen_dataset_revision(capsys, tmp_path, block_disallowed_imports):
    """The blocker code path itself must still fire for a config that has
    NOT frozen its dataset revision -- proven here against a synthetic
    config, since the real repo config no longer exercises it."""
    unfrozen_config = tmp_path / "unfrozen.yaml"
    unfrozen_config.write_text(
        "model:\n"
        "  name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B\n"
        "  revision: 6a6f4aa4197940add57724a7707d069478df56b1\n"
        "  tokenizer_name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B\n"
        "  tokenizer_revision: 6a6f4aa4197940add57724a7707d069478df56b1\n"
        "  model_type: llama\n"
        "  dtype: bfloat16\n"
        "dataset:\n"
        "  name: MATH-500\n"
        "rkv:\n"
        "  budget: 1024\n"
        "  upstream_revision: 45eaa7d69d20b7388321f077020a610d9afb65bd\n",
        encoding="utf-8",
    )
    cmd_plan_discovery(Namespace(config=str(unfrozen_config), dry_run=True))
    out = capsys.readouterr().out
    assert "revision_frozen=False" in out
    assert "dataset.revision is not frozen" in out


def test_plan_discovery_creates_no_result_files(tmp_path, monkeypatch, block_disallowed_imports):
    # Run from the real repo root (so the relative config path resolves),
    # but assert no new file/directory appears anywhere under a scratch
    # directory this command has no reason to touch.
    before = set(tmp_path.rglob("*"))
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    cmd_plan_discovery(Namespace(config=DISCOVERY_CONFIG, dry_run=True))
    after = set(tmp_path.rglob("*"))
    assert before == after


def test_plan_discovery_registered_in_build_parser():
    parser = build_parser()
    args = parser.parse_args(["plan-discovery", "--config", DISCOVERY_CONFIG, "--dry-run"])
    assert args.func is cmd_plan_discovery
    assert args.config == DISCOVERY_CONFIG
    assert args.dry_run is True


def test_plan_discovery_rejects_bad_config_before_printing_anything(capsys, tmp_path, block_disallowed_imports):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        "model:\n"
        "  name: x\n"
        "  revision: main\n"
        "  tokenizer_name: x\n"
        "  tokenizer_revision: main\n"
        "  model_type: llama\n"
        "  dtype: bfloat16\n"
        "dataset:\n"
        "  name: MATH-500\n"
        "rkv:\n"
        "  budget: 1024\n"
        "  upstream_revision: 45eaa7d69d20b7388321f077020a610d9afb65bd\n",
        encoding="utf-8",
    )
    with pytest.raises(Exception):
        cmd_plan_discovery(Namespace(config=str(bad_config), dry_run=True))
