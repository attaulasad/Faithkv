"""Structural tests for `kvcot prepare-b2a-manifest` (B1B-R3 §5/§20)."""
from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from kvcot.cli import build_parser, cmd_prepare_b2a_manifest

DISCOVERY_CONFIG = "configs/discovery/llama8b_math500_b1024.yaml"


def _dry_run_args(**overrides):
    defaults = dict(config=DISCOVERY_CONFIG, dry_run=True, execute=False, force=False)
    defaults.update(overrides)
    return Namespace(**defaults)


def test_registered_in_build_parser():
    parser = build_parser()
    args = parser.parse_args(["prepare-b2a-manifest", "--config", DISCOVERY_CONFIG, "--dry-run"])
    assert args.func is cmd_prepare_b2a_manifest
    assert args.dry_run is True


def test_dry_run_prints_plan_and_writes_nothing(capsys):
    before = set(Path("configs/discovery").glob("*"))
    exit_code = cmd_prepare_b2a_manifest(_dry_run_args())
    after = set(Path("configs/discovery").glob("*"))
    out = capsys.readouterr().out
    assert "dataset: HuggingFaceH4/MATH-500" in out
    assert "tokenizer: deepseek-ai/DeepSeek-R1-Distill-Llama-8B" in out
    assert "no download, no write performed by --dry-run" in out
    assert exit_code == 0
    assert before == after


def test_requires_dry_run_or_execute():
    with pytest.raises(SystemExit, match="requires exactly one of --dry-run or --execute"):
        cmd_prepare_b2a_manifest(Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=False, force=False))


def test_execute_without_force_refuses_on_already_resolved_manifest():
    """The repository's committed manifest is already resolved (B1B-R3) --
    --execute without --force must refuse rather than silently overwrite it."""
    with pytest.raises(SystemExit):
        cmd_prepare_b2a_manifest(Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, force=False))
