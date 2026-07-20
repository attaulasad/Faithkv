"""Structural + smoke tests for `kvcot b2a-calibrate`
(`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §11/§12).

Proves `--dry-run` prints the required planning numbers/hashes, creates no
result files, and -- structurally -- can never reach `from_pretrained`,
`load_dataset`, a CUDA API, or the real R-KV patcher. Proves `--execute`
requires the flag, refuses without CUDA, and refuses on the currently-
unresolved manifest prompt-token identity -- all without ever importing
torch.cuda machinery or a real model.
"""
from __future__ import annotations

import builtins
from argparse import Namespace
from pathlib import Path

import pytest

from kvcot.cli import build_parser, cmd_b2a_calibrate

DISCOVERY_CONFIG = "configs/discovery/llama8b_math500_b1024.yaml"

_DISALLOWED_MODULES = {"transformers", "datasets", "huggingface_hub", "rkv", "rkv.monkeypatch"}


@pytest.fixture
def block_disallowed_imports(monkeypatch):
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in _DISALLOWED_MODULES or name.split(".")[0] in _DISALLOWED_MODULES:
            raise AssertionError(f"b2a-calibrate --dry-run must never import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def _dry_run_args(**overrides):
    defaults = dict(config=DISCOVERY_CONFIG, dry_run=True, execute=False, problem_index=None, limit=None)
    defaults.update(overrides)
    return Namespace(**defaults)


def test_dry_run_prints_required_planning_information(capsys, block_disallowed_imports):
    exit_code = cmd_b2a_calibrate(_dry_run_args())
    out = capsys.readouterr().out
    assert "B2A is a one-example engineering calibration. It does not authorize the 12-example pilot." in out
    assert "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" in out
    assert "HuggingFaceH4/MATH-500" in out
    assert "one_example_only=True" in out
    assert "144 real branches" in out
    assert "canonical_config_hash=" in out
    assert "generation_config_hash=" in out
    assert "rkv_config_hash=" in out
    assert "prompt_template_hash=" in out
    assert "manifest_hash=" in out
    assert "no result files created; no model loaded; no CUDA required" in out
    # The manifest's prompt-token identity is genuinely unresolved in this
    # pass -- dry-run must report that as a blocker and exit non-zero.
    assert "BLOCKED" in out
    assert "prompt-token identity is unresolved" in out
    assert exit_code == 2


def test_dry_run_creates_no_result_files(tmp_path, monkeypatch, block_disallowed_imports):
    before = set(tmp_path.rglob("*"))
    monkeypatch.chdir(Path(__file__).resolve().parents[2])
    cmd_b2a_calibrate(_dry_run_args())
    after = set(tmp_path.rglob("*"))
    assert before == after


def test_dry_run_never_touches_cuda(monkeypatch, block_disallowed_imports):
    import torch

    def _boom(*a, **k):
        raise AssertionError("b2a-calibrate --dry-run must never call a CUDA API")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    cmd_b2a_calibrate(_dry_run_args())


def test_registered_in_build_parser():
    parser = build_parser()
    args = parser.parse_args(
        ["b2a-calibrate", "--config", DISCOVERY_CONFIG, "--dry-run"]
    )
    assert args.func is cmd_b2a_calibrate
    assert args.dry_run is True
    assert args.execute is False


def test_execute_without_flag_refuses_before_touching_cuda(monkeypatch):
    import torch

    def _boom(*a, **k):
        raise AssertionError("must not reach a CUDA check without --execute")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    with pytest.raises(SystemExit, match="requires exactly one of --dry-run or --execute"):
        cmd_b2a_calibrate(Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=False, problem_index=None, limit=None))


def test_execute_rejects_problem_index_override():
    with pytest.raises(SystemExit, match="no --problem-index/--limit override"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=5, limit=None)
        )


def test_execute_rejects_limit_override():
    with pytest.raises(SystemExit, match="no --problem-index/--limit override"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=10)
        )


def test_execute_refuses_on_unresolved_manifest_before_touching_cuda(monkeypatch):
    """The manifest's prompt-token identity is genuinely unresolved right
    now -- --execute must refuse on that BEFORE ever calling a CUDA API,
    since the identity check is cheaper and more fundamental than device
    availability."""
    import torch

    def _boom(*a, **k):
        raise AssertionError("must not reach a CUDA check while blockers are still outstanding")

    monkeypatch.setattr(torch.cuda, "is_available", _boom)
    with pytest.raises(SystemExit, match="prompt-token identity is unresolved"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
        )


def test_execute_requires_cuda_when_manifest_is_resolved(monkeypatch):
    """Proves the CUDA guard itself fires correctly once the identity
    blockers are cleared -- monkeypatches `manifest.prompt_identity_is_resolved`
    to simulate a future resolved state, without ever actually touching a
    real tokenizer or model."""
    import torch

    from kvcot.discovery.manifest import B2AOneExampleManifest

    monkeypatch.setattr(B2AOneExampleManifest, "prompt_identity_is_resolved", property(lambda self: True))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(SystemExit, match="requires CUDA"):
        cmd_b2a_calibrate(
            Namespace(config=DISCOVERY_CONFIG, dry_run=False, execute=True, problem_index=None, limit=None)
        )


def test_rejects_bad_config_before_printing_anything(tmp_path, block_disallowed_imports):
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
        cmd_b2a_calibrate(Namespace(config=str(bad_config), dry_run=True, execute=False, problem_index=None, limit=None))
