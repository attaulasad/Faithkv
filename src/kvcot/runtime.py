"""Run orchestration: environment capture, resume bookkeeping, the Stage 2
operating-point guard, and the "rkv_selected" condition-placeholder
resolution used by configs/stage2_main.yaml.

This module itself does not import torch at scope — GPU-dependent
orchestration (kvcot.generation.*) is imported lazily inside the specific
functions that need it, so `--dry-run` and config/manifest validation work
without a GPU-capable environment (see pyproject.toml's deferred-import
note).
"""
from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

import yaml

from kvcot.config import FrozenSettings, StageConfig
from kvcot.schemas import VersionInfo


class OperatingPointMissingError(RuntimeError):
    pass


def require_operating_point(path: str = "configs/selected_operating_point.yaml") -> dict:
    """§10: "Stage 2 must refuse to start if this file is absent." Called
    before any Stage 2 work — including `--dry-run`, since the point of
    dry-run is to catch exactly this kind of missing-prerequisite error
    before spending GPU time.
    """
    p = Path(path)
    if not p.exists():
        raise OperatingPointMissingError(
            f"{path} does not exist. Stage 2 requires a completed, reviewed Stage 1B decision. "
            f"Copy configs/selected_operating_point.yaml.example, fill it in from the real Stage 1B "
            f"decision JSON at results/decisions/stage1b_budget_<N>.json, and re-run."
        )
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "selected_budget" not in data or not isinstance(data["selected_budget"], int):
        raise OperatingPointMissingError(f"{path} is missing a valid integer 'selected_budget' field")
    return data


def resolve_conditions(stage: StageConfig) -> list[str]:
    """Resolves the `rkv_selected` placeholder condition (used only in
    configs/stage2_main.yaml) into a concrete `rkv_b{budget}` condition
    name by reading configs/selected_operating_point.yaml. Every other
    condition name passes through unchanged."""
    resolved = []
    for c in stage.conditions:
        if c == "rkv_selected":
            op = require_operating_point()
            resolved.append(f"rkv_b{op['selected_budget']}")
        else:
            resolved.append(c)
    return resolved


def _git(args: list[str]) -> str:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_commit() -> str:
    return _git(["rev-parse", "HEAD"])


def git_is_dirty() -> bool:
    """True iff any TRACKED file differs from `HEAD` (staged or unstaged).
    Deliberately ignores untracked files (`--untracked-files=no`) — 2026-07-19
    review found that every real command in this repository's own workflow
    writes a new, intentionally-committed-but-not-yet-committed artifact
    (`results/run_manifests/*.json`, `results/selections/*.json`,
    `results/decisions/*.json` — all deliberately NOT in `.gitignore`, per
    `README.md`'s documented layout) after the very first invocation of a
    GPU session. Counting those as "dirty" made `git_dirty` report `True` on
    every record after the first command, regardless of whether the actual
    CODE matched a committed state — the opposite of what this field exists
    to detect (§13: reproducibility from a known commit). A real change
    worth flagging is always a MODIFICATION TO A TRACKED FILE (source,
    config, or a previously-committed `requirements-lock.txt` — `git status
    --porcelain` still reports those with an `M`/` M` prefix, which
    `--untracked-files=no` does not suppress), never a brand-new output file
    the tooling itself just wrote as part of doing its job.
    """
    status = _git(["status", "--porcelain", "--untracked-files=no"])
    return status != "" and status != "unknown"


def capture_version_info() -> VersionInfo:
    """§12: version fields on every record. torch/transformers/cuda/
    flash_attn are None on this build machine (not installed) — populated
    for real only when this runs inside a GPU generation process."""
    torch_version = None
    cuda_version = None
    transformers_version = None
    flash_attn_version = None
    try:
        import torch as _torch

        torch_version = _torch.__version__
        cuda_version = _torch.version.cuda
    except ImportError:
        pass
    try:
        import transformers as _transformers

        transformers_version = _transformers.__version__
    except ImportError:
        pass
    try:
        import flash_attn as _flash_attn

        flash_attn_version = getattr(_flash_attn, "__version__", None)
    except ImportError:
        pass
    return VersionInfo(
        python=platform.python_version(),
        torch=torch_version,
        cuda=cuda_version,
        transformers=transformers_version,
        flash_attn=flash_attn_version,
    )


def gpu_model_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return None


def upstream_submodule_commit(lock: FrozenSettings) -> str:
    """Reads the actually-checked-out submodule commit (not just the
    configured pin) so a record's provenance reflects reality even if the
    submodule were ever out of sync with configs/lock.yaml — that
    divergence would itself be worth catching, not silently trusting the
    config value."""
    submodule_head = _git(["-C", lock.upstream.submodule_path, "rev-parse", "HEAD"])
    if submodule_head != lock.upstream.commit and submodule_head != "unknown":
        raise RuntimeError(
            f"submodule at {lock.upstream.submodule_path} is checked out at {submodule_head}, "
            f"but configs/lock.yaml pins {lock.upstream.commit} — run "
            f"`git -C {lock.upstream.submodule_path} checkout {lock.upstream.commit}`"
        )
    return lock.upstream.commit
