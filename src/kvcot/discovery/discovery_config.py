"""Discovery-track-only configuration schema (Blocker 6,
`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`). Loads
`configs/discovery/llama8b_math500_b1024.yaml` -- never
`configs/lock.yaml`, which remains the frozen Qwen-1.5B primary pipeline's
executable source of truth, untouched by this module (`CLAUDE.md` §1a/§4a).

Pure Python (pydantic + yaml only) — no torch import, so this module is
usable from CPU-only planning code (`kvcot plan-discovery --dry-run`)
exactly like `kvcot.config`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

from kvcot.discovery.dispatch import MODEL_TYPE_TO_PATCHER_NAME

# The SAME pinned R-KV commit used everywhere else in this repository
# (configs/lock.yaml `upstream.commit`, CLAUDE.md, docs/UPSTREAM_AUDIT.md) —
# the discovery track's own config must never independently drift to a
# different R-KV revision.
PINNED_RKV_UPSTREAM_REVISION = "45eaa7d69d20b7388321f077020a610d9afb65bd"

_FULL_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


def validate_full_revision(value: object, field_name: str) -> str:
    """A full 40-character lowercase hex git revision, and nothing else.
    Rejects `None`, `"main"`, `"latest"`, and any short/abbreviated hash --
    a single regex on a required `str` field structurally rejects all four:
    `None` fails pydantic's own type coercion before this even runs,
    `"main"`/`"latest"` are not 40 lowercase hex characters, and a short
    hash fails the length anchor.
    """
    if not isinstance(value, str) or not _FULL_REVISION_RE.match(value):
        raise ValueError(
            f"{field_name} must be a full 40-character lowercase hex git revision, got {value!r} -- "
            "never a branch name (\"main\"), a moving tag (\"latest\"), null, or an abbreviated hash."
        )
    return value


class DiscoveryModelLock(BaseModel):
    name: str
    revision: str
    tokenizer_name: str
    tokenizer_revision: str
    model_type: str
    dtype: Literal["bfloat16"]

    @field_validator("revision")
    @classmethod
    def _revision_full(cls, v: str) -> str:
        return validate_full_revision(v, "model.revision")

    @field_validator("tokenizer_revision")
    @classmethod
    def _tokenizer_revision_full(cls, v: str) -> str:
        return validate_full_revision(v, "model.tokenizer_revision")

    @field_validator("model_type")
    @classmethod
    def _model_type_has_verified_dispatch(cls, v: str) -> str:
        if v not in MODEL_TYPE_TO_PATCHER_NAME:
            raise ValueError(
                f"model.model_type={v!r} has no verified R-KV monkeypatch dispatch entry "
                f"(kvcot.discovery.dispatch.MODEL_TYPE_TO_PATCHER_NAME: {sorted(MODEL_TYPE_TO_PATCHER_NAME)}) "
                "-- never claim support for an architecture the dispatch table cannot actually patch."
            )
        return v


class DiscoveryDatasetLock(BaseModel):
    """`revision` is deliberately `str | None` and defaults to `None` --
    per the task brief, this pass must NOT invent a dataset repository ID
    or dataset revision that has not been independently verified from an
    authoritative source. A `None` revision is not a bug in this schema; it
    is the accurate current state, and `revision_is_frozen` makes that gap
    machine-checkable so a future B2A gate can refuse to proceed on it
    rather than silently treating an absent revision as "anything goes"."""

    name: Literal["MATH-500"]
    revision: str | None = None

    @property
    def revision_is_frozen(self) -> bool:
        return self.revision is not None


class DiscoveryRkvLock(BaseModel):
    budget: int
    upstream_revision: str

    @field_validator("budget")
    @classmethod
    def _positive_budget(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"rkv.budget must be positive, got {v}")
        return v

    @field_validator("upstream_revision")
    @classmethod
    def _upstream_revision_full(cls, v: str) -> str:
        return validate_full_revision(v, "rkv.upstream_revision")

    @model_validator(mode="after")
    def _upstream_revision_matches_pinned_submodule(self) -> "DiscoveryRkvLock":
        if self.upstream_revision != PINNED_RKV_UPSTREAM_REVISION:
            raise ValueError(
                f"rkv.upstream_revision ({self.upstream_revision!r}) must equal the pinned R-KV "
                f"submodule commit used everywhere else in this repository ({PINNED_RKV_UPSTREAM_REVISION!r}) "
                "-- the discovery track must never independently drift to a different R-KV revision."
            )
        return self


class DiscoveryConfig(BaseModel):
    model: DiscoveryModelLock
    dataset: DiscoveryDatasetLock
    rkv: DiscoveryRkvLock


def load_discovery_config(path: str | Path) -> DiscoveryConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DiscoveryConfig.model_validate(raw)
