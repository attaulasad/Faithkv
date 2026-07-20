"""Discovery-track-only configuration schema (Blocker 6,
`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`; frozen and hashed in full by
`docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §7/§8). Loads
`configs/discovery/llama8b_math500_b1024.yaml` -- never
`configs/lock.yaml`, which remains the frozen Qwen-1.5B primary pipeline's
executable source of truth, untouched by this module (`CLAUDE.md` §1a/§4a).

Pure Python (pydantic + yaml only) — no torch import, so this module is
usable from CPU-only planning code (`kvcot plan-discovery --dry-run`,
`kvcot b2a-calibrate --dry-run`) exactly like `kvcot.config`.

## Complete freeze, never `"auto"` or a hidden library default (B1B-R2 §7)

Every scientifically relevant field is a required, explicitly-typed
schema field -- there is no `Literal["auto"]` anywhere in this module, and
no field silently falls back to a transformers/library default. A value
that cannot be independently verified (currently: the MATH-500 dataset
revision was UNVERIFIED before this section; the exact tokenized prompt
still IS, since that requires a live tokenizer -- see
`kvcot.discovery.manifest`) is represented as `None` and the affected
gate/dry-run path reports it as an explicit blocker, never a fabricated
placeholder.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator, model_validator

from kvcot.discovery.dispatch import MODEL_TYPE_TO_PATCHER_NAME
from kvcot.utils.hashing import sha256_json, sha256_text

# The SAME pinned R-KV commit used everywhere else in this repository
# (configs/lock.yaml `upstream.commit`, CLAUDE.md, docs/UPSTREAM_AUDIT.md) —
# the discovery track's own config must never independently drift to a
# different R-KV revision.
PINNED_RKV_UPSTREAM_REVISION = "45eaa7d69d20b7388321f077020a610d9afb65bd"

# MATH-500 dataset revision, resolved 2026-07-20 directly against the
# Hugging Face Hub API (`GET /api/datasets/HuggingFaceH4/MATH-500`, `sha`
# field) -- verified via two independent lookups against the live source,
# never guessed and never a floating branch name. See
# `docs/B1B_R2_REAL_MODEL_BOUNDARY_AND_B2A_PREFLIGHT.md` §8 for the exact
# verification method.
MATH500_DATASET_REPO = "HuggingFaceH4/MATH-500"
MATH500_DATASET_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

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
    """`revision` is `str | None`. As of B1B-R2 (§8), the frozen config
    file sets it to the independently-verified
    `MATH500_DATASET_REVISION` -- `None` remains a legal (and structurally
    handled) value for this field so that a config which has NOT resolved
    it fails closed (`revision_is_frozen == False`) rather than silently
    treating an absent revision as "anything goes"."""

    name: Literal["MATH-500"]
    config: str = "default"
    split: str = "test"
    revision: str | None = None

    @field_validator("revision")
    @classmethod
    def _revision_full_if_present(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return validate_full_revision(v, "dataset.revision")

    @property
    def revision_is_frozen(self) -> bool:
        return self.revision is not None


class DiscoveryRkvLock(BaseModel):
    """Every R-KV hyperparameter that affects the compression/eviction
    decision (B1B-R2 §7: "Do not guess R-KV values... resolve them from the
    existing authority files") -- reused from `configs/lock.yaml`'s own
    frozen R-KV row (CLAUDE.md §4), applied to the discovery-track budget."""

    budget: int
    upstream_revision: str
    window_size: int = 8
    mix_lambda: float = 0.1
    retain_ratio: float = 0.2
    retain_direction: Literal["last"] = "last"
    divide_method: Literal["step_length"] = "step_length"
    divide_length: int = 128
    compression_content: Literal["all"] = "all"
    kernel_size: int = 3

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


class DiscoveryGenerationLock(BaseModel):
    """Deterministic greedy generation, frozen explicitly (B1B-R2 §7) --
    every field here is a concrete, typed value; nothing is `"auto"` or a
    hidden transformers default. `framework_seed` is set even though token
    selection itself is greedy/argmax (no sampling), for reproducibility of
    any framework-level nondeterminism (dict/set iteration order, etc.) —
    reuses `13`, the SAME seed already used to freeze this repository's
    GSM8K/MATH-500 manifests (`kvcot.cli.cmd_freeze_manifests`), rather than
    inventing a new one."""

    generation_mode: Literal["greedy"] = "greedy"
    do_sample: Literal[False] = False
    temperature: None = None
    top_p: None = None
    batch_size: Literal[1] = 1
    max_new_tokens: int = 6144
    framework_seed: int = 13
    attention_backend: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2"
    cache_implementation: Literal["DynamicCache"] = "DynamicCache"
    no_offload_required: Literal[True] = True


class DiscoveryConfig(BaseModel):
    model: DiscoveryModelLock
    dataset: DiscoveryDatasetLock
    rkv: DiscoveryRkvLock
    generation: DiscoveryGenerationLock = DiscoveryGenerationLock()


def load_discovery_config(path: str | Path) -> DiscoveryConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DiscoveryConfig.model_validate(raw)


def generation_config_hash(generation: DiscoveryGenerationLock) -> str:
    return sha256_json(generation.model_dump(mode="json"))


def rkv_config_hash(rkv: DiscoveryRkvLock) -> str:
    return sha256_json(rkv.model_dump(mode="json"))


def prompt_template_hash() -> str:
    """Hash of the LITERAL prompt template string (pure Python, resolvable
    without any tokenizer/model) -- the discovery track reuses the SAME
    frozen `BASE_USER_TEMPLATE` the primary pipeline uses
    (`kvcot.probes.templates`), rendered against a MATH-500 problem instead
    of a GSM8K one. This is NOT the same thing as a rendered prompt's
    TOKEN-ID hash, which requires a live tokenizer and is deliberately left
    unresolved (`kvcot.discovery.manifest`)."""
    from kvcot.probes.templates import BASE_USER_TEMPLATE

    return sha256_text(BASE_USER_TEMPLATE)


def canonical_config_hash(config: DiscoveryConfig) -> str:
    """One stable hash over every scientifically relevant, CPU-resolvable
    configuration field -- included in `plan-discovery`/`b2a-calibrate
    --dry-run` output and any future B2A artifact (B1B-R2 §7). Deliberately
    excludes the dataset row/prompt-token identity (that is
    `kvcot.discovery.manifest`'s own, separately-reported hash) so a config
    hash change and a manifest hash change are never conflated."""
    return sha256_json(
        {
            "model": config.model.model_dump(mode="json"),
            "dataset_name": config.dataset.name,
            "dataset_config": config.dataset.config,
            "dataset_split": config.dataset.split,
            "dataset_revision": config.dataset.revision,
            "rkv": config.rkv.model_dump(mode="json"),
            "generation": config.generation.model_dump(mode="json"),
            "prompt_template_hash": prompt_template_hash(),
        }
    )
