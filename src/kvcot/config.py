"""Frozen-settings and stage-config loading (§4, §10, §13).

Two layers:
  - `FrozenSettings` (from configs/lock.yaml): the settings that are frozen
    per §4 and must not silently drift. Changing any of these requires a
    dated CHANGELOG.md entry per the brief — this module does not enforce
    that policy mechanically, but every stage config is required to load
    lock.yaml rather than re-specifying these values, so there is exactly
    one place to change them.
  - `StageConfig` (from configs/stageN_*.yaml): what varies per stage
    (manifest, conditions, budgets under calibration, output paths, limits).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from kvcot.utils.hashing import sha256_file

PROBE_FRACTIONS_ALL = (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
PROBE_FRACTIONS_SCORED = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)  # excludes f=0 and f=1, §8.1/§8.2


class ModelLock(BaseModel):
    name: str
    revision: str
    tokenizer_name: str
    tokenizer_revision: str
    dtype: Literal["bfloat16"]


class UpstreamLock(BaseModel):
    repo: str
    commit: str
    submodule_path: str


class AttentionLock(BaseModel):
    primary: Literal["flash_attention_2"]
    determinism_test: Literal["sdpa"]


class GenerationLock(BaseModel):
    batch_size: Literal[1]
    base_temperature: float
    base_top_p: float
    base_max_new_tokens: int

    @field_validator("base_max_new_tokens")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("base_max_new_tokens must be positive")
        return v


class RKVLock(BaseModel):
    window_size: int
    mix_lambda: float
    retain_ratio: float
    retain_direction: Literal["last", "first", "last_percent", "first_percent"]
    divide_method: Literal["newline", "step_length"]
    divide_length: int
    compression_content: Literal["think", "all"]


class ProbesLock(BaseModel):
    fractions_all: list[float]
    fractions_scored: list[float]
    max_new_tokens: int
    decoding: Literal["greedy"]

    @model_validator(mode="after")
    def _fractions_match_frozen_spec(self) -> "ProbesLock":
        if tuple(self.fractions_all) != PROBE_FRACTIONS_ALL:
            raise ValueError(
                f"probes.fractions_all must equal the frozen set {PROBE_FRACTIONS_ALL}, "
                f"got {tuple(self.fractions_all)}. Changing this requires a dated "
                "CHANGELOG.md entry per §4 of the build brief."
            )
        if tuple(self.fractions_scored) != PROBE_FRACTIONS_SCORED:
            raise ValueError(
                f"probes.fractions_scored must equal the frozen 7-fraction EAS set "
                f"{PROBE_FRACTIONS_SCORED} (f=0 and f=1 excluded, §8.1/§8.2), "
                f"got {tuple(self.fractions_scored)}"
            )
        return self


class FrozenSettings(BaseModel):
    model: ModelLock
    upstream: UpstreamLock
    attention: AttentionLock
    generation: GenerationLock
    seeds: list[int]
    rkv: RKVLock
    probes: ProbesLock

    @field_validator("seeds")
    @classmethod
    def _three_seeds(cls, v: list[int]) -> list[int]:
        if tuple(v) != (13, 42, 2026):
            raise ValueError(f"seeds must be exactly [13, 42, 2026] per §4, got {v}")
        return v


class FixedTraceSettings(BaseModel):
    """Settings for the secondary, additive fixed-trace prefix-sufficiency
    screen (`kvcot replay-fixed-trace`/`analyze-fixed-trace`,
    kvcot.analysis.fixed_trace) — deliberately separate from `ProbesLock`
    (§4's frozen `probes.max_new_tokens: 48`). The fixed-trace probe's
    answer-elicitation strategy (teacher-forced boxed-answer prefix, §
    kvcot.probes.templates.FIXED_TRACE_SUFFIX_TEXT) is not the frozen
    primary replay-probe protocol, so it is allowed its own decoding budget
    and eligibility thresholds without touching `configs/lock.yaml` — mixing
    the two would let a fixed-trace-motivated change silently alter the
    frozen primary EAS experiment.
    """

    probe_max_new_tokens: int = Field(default=64, gt=0)
    # A frozen Literal, not a plain bool (§ external review 2026-07-16): a
    # non-boxed (fallback) f=1 anchor is documented noise
    # (kvcot.analysis.fixed_trace._pss_for_side/FixedTraceEligibility) —
    # there is no supported "off" mode, so the field cannot silently be set
    # to False and quietly do nothing (a plain `bool = True` field that no
    # code ever reads is worse than no field at all). Kept as an explicit
    # field, rather than removed, so a stage config can still document the
    # requirement inline.
    require_boxed_extraction: Literal[True] = True
    min_eligible_examples: int = Field(default=5, ge=1)
    min_actual_compression_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    max_mean_f1_retention_ratio: float = Field(default=0.7, gt=0.0, le=1.0)


class StageConfig(BaseModel):
    stage_name: str
    lock_config_path: str = "configs/lock.yaml"
    dataset_manifest: str
    conditions: list[str]
    rkv_budgets: list[int] | None = None  # only set for stage1b calibration sweeps
    output_dir: str
    limit: int | None = None
    # Stages 0/1A/1B are single-seed by design (§10); None means "use every
    # seed in lock.seeds" (Stage 2 only).
    seeds_override: list[int] | None = None
    notes: str | None = None
    # Only set (and only required) for the secondary fixed-trace screen's
    # own stage configs (early_gap_b*.yaml) — see FixedTraceSettings and
    # kvcot.cli.cmd_replay_fixed_trace/cmd_analyze_fixed_trace, which refuse
    # to run without it rather than silently falling back to the frozen
    # primary probes.* settings.
    fixed_trace: FixedTraceSettings | None = None

    def resolve_seeds(self, lock: "FrozenSettings") -> list[int]:
        if self.seeds_override is not None:
            for s in self.seeds_override:
                if s not in lock.seeds:
                    raise ValueError(
                        f"seeds_override value {s} is not one of the frozen seeds {lock.seeds}"
                    )
            return self.seeds_override
        return lock.seeds

    @field_validator("conditions")
    @classmethod
    def _no_ten_percent_naming(cls, v: list[str]) -> list[str]:
        for c in v:
            if "%" in c:
                raise ValueError(
                    f"condition name {c!r} contains '%' — conditions must be named "
                    "'full', 'patched_noop', or 'rkv_b{budget}' (integer token budget), "
                    "never a percentage (§9: realized retention is measured, not a name)."
                )
        return v


def load_lock_config(path: str | Path) -> FrozenSettings:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return FrozenSettings.model_validate(raw)


def load_stage_config(path: str | Path) -> tuple[StageConfig, FrozenSettings]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    stage = StageConfig.model_validate(raw)
    lock_path = Path(path).parent / Path(stage.lock_config_path).name
    if not lock_path.exists():
        lock_path = Path(stage.lock_config_path)
    lock = load_lock_config(lock_path)
    return stage, lock


def config_identity(path: str | Path) -> str:
    """SHA-256 of the config file's exact bytes, stored on every record that
    was produced under it (§12: "config path + SHA-256")."""
    return sha256_file(path)
