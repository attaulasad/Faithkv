"""Active B1A discovery pairwise provenance schema
(`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md` Part II / Part VII.16).

This is a fresh, standalone Pydantic model — the discovery pilot itself
(B1B/B2A/B2B) is not authorized and this schema is never written to by any
code path in this repository yet. It supersedes every prior *documented*
schema draft (`docs/B0_5_R2_1_FINAL_PROTOCOL.md` §10 and earlier), and is
the first version of this schema actually coded in `src/`.

Corrects the ambiguous `first_affected_logit_absolute_position` field
(which stored `t+1`, an *input* position, while its own docstring already
said the *logits it produces* target `t+2` — two different things sharing
one name) into two explicit fields:

  event_token_absolute_position                    = t
  bridge_token_absolute_position                    = t + 1
  first_affected_forward_input_absolute_position    = t + 1
  first_affected_logit_target_absolute_position     = t + 2
  first_scored_absolute_position                    = t + 2

No prior committed record ever used the old field name (nothing under
`src/` wrote this schema before this module existed), so there is no
migration to perform — this is simply the first correctly-named version.

Pure Python (pydantic only) — no torch import, matching the discipline
`kvcot.analysis`/`kvcot.utils` already follow (this module could live next
to them import-wise, even though it is organized under `kvcot.discovery`
for cohesion with the rest of the B1A prerequisite code).
"""
from __future__ import annotations

import math
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DISCOVERY_SCHEMA_VERSION = "b0_5_r2_2.v1"
SCORED_HORIZON = 48
UNCERTAINTY_SIGNAL_SOURCE = "raw_next_token_logits_at_token_prediction_time"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_FLOAT_DIFF_TOLERANCE = 1e-9


class SwapPairRecord(BaseModel):
    """One JSON-lines record per swap-branch (pair) — the active successor
    to every historical schema draft in `docs/B0_5_R2_1_FINAL_PROTOCOL.md`
    §10 and earlier. Field-level and cross-field validation below rejects
    malformed records outright; it never silently repairs one."""

    schema_version: Literal["b0_5_r2_2.v1"] = DISCOVERY_SCHEMA_VERSION

    # --- identity ---
    example_id: str
    model_revision: str
    rkv_revision: str

    # --- compaction event identity ---
    compaction_event_id: int
    chronological_event_ordinal: int = Field(description="0, 1, or 2 -- draw order, NOT depth")
    depth_stratum: int = Field(description="0 (early third), 1 (middle third), or 2 (late third) -- independently permuted, Part III")

    # --- layer / KV-head identity ---
    layer_index: int
    kv_head_index: int

    # --- corrected timing fields (Part II) ---
    event_token_absolute_position: int = Field(description="t: the compaction event token")
    bridge_token_absolute_position: int = Field(description="t + 1: fed identically into both branches, unscored")
    first_affected_forward_input_absolute_position: int = Field(description="t + 1: first forward-call INPUT reading the diverged cache")
    first_affected_logit_target_absolute_position: int = Field(description="t + 2: first token whose LOGITS can differ between branches")
    first_scored_absolute_position: int = Field(description="t + 2: first scored reference token")

    # --- candidate / donor identity and physical positions ---
    evicted_absolute_token_position: int
    evicted_pre_storage_position: int
    retained_absolute_token_position: int
    retained_pre_storage_position: int
    retained_post_storage_position: int

    # --- recomputed R-KV score and component values ---
    score_e: float
    score_r: float
    score_margin_e_minus_r: float
    attention_component_diff: float
    similarity_component_diff: float
    recency_diff: int
    key_norm_diff: float
    value_norm_diff: float

    # --- entropy / logit-margin: source values AND difference (Part IV.8 — never diff-only) ---
    entropy_e: float | None = None
    entropy_r: float | None = None
    entropy_diff: float | None = None
    logit_margin_e: float | None = None
    logit_margin_r: float | None = None
    logit_margin_diff: float | None = None
    uncertainty_signal_source: Literal["raw_next_token_logits_at_token_prediction_time"] = UNCERTAINTY_SIGNAL_SOURCE

    # --- parity / no-op / validity status ---
    parity_check_passed: bool
    parity_failure_reason: str | None = None
    is_noop_control: bool
    net_physical_bytes_changed: int
    cap_hit_flag: bool
    valid_flag: bool
    invalid_reason: str | None = None

    # --- reference horizon and NLL branches ---
    reference_horizon_sha256: str
    swap_gain: float
    baseline_per_token_nll: list[float] = Field(min_length=SCORED_HORIZON, max_length=SCORED_HORIZON)
    swapped_per_token_nll: list[float] = Field(min_length=SCORED_HORIZON, max_length=SCORED_HORIZON)

    # --- optional / context / oracle fields (already authorized, §8.2) ---
    hidden_state_delta: float | None = None
    recurrence_flag: bool | None = None
    future_attention_oracle: float | None = None
    oracle_non_deployable: Literal[True] = True

    @field_validator("chronological_event_ordinal")
    @classmethod
    def _chronological_ordinal_range(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"chronological_event_ordinal must be 0, 1, or 2, got {v}")
        return v

    @field_validator("depth_stratum")
    @classmethod
    def _depth_stratum_range(cls, v: int) -> int:
        if v not in (0, 1, 2):
            raise ValueError(f"depth_stratum must be 0, 1, or 2, got {v}")
        return v

    @field_validator("reference_horizon_sha256")
    @classmethod
    def _horizon_hash_shape(cls, v: str) -> str:
        if not _SHA256_HEX_RE.match(v):
            raise ValueError(f"reference_horizon_sha256 must be 64 lowercase hex characters, got {v!r}")
        return v

    @model_validator(mode="after")
    def _timing_invariants(self) -> "SwapPairRecord":
        t = self.event_token_absolute_position
        if self.bridge_token_absolute_position != t + 1:
            raise ValueError("bridge_token_absolute_position must equal event_token_absolute_position + 1")
        if self.bridge_token_absolute_position != self.first_affected_forward_input_absolute_position:
            raise ValueError("bridge_token_absolute_position must equal first_affected_forward_input_absolute_position")
        if self.first_affected_logit_target_absolute_position != self.first_scored_absolute_position:
            raise ValueError("first_affected_logit_target_absolute_position must equal first_scored_absolute_position")
        if self.first_affected_logit_target_absolute_position != self.first_affected_forward_input_absolute_position + 1:
            raise ValueError(
                "first_affected_logit_target_absolute_position must equal "
                "first_affected_forward_input_absolute_position + 1"
            )
        if self.first_scored_absolute_position != self.bridge_token_absolute_position + 1:
            raise ValueError("first_scored_absolute_position must equal bridge_token_absolute_position + 1")
        return self

    @model_validator(mode="after")
    def _validity_and_reason(self) -> "SwapPairRecord":
        if not self.valid_flag and not self.invalid_reason:
            raise ValueError("valid_flag=False requires a non-empty invalid_reason")
        if self.valid_flag and self.invalid_reason is not None:
            raise ValueError("valid_flag=True requires invalid_reason=None")
        return self

    @model_validator(mode="after")
    def _net_bytes_zero_for_valid_records(self) -> "SwapPairRecord":
        if self.valid_flag and self.net_physical_bytes_changed != 0:
            raise ValueError("net_physical_bytes_changed must be 0 for a valid record (fixed-shape swap invariant)")
        return self

    @model_validator(mode="after")
    def _noop_requires_matching_identity(self) -> "SwapPairRecord":
        if self.is_noop_control and self.evicted_absolute_token_position != self.retained_absolute_token_position:
            raise ValueError(
                "is_noop_control=True requires candidate and donor token identity to match "
                "(evicted_absolute_token_position == retained_absolute_token_position)"
            )
        return self

    @model_validator(mode="after")
    def _nll_finiteness_for_valid_records(self) -> "SwapPairRecord":
        if self.valid_flag:
            for name in ("baseline_per_token_nll", "swapped_per_token_nll"):
                values = getattr(self, name)
                if not all(math.isfinite(v) for v in values):
                    raise ValueError(f"{name} must be all-finite for a valid record")
        return self

    @model_validator(mode="after")
    def _uncertainty_diffs_consistent(self) -> "SwapPairRecord":
        if self.entropy_e is not None and self.entropy_r is not None:
            expected = self.entropy_e - self.entropy_r
            if self.entropy_diff is None or abs(self.entropy_diff - expected) > _FLOAT_DIFF_TOLERANCE:
                raise ValueError("entropy_diff must equal entropy_e - entropy_r when both source values exist")
        if self.logit_margin_e is not None and self.logit_margin_r is not None:
            expected = self.logit_margin_e - self.logit_margin_r
            if self.logit_margin_diff is None or abs(self.logit_margin_diff - expected) > _FLOAT_DIFF_TOLERANCE:
                raise ValueError("logit_margin_diff must equal logit_margin_e - logit_margin_r when both source values exist")
        return self
