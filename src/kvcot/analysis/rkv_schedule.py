"""Exact CPU simulator of R-KV's schedule/trigger mechanics — protocol v3
(2026-07-17, CHANGELOG.md). Predicts physical cache length and realized
retention at arbitrary absolute positions from a FullKV base record alone
(prompt length + how many tokens were fed), WITHOUT running the model. This
is what makes deterministic, outcome-blind trace selection possible on this
CPU-only machine (`kvcot inspect-fixed-trace --write-selection`): candidate
traces can be ranked by predicted realized compression before any GPU time
is spent replaying them.

Grounded entirely in `docs/UPSTREAM_AUDIT.md` H4 and §3.1/3.3, which cite
exact file:line ranges in the pinned upstream commit
(`third_party/R-KV/HuggingFace/rkv/modeling.py`,
`third_party/R-KV/HuggingFace/rkv/compression/r1_kv.py`). Restated here as
the two independent mechanics this simulator reproduces:

  SCHEDULE (when compression is attempted): `self.length` is a cumulative
  absolute-token counter on the top-level `CausalLM` object, incremented
  once per top-level forward call (prefill: += prompt_length in one call;
  each decode step: += 1). At the END of every forward call, upstream sets
  `is_newline = self.length % divide_length == 0` (this repo's frozen
  `divide_method=step_length`, `configs/lock.yaml`) and assigns it as EVERY
  layer's `compression` flag — so a flag computed from call N's `self.length`
  takes effect starting with call N+1, never the call that set it
  (modeling.py:598-611, cited in UPSTREAM_AUDIT.md H4).

  TRIGGER (whether an attempted compression actually evicts anything):
  `R1KV.update_kv` is a no-op whenever the current call's `kv_cache_len`
  (physical cache length plus this call's own newly-appended tokens) is
  still below `budget` — eviction only fires, and only ever evicts down to
  exactly `budget` slots, once that length has reached `budget`
  (r1_kv.py:46-47, 82; UPSTREAM_AUDIT.md H3). Because SCHEDULE and TRIGGER
  are independent, physical cache length between compaction events grows
  unboundedly past `budget` (a "sawtooth" pattern) until the next
  schedule-eligible call finds it at or above budget and snaps it back down.

  ONE MORE MECHANIC, easy to miss: on the very first forward call this
  process/model has ever made, every layer's `compression` flag is `None`
  (not `True`/`False`) — the compression-config initial value
  (`kvcot.generation.policies._PatchedPolicyBase._compression_config`,
  mirroring upstream's own `run_math.py:230`). The `compression is None`
  branch in the patched attention forward (modeling.py:288-308) ALWAYS
  attempts an eviction check on that first call, regardless of what the
  schedule would otherwise say — i.e. prefill always attempts compression,
  never waits for a `divide_length` boundary. Every call after the first
  ever sees a concrete `True`/`False`, never `None` again.

Must never import torch (tests/unit/test_no_analysis_torch_import.py) —
every input here is plain ints, exactly like kvcot.analysis.fixed_trace/
.metrics/.pipeline.
"""
from __future__ import annotations

from collections.abc import Iterable


def simulate_rkv_cache_lengths(
    prompt_length: int,
    target_absolute_positions: Iterable[int],
    budget: int,
    divide_length: int,
) -> dict[int, int]:
    """Predicted physical R-KV cache length at each requested absolute
    position (prompt_length + number of generated tokens fed so far — the
    same `absolute_position` convention `kvcot.generation.replay.
    replay_and_snapshot`/`capture_snapshot` use, so predictions here are
    directly comparable to a real snapshot's measured cache length).

    `target_absolute_positions` values must each be >= prompt_length (a
    position before the end of prefill has no cache state to simulate).
    Walks the schedule/trigger mechanics one simulated forward call at a
    time (prefill as one call, then one call per token from prompt_length
    up to the largest requested position) — never a closed-form shortcut,
    since the sawtooth pattern is only exactly reproducible by actually
    stepping through it.
    """
    targets = set(target_absolute_positions)
    for pos in targets:
        if pos < prompt_length:
            raise ValueError(
                f"target absolute position {pos} is before the end of prefill "
                f"({prompt_length}) — nothing to simulate before prefill completes"
            )
    max_target = max(targets) if targets else prompt_length

    # --- prefill: one call, self.length = prompt_length, compression=None
    # for this call only (always attempts eviction; min() captures both the
    # "no-op below budget" and "evict down to exactly budget" cases) ---
    physical_length = min(prompt_length, budget)
    self_length = prompt_length
    results: dict[int, int] = {}
    if prompt_length in targets:
        results[prompt_length] = physical_length

    # Flag that will be READ by the NEXT call (the first decode step),
    # computed from this call's own self.length at its end.
    compression_flag_next = (self_length % divide_length == 0)

    absolute_position = prompt_length
    while absolute_position < max_target:
        self_length += 1
        absolute_position += 1
        if compression_flag_next:
            # attempted: no-op if still below budget, else evict to exactly
            # budget — both cases are min(candidate, budget).
            physical_length = min(physical_length + 1, budget)
        else:
            # not scheduled this call: pure growth, no eviction attempt at all.
            physical_length += 1
        compression_flag_next = (self_length % divide_length == 0)
        if absolute_position in targets:
            results[absolute_position] = physical_length

    return results


def retention_ratio(physical_length: int, fullkv_equivalent_slots: int) -> float:
    """Same definition as kvcot.schemas.RetentionSummary.instantaneous_retention_ratio
    (physical_cache_slots / fullkv_equivalent_slots) — never budget / prompt
    length (§9: realized retention is measured, never derived from the
    configured budget alone)."""
    if fullkv_equivalent_slots <= 0:
        return 1.0
    return physical_length / fullkv_equivalent_slots


def predict_retention_by_fraction(
    prompt_length: int,
    target_absolute_positions: dict[float, int],
    budget: int,
    divide_length: int,
) -> dict[float, float]:
    """fraction -> predicted instantaneous_retention_ratio, for every
    fraction key in `target_absolute_positions` (typically
    kvcot.config.PROBE_FRACTIONS_ALL mapped through
    kvcot.probes.early_answering.absolute_cut_position + prompt_length,
    exactly as kvcot.generation.replay.replay_and_snapshot's
    `snapshot_absolute_positions` argument is built)."""
    lengths = simulate_rkv_cache_lengths(
        prompt_length, target_absolute_positions.values(), budget, divide_length
    )
    return {
        fraction: retention_ratio(lengths[pos], pos)
        for fraction, pos in target_absolute_positions.items()
    }


def meaningfully_compressed_fractions(
    retention_by_fraction: dict[float, float], meaningful_retention_ceiling: float
) -> set[float]:
    """Fractions whose predicted retention is <= the ceiling — i.e. a
    SUBSTANTIAL predicted compression, never just "not exactly 1.0" (the
    "any eviction" ambiguity documented in CHANGELOG.md's 2026-07-17 entry
    as protocol v2's actual failure mode: a 0.9959-retention example counted
    as "compression active" there)."""
    return {f for f, r in retention_by_fraction.items() if r <= meaningful_retention_ceiling}
