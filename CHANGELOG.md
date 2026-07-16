# Changelog

Frozen settings (`configs/lock.yaml`, and Sections 1/4/8/9 mirrored into
`CLAUDE.md`) may only change via a dated entry here, added **before** the
run that depends on the change (per the build brief). Entries are ordered
newest first.

## 2026-07-16 — Fixed-trace prefix-sufficiency screen (secondary, additive; no frozen §1/§4/§8/§9 value changed)

Added on branch `early-gap-fixed-trace`, still pre-GPU. This is an
**addition alongside** the frozen `replay-probe`/EAS/Delta_EAS pipeline, not
a replacement or a modification of it — every §1/§4/§8/§9 frozen value in
`CLAUDE.md`/`configs/lock.yaml` is unchanged, `replay-probe` itself is
byte-for-byte unmodified, and the frozen research question (§1) remains this
repository's headline claim.

**Motivation.** `replay-probe`/EAS scores each condition's probe answer
against that SAME condition's own sampled base answer (§8). FullKV and R-KV
each generate their own natural trace, so a Delta_EAS effect could in
principle be partly attributable to the traces themselves differing between
conditions, not only to the cache policy — the frozen design already
controls for this at the level that matters for the primary claim (§8.5's
both-correct-and-compression-active subset conditions on correctness per
problem), but it does not isolate the cache-policy question in the most
literal possible way: replaying one identical token sequence under two
different policies. This addition does exactly that, as a secondary,
smaller-sample screen — a kill/continue check, not a second primary result.

- **New commands**: `kvcot replay-fixed-trace` and `kvcot analyze-fixed-trace`
  (`src/kvcot/cli.py`). `replay-fixed-trace` reads its canonical token
  sequence from one condition's base file (`--trace-condition`, default
  `full`) but loads the model and applies cache-policy replay under a
  possibly-different condition (`--replay-condition`) — both replay
  policies teacher-force identical prompt and reasoning tokens; only the
  cache policy varies. `replay-probe` is untouched and remains the on-policy
  diagnostic; both commands can coexist against the same stage's output
  directory.
- **New metric**: Prefix-Sufficiency Sensitivity (PSS) / Delta_PSS
  (`src/kvcot/analysis/fixed_trace.py`) — mean mismatch rate against each
  replay policy's own greedy f=1 answer (never the trace source's sampled
  natural answer, which would reintroduce the sampled-vs-greedy confound
  §7 of `docs/EXPERIMENT.md` already documents for the original f=1
  stability probe). `Delta_PSS = PSS_full - PSS_rkv`, same subtraction
  order and sign meaning as `Delta_EAS` (positive => R-KV less sensitive to
  truncation). This is a **different metric** from EAS/Delta_EAS — never
  pool or directly compare the two. No p-value or confidence interval is
  computed at this sample size (`configs/early_gap_b512.yaml`: n=10,
  one seed) — descriptive counts only.
- **New schema**: `FixedTraceProbeRecord` (`src/kvcot/schemas.py`),
  distinguishing `trace_source_condition` from `replay_policy_condition` —
  a distinction `ProbeRunRecord` has no field for, since it never needed
  one. `SCHEMA_VERSION` bumped `1.0.0` -> `1.1.0`. `kvcot validate-run` now
  dispatches on each record's own `record_type` field instead of filename
  pattern-matching (`_schema_for_record`) — a `..._fixed_trace_probes.jsonl`
  file still ends in `_probes.jsonl`, so filename-based dispatch would have
  silently misvalidated it against `ProbeRunRecord`.
- **New configs**: `configs/early_gap_b512.yaml` (primary, 10-example,
  seed=42 screen) plus `early_gap_b256.yaml`/`early_gap_b1024.yaml`
  (budget-escalation fallbacks — step down only if compression rarely
  fires at 512, step up only if it fires but breaks accuracy, never step
  down after breaking accuracy).
- **Scope note**: an earlier draft of this change also proposed a
  mistake-insertion probe (corrupting a verified intermediate arithmetic
  step and testing whether the answer changes). That is **not implemented**
  here — `CLAUDE.md` §1 and `README.md`'s Scope section both explicitly and
  repeatedly list "mistake insertion" as out of scope for this repository,
  and the technique is a standard chain-of-thought-faithfulness probe from
  the literature, which is exactly the category of conclusion §1's
  "Forbidden conclusions" clause exists to rule out. Implementing it would
  require un-freezing that boundary first, with its own dated entry here —
  deliberately deferred rather than done silently alongside an otherwise
  in-scope addition.

## 2026-07-15 — Second pre-GPU audit: orchestration/pipeline completeness (no frozen §4 setting changed)

A second audit found the orchestration layer (CLI commands and the shell
scripts driving them) incomplete in ways that would have made the frozen
`scripts/run_stage1b.sh`/`run_stage2.sh` fail outright or silently skip
work, independent of the generation/replay correctness fixes above.

- **`kvcot calibrate-budget` was a stub** that always printed "no results
  exist" and returned failure. Implemented for real: reads each candidate
  budget's `results/decisions/stage1b_budget_<N>.json` (now actually written
  by a new `cmd_analyze` branch for `stage1b_budget_*` stages, using only
  `generate` output — Stage 1B's calibration decision never needed probes),
  reports the smallest budget passing both gates to
  `results/decisions/stage1b_recommendation.json`, and deliberately does NOT
  auto-write `configs/selected_operating_point.yaml` (§10: that stays a
  manual, reviewed step).
- **`replay-probe --condition rkv_selected` was broken** — `generate`
  resolved the placeholder to `rkv_b{budget}`, but `replay-probe` never did,
  so it looked for a nonexistent `rkv_selected.jsonl` and passed the literal
  placeholder to `build_policy()`. Both commands now share one
  `_resolve_condition` helper so this can't drift again.
- **Compaction events were still recorded once per R-KV layer inside
  `kvcot.generation.replay`'s `CompactionTracker`** (`compaction.note_event()`
  was called inside the per-layer sync loop) — the same "events x n_layers"
  inflation the first audit fixed in `cli.py`/`decode.py`, still present in
  the replay path itself. `_sync_layer_after_call` now only reports whether
  ITS layer fired; a new `_note_event_once` cross-checks all R-KV layers
  agree and records the event exactly once.
- **`--resume` never actually checked identity.** `kvcot.utils.io`'s own
  docstring promises "schema-valid completed records with matching
  config/model/upstream hashes"; the real logic only checked `record_id`
  membership. Added `_verify_resumable_record_ids` (schema validation +
  config/model/tokenizer/upstream-commit match, dotted-path comparison) and
  wired it into both `generate` and `replay-probe`'s resume paths — a
  mismatch now refuses loudly with a clear diagnostic instead of silently
  mixing identities in one output file.
- **Question hashes were computed but never checked.** `cmd_generate` now
  re-hashes each manifest row's question text and compares it to the
  manifest's own recorded `question_hash` before generating anything against
  it (catches a corrupted/hand-edited manifest one layer earlier, §5).
- **`RetentionSummary` was defined but never populated.** `cmd_generate` now
  measures it at end of each R-KV/patched-noop base generation from data
  already computed in that command (physical cache lengths, final absolute
  position) — no extra GPU passes. `ProvenanceRetentionSummary` (prompt/
  reasoning-token retention) and `BaseRunRecord.replay_state_hash` remain
  unpopulated — computing them without extra GPU cost would require
  restructuring `generate_base`'s hot decode loop to track full per-KV-head
  provenance, which risked meaningfully increasing Stage 2 wall-clock; out
  of scope for this pass, flagged here rather than silently left as a gap.
- **The three snapshot hashes on `ProbeRunRecord` hashed proxies, not
  content** (`snapshot_cache_hash` hashed only cache-length shapes;
  `snapshot_provenance_hash` hashed only the event-step list;
  `snapshot_state_hash` hashed two integers) — none would actually detect a
  divergence in the data they're named after. `replay-probe` now hashes the
  real K/V tensor bytes, the real per-layer/per-KV-head absolute source
  positions, and the real scheduling/bookkeeping state respectively.
- **`RunManifest` was imported but never constructed**; `kvcot.analysis.plots`
  functions existed but were never called; the stage0_smoke.yaml-advertised
  "throughput measurement + Stage 2 wall-clock extrapolation" didn't exist.
  `generate`/`replay-probe` now write a `RunManifest` per invocation
  (`{condition}_generate_manifest.json` / `{condition}_replay_probe_manifest.json`);
  `analyze` now writes `results/figures/agreement_curve.png` and
  `delta_eas_distribution.png` for stage2-shaped stages, and a rough
  throughput/wall-clock extrapolation decision JSON for `stage0_smoke`.
  `plot_realized_retention` remains unwired — no per-snapshot retention data
  source exists without the schema change noted above.
- **`Makefile`'s `dry-run` target used `--condition rkv`**, which is not a
  condition any stage config defines (`stage0_smoke.yaml` defines `rkv_b96`)
  — fixed to `rkv_b96`.

## 2026-07-15 — Pre-GPU correctness fixes (no frozen §4 setting changed)

Bug fixes found in a pre-run audit against the pinned upstream + transformers
4.55.4 semantics. None of these alters a `configs/lock.yaml` frozen setting;
they fix defects that would otherwise crash the probe stage or corrupt
provenance. Recorded here for traceability.

- **Blocker 1 — probe branch from an empty cache.** `restore_snapshot`
  item-assigned into `cache.key_cache[i]` on a freshly constructed
  `DynamicCache()`. On transformers 4.55.4 `key_cache` is a deprecated
  `KeyValuesWrapper` property whose `__setitem__` does `setattr(layers[idx],…)`
  with no growth, and a bare `DynamicCache()` pre-creates only ONE layer → the
  probe stage raised `IndexError` at layer 1. Fixed by populating the fresh
  cache through the public `cache.update(...)` path (`_populate_fresh_cache`).
- **Blocker 2 — inflated compaction count.** Base generation counted
  `events × n_layers` and stored a per-layer `[0,1,2,…]` enumeration in
  `compaction_event_steps`. `generate_base` now tracks true events at their
  absolute positions (one count, assert all R-KV layers agree). GPU test's
  `n_compactions` and the `>=2` hard gate corrected to count events, not
  events×layers.
- **#3 — cut-position arithmetic** in `replay-probe` dropped `think_start_index`
  (masked only by the pre-opened `<think>` template); fixed to match the
  documented replay contract, `cut_index` recomputed accordingly.
- **#4 — replay EOS asymmetry**: replay now stops once every requested snapshot
  is captured (all ≤ think_end), so it no longer feeds the trailing answer/EOS
  the base run never fed.
- **#5 — exact-budget assert crash**: `kv_cache_len == budget` records a
  zero-eviction compaction event; the equality assert in `_sync_layer_after_call`
  now treats bookkeeping growth as the event ground-truth and only raises on the
  genuinely-impossible reverse (cache shrank with no event).
- **#6** deleted dead `generate_probe_answer`. **#7** Stage-2 `--dry-run` now
  prints a clean prerequisite error instead of a traceback; base records now
  carry `dataset_config/revision/fingerprint` and the full R-KV `method_config`.
- **Analysis wiring**: `kvcot analyze` now actually computes the Stage-2 primary
  result (EAS → Delta_EAS → Wilcoxon/bootstrap/accuracy + attrition funnel) via
  the new `kvcot.analysis.pipeline`, instead of importing those helpers and
  never calling them. Stage 1A measurability now counts real answer-changes.

## 2026-07-15 — Initial build

- Repository built from scratch per the original build brief, on a
  CPU-only machine (no GPU, no model weights downloaded).
- Upstream R-KV pinned at commit `45eaa7d69d20b7388321f077020a610d9afb65bd`
  (verified to exist before use — `docs/UPSTREAM_AUDIT.md` §0).
- Model/tokenizer revision pinned at
  `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562` (resolved via HF metadata API,
  no weights).
- All frozen settings in `configs/lock.yaml` set to their brief-specified
  values for the first time — nothing to diff against, so no prior value is
  listed.
- Four GSM8K manifests (smoke=20, calibration=50, main=200, disjoint) and
  one MATH-500 backup manifest (100 rows, levels 3-5) frozen with real
  network access, seed=13.
- No GPU code executed; no Stage 0-2 run.
