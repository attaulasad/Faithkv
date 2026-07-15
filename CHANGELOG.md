# Changelog

Frozen settings (`configs/lock.yaml`, and Sections 1/4/8/9 mirrored into
`CLAUDE.md`) may only change via a dated entry here, added **before** the
run that depends on the change (per the build brief). Entries are ordered
newest first.

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
