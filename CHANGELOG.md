# Changelog

Frozen settings (`configs/lock.yaml`, and Sections 1/4/8/9 mirrored into
`CLAUDE.md`) may only change via a dated entry here, added **before** the
run that depends on the change (per the build brief). Entries are ordered
newest first.

## 2026-07-18 — Fixed-trace protocol v3, second pass: selection-to-replay wiring, strict accuracy gate, screen-validity fix (secondary, additive; no frozen §1/§4/§8/§9 value changed)

A second review of the 2026-07-17 protocol-v3 commit (`45bf2d8`) found the
frozen-cache design and cache-schedule simulator sound (independently
re-checked against the archived protocol-v2 raw GPU data: 0 mismatches
across all 90 (example, fraction) comparisons), but four gaps that would
have wasted GPU time or produced an untrustworthy screen on the next real
run. Fixed here, still before any GPU spend:

- **The selection file was written but never consumed.** `kvcot
  inspect-fixed-trace --write-selection` wrote `results/selections/
  {stage}.json`, but neither `replay-fixed-trace` nor `analyze-fixed-trace`
  had any way to read it back — the only way to run a selected subset was
  one `--problem-index` invocation per example, reloading the model every
  time. Fixed: both commands gained `--selection-file`
  (`kvcot.cli._load_fixed_trace_selection`), which verifies the selection
  was computed against THIS exact config/base-file/budget/divide_length/
  stage_name before trusting it (raises `SelectionFileMismatchError`
  otherwise), then restricts the run to exactly the selected
  `base_record_id`s. `replay-fixed-trace` refuses to proceed (`SystemExit`)
  if the selection's `n_selected` is already below `min_eligible_examples`.
  The selection path and its SHA-256 are now recorded on every
  `replay-fixed-trace` `RunManifest` (`RunManifest.selection_path`/
  `selection_file_sha256`, new optional fields).
- **`--max-selected` capped the ranked candidate list BEFORE filtering to
  predicted-eligible ones** (`kvcot.cli._write_fixed_trace_selection`) — a
  real bug: if the first `--max-selected` candidates by `source_row_index`
  happened to be predicted-ineligible, the selection could come back with
  `n_selected=0` even though plenty of eligible candidates existed later in
  the ranking. Fixed: candidates are ranked (uncapped) first, filtered to
  `predicted_eligible`, and ONLY THEN capped. New `FixedTraceSettings.
  max_selected_examples` lets the cap be pre-registered in the stage config
  itself, not chosen after seeing the selection; `--max-selected` on the
  CLI still overrides it when explicitly given.
- **The documented accuracy-gate run order was impossible.**
  `docs/GPU_VALIDATION_PLAN.md` said to run `analyze-fixed-trace` right
  after natural R-KV generation, before any fixed-trace replay, to get the
  pilot accuracy screen — but `analyze-fixed-trace`/`run_fixed_trace_
  analysis` unconditionally reads the fixed-trace probe files too, which do
  not exist yet at that point in the order; the command would simply crash.
  Separately, `build_accuracy_screen`'s intersection-only pairing means one
  matching `(source_row_index, global_seed)` pair could report
  `pilot_accuracy_plausible: True` even while 49 of 50 R-KV records were
  never generated at all. Fixed with a new CPU-only command, **`kvcot
  check-fixed-trace-accuracy`**, that reads ONLY the natural `full.jsonl`/
  `{replay_condition}.jsonl` files (never fixed-trace probes) and wraps the
  pairing in `kvcot.analysis.fixed_trace.build_strict_accuracy_gate`, which
  requires (all independently, never silently skipped): an exact expected
  record count on both sides, no duplicate `(source_row_index, global_seed)`
  keys, an IDENTICAL key set between the two files (not just a
  large-enough intersection), every record schema-valid, one shared
  `(config_sha256, upstream_rkv_commit, model_revision, tokenizer_revision)`
  identity matching the current invocation's own config/lock, consistent
  `condition` fields, and `pilot_accuracy_plausible`. Writes `results/
  decisions/{stage_name}_accuracy_gate.json`, exits nonzero on any failure.
  `build_accuracy_screen` itself is unchanged (still an honest, low-level
  intersection-based pairing helper) — the strict gate wraps it, rather
  than changing what it means.
- **Screen-level validity still gated on `actual_compression_rate`, not
  `meaningful_compression_rate`, even under `require_meaningful_compression
  =True`.** The per-PAIR eligibility gate (`FixedTraceEligibility.
  rkv_meaningful_compression_at_f1`, 2026-07-17) worked correctly, but
  `build_screen_validity` kept checking the old any-eviction rate at the
  SCREEN level regardless — a batch where every example evicted at least
  one token (`actual_compression_rate=1.0`) but almost none evicted enough
  to be "meaningful" could still report `screen_valid: True`. Fixed: when
  `require_meaningful_compression=True`, the screen gate now checks
  `meaningful_compression_rate` against new setting `FixedTraceSettings.
  min_meaningful_compression_rate` (default 0.7) INSTEAD of
  `actual_compression_rate`/`min_actual_compression_rate`; v2 stages
  (`require_meaningful_compression=False`, the default) are completely
  unaffected.
- **Analysis completeness guard added**: `run_fixed_trace_analysis` gained
  `selected_base_record_ids` (wired from `--selection-file`) — when given,
  `kvcot.analysis.fixed_trace._verify_selection_completeness` requires
  every selected example to have all 9 fractions recorded under BOTH
  replay policies, and raises `ValueError` (abort, never silent attrition)
  if any are missing — a partially-completed replay (e.g. 5 of 20 selected
  examples actually written) could otherwise still clear
  `min_eligible_examples` and produce a decision JSON as if the screen were
  actually complete. All curves/counts are also scoped to exactly the
  selected set once a selection is active (`n_shared == n_selected`
  becomes an enforced invariant, not just a hope).
- Still true from the 2026-07-17 entry: no GPU exists on this machine.
  `tests/integration/test_rkv_schedule_prediction_gpu.py` (new) and
  `tests/integration/test_frozen_probe_gpu.py` (new) implement the two
  mandatory GPU gates that entry promised but did not yet contain — both
  collect and auto-skip cleanly here (13 GPU tests skipped, up from 11) and
  must be run for real, and pass, before trusting either the schedule
  simulator or `frozen_at_cut` against a real model.

## 2026-07-17 — Fixed-trace protocol v3: frozen-cache causal probe, meaningful-compression gating, exact cache-schedule simulator (secondary, additive; no frozen §1/§4/§8/§9 value changed)

**Protocol v2's real GPU screen (`configs/early_gap_v2_b128.yaml`, committed
result `results/decisions/early_gap_v2_b128_fixed_trace.json`) produced
`n_eligible=3` against a `min_eligible_examples=5` floor and
`mean_f1_rkv_retention_ratio=0.7456` against a `max_mean_f1_retention_ratio
=0.7` ceiling — `screen_valid: false`, `hypothesis_status: "not_tested"`, as
designed.** This is a valid negative screening outcome, not a bug in the
correctness machinery (patched-noop parity, no-state-leak, and replay
identity GPU gates all passed; every one of the 180 fixed-trace probes
produced a valid boxed answer; cross-file/schema identity checks held). Two
mechanisms specifically explain why it came back invalid, diagnosed from the
raw per-example records already in the committed decision JSON:

1. **`n_actual_compression_active=10/10` overstated real compression.**
   `rkv_actual_compression_at_f1` (added in v2) only requires the physical
   cache to be smaller than the FullKV-equivalent slot count by ANY amount —
   `source_row_index=148` counts as "compression active" at
   `rkv_f1_retention_ratio=0.9959`, i.e. one token evicted out of ~240. Six
   of the ten shared examples sit at retention ≥0.94; the schedule mechanics
   audited in `docs/UPSTREAM_AUDIT.md` H4 (periodic `divide_length=128`
   compaction checks, real eviction gated on `kv_cache_len >= budget`) mean a
   trace can cross the schedule boundary long before it has accumulated
   enough tokens for the eviction itself to matter. An "any eviction"
   boolean cannot distinguish this from a trace that lost most of its cache.
2. **The v2 probe protocol lets R-KV compact again while writing its own
   answer.** Five of the ten shared examples (`source_row_index` 30, 176,
   262, 271, 307) failed eligibility via `rkv_evicted_during_answer_probe` —
   a real compaction event fired during the teacher-forced closing-marker/
   suffix/greedy-answer tokens fed by `branch_and_probe`, after the snapshot
   the probe was supposed to measure. This is a protocol confound, not
   evidence about the hypothesis: it measures a cache state that moved again
   after the reasoning cut, not the cut itself.

Both are addressed here, **before any further GPU spend**, without touching
the frozen v2 output (`configs/early_gap_v2_b128.yaml`,
`results/decisions/early_gap_v2_b128_fixed_trace.json`, and the raw v2
probes are never modified, resumed into, or reinterpreted under v3 —
protocol v2 is an archived, invalid-but-real result, not a draft).

- **`kvcot.analysis.rkv_schedule`** (new, torch-free): an exact CPU
  simulator of the upstream schedule/trigger mechanics audited in
  `docs/UPSTREAM_AUDIT.md` H4 and §3.1/3.3 (`self.length` cumulative
  counter incremented once per top-level forward call; the compression flag
  for call N+1 is `self.length % divide_length == 0` computed at the end of
  call N; the very first forward call in a fresh process always attempts
  eviction regardless of schedule, since `compression` initializes to
  `None`; an attempted eviction is a no-op whenever the current physical
  cache length is still below `budget`). Used to *predict*, from a FullKV
  base record alone (prompt length + generated-token count), the physical
  R-KV cache length and retention ratio at every probe fraction before
  spending any GPU time — this is what makes deterministic, outcome-blind
  trace selection (below) possible on this CPU-only machine.
- **`kvcot.generation.replay.branch_and_probe`** gained
  `probe_cache_mode: Literal["native", "frozen_at_cut"] = "native"`
  (default preserves exact v2/existing behavior and all existing tests
  unchanged). `"frozen_at_cut"` forces `compression=False` on every R-KV
  layer before every teacher-forced/generated token fed during the probe
  (close marker, control suffix, and each greedily generated answer token) —
  addresses failure mode 2 above by construction, not by post-hoc filtering:
  the cache snapshot the probe branched from cannot be disturbed by the
  model's own answer-writing. A fresh per-layer cache-length assertion
  (`final_length == snapshot_length + tokens_fed`, checked every layer) now
  raises loudly if this is ever violated, rather than silently producing an
  ineligible pair after the GPU time to produce it is already spent.
- **`kvcot.config.FixedTraceSettings`** gained (all with defaults that leave
  `configs/early_gap_v2_b128.yaml` byte-behavior-identical):
  `meaningful_retention_ceiling` (default 0.7), `require_meaningful_compression`
  (default `False`), `min_meaningfully_compressed_scored_fractions` (default
  0), `probe_cache_mode` (default `"native"`), `max_pilot_accuracy_drop`
  (default 0.10).
- **`kvcot.analysis.fixed_trace`**: new `rkv_meaningful_compression_at_f1`
  eligibility field — `rkv_f1_retention_ratio <= meaningful_retention_ceiling`
  — additive alongside (never replacing) the existing
  `rkv_actual_compression_at_f1` "any eviction" diagnostic; only enforced as
  an eligibility gate when `require_meaningful_compression=True` (v2 configs
  never set this, so v2's frozen eligibility semantics and its archived
  decision JSON are unaffected). New CPSS (Compression-Active
  Prefix-Sufficiency Sensitivity) metric — `compute_cpss`/`compute_delta_cpss`
  — restricted to the subset of the 7 scored fractions where
  `rkv_retention_at_fraction <= meaningful_retention_ceiling` (requires at
  least `min_compressed_scored_fractions_for_cpss`, default 2, else `None`) —
  a DIFFERENT metric from PSS/Delta_PSS, additive, never pooled with it. New
  per-fraction `retention_summary_by_fraction` (count/mean/median/min/max/
  meaningful-compression-rate) and `compression_rate_by_fraction` in the
  decision JSON — this is what would have shown the sawtooth retention
  pattern (0.994 at f=0.125 down to 0.746 at f=1) without hand-decoding raw
  records. New `all_shared_full_curve`/`all_shared_rkv_curve` keys carry
  forward the EXACT existing `full_curve`/`rkv_curve` semantics (still
  present, unchanged, for v2 backward compatibility) under clearer names;
  new `*_eligible_only` curve variants are additive.
- **New natural-accuracy screen** (`build_accuracy_screen` in
  `kvcot.analysis.fixed_trace`, wired automatically into `kvcot
  analyze-fixed-trace`/`run_fixed_trace_analysis` — no separate CLI command,
  one artifact per stage rather than two): pairs natural (non-fixed-trace)
  `full.jsonl`/`rkv_b{budget}.jsonl` base accuracy on the SAME manifest rows
  — v2 never generated a natural R-KV b128 run, so it could not establish
  `pilot_accuracy_plausible` (deliberately not named `accuracy_neutral` —
  §8.5 of `CLAUDE.md`/the build brief already states the primary paired
  200-problem accuracy check, `kvcot.analysis.stats.paired_accuracy_diff`,
  is the only test allowed to claim distributional accuracy preservation;
  this is a small-n stop/continue gate only). Only attempted for stages that
  set `require_meaningful_compression: true` (the v2/v3 discriminator) —
  `screen_valid` in the v3 fixed-trace decision is `False` whenever this
  screen is missing (natural R-KV base file not found) or
  `accuracy_difference_rkv_minus_full < -max_pilot_accuracy_drop`.
- **`kvcot inspect-fixed-trace --write-selection`** (new flag): deterministic
  trace selection using ONLY the FullKV base file's own correctness/cap/
  think-parse validity plus `kvcot.analysis.rkv_schedule`'s predicted
  retention — never any fixed-trace probe answer, PSS, or CPSS value (a
  selection that could see outcomes would not be a pre-registered screen).
  Writes `results/selections/{stage_name}.json` with the config hash, base
  file SHA-256, per-candidate predicted retention at every fraction, and
  selected/rejected row indices with reasons.
- **`configs/early_gap_v3_b128.yaml`** (new stage, never a resumption of
  `early_gap_v2_b128.yaml`'s `output_dir`): `limit: 50` (natural generation
  only — selection then narrows to a smaller replayed set),
  `probe_cache_mode: frozen_at_cut`, `require_meaningful_compression: true`,
  `min_meaningfully_compressed_scored_fractions: 2`.
- **Schema bump 1.3.0 -> 1.4.0** (`kvcot.schemas`): `FixedTraceProbeRecord`
  gained `protocol_version` (`Literal["v2", "v3"]`, default `"v2"` so
  existing v2 records/fixtures continue to validate unchanged),
  `probe_cache_mode`, `meaningful_compression_at_cut`,
  `compressed_scored_fraction`. `SCHEMA_VERSION` bump means a stale
  1.3.0-shaped record fails validation outright under the new
  `Literal["1.4.0"]` — exactly the same discipline the 1.2.0->1.3.0 and
  1.1.0->1.2.0 bumps established; a v2 output directory must never be
  resumed under this schema.
- **Still open**: this is all CPU-buildable/testable infrastructure — no
  GPU exists on this machine (unchanged from every prior entry). The exact
  schedule simulator's predictions have not been cross-checked against real
  GPU-measured retention (the raw v2 fixed-trace probe JSONL is gitignored
  per `README.md`'s stated layout and was not preserved outside the GPU
  host), only against the audited mechanics and hand-derived unit cases —
  `tests/integration/test_rkv_schedule_prediction_gpu.py` (new, GPU-only)
  is the mandatory one-example gate before trusting it against real
  generations. Natural R-KV b128 generation on the 50-example manifest has
  not been run — the accuracy screen above cannot report a real number
  until it is. `docs/GPU_VALIDATION_PLAN.md` §(new) documents the required
  run order; do not skip the one-example frozen-probe/schedule-prediction
  gate before committing to the full 20-example replay.

## 2026-07-16 — Fixed-trace protocol v2, fourth pass: policy-role validation, resume identity gap (secondary, additive; no frozen §1/§4/§8/§9 value changed)

A fourth external review of the third-pass cross-file identity commit found
the identity checks it added still had two gaps, both in the same spirit
(catching a mislabeled/stale file before it silently corrupts a comparison):

- **A record's own `replay_policy_condition` was never checked against the
  file it was loaded from.** `_assert_consistent_identity`/
  `_assert_shared_trace_source` (prior entry) check `config_sha256`/
  `upstream_rkv_commit`/`model_revision`/`tokenizer_revision` and
  `trace_source_condition` agreement, but nothing checked that a probe
  file's records actually declare the replay policy the filename convention
  implies. A `full_on_full_fixed_trace_probes.jsonl` file whose records
  declare `replay_policy_condition="rkv_b128"` (e.g. from an accidental file
  swap or rename) was silently accepted, which can flip which curve gets
  called FullKV vs. R-KV. Fixed: `load_fixed_trace_records` now raises if any
  row's `replay_policy_condition` disagrees with the `replay_condition` it
  was called with. Also added: `_validate_base_records` now checks the
  canonical base file's own `condition` field against `trace_condition`, so
  a base file recorded under a different condition (e.g. an R-KV file
  passed in as the canonical trace by mistake) is rejected too.
- **`cmd_replay_fixed_trace`'s `--resume` identity check omitted
  `model_revision`/`tokenizer_revision`.** `FixedTraceProbeRecord` has
  carried both fields since the 1.2.0 -> 1.3.0 bump (prior entry)
  specifically so cross-file identity could be checked — but the
  `expected_identity` dict `cmd_replay_fixed_trace` builds for `--resume`
  still only carried `config_sha256`/`upstream_rkv_commit`, so resuming into
  a fixed-trace probe file recorded under a stale model/tokenizer revision
  was accepted at resume time. `run_fixed_trace_analysis` would eventually
  reject the resulting mixed-identity output directory, but only after
  wasting GPU time producing it. Fixed: `expected_identity` now also carries
  `model_revision`/`tokenizer_revision`, matching what `cmd_generate` already
  did.
- Also corrected two stale comments found during this pass: a
  `_validate_fixed_trace_probe_records` docstring still claimed
  `load_fixed_trace_records` "keeps the last" duplicate `(base_record_id,
  fraction)` row, when it has raised on that case since the prior entry; and
  a schema test's comment still described the fixed-trace suffix as always
  empty, a protocol-v1 behavior protocol v2 (two entries ago) replaced with a
  non-empty teacher-forced boxed-answer prefix. `docs/SCHEMA.md` and the
  archived protocol-v1 README were also still citing schema `"1.2.0"` after
  the prior entry's bump to `"1.3.0"`.
- 224 CPU tests pass (up from 217 — new coverage for both validation gaps
  plus the two stale-comment fixes). GPU test files still only collect and
  skip (11 skipped) — no GPU exists in this environment. Same still-open
  items as the prior entry (unrecoverable b256/b1024 raw data, unimplemented
  MATH-500 answer equivalence, no reviewed PR for this pass either).

## 2026-07-16 — Fixed-trace protocol v2, third pass: config-limit ignored, cross-file identity, duplicate rows (secondary, additive; no frozen §1/§4/§8/§9 value changed)

A third external review of the protocol-v2 hardening commit found two
blocking repository bugs (not scientific-mechanics bugs — the anchor
extraction, budget selection, and f=1-eviction fixes from the prior two
entries were all confirmed correct) that would have let a GPU rerun either
silently process the wrong number of examples or silently pair
inconsistent data:

- **`StageConfig.limit` was completely ignored.** `_load_manifest_filtered`
  only ever consulted `args.limit` (the CLI `--limit` flag); a stage
  config's own `limit:` (e.g. `early_gap_v2_b128.yaml`'s `limit: 10`
  against a 50-row manifest) had no effect at all. Every documented
  fixed-trace command in `docs/GPU_VALIDATION_PLAN.md` omits an explicit
  `--limit`, relying entirely on the config's declared limit — so the
  documented "ten-example screen" was actually running against all 50
  rows. Fixed: `effective_limit = args.limit if args.limit is not None
  else stage.limit`. Verified end-to-end with the real config
  (`--dry-run`, no `--limit`): `generate` now reports `rows: 10`,
  `replay-fixed-trace` reports `planned examples: 10` /
  `planned probe records: 90`, matching the documented n=10 exactly.
- **Cross-file identity was never checked.** `_validate_base_records`/
  `_validate_fixed_trace_probe_records` (added in the prior entry) each
  only verified ONE file's own internal consistency — nothing compared the
  canonical base file against either fixed-trace probe file, or either
  probe file against the other. A base file from one config/model/upstream
  pin could be silently paired against probe files from a different run.
  Fixed: `FixedTraceProbeRecord` gained `model_revision`/
  `tokenizer_revision` fields (schema bumped `1.2.0` -> `1.3.0`) so its
  identity is directly comparable to `BaseRunRecord`'s; a new
  `_assert_consistent_identity` cross-checks all three files' identities
  against each other, and `cmd_analyze_fixed_trace` now also passes the
  CURRENT invocation's own `(config_sha256, upstream_commit, model_revision,
  tokenizer_revision)` — computed from `args.config` and the freshly loaded
  lock, previously loaded and silently discarded — so stale data cannot be
  analyzed even if it happens to be internally self-consistent.
- **Duplicate `(base_record_id, fraction)` rows were silently overwritten.**
  `load_fixed_trace_records` now raises on a duplicate key instead of
  letting the later row win silently — such a duplicate can only arise
  from a corrupted, hand-edited, or improperly concatenated file (the
  writer itself already refuses a duplicate `record_id` within one run).
- **`require_boxed_extraction` was a dead config field.** Declared in every
  fixed-trace stage config but never read by any code (boxed extraction
  was always required unconditionally). Changed from a plain `bool` to a
  frozen `Literal[True]` — settable to `True` (or omitted), never silently
  disabled to `False` with no effect.
- Fixed a stale docstring still naming the retired
  `no_rkv_eviction_during_scored_probes` field (renamed
  `no_rkv_eviction_during_answer_probes` two entries ago) and corrected
  `docs/GPU_VALIDATION_PLAN.md`'s one-example-gate instructions, which
  implied `--limit 1` applies to `analyze-fixed-trace` too (it takes no
  such flag — it only reads what `replay-fixed-trace` already wrote).
- **Still open, unchanged**: no GPU exists in this environment to actually
  exercise any of this — 217 CPU tests pass (up from 203), GPU test files
  still only collect and skip. Raw b256/b1024 probe data remains
  unrecoverable through code (§ prior entry). MATH-500 answer equivalence
  remains unimplemented. This round also went directly to `main` without a
  reviewed PR, same as the prior two — flagged again here since external
  review has now raised it twice; a subsequent change may switch to a
  branch+PR flow if that continues to matter.

## 2026-07-16 — Fixed-trace protocol v2 hardening: f=1 eviction gap, budget too large, analysis-input validation (secondary, additive; no frozen §1/§4/§8/§9 value changed)

External review of the first protocol-v2 commit (`20e2ad6`, merged as
`b883fd3`) found the anchor-extraction fix correct but three remaining gaps
that would have let a GPU rerun waste money on a screen that still cannot
produce a valid result, plus one real eligibility bug. Fixed here, still
before any GPU spend:

- **b512/b1024 cannot compress on this manifest; b256 falls short too.**
  Recalculating from the real GPU data already collected
  (`logs/b512_accuracy_compaction.log`): observed prompt+think lengths on
  the `gsm8k_calibration_50` sample never exceed budget 512 or 1024 at all
  (`mean_final_retention_ratio: 0.98` — confirms this), and exceed budget
  256 on at most ~6/10 traces — structurally below
  `FixedTraceSettings.min_actual_compression_rate` (0.70), and even
  maximally aggressive compaction at 256 cannot bring mean retention under
  the 0.70 ceiling on traces this short. **Added
  `configs/early_gap_v2_b128.yaml`** (new `stage_name`/`output_dir`, never a
  resumption of an `early_gap_b*.yaml` directory) as the first budget with
  a realistic chance of clearing both thresholds on this manifest.
  Thresholds themselves were **not** weakened to make an existing budget
  pass — per the review's explicit instruction, a budget too large for the
  data is fixed by picking a smaller budget (or longer traces), not by
  lowering the bar.
- **`kvcot inspect-fixed-trace` strengthened** (`src/kvcot/cli.py`) with two
  new arithmetic-only stop conditions, on top of the existing "nothing
  exceeds the budget" check: (1) `fraction_of_traces_longer_than_budget` is
  an upper bound on the achievable `actual_compression_rate` — if that
  bound is already below `min_actual_compression_rate`, the eligibility
  gate is mathematically unreachable at this budget; (2)
  `mean_optimistic_retention` (`budget/length` per trace, the most
  aggressive possible compaction) is a lower bound on achievable mean
  retention — if even that best case exceeds
  `max_mean_f1_retention_ratio`, no real run can pass either. Both checks
  only run when the stage config declares `fixed_trace:` settings.
- **Eligibility gap: answer-time eviction was never checked for the f=1
  anchor itself** (`src/kvcot/analysis/fixed_trace.py`,
  `FixedTraceEligibility`) — the check only scanned the 7 scored fractions.
  Every scored fraction's match is scored against the f=1 anchor's own
  answer, so an eviction while the anchor was writing ITS OWN answer is
  exactly as disqualifying as one on a scored fraction; a synthetic
  f=1-only-eviction case previously came back eligible with zero failure
  reasons. `no_rkv_eviction_during_answer_probes` (renamed from
  `no_rkv_eviction_during_scored_probes`) now covers
  `PROBE_FRACTIONS_SCORED + (1.0,)`. Regression test added:
  `test_f1_only_answer_time_eviction_makes_pair_ineligible`.
- **`run_fixed_trace_analysis` now validates every input record**
  (`src/kvcot/analysis/fixed_trace.py`, `_validate_base_records`/
  `_validate_fixed_trace_probe_records`) against `BaseRunRecord`/
  `FixedTraceProbeRecord` (rejecting a stale `schema_version` outright, via
  the `Literal["1.2.0"]` field) and checks every record shares one coherent
  `(config_sha256, upstream_rkv_commit[, model_revision, tokenizer_revision])`
  identity, before any pairing/scoring happens. Previously the analysis read
  JSONL as plain dicts with no schema check at load time, so a protocol-v1
  directory (or one mixing two different runs) could be silently
  "analyzed" as if it were valid current input.
- **Archived stale protocol-v1 decision JSONs**
  (`results/decisions/early_gap_b{256,512,1024}_fixed_trace.json`,
  `early_gap_b512_accuracy_compaction.json`) to
  `results/decisions/archive/protocol_v1_2026-07-16/` (with a README
  explaining why) rather than leaving them under names that look like
  current results. These files are `schema_version "1.1.0"`, `git_dirty:
  true`, and — per the diagnosis above — describe a screen that produced
  zero eligible examples; they must never be read as evidence for or
  against G1. Corresponding raw probe data for b256/b1024 (only b512 was
  ever actually generated) does not exist and cannot be reconstructed —
  a fresh GPU run under `early_gap_v2_b128.yaml` is required.
- **Still open, unchanged from the prior entry**: MATH-500 answer
  equivalence is still not implemented (plain string equality only) — do
  not switch to MATH-500 traces until that lands, tested. Protocol v2 has
  not yet been exercised on a real GPU — CPU tests (203 passing) and
  `python -m py_compile` are the only checks possible on this build
  machine; the one-example GPU gate in `docs/GPU_VALIDATION_PLAN.md` is
  still required before any 10-example rerun.

## 2026-07-16 — Fixed-trace protocol v2: boxed-answer prefix, realized-compression gating (secondary, additive; no frozen §1/§4/§8/§9 value changed)

**Protocol v1 produced no scientific result.** The first fixed-trace GPU
screen (b512, seed=42, n=10) ran end-to-end cleanly — sampled base accuracy
9/10 under both conditions, zero cap hits on generation — but `n_eligible =
0` at every budget tested, for two independent, diagnosable reasons found by
decoding the raw probe text:

1. **The f=1 anchor was garbage on every example.** The fixed-trace suffix
   was deliberately empty (`FIXED_TRACE_SUFFIX_TEXT = ""`, to avoid cueing
   recomputation), and probe decoding used the frozen 48-token budget
   (`configs/lock.yaml`'s `probes.max_new_tokens`). R1-Distill's answer mode
   is a verbose structured write-up that essentially never reaches a
   `\boxed{...}` (or even an explicit `Final answer:`) within 48 tokens, so
   extraction fell through to the conservative final-number fallback tier on
   nearly every probe and grabbed an incidental mid-sentence number as the
   "anchor" — noise, not an answer. Every reported PSS/curve value from that
   screen was contaminated (fallback-extracted noise compared against
   fallback-extracted noise) and must not be read as evidence in either
   direction.
2. **Eligibility gated on a recorded compaction EVENT COUNT, not realized
   compression.** At the exact budget boundary R-KV can record a compaction
   event that evicts zero tokens (`kvcot.generation.replay`'s documented
   boundary case) — `rkv_had_replay_compaction` (`count > 0`) let such pairs
   through as "eligible" even though the physical cache never actually
   shrank.

Neither failure says anything about the underlying hypothesis (G1) — this
screen tested the elicitation machinery and found it broken, not the
compression question. All kill criteria from the earlier design chats remain
live and untriggered; the infra (gates, replay, schemas, eligibility logic)
is fully reusable once these two defects are fixed. Fixed here, before any
rerun:

- **`src/kvcot/probes/templates.py`**: `FIXED_TRACE_SUFFIX_TEXT` changed from
  `""` to `"\n\nFinal answer: \\boxed{"` — a teacher-forced FORMAT prefix
  (identical across conditions, fed as plain tokens exactly like the closing
  `</think>` marker), never a natural-language recomputation instruction
  ("solve again"/"recalculate"/"use the question"/"explain your answer" are
  all still forbidden, per the module's own documented rationale).
- **`src/kvcot/config.py`**: new `FixedTraceSettings` (own
  `probe_max_new_tokens` default 64, `min_eligible_examples`,
  `min_actual_compression_rate`, `max_mean_f1_retention_ratio`), attached as
  `StageConfig.fixed_trace`, required (not optional) by
  `cmd_replay_fixed_trace`/`cmd_analyze_fixed_trace` — deliberately
  **separate** from the frozen `configs/lock.yaml` `probes.max_new_tokens:
  48`, so a fixed-trace-motivated change can never silently alter the frozen
  primary EAS experiment. `configs/early_gap_b{256,512,1024}.yaml` each gained
  a `fixed_trace:` block.
- **`src/kvcot/utils/answers.py`**: `has_complete_boxed_answer` (stop
  predicate for probe decoding) and `answers_match_or_none` (three-valued
  match — `None` means "could not extract," `False` means "a valid but
  different answer"; the two must never be conflated, since coercing the
  first into the second hides extraction breakage inside what looks like a
  normal disagreement rate). `answers_match` (the frozen primary path's
  two-valued match) is unchanged.
- **`src/kvcot/generation/replay.py`**: `branch_and_probe` accepts an
  optional `stop_predicate` (checked after every generated token, in
  addition to EOS) so fixed-trace decoding halts the instant a box closes —
  never used by the frozen primary replay-probe path. `ProbeResult` gained
  `stop_reason`, `final_absolute_position`, `final_cache_lengths_per_layer`
  so callers can detect an eviction that happened *while writing the answer*
  itself, not just at the reasoning cut.
- **`src/kvcot/cli.py`** (`cmd_replay_fixed_trace`): extraction now runs over
  the reconstructed prefix+generated text (`probe_extraction_text`), never
  generated tokens alone; the stop predicate is wired in; realized retention
  and actual-compression are measured at every snapshot
  (`replay_retention_at_cut`, `actual_compression_at_cut` — physical cache
  length vs. FullKV-equivalent slots, never the configured budget);
  answer-time eviction is detected (`probe_actual_eviction_during_answer`).
  New CPU-only `kvcot inspect-fixed-trace` command: reports think-span/
  prompt+think-span length statistics against the configured R-KV budget and
  refuses to proceed if no trace in the file is even longer than the budget
  (this cannot prove compression will happen, only rule out the case where
  it definitely cannot) — run this before spending GPU time on
  `replay-fixed-trace`.
- **`src/kvcot/schemas.py`**: `FixedTraceProbeRecord` gained
  `probe_extraction_text`, `probe_stop_reason`, `probe_cap_hit`,
  `replay_retention_at_cut`, `actual_compression_at_cut`,
  `probe_cache_length_final_per_layer`, `probe_actual_eviction_during_answer`.
  `SCHEMA_VERSION` bumped `1.1.0` -> `1.2.0`, and every record's
  `schema_version` is now `Literal["1.2.0"]` (not just a string default) —
  a stale-schema record now fails Pydantic validation outright instead of
  being silently accepted. **Old protocol-v1 output directories must not be
  resumed under protocol v2** — start a fresh `output_dir`.
- **`src/kvcot/analysis/fixed_trace.py`**: eligibility (`FixedTraceEligibility`)
  reworked around realized compression (`rkv_actual_compression_at_f1`,
  `no_rkv_eviction_during_scored_probes`) instead of a recorded event count,
  plus new gates on each side's own f=1 anchor being a `"boxed"` extraction
  (`full_f1_anchor_boxed`/`rkv_f1_anchor_boxed` — a fallback anchor is never
  accepted) and on the canonical trace's own base answer being correct
  (`canonical_trace_base_correct`). PSS is `None` (never `0.0`) whenever a
  side's own anchor is invalid/fallback or any scored fraction failed to
  extract; `Delta_PSS` is additionally `None` whenever the pair fails full
  eligibility (in particular: no actual R-KV compression, or an answer-time
  eviction) even if both PSS values are individually defined. Descriptive
  curves (`fixed_trace_curve_by_fraction`) now return `None` for a fraction
  with zero valid measurements, never `0.0` — the two are different claims.
  New screen-level validity gate (`build_screen_validity`,
  `build_fixed_trace_decision`): `screen_valid` requires enough eligible
  examples, a high enough realized-compression rate, and low enough realized
  retention (all from `FixedTraceSettings`); `hypothesis_status` is
  `"not_tested"` when any of those fail, and even when the screen is valid
  this module never reports "positive"/"negative"/"gap exists"/"gap does not
  exist" — only descriptive counts, per its existing kill/continue-screen
  discipline.
- **GPU test process isolation** (`tests/integration/test_replay_gpu.py`,
  `test_probe_stability_gpu.py`): every patched-R-KV test and the FullKV
  identity test now run inside their own `multiprocessing.get_context
  ("spawn")` subprocess (never `fork`), mirroring the pattern already used in
  `test_patched_noop_parity_gpu.py`. Previously, several of these tests ran
  directly in the shared pytest process — since the R-KV monkeypatch on
  `transformers.models.qwen2` is process-global with no per-instance undo
  (`docs/UPSTREAM_AUDIT.md` H1), and `kvcot.generation.state.
  declare_process_mode` already refuses a second, conflicting mode in one
  process, mixing stock/patched tests (or two different R-KV configs) in one
  process was unsafe or outright broken (`reset_active_mode_for_testing()`
  only clears kvcot's own tracking variable, not the underlying monkeypatch).
  `_load_rkv_model` now calls `declare_process_mode("patched")` before
  `replace_qwen2(...)`, matching every real loader
  (`kvcot.generation.policies._PatchedPolicyBase.load`).
- **Scope note**: MATH-500 support (verified answer-equivalence for
  fractions/radicals/decimals, distinct from GSM8K's plain string
  equality) and longer-trace budget calibration are deliberately **not**
  included in this entry — planned as a follow-up once the corrected
  protocol passes its one-example GPU gate on GSM8K, per the original
  design's stated validation order.

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
