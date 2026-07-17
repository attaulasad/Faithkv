# GPU validation plan

Ordered list of exact commands to run on the rented GPU host, the moment
it's available. **Do not skip ahead** — nothing downstream of replay
identity (`test_replay_gpu.py`) is meaningful until it passes, since every
later stage's provenance/retention accounting depends on the assumptions
`docs/REPLAY_DESIGN.md` §5 lists as unverified.

## 0. Setup (once)

```bash
export HF_HOME=/workspace/hf_cache   # or another large-disk path
bash scripts/setup_vast.sh
```

Pass criterion: script exits 0, prints "setup complete", and
`requirements-lock.txt` is no longer the placeholder (real `pip freeze`
output).

## 1. Environment verification

```bash
bash scripts/verify_environment.sh
```

Pass criterion: exits 0. Checks (in order): torch importable and
CUDA-available; BF16 supported; `flash_attn` importable; `transformers`
version matches (or warns loudly if it diverges from) the upstream-
validated `4.55.4`; the `third_party/R-KV` submodule is checked out at
exactly `45eaa7d69d20b7388321f077020a610d9afb65bd`;
`requirements-lock.txt` matches installed state-critical package versions.

**On failure:** fix the environment before proceeding — do not silently
fall back to `sdpa` or a different transformers version. If
`transformers` can't be pinned to `4.55.4` on this host, stop and report;
do not guess whether a different version is safe (docs/UPSTREAM_AUDIT.md
H6 explains exactly why the version ceiling is load-bearing, not
cosmetic).

## 2. CPU unit tests (should already pass; re-run here as a sanity check that the GPU host's checkout matches this build)

```bash
pytest -m "not gpu" tests/ -q
```

Pass criterion: all tests pass (107 passed on the build machine — see the
final report for the exact count and any environment-specific skips).

## 3. Stock-vs-patched parity (§3.2)

```bash
pytest -m gpu tests/integration/test_patched_noop_parity_gpu.py -v
```

Pass criterion (hard gates, see the test file for the full list):
- `patched_noop`'s compaction count is exactly 0.
- Teacher-forced argmax token sequence is identical between stock FullKV
  and `patched_noop`.
- Prompt token IDs identical (trivial, but checked).

**On failure:** compression is confounded with the patched forward
implementation — the main experiment must not run. Report the exact
mismatch (which step, what token IDs) before touching anything else.

## 4. Cross-example state-leak test (§3.1)

```bash
pytest -m gpu tests/integration/test_no_state_leak_gpu.py -v
```

Pass criterion: example A's results (tokens, logits argmax, compaction
event steps, kept-position hashes) are byte-identical whether example B ran
before it or not, in both orders.

**On failure:** `kvcot.generation.state.reset_patched_state` is missing a
piece of mutable state — cross-reference `docs/UPSTREAM_AUDIT.md` §3.2's
inventory against whatever leaked, since that inventory is this function's
specification.

## 5. Replay identity — the load-bearing test (§6.1)

```bash
pytest -m gpu tests/integration/test_replay_gpu.py -v
```

Pass criteria (all hard gates — see the test file and
`docs/REPLAY_DESIGN.md` §2 and §5 for the reasoning behind each):
- Full replay reproduces the identical token sequence.
- Identical compaction event steps and identical per-layer/per-KV-head
  surviving absolute source position sets, at every event.
- Identical cache shapes, query-cache shapes.
- Two independent replays of the same trace are identical on all of the
  above.
- Restoring the same snapshot twice and probing yields identical probe
  tokens.
- FullKV replay passes the corresponding identity test (never evicts, cache
  length always equals total processed tokens).
- The long R-KV fixture triggers ≥2 real compactions, cache length stays
  within `budget + divide_length`.
- The bulk-prefill negative control produces a *different* compaction-event
  count than streaming replay (docs/REPLAY_DESIGN.md §1-2's named
  assumption).

K/V tensor closeness and logit deltas are printed as diagnostics, not hard
gates — read them, but a diagnostic failing alone does not block Stage 0.

**On failure:** this is the single most important failure this build can
have. Do not patch around it by loosening a hard gate to a diagnostic.
Read `docs/REPLAY_DESIGN.md` §5's assumption list, identify which one broke,
and report before writing any more code against the replay module.

## 6. Probe stability (§10 Stage 0 criterion)

```bash
pytest -m gpu tests/integration/test_probe_stability_gpu.py -v
```

Pass criterion: ≥90% f=1 stability for both `full` and the smoke R-KV
budget, on the 20-row smoke manifest.

**On failure:** per the build brief, do not tune the statistic. Inspect
`kvcot.probes.templates.CONTROL_SUFFIX_TEXT` and the greedy probe decoding
config first (temperature/max_new_tokens), then report and ask before
changing anything.

## 7. Stage 0

```bash
bash scripts/run_stage0.sh
```

Pass criteria: see `docs/EXPERIMENT.md` §7. Also prints a Stage 2
wall-clock extrapolation from measured tokens/sec — read it before
authorizing Stage 1B/2, since batch-1 token-by-token Python decode is slow
and the estimate may change the plan.

## 8. Stage 1A, then 1B, then 2

Only after Stage 0 passes:

```bash
bash scripts/run_stage1a.sh   # read results/decisions/stage1a_baseline_measurability.json
bash scripts/run_stage1b.sh   # read results/decisions/stage1b_budget_*.json, then manually fill in
                               # configs/selected_operating_point.yaml from configs/selected_operating_point.yaml.example
bash scripts/run_stage2.sh    # refuses to start without configs/selected_operating_point.yaml
```

## 9. Before terminating the instance

```bash
export KVCOT_SYNC_DEST=user@host:/path/to/backup/
bash scripts/sync_results.sh
```

`results/raw/` is gitignored — this is the only copy of raw generations
unless synced.

## 10. Optional: fixed-trace prefix-sufficiency screen (secondary, additive — not part of the ordered sequence above)

Not a Stage 0-2 step and not gated behind any of the stages above — this can
run any time after `test_replay_gpu.py` passes (§5), since it depends on the
same replay engine. **Use `configs/early_gap_v2_b128.yaml` first** (not
`early_gap_b512.yaml`) — see "Which budget to run" below for why. Requires
FullKV base generation for the screen's manifest first:

```bash
kvcot generate --config configs/early_gap_v2_b128.yaml --condition full
kvcot inspect-fixed-trace --config configs/early_gap_v2_b128.yaml --trace-condition full  # CPU preflight; stops on any of 3 arithmetic-only failure conditions, see below
kvcot replay-fixed-trace --config configs/early_gap_v2_b128.yaml --trace-condition full --replay-condition full
kvcot replay-fixed-trace --config configs/early_gap_v2_b128.yaml --trace-condition full --replay-condition rkv_b128
kvcot analyze-fixed-trace --config configs/early_gap_v2_b128.yaml --trace-condition full --replay-condition rkv_b128
```

Reads `results/decisions/early_gap_v2_b128_fixed_trace.json` — descriptive
counts only (n=10, one seed), no p-value. See `docs/EXPERIMENT.md` §11 and
`CHANGELOG.md`'s 2026-07-16 entries for what this measures and why it is
additive to, not a substitute for, Stage 0-2 above.

**Protocol v2, do not resume protocol-v1 output.** Any `results/raw/
early_gap_b*` directory generated before 2026-07-16 is protocol v1
(`schema_version` `"1.1.0"`, empty fixed-trace suffix, event-count
eligibility) and produced zero eligible examples — `kvcot analyze-fixed-trace`
now refuses to read it at all (schema/identity validation,
`kvcot.analysis.fixed_trace._validate_base_records`). Start a fresh
`output_dir` for any rerun; never point a v2 config at an old v1 directory.
The four stale protocol-v1 decision JSONs from the first GPU screen are kept,
for provenance only, under `results/decisions/archive/protocol_v1_2026-07-16/`
— do not read them as evidence for or against G1.

**Which budget to run, and why not b512/b256/b1024.** The real GPU data
already collected under b512 (`logs/b512_accuracy_compaction.log`) shows
prompt+think lengths that never exceed budget 512 or 1024 at all
(`mean_final_retention_ratio: 0.98`), and exceed budget 256 on at most
~6/10 traces — below `min_actual_compression_rate` (0.70) even before
accounting for how much retention itself would need to drop. `early_gap_b256
.yaml`/`early_gap_b1024.yaml` remain in the repo as historical artifacts of
that (now-superseded) escalation plan; do not run them expecting a different
outcome without first re-checking their trace-length assumptions with
`inspect-fixed-trace`. `early_gap_v2_b128.yaml` is the current starting
point. If it also fails `inspect-fixed-trace`'s checks on the real data, the
next step is a longer-trace manifest (MATH-500), not a smaller GSM8K budget
and not weaker thresholds.

### One-example gate before a full rerun

Per the corrected protocol's own validation order: run ONE fixed-trace
example first — add `--limit 1 --problem-index <n>` to the `generate` and
`replay-fixed-trace` commands above (`analyze-fixed-trace` takes neither
flag; it just reads whatever `replay-fixed-trace` already wrote, so limiting
upstream is sufficient) — and confirm, from the written records/decision
JSON, all of the following before
running the full n=10 screen:

- Both FullKV and R-KV f=1 extraction status is `"boxed"` (never
  `final_number_fallback`).
- Both f=1 answers equal gold.
- Neither f=1 probe hit its token cap (`probe_cap_hit` is `False`).
- R-KV shows `actual_compression_at_cut: true` at f=1 with
  `instantaneous_retention_ratio < 1.0`.
- `probe_actual_eviction_during_answer` is `False` for R-KV's scored probes
  **and for the f=1 anchor itself** — an eviction while the anchor was
  writing its own answer is exactly as disqualifying, since every fraction
  is compared against that anchor (`no_rkv_eviction_during_answer_probes`,
  `kvcot.analysis.fixed_trace`).
- `git status` shows a clean tree (`git_dirty: false` in the record's own
  provenance) before the run started.

If any of these fail, stop and fix the specific cause rather than running
the full n=10 screen — the full screen only adds sample size to a result
that is already known-invalid at n=1.

### Protocol v2's real result, and why protocol v3 exists (2026-07-17)

The one-example gate above passed and the full n=10 screen under
`configs/early_gap_v2_b128.yaml` ran for real:
`results/decisions/early_gap_v2_b128_fixed_trace.json` —
`n_shared=10 n_eligible=3 mean_f1_rkv_retention_ratio=0.7456
screen_valid=false hypothesis_status=not_tested`. This is a valid negative
screening outcome (both conditions' base accuracy 9/10, zero cap hits, all
180 fixed-trace probes produced a valid boxed answer, every correctness
gate above passed) — not evidence against the hypothesis, since the screen
itself did not clear its own validity bar. Two diagnosed, fixable causes
(full detail in `CHANGELOG.md`'s 2026-07-17 entry):

1. `actual_compression_at_cut` (any nonzero eviction) let a
   0.9959-retention example count as "compression active" — R-KV's
   periodic `divide_length=128` schedule can cross a compaction checkpoint
   long before the cache has accumulated enough tokens for the eviction to
   matter (`docs/UPSTREAM_AUDIT.md` H4).
2. 5 of 10 shared examples failed via `rkv_evicted_during_answer_probe` — a
   real compaction event fired while the probe was teacher-forcing the
   closing marker/suffix/greedily generating its own answer, after the
   snapshot the probe was supposed to measure.

### Protocol v3 (use this, not v2, for any new run)

`configs/early_gap_v3_b128.yaml` — new `stage_name`/`output_dir`, never a
resumption of protocol v2's `results/raw/protocol_v2/early_gap_b128`.
Fixes both causes above: `probe_cache_mode: frozen_at_cut`
(`kvcot.generation.replay.branch_and_probe`) forces R-KV compression off
for the duration of every probe and hard-asserts the cache did not move;
`require_meaningful_compression: true` with `meaningful_retention_ceiling:
0.7` requires a SUBSTANTIAL measured retention drop, never just a nonzero
one.

**Required order (revised 2026-07-18 — the original v3 pass's documented
order called `analyze-fixed-trace` before any fixed-trace replay had ever
run, which cannot work: that command unconditionally reads the fixed-trace
probe files, which do not exist yet at that point):**

```bash
# 1. Mandatory GPU correctness gates, including the two new v3-specific ones.
pytest -m "not gpu" -q
pytest -m gpu tests/integration/test_patched_noop_parity_gpu.py -v
pytest -m gpu tests/integration/test_no_state_leak_gpu.py -v
pytest -m gpu tests/integration/test_replay_gpu.py -v
pytest -m gpu tests/integration/test_probe_stability_gpu.py -v
pytest -m gpu tests/integration/test_rkv_schedule_prediction_gpu.py -v   # new: simulator vs real measured cache length
pytest -m gpu tests/integration/test_frozen_probe_gpu.py -v              # new: frozen_at_cut vs native, real model

# 2. Natural FullKV generation, then CPU-only, outcome-blind selection --
#    BEFORE any R-KV replay.
kvcot generate --config configs/early_gap_v3_b128.yaml --condition full
kvcot inspect-fixed-trace --config configs/early_gap_v3_b128.yaml --trace-condition full --write-selection
# ^ Predicts realized retention per example via kvcot.analysis.rkv_schedule.
#   Writes results/selections/early_gap_v3_b128.json. Exit code is already 1
#   if n_selected < fixed_trace.min_eligible_examples -- stop here if so.

# 3. Natural R-KV generation on the SAME 50 problems, then the CPU-only
#    strict accuracy gate -- this now has no ordering dependency on any
#    fixed-trace replay, unlike the superseded 2026-07-17 order.
kvcot generate --config configs/early_gap_v3_b128.yaml --condition rkv_b128
kvcot check-fixed-trace-accuracy --config configs/early_gap_v3_b128.yaml --replay-condition rkv_b128
# ^ Reads ONLY full.jsonl/rkv_b128.jsonl (never fixed-trace probes). Requires
#   an exact expected record count, an IDENTICAL key set between the two
#   files, schema-valid records, one shared identity, and
#   pilot_accuracy_plausible -- exits nonzero (and writes
#   results/decisions/early_gap_v3_b128_accuracy_gate.json documenting why)
#   if any of those fail. Stop here if it does not pass.

# 4. One-example frozen-probe/schedule-prediction gate (below), using
#    --selection-file + --problem-index together to replay exactly one
#    SELECTED example without touching the rest.
FIRST_SELECTED=<first entry of selected_source_row_indices in the selection file>
kvcot replay-fixed-trace --config configs/early_gap_v3_b128.yaml --trace-condition full --replay-condition full \
  --selection-file results/selections/early_gap_v3_b128.json --problem-index "$FIRST_SELECTED"
kvcot replay-fixed-trace --config configs/early_gap_v3_b128.yaml --trace-condition full --replay-condition rkv_b128 \
  --selection-file results/selections/early_gap_v3_b128.json --problem-index "$FIRST_SELECTED"
# Validate against the checklist below. Only then:

# 5. Replay the REST of the selected examples in two model loads --
#    --resume skips the one example already written above.
kvcot replay-fixed-trace --config configs/early_gap_v3_b128.yaml --trace-condition full --replay-condition full \
  --selection-file results/selections/early_gap_v3_b128.json --resume
kvcot replay-fixed-trace --config configs/early_gap_v3_b128.yaml --trace-condition full --replay-condition rkv_b128 \
  --selection-file results/selections/early_gap_v3_b128.json --resume

# 6. Analysis, restricted to exactly the selected set -- aborts (does not
#    silently under-report) if any selected example is missing or
#    incomplete under either policy.
kvcot analyze-fixed-trace --config configs/early_gap_v3_b128.yaml --replay-condition rkv_b128 \
  --selection-file results/selections/early_gap_v3_b128.json
```

**One-example frozen-probe/schedule-prediction gate — mandatory, before
step 5 above.** This is IN ADDITION to (does not replace) the one-example
gate two sections above (extraction/correctness/no-cap/clean-tree still
apply identically). Additionally confirm, for that one example:

- `probe_cache_mode` on every written `FixedTraceProbeRecord` is
  `"frozen_at_cut"`, and `protocol_version` is `"v3"`.
- No `RuntimeError` was raised by `branch_and_probe` (it would have aborted
  the run outright — a silent pass here means the frozen-cache assertion
  held for real, not just in unit tests).
- `kvcot.analysis.rkv_schedule.simulate_rkv_cache_lengths`'s prediction for
  this example (from `results/selections/early_gap_v3_b128.json`) matches
  the example's own written `replay_retention_at_cut.instantaneous_
  retention_ratio` at every fraction to within rounding. A disagreement of
  even one token means the CPU simulator does not actually reproduce real
  R-KV behavior on this build/version combination — stop and re-derive the
  simulator against the currently-pinned upstream commit before trusting it
  for selection on any further example. (`test_rkv_schedule_prediction_gpu.
  py` in step 1 already checks this on a synthetic fixture; this step
  re-checks it on a REAL selected example from this actual manifest.)

### GPU test process isolation (§ Step 13/14, 2026-07-16)

`tests/integration/test_replay_gpu.py` and `test_probe_stability_gpu.py` now
run every patched-R-KV test (and the FullKV identity test in
`test_replay_gpu.py`) inside its own `spawn`ed subprocess — never in the
shared pytest process. This matters because the R-KV monkeypatch on
`transformers.models.qwen2` is process-global with no per-instance undo
(`docs/UPSTREAM_AUDIT.md` H1); mixing a stock and a patched test (or two
different R-KV configs) in one process is unsafe even though
`kvcot.generation.state.declare_process_mode` will usually catch and refuse
the conflict first. Both files must still pass as complete files, not just as
individually-run tests — a change that passes one test alone but breaks the
file as a whole reintroduces exactly this class of bug:

```bash
pytest -m gpu tests/integration/test_replay_gpu.py -v
pytest -m gpu tests/integration/test_probe_stability_gpu.py -v
```
