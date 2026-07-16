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
same replay engine. Requires FullKV base generation for the screen's
manifest first:

```bash
kvcot generate --config configs/early_gap_b512.yaml --condition full
kvcot replay-fixed-trace --config configs/early_gap_b512.yaml --trace-condition full --replay-condition full
kvcot replay-fixed-trace --config configs/early_gap_b512.yaml --trace-condition full --replay-condition rkv_b512
kvcot analyze-fixed-trace --config configs/early_gap_b512.yaml --trace-condition full --replay-condition rkv_b512
```

Reads `results/decisions/early_gap_b512_fixed_trace.json` — descriptive
counts only (n=10, one seed), no p-value. See `docs/EXPERIMENT.md` §11 and
`CHANGELOG.md`'s 2026-07-16 entry for what this measures and why it is
additive to, not a substitute for, Stage 0-2 above. `early_gap_b256.yaml`/
`early_gap_b1024.yaml` are budget-escalation fallbacks (same three commands,
substituting the condition/config names).
