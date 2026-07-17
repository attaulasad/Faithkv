# kv-cot-dependence

Does decoding-time R-KV KV-cache compression change a reasoning model's
behavioral dependence on the omitted suffix of its own visible
chain-of-thought, at an accuracy-preserving operating point?

Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. Method:
[R-KV](https://github.com/Zefan-Cai/R-KV) (pinned submodule,
`third_party/R-KV`). Dataset: GSM8K (frozen MATH-500 backup if Stage 1A
shows insufficient measurement range). Full design: `docs/EXPERIMENT.md`.
Upstream audit: `docs/UPSTREAM_AUDIT.md`. Replay design and open
assumptions: `docs/REPLAY_DESIGN.md`.

**Claim boundary:** this measures counterfactual behavioral dependence on
*visible, generated* reasoning tokens under an early-answering
intervention. It does not measure, and this repository never claims to
measure, whether the chain-of-thought is "real," faithful to internal
cognition, or decorative. See `docs/EXPERIMENT.md` §1.

**Status: GPU correctness gates passed; protocol-v2 fixed-trace screen ran
and came back invalid; protocol v3 implemented, not yet GPU-run.** This
repository is built/maintained on a machine with no GPU — all development
here is CPU-only (`pytest -m "not gpu" tests/`, `--dry-run`). GPU-dependent
work happens on a rented host and results are synced back
(`scripts/sync_results.sh`): `test_patched_noop_parity_gpu.py`,
`test_no_state_leak_gpu.py`, and all seven `test_replay_gpu.py` cases have
real passing runs on record (`logs/gpu_validation/*.log`). The protocol-v2
fixed-trace screen (`configs/early_gap_v2_b128.yaml`) also ran for real —
`results/decisions/early_gap_v2_b128_fixed_trace.json`,
`n_shared=10 n_eligible=3 mean_f1_rkv_retention_ratio=0.7456
screen_valid=false hypothesis_status=not_tested` — a valid negative
screening outcome, not a crash or a bug in the correctness machinery. See
`CHANGELOG.md`'s 2026-07-17 entry for the two diagnosed causes and the
protocol-v3 fixes (`configs/early_gap_v3_b128.yaml`), which have not yet
been run on a GPU. No Stage 0-2 run (the frozen primary EAS/Delta_EAS
pipeline) has happened yet. See `docs/GPU_VALIDATION_PLAN.md`.

`logs/git_commit.txt`/`logs/git_status.txt` reflect an OLDER commit
(`bb1917a...`) than this repository's current history — they were captured
during the GPU run that produced the protocol-v2 result above and were
never refreshed afterward. Treat them as a historical record of what the
GPU host had checked out for that specific run, not as current provenance;
regenerate them (`git rev-parse HEAD`, `git status --short`) on every future
GPU invocation rather than trusting stale copies.

## Layout

```
configs/        frozen settings (lock.yaml) + per-stage configs
data/manifests/ frozen, committed dataset manifests (question + hash + gold only)
docs/           design docs — read UPSTREAM_AUDIT.md and REPLAY_DESIGN.md first
src/kvcot/      the package
scripts/        thin shell wrappers around the CLI
tests/          unit/ (CPU, run here) + integration/ (GPU, skipped here)
third_party/R-KV/  pinned upstream submodule (sparse-checked-out here — see below)
results/        raw/ is gitignored; run_manifests/, decisions/, tables/, figures/ are committed
```

## Setup

**On this machine (CPU-only, for development/analysis/tests):**

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[cpu-tools,dev]"    # no torch — analysis, tests, manifest freezing, tokenizer only
pytest -m "not gpu" tests/           # see CHANGELOG.md's latest entry for the current passing count
```

**On a GPU host (e.g. a rented Vast.ai instance) — required for anything that generates or replays:**

```bash
tmux new -s kvcot          # long-running work; don't lose it to a dropped SSH session
export HF_HOME=/workspace/hf_cache   # point at the instance's large disk, not the root volume
bash scripts/setup_vast.sh
bash scripts/verify_environment.sh
```

Model access: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` is a public HF
model — no gated-access request needed. `setup_vast.sh` does not download
weights itself; the first `kvcot generate` call does, into `$HF_HOME`.

No Weights & Biases, no external LLM APIs anywhere in this repository —
all logging is local JSONL/CSV/JSON under `results/`.

## Running the stages

Exact order, with pass criteria, in `docs/GPU_VALIDATION_PLAN.md`. Summary:

```bash
bash scripts/verify_environment.sh
pytest -m gpu tests/integration/test_patched_noop_parity_gpu.py -v
pytest -m gpu tests/integration/test_no_state_leak_gpu.py -v
pytest -m gpu tests/integration/test_replay_gpu.py -v          # load-bearing — do not skip ahead of this
pytest -m gpu tests/integration/test_probe_stability_gpu.py -v
bash scripts/run_stage0.sh
bash scripts/run_stage1a.sh
bash scripts/run_stage1b.sh    # then manually fill in configs/selected_operating_point.yaml
bash scripts/run_stage2.sh     # refuses to start without that file
```

Every `kvcot generate`/`replay-probe` call supports `--limit`,
`--problem-index`, `--seed`, `--resume`, `--dry-run`. `--dry-run` works
without a GPU — it resolves the config, validates the manifest/schema, and
prints the planned record count; it's the only end-to-end check available
on a non-GPU machine, and every stage's dry-run was exercised during this
build (see the final build report).

### Secondary diagnostic: fixed-trace prefix-sufficiency screen

`kvcot replay-fixed-trace` / `kvcot analyze-fixed-trace`
(`configs/early_gap_v2_b128.yaml` — see below for why not the older
`early_gap_b512.yaml`/`b256`/`b1024` siblings) replay
ONE canonical trace (FullKV's own generated tokens) under both FullKV and
R-KV cache policies, so both conditions teacher-force identical reasoning
tokens — only the cache policy varies. This is a smaller-sample,
kill/continue screen (Prefix-Sufficiency Sensitivity / Delta_PSS,
`kvcot.analysis.fixed_trace`), additive alongside the frozen
`replay-probe`/EAS/Delta_EAS pipeline above, not a replacement for it — see
`CHANGELOG.md`'s 2026-07-16 entry. Same `--limit`/`--problem-index`/`--seed`/
`--resume`/`--dry-run` support.

**Protocol v2 (2026-07-16):** the first GPU screen under protocol v1 produced
zero eligible examples — not a negative result, no result at all (root cause
and fix in `CHANGELOG.md`). Protocol v2 uses a teacher-forced boxed-answer
format prefix instead of an empty suffix, its own `FixedTraceSettings`
(separate from the frozen primary `probes.max_new_tokens: 48`), and gates
eligibility on realized (measured) compression rather than a recorded
compaction event count. Run the CPU-only `kvcot inspect-fixed-trace`
preflight against an already-generated FullKV trace before spending any GPU
time on `replay-fixed-trace` — it stops if no trace exceeds the configured
budget, if the fraction of traces that could even possibly exceed it is
already below the required compression rate, or if even best-case
compaction couldn't clear the retention ceiling on this data. Old
protocol-v1 output directories (`schema_version` `"1.1.0"`) are rejected
outright at analysis time (schema/identity validation,
`kvcot.analysis.fixed_trace`) — start a fresh `output_dir` instead of
resuming one.

**2026-07-16 follow-up hardening (external review):** the real b512 GPU
data already collected shows prompt+think lengths that never exceed budget
512/1024 at all, and exceed 256 on at most ~6/10 traces — structurally
below the required compression rate on this manifest. `early_gap_v2_b128.yaml`
is the current starting config (new `stage_name`/`output_dir`, isolated from
any protocol-v1 data); the old `early_gap_b*.yaml` decision JSONs are
archived under `results/decisions/archive/protocol_v1_2026-07-16/`, not
deleted. Also fixed: eligibility now checks answer-time cache eviction for
the f=1 anchor itself, not only the 7 scored fractions (a synthetic
f=1-only-eviction case previously slipped through as eligible).

**Protocol v2's real GPU screen came back invalid, protocol v3 fixes why
(2026-07-17):** `results/decisions/early_gap_v2_b128_fixed_trace.json`
(n=10, seed=42) came back `screen_valid: false` —
`n_eligible=3` (< 5 required) and `mean_f1_rkv_retention_ratio=0.7456`
(> the 0.7 ceiling). Two diagnosed, fixable causes (full detail in
`CHANGELOG.md`): (1) the "any nonzero eviction" compression check let a
0.9959-retention example count as "compression active"; (2) 5/10 examples
failed via `rkv_evicted_during_answer_probe` — R-KV kept compacting while
the probe wrote its own answer. `configs/early_gap_v3_b128.yaml` adds a
`meaningful_retention_ceiling`-gated eligibility check
(`require_meaningful_compression`), a new Compression-Active
Prefix-Sufficiency Sensitivity metric (CPSS/Delta_CPSS, restricted to
fractions where R-KV actually compressed meaningfully), a
`probe_cache_mode: frozen_at_cut` that prevents answer-time compaction by
construction (hard runtime assertion, `kvcot.generation.replay.
branch_and_probe`), a CPU-only exact cache-schedule simulator
(`kvcot.analysis.rkv_schedule`) for deterministic, outcome-blind trace
selection (`kvcot inspect-fixed-trace --write-selection`) before any GPU
replay, and a natural-accuracy pilot screen
(`build_accuracy_screen`/`pilot_accuracy_plausible`) since v2 never
generated a natural R-KV b128 run to check against. Not yet exercised on a
GPU — the one-example frozen-probe/schedule-prediction gate
(`docs/GPU_VALIDATION_PLAN.md`) is required before any full replay under
this config.

**Second-pass v3 hardening (2026-07-18):** a follow-up review found the
selection file from `--write-selection` was never actually consumed by
`replay-fixed-trace`/`analyze-fixed-trace` (both now take
`--selection-file`), a real bug in `--max-selected` (capped the ranked list
before filtering to predicted-eligible candidates, fixed — filter first,
cap second), the documented run order calling `analyze-fixed-trace` before
any fixed-trace replay existed (impossible — fixed with a new CPU-only
`kvcot check-fixed-trace-accuracy` command that reads only the natural
`full.jsonl`/`{condition}.jsonl` files and requires an exact, identical
record count/key-set/schema/identity match before trusting
`pilot_accuracy_plausible`), and a screen-validity gap where
`require_meaningful_compression=True` still checked the loose any-eviction
rate at the SCREEN level (fixed: checks `meaningful_compression_rate`
against new `min_meaningful_compression_rate` instead). Also added: the two
GPU tests the 2026-07-17 entry promised but did not yet contain
(`tests/integration/test_rkv_schedule_prediction_gpu.py`,
`test_frozen_probe_gpu.py`), plus a CPU-only fake-model regression suite
(`tests/integration/test_frozen_probe_fake_model.py`, runs on any machine
with a plain CPU torch install — no GPU needed) that directly verifies
`frozen_at_cut` forces compression off before every fed token, not just
once. See `CHANGELOG.md`'s 2026-07-18 entry for full detail.

**Resume behavior:** re-running the same command with `--resume` skips any
`record_id` already present in the output JSONL file
(`kvcot.utils.io.JsonlWriter`/`read_existing_record_ids`) — safe to
interrupt and restart at any point, including mid-generation.

## Expected artifacts

- `results/raw/<stage>/<condition>.jsonl`, `<condition>_probes.jsonl` —
  gitignored, per-record generations/probes.
- `results/run_manifests/` — one JSON per invocation, committed.
- `results/decisions/` — `stage1a_baseline_measurability.json`,
  `stage1b_budget_<N>.json`, committed.
- `results/tables/attrition_funnel.csv` and the primary analysis JSON,
  committed.
- `results/figures/*.png`, committed.

**Before terminating a GPU instance**, sync `results/raw/` off it — it is
gitignored and exists only there and wherever you sync it to:

```bash
export KVCOT_SYNC_DEST=user@host:/path/to/backup/
bash scripts/sync_results.sh   # never deletes anything, local or remote
```

## Scope

This repository implements exactly one method (R-KV) on exactly one model,
measuring exactly one thing (early-answering sensitivity under truncation).
It does not implement faithfulness-aware eviction, KIVI, mistake insertion,
7B model support, vLLM/SGLang serving, multi-GPU, an LLM judge, or a
benchmark suite. See `docs/EXPERIMENT.md` for why, and the build brief this
repository was built from for the full scope boundary.

The fixed-trace prefix-sufficiency screen above (`replay-fixed-trace`/
`analyze-fixed-trace`) stays within this boundary — same method, same model,
still a truncation-sensitivity measurement, added alongside the frozen
pipeline rather than replacing it (`CHANGELOG.md`, 2026-07-16). Mistake
insertion in particular remains **not implemented**, per the line above.

## License

**Not yet chosen.** No `LICENSE` file exists in this repository — this was
deliberate (the build brief instructs "ask me which license; do not pick
one silently"). Add one before treating this repository as distributable.

## Getting help with this build

`docs/UPSTREAM_AUDIT.md` — what was verified about the pinned R-KV source,
with file:line citations, and what remains an open assumption.
`docs/REPLAY_DESIGN.md` — the replay engine's design and the specific,
named assumptions GPU validation needs to confirm. `CLAUDE.md` — frozen
decisions and conventions for anyone (human or agent) picking this repo
back up in a future session.
