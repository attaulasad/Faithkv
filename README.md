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

**Status: implementation complete; GPU validation pending.** This
repository was built on a machine with no GPU and no model weights
downloaded. Every GPU-dependent test is implemented in full and marked
`@pytest.mark.gpu`; none has been run. See `docs/GPU_VALIDATION_PLAN.md`.

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
pytest -m "not gpu" tests/           # 107 passed on the build machine
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
(`configs/early_gap_b512.yaml` and its budget-escalation siblings) replay
ONE canonical trace (FullKV's own generated tokens) under both FullKV and
R-KV cache policies, so both conditions teacher-force identical reasoning
tokens — only the cache policy varies. This is a smaller-sample,
kill/continue screen (Prefix-Sufficiency Sensitivity / Delta_PSS,
`kvcot.analysis.fixed_trace`), additive alongside the frozen
`replay-probe`/EAS/Delta_EAS pipeline above, not a replacement for it — see
`CHANGELOG.md`'s 2026-07-16 entry. Same `--limit`/`--problem-index`/`--seed`/
`--resume`/`--dry-run` support.

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
