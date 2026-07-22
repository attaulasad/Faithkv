# B2A one-example GPU authorization (dated 2026-07-22)

## 1. Audit decision

**AUTHORIZE B2A — ONE-EXAMPLE ENGINEERING CALIBRATION ONLY.**

This is the separate, explicit, dated authorization that CLAUDE.md §1a and
§1b both stated was still required before any B2A (one-example GPU
calibration) activity. It authorizes exactly one `b2a-calibrate --execute`
attempt against the committed one-example manifest, nothing more. It does
not authorize B2B, a discovery pilot, a method implementation, a threshold
change, or any additional scientific experiment.

## 2. Evidence supporting this decision

At commit `a4f6e4298eba10d037ca7e6570fe6d69aad2472f` on
`research/b1b-r4-final-b2a-closure`:

- CPU CI run [29892965613](https://github.com/asad073-ui/Faithkv/actions/runs/29892965613)
  concluded `success` for this exact SHA (checkout, submodule init, CPU-only
  torch 2.6.0 install/verify, byte-compile, full collection, full
  non-GPU-marked suite, `git diff --check`, all green).
- The full non-GPU test suite passed locally in `.venv-ci` (Python 3.11,
  CPU-only torch 2.6.0): 1204 passed, 14 deselected (`-m gpu`).
- On the Vast RTX 3090 host, in `.venv-gpu` (Python 3.12, torch 2.6.0+cu124,
  transformers 4.55.4, flash-attn 2.7.4.post1), the following GPU-marked
  tests passed:
  - `test_replay_gpu.py::test_fullkv_replay_passes_identity_test` (replay
    identity);
  - `test_patched_noop_parity_gpu.py::test_stock_vs_patched_noop_parity`
    (stock-vs-patched no-op parity);
  - `test_no_state_leak_gpu.py::test_example_results_independent_of_run_order`
    (cross-example state isolation);
  - `test_replay_gpu.py::test_restoring_same_snapshot_twice_yields_identical_probe_tokens`
    (snapshot restoration);
  - `test_rkv_schedule_prediction_gpu.py::test_simulator_matches_real_measured_cache_length_at_every_fraction`
    (R-KV schedule prediction);
  - every other collected non-`test_probe_stability_gpu.py` GPU test
    (`test_frozen_probe_gpu.py`, remaining `test_replay_gpu.py` cases,
    `test_capture.py`'s CUDA/CPU provenance test) — 12/12 passed.
- The CUDA environment is pinned and was verified exactly: one visible RTX
  3090, torch 2.6.0+cu124, `torch.version.cuda == "12.4"`, transformers
  4.55.4, flash-attn 2.7.4.post1, `pip check` clean.
- No CPU offload was observed anywhere in the tests exercised
  (`device_map="cuda"` explicit in every loader; no `device_map="auto"`).

## 3. Historical probe classification — not altered, not a B2A blocker

`tests/integration/test_probe_stability_gpu.py::test_f1_probe_stability_meets_threshold`
failed both parametrized conditions on the frozen 20-row
`gsm8k_smoke_20.jsonl` manifest:

- FullKV f=1 stability = **17/20** (0.85, below the 0.90 threshold);
- R-KV b256 f=1 stability = **15/20** (0.75, below the 0.90 threshold).

These remain **failed historical Stage 0 results** for the archived
Qwen-1.5B/GSM8K early-answering protocol (§1/§4 of CLAUDE.md). They are not
altered, rerun, or redescribed as passes anywhere in this repository. No
change was made to `CONTROL_SUFFIX_TEXT`, `STABILITY_THRESHOLD`, answer
normalization, `answers_match`, probe decoding, or any Qwen/GSM8K
configuration or test assertion.

They do not gate B2A because they measure a different mechanism than the
one B2A exercises: that protocol's base continuation is sampled
(temperature 0.6, top-p 0.95) while its probe continuation is greedy and
conditioned on an inserted control suffix (`render_control_suffix()`) —
the test mixes sampling noise and suffix sensitivity into what it reports
as "stability." The B2A track (CLAUDE.md §1a/§1b/§4a/§4b) uses a disjoint
model, dataset, and mechanism: `DeepSeek-R1-Distill-Llama-8B` on MATH-500,
greedy decoding throughout, no early-answering control suffix, and causal
candidate/donor swaps with a teacher-forced bridge plus 48 scored tokens,
evaluated against its own mandatory gate contract
(`kvcot.discovery.b2a_execute_coordinator`). Nothing about the Qwen/GSM8K
probe-stability shortfall constitutes evidence about this disjoint
mechanism.

## 4. Exact scope of this authorization

Authorized — exactly once:

- one frozen MATH-500 example, identified by the committed manifest
  `configs/discovery/b2a_one_example_manifest.json`;
- the exact discovery config `configs/discovery/llama8b_math500_b1024.yaml`,
  unmodified;
- `DeepSeek-R1-Distill-Llama-8B` at its pinned model/tokenizer revision, as
  resolved by that config — no other model, revision, or floating branch;
- FullKV and R-KV (budget 1024) workers, the existing 12-real-pair plus
  one-no-op design, unmodified;
- greedy generation, batch size 1, one visible RTX 3090, local exact-snapshot
  model loading, no CPU/disk/meta offload;
- hard limits: peak tracked CUDA memory <= 22 GiB; projected complete-pilot
  runtime <= 4.00 GPU-hours; one execution attempt only; 2-hour command
  timeout.

Not authorized by this document: B2B, a 12-example pilot, any rerun after a
completed scientific attempt, any method implementation, any new eviction
criterion, or any change to a scientific threshold, event selection, pair
selection, candidate/donor definition, model, dataset row, revision, seed,
budget, or generation config.

## 5. Failure policy

- A gate failure is itself the B2A result — evidence, not an error to
  engineer around.
- No tuning of thresholds, configuration, or code after observing the
  result.
- No automatic second scientific attempt. A new attempt requires new,
  separate human authorization.
- Environment or preflight failures that occur **before** any worker
  inference begins (missing dependency, bad path, config-resolution error,
  etc.) may be diagnosed and repaired, and the run retried, because no
  scientific measurement has started.
- Once FullKV or R-KV inference has actually begun, the attempt is
  scientifically consumed: any failure past that point (including OOM,
  timeout, or an identity/parity mismatch) is reported as the B2A result,
  not repaired and retried.
