# B2A-R1 failure and B2A-R2 pre-registered protocol (dated 2026-07-22)

This document is the separate, explicit, dated authorization and
pre-registration CLAUDE.md §1c's B2A-R1 authorization said any further B2A
attempt would require. It is written and committed **before** any B2A-R2
qualification inference runs, per its own rule (§4 below): the selection
procedure, its criteria, and its stopping rule are fixed here, and nothing
about them may be changed after candidate outputs are observed.

## 1. B2A-R1 failure

The single attempt authorized by CLAUDE.md §1c executed
`b2a-calibrate --execute` against `example_index=0` of the frozen MATH-500
one-example manifest. Two sub-attempts occurred:

- **Preflight-only attempt** (`b2a_attempt_20260722T071154...`): crashed
  before any worker inference began --
  `AttributeError: module 'torch.cuda' has no attribute 'cudnn'` in
  `verify_single_rtx3090` (`src/kvcot/discovery/strict_device.py`). Fixed in
  commit `0f86e27` (already on this branch): `torch.backends.cudnn.version()`
  is the correct torch 2.6.0 API, not `torch.cuda.cudnn.version()`.
- **Consumed attempt** (`b2a_attempt_20260722T072823...`): preflight passed
  (one verified RTX 3090); both FullKV and R-KV workers completed process
  execution (return code 0 each); but the coordinator crashed with
  `ValueError: runtime projection requires exactly 12 B2A real-pair
  durations` in `build_runtime_projection`
  (`src/kvcot/discovery/execution_measurement.py`).

Root cause of the second failure: `example_index=0`'s frozen row produced a
prompt of 105 tokens and a FullKV natural generation of 449 tokens --
processed length ≈554, far under R-KV's configured budget of 1024. R-KV's
eviction trigger (`kv_cache_len >= budget`) never fired even once, so **zero
compaction events occurred**, hence **zero real pairs** could ever be
constructed. This is a genuine scientific ineligibility (the causal-swap
hypothesis was never tested -- there was no eviction to swap around), not a
software crash by itself. The coordinator's defect was that this condition
raised an uncaught exception instead of resolving to a clean, internally
consistent `gate_failed` outcome.

Both preserved attempts, their complete file inventories, and their
external archives are indexed in
`docs/evidence/B2A_R1_ATTEMPT_INDEX_2026-07-22.json`. Per the classification
rule below, **B2A-R1 counts as the single attempt CLAUDE.md §1c
authorized** -- inference began, so it is consumed. Everything in this
document is preparation for a separately, explicitly authorized second
attempt, B2A-R2.

**Do not repeat B2A-R1's row.** `example_index=0` is not eligible for
re-selection at budget 1024 -- it structurally cannot produce a compaction
event.

## 2. Repairs made before B2A-R2 (no scientific config touched)

### 2.1 Zero-event coordinator defect (H3, test-harness/coordinator)

`build_runtime_projection` now reports an insufficient real-pair count
(0-11 durations) as an explicit, structured "unavailable" outcome
(`available=False`, `unavailable_reason="insufficient_real_pair_durations"`,
`observed_real_pair_duration_count`, `required_real_pair_duration_count`,
`conservative_real_pair_seconds=None`, `projected_total_seconds=None`,
`projected_complete_pilot_gpu_hours=None`) instead of raising. **Never**
`0`, `inf`, or the `4.00`-hour limit itself as a stand-in value.
`evaluate_b2a_gate` derives `runtime_within_limit=False` from that `None`,
failing closed rather than fabricating a passing number. The exactly-12
"available" path's arithmetic is completely unchanged (verified by the
pre-existing test asserting its exact formula still holds). A genuinely
malformed measurement (a present but non-finite/non-positive real-pair
duration) still raises -- only the *count* mismatch became non-exceptional.
Regression tests:
`tests/unit/discovery/test_execution_measurement.py`,
`tests/unit/discovery/test_b2a_contract.py`,
`tests/unit/discovery/test_b2a_execute_coordinator.py` (a full
coordinator-level test reproducing zero compaction events end-to-end,
proving `completion.json` gets `outcome="gate_failed"`, `exit_code=2`,
`gate_passed=False`, and that `final.json` is written and independently
verified -- and a contrast test proving a genuinely malformed measurement
still raises and writes only `failure.json`).

### 2.2 Answer-verifier defect (confirmed, general fix -- not tuple-specific)

Audited directly against the installed `math-verify==0.9.0` package (not
assumed): the raw B2A-R1 artifacts show FullKV's answer
(`\left(3, \frac{\pi}{2}\right)`) marked `"unverifiable"` against the gold
answer (`\left( 3, \frac{\pi}{2} \right)`), despite being the identical
ordered pair. Direct experimentation established the actual root cause:
`math_verify.parse`'s non-anchored fallback extraction (used whenever the
text handed to it carries no `\boxed{}`/"final answer" anchor) is
unreliable for compound expressions -- it parsed the two strings above as
`[3, '3']` (silently dropping the second tuple component) and `[]` (no
candidate at all) respectively, purely from an incidental internal-
whitespace difference the anchored path does not care about. Re-wrapping
each already-extracted final answer in its own `\boxed{...}` before
verification (`kvcot.discovery.math500_verification.Math500AnswerVerifier
.__call__`) routes both through `math_verify`'s well-tested boxed-
extraction path instead -- confirmed double-wrap-safe
(`\boxed{\boxed{x}}` normalizes identically to `\boxed{x}`), whitespace-
insensitive, `\left`/`\right`-insensitive, order-preserving for ordered
pairs, and correct for scalar fractions and symbolic constants (`\pi`,
`e`). This is a general parsing-boundary fix, not a custom tuple splitter,
and not a hard-coded acceptance of the specific observed answer -- no
regex-number matching, no tuple-order relaxation, fails closed on malformed
syntax. 12 regression tests in
`tests/unit/discovery/test_math500_verification.py`, including the exact
B2A-R1 observed strings.

## 3. Candidate population and deterministic ordering (pre-registered)

`kvcot.discovery.b2a_r2_candidates.build_candidate_manifest`, committed as
`configs/discovery/b2a_r2_candidate_manifest.json`:

1. Population: every level-5 row (the hardest, longest MATH-500 problems --
   most likely to produce a generated trace long enough to actually
   exercise a 1024-token budget) from the SAME pinned dataset revision
   (`6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`) already frozen in
   `configs/discovery/llama8b_math500_b1024.yaml`. 134 of 500 rows are
   level 5.
2. Duplicate `unique_id` is a hard refusal.
3. Ordering key, fixed BEFORE any qualification inference and a pure
   function of identity alone (never of anything observed about
   generation):
   ```
   sha256("faithkv-b2a-r2-row-order-v1" + "|" + dataset_revision + "|"
          + model_revision + "|budget=1024" + "|" + unique_id)
   ```
4. Ascending sort by that hash; first 12 rows kept.

Canonical manifest hash: `ac2dcc4550a89f2cfa701acd608a8087b4a1ebaa0ea05eb15d8f71e3434ee0ec`.
13 tests in `tests/unit/discovery/test_b2a_r2_candidates.py` prove the
order depends only on identity fields (not on row content/length),
depends on `budget`, is stable across repeated calls and across different
input row orders, and rejects duplicates/malformed rows.

## 4. FullKV-only qualification (pre-registered, before any inference)

`kvcot.discovery.b2a_qualification`, CLI: `kvcot qualify-b2a-row`.

- **R-KV is never imported, patched, or loaded during qualification.**
  Qualification uses `kvcot.discovery.b2a_workers.run_fullkv_worker`
  (the SAME FullKV natural-generation code the real B2A execution's FullKV
  worker uses) to generate ONE greedy natural response per candidate, and
  predicts where R-KV WOULD compact from that generation's length alone,
  via `kvcot.analysis.rkv_schedule.predicted_compaction_event_positions`
  (a pure-Python, no-torch simulator of R-KV's exact schedule/trigger
  mechanics, extended -- not duplicated -- from the existing, GPU-verified
  `simulate_rkv_cache_lengths`). Event eligibility (never first/last, never
  mid-prefill, ≥49 real future tokens) reuses
  `kvcot.discovery.pass1.eligible_event_positions`, the same rule
  `eligible_event_ids` uses for the real B1B harness, refactored into one
  shared function rather than two independently maintained copies.
- Candidates are attempted **in the committed manifest order**, maximum 12,
  **stopping immediately at the first one that satisfies every one of the
  10 mandatory conditions below** -- no later candidate is even attempted
  once one qualifies, and there is no "best of N" selection.
- `--dry-run` touches no CUDA, no model, no R-KV (structurally verified by
  a test that raises if any of `torch`/`transformers`/`rkv`/
  `huggingface_hub`/`datasets` is imported).
- `--execute` loads the exact local `DeepSeek-R1-Distill-Llama-8B` snapshot
  already pinned by `configs/discovery/llama8b_math500_b1024.yaml`, no CPU
  offload, one RTX 3090, and writes one immutable qualification artifact
  referencing the exact candidate-manifest hash it ran against.

**Mandatory qualification conditions (all 10, every one required):**

1. `no_cap_hit` -- FullKV's natural generation did not hit `max_new_tokens`.
2. `fullkv_answer_verifiable` -- the extracted answer was not `unverifiable`.
3. `fullkv_answer_correct` -- the extracted answer matches the gold answer.
4. `predicted_schedule_has_at_least_three_events` -- the pure schedule
   simulator predicts ≥3 total R-KV compaction events at budget=1024,
   divide_length=128, from this generation's length.
5. `at_least_three_events_have_49_future_tokens` -- at least 3 of those
   predicted events pass the FULL eligibility rule (never first/last, ≥49
   real future tokens after the event) -- i.e. `eligible_event_count >= 3`,
   the same threshold the real B2A gate's `sufficient_eligible_events`
   condition checks.
6. `identity_checks_pass` -- the worker's own reported dataset/model/
   tokenizer revision agree with the config and the candidate under test.
7. `batch_size_is_one`.
8. `all_parameters_on_cuda`.
9. `no_offload` -- `parameter_placement.no_offload_verified`.
10. `peak_memory_within_limit` -- `max(peak_allocated, peak_reserved) <= 22 GiB`.

**No qualified row means immediate stop.** If none of the (at most 12)
candidates satisfies every condition, qualification records a "no
selection" result, B2A-R2/B2B remain blocked, and no manifest is frozen.
This is decided by the same fixed rule regardless of outcome -- there is no
point at which a human or automated chooser may pick a candidate that
failed a condition, weaken a threshold, or extend the candidate count
past 12.

## 5. Freezing the selected row

`kvcot.discovery.b2a_r2_freeze.freeze_qualified_row` is the only function
that may write a replacement `configs/discovery/b2a_one_example_manifest.json`.
It accepts only: the qualification artifact, its canonical hash, the
candidate manifest, and the config -- **never** a caller-supplied
`example_index` or ordinal. It rejects, before touching a tokenizer:

- a candidate manifest whose `canonical_sha256` does not match the
  qualification artifact's own recorded `candidate_manifest_hash`;
- a config whose dataset/model/tokenizer revision or budget disagree with
  what the qualification artifact recorded;
- a qualification artifact with no selected row (`selected_ordinal is None`);
- a selected candidate whose own recorded `qualified` flag, or ANY entry in
  its condition map, is not `True` (defense in depth -- re-checked here,
  never merely trusted from the artifact's summary field);
- a candidate manifest row, at the selected ordinal, whose `unique_id`
  disagrees with what the qualification artifact selected;
- row content that no longer reproduces its own recorded raw-content hash.

Only after every one of those checks passes does it resolve the new row's
prompt identity (reusing `kvcot.discovery.manifest_prepare
._render_and_tokenize`, the same rendering call every other manifest
resolution in this repository uses) and atomically write the replacement
manifest, plus a companion selection-provenance record (qualification
artifact path/hash, candidate manifest path/hash, selected ordinal/
unique_id, selection protocol version, row raw hash, prompt hash) --
`B2AOneExampleManifest`'s own schema is completely unchanged; every
existing field in it means exactly what it already meant. 10 tests in
`tests/unit/discovery/test_b2a_r2_freeze.py` prove every rejection above is
real, including a simulated row-substitution-after-qualification attack.

## 6. Exactly one B2A-R2 execute attempt

Once a row is qualified and frozen, exactly one
`kvcot b2a-calibrate --execute` attempt is authorized against it -- the
existing, unmodified B2A harness (12-real-pair-plus-one-no-op design, one
RTX 3090, batch size 1, no offload, `budget=1024`, `divide_length=128`,
`bridge_tokens=1`, `scored_horizon=48`,
`minimum_future_tokens_after_event=49`, peak tracked CUDA memory ≤22 GiB,
projected complete-pilot runtime ≤4.00 GPU-hours -- every one of these
values read from existing frozen config/constants, none changed by this
document). Once FullKV or R-KV inference begins under this attempt, it is
scientifically consumed -- no automatic or unauthorized further attempt.
Allowed outcomes: gate passed (exit 0), a clean scientific gate failure
(exit 2), or an environment/implementation failure discovered strictly
before either worker's inference began (repairable and retriable only in
that specific case, per CLAUDE.md's existing H1-H6 failure taxonomy).

## 7. What this document does not authorize

No change to `configs/lock.yaml`, `third_party/R-KV`, the pinned R-KV
revision, `budget`, `divide_length`, `bridge_tokens`, `scored_horizon`,
`minimum_future_tokens_after_event`, the selected model/dataset/revisions,
or any parity/provenance/memory/timing/device-placement gate. No B2B. No
FaithKV method implementation. No second B2A-R2 attempt once inference
begins. No CPU offload, no quantization, no multi-GPU.
