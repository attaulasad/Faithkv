# Record schema

Human-readable companion to `src/kvcot/schemas.py`, the executable source
of truth. Every JSONL record is validated against one of these Pydantic
models before it is appended (`kvcot.utils.io.JsonlWriter`); this document
explains *why* each field exists, not just its type.

## Design principles

- **No stale fields.** `docs/UPSTREAM_AUDIT.md` H8 documents a real upstream
  incident where a leftover `generation` field in a shipped dataset file
  silently got scored instead of fresh output. Every record here is
  constructed field-by-field from the current run; nothing is ever copied
  wholesale from a dataset row or a prior record.
- **Configured vs. measured, never conflated (§9).** `MethodConfig` holds
  what was *asked for* (budget, window, mix_lambda, ...). `RetentionSummary`
  holds what *actually happened* (physical cache slots vs. FullKV-equivalent
  slots). A condition is never named after a configured or measured
  percentage — see `tests/unit/test_no_ten_percent_naming.py`.
- **Undefined is a real state, not coerced to False/0.** `is_correct: bool |
  None` — `None` means extraction failed, and is never silently treated as
  incorrect. `RetentionSummary | None` — `None` for `condition == "full"`,
  since FullKV has no compression to measure.

## `BaseRunRecord` (`record_type = "base_generation"`)

One row per (problem, condition, seed) independent base generation.

| Field | Why |
|---|---|
| `record_id` | Stable, deterministic (`base-{condition}-{dataset}-{row}-seed{seed}`) — the unit resume dedup operates on. |
| `parent_record_id` | `None` for base records; set on derived records (not used here, reserved for future record types). |
| `provenance.upstream_rkv_commit` | The pinned R-KV SHA this record was produced under — lets a later analysis distinguish records produced before/after a submodule pin change. |
| `versions` | python/torch/cuda/transformers/flash_attn — a version drift between two runs being compared is exactly the kind of thing that should be visible in the data, not just in a log file. |
| `dataset.question_hash` | Verified against the live dataset on every load (`kvcot.data.verify_manifest_row_against_live_question`) — the mechanism that would have caught the H8-class bug one layer earlier. |
| `condition` | `"full"` \| `"patched_noop"` \| `"rkv_b{budget}"`. Never a percentage. |
| `method_config` | Configured parameters only (§9) — see `RetentionSummary` for measured. |
| `global_seed` / `derived_seed` | `derived_seed` is `kvcot.utils.seeding.derive_seed(global_seed, dataset_name, problem_index)` — reproducible, order-independent, identical across FullKV and R-KV for the same example. |
| `think_span` | `ThinkSpanInfo` — parse status is always recorded, even on failure (§3.5: "never guess"). |
| `extraction_method` / `extraction_failure_reason` | Always recorded, even when `extracted_answer is None`. |
| `is_correct` | `None` iff extraction failed — never coerced to `False`. |
| `cap_hit` | `True` iff `max_new_tokens` was reached without a natural EOS — a diagnostic in its own right (§8.3/§8.4), and part of the eligibility filter. |
| `compaction_event_steps` | Absolute token positions at which a real eviction fired (docs/UPSTREAM_AUDIT.md H3-H5) — `[]` for FullKV and for `patched_noop` (should always be `[]` there — see `test_patched_noop_parity_gpu.py`). |
| `retention` | `None` for `condition == "full"`. See `RetentionSummary` below. |
| `provenance_retention` | Prompt- vs. think-token survival, aggregated across layers/KV-heads (§3.4, §9). `None` for FullKV (nothing is ever evicted, so the distinction is moot). |
| `replay_state_hash` | Filled in once a replay pass has validated this base run reproduces identically (§6.1) — `None` until then. |

## `RetentionSummary` (measured, §9)

| Field | Meaning |
|---|---|
| `fullkv_equivalent_slots` | Absolute count of tokens processed so far — what FullKV's cache length would be at this point. |
| `physical_cache_slots_per_layer` | What R-KV's cache actually holds, per layer (should be equal across layers — verified, not assumed). |
| `instantaneous_retention_ratio` | `physical / fullkv_equivalent` at this snapshot. Never a single fixed number for a whole run — it sawtooths between compaction events (docs/UPSTREAM_AUDIT.md H4 refinement). |
| `post_compaction_budget_tokens` | The configured budget the cache was most recently compacted down to. |
| `tokens_since_last_compaction` | How far into the current "sawtooth" growth phase this snapshot sits. |

## `ProbeRunRecord` (`record_type = "probe"`)

One row per (base record, probe fraction).

| Field | Why |
|---|---|
| `fraction` | One of the 9 frozen probe fractions (`kvcot.config.PROBE_FRACTIONS_ALL`). |
| `think_span_length` | `L` — number of generated tokens strictly inside the think span (§6 step 6). |
| `cut_index` | `floor(fraction * L)` (§6 step 7). |
| `control_suffix_token_ids` | The exact tokens teacher-forced after the closing-think marker — always identical across records at the same fraction, since `kvcot.probes.templates.CONTROL_SUFFIX_TEXT` is the single source. |
| `matches_own_condition_base_answer` | `match_{i,c,s}(f)` (§8) — `None` iff either side's extraction failed; matched against the **same condition's own** untruncated base answer, never gold, never across conditions. |
| `is_f1_stability_probe` | `True` iff `fraction == 1.0` — the stability control (§4/§8.1), excluded from EAS. |
| `snapshot_cache_hash` / `snapshot_provenance_hash` / `snapshot_state_hash` | Cheap fingerprints for dataset-scale runs, which never persist full K/V tensors (§6) — only the tiny replay-validation fixture does that. |

## `FixedTraceProbeRecord` (`record_type = "fixed_trace_probe"`)

Secondary, additive diagnostic (`kvcot replay-fixed-trace`/`analyze-fixed-trace`,
`kvcot.analysis.fixed_trace`) — NOT the frozen primary record type. Replays
ONE canonical trace (always FullKV's own generated tokens,
`trace_source_condition`) under a possibly-different cache policy
(`replay_policy_condition`), so both conditions teacher-force identical
prompt and reasoning tokens and only the cache policy varies.

Protocol v2 (2026-07-16, `CHANGELOG.md`): protocol v1 used an empty
`control_suffix_token_ids` and the frozen 48-token probe budget, and gated
eligibility on a recorded compaction *event count* alone. Both were wrong —
see the `kvcot.analysis.fixed_trace` module docstring for the full root-cause
analysis. Protocol v2's fields:

| Field | Why |
|---|---|
| `control_suffix_token_ids` | **No longer always `[]`.** A teacher-forced boxed-answer format prefix (`kvcot.probes.templates.FIXED_TRACE_SUFFIX_TEXT`, `"\n\nFinal answer: \\boxed{"`), identical across conditions, so extraction reliably reaches a `\boxed{...}` within a short budget instead of falling through to the conservative final-number fallback. |
| `probe_extraction_text` | The text `extract_answer` actually ran on: `control_suffix_token_ids` decoded + `probe_output_token_ids` decoded. `probe_output_text` (generated tokens alone) does **not** contain the opening `\boxed{` — it lives in the prefix. |
| `probe_stop_reason` | `"eos"` \| `"boxed_answer_complete"` \| `"max_new_tokens"`. A fixed-trace probe stopping on `"max_new_tokens"` (`probe_cap_hit`) without a complete box is a red flag, not a normal case — the stop predicate (`kvcot.utils.answers.has_complete_boxed_answer`) is expected to fire first. |
| `replay_retention_at_cut` / `actual_compression_at_cut` | Realized, measured retention AT THE SNAPSHOT this probe branched from (§9) — `actual_compression_at_cut` requires the physical cache to have actually shrunk relative to the FullKV-equivalent slot count, never just a nonzero `replay_compaction_count_at_cut`. R-KV can record a compaction event that evicts zero tokens at the exact budget boundary (`kvcot.generation.replay`'s documented boundary case) — an event count alone is not evidence of compression. |
| `probe_cache_length_final_per_layer` / `probe_actual_eviction_during_answer` | Cache state AFTER this probe's own answer decoding, to detect a further eviction that happened WHILE writing the answer (as opposed to at the reasoning cut) — such a probe is excluded from PSS (`kvcot.analysis.fixed_trace`). |
| `matches_f1_anchor_answer` | Never accepted from a non-`"boxed"` (e.g. `final_number_fallback`) `normalized_f1_anchor_answer` — a fallback anchor is documented noise, not a usable reference point (`kvcot.analysis.fixed_trace._pss_for_side`). |

## `RunManifest`

One per `kvcot generate` / `kvcot replay-probe` / `kvcot replay-fixed-trace`
invocation. Counts `n_attempted`/`n_completed`/`n_skipped_resumed`/`n_failed`
are always reported, even (especially) when resuming — a resumed run that
silently under-reports what it actually did defeats the point of having a
manifest.

## Versioning

`schema_version` is `Literal["1.2.0"]` on every record — a Pydantic literal,
not just a string default, so an old-schema record (e.g. a stray `"1.1.0"`
row from before the fixed-trace protocol v2 fields existed) fails validation
outright rather than being silently accepted and reinterpreted under the new
model. History: `"1.0.0"` (initial build) -> `"1.1.0"` (added
`FixedTraceProbeRecord`, protocol v1) -> `"1.2.0"` (2026-07-16: protocol v2
fixed-trace fields — boxed-answer prefix, realized-compression tracking,
answer-time eviction detection). A breaking schema change bumps this and is
documented in `CHANGELOG.md` — `kvcot validate-run` checks it. **Do not
resume an old output directory produced under a prior schema version** — start
a fresh `output_dir` instead (protocol v1's `results/raw/early_gap_b*`
directories from before 2026-07-16 are not valid inputs to protocol v2
commands).
