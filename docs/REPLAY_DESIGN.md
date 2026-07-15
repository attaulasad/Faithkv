# Replay design

This document is the design specification for `kvcot.generation.decode` and
`kvcot.generation.replay` — the part of this repository that cannot be
tested on this build machine (no GPU) and must be reviewed as a design
before it is trusted. Every claim about upstream behavior below cites
`docs/UPSTREAM_AUDIT.md`; every claim about *our own* code is implemented in
`src/kvcot/generation/`, not just described here.

## 1. Why bulk-prefilling a truncated prefix is not replay

R-KV is a streaming, periodic, decode-time policy (docs/UPSTREAM_AUDIT.md
§2 "H4 — CONFIRMED, with an important refinement", §3.1). Its cache state at
token position N depends on the exact sequence of forward calls that
produced positions 0..N — specifically:

- the schedule counter `self.length` (cumulative tokens processed across all
  forward calls, `modeling.py:560-563`),
- which of those calls crossed a `divide_length` multiple and thus set the
  per-layer `compression` flag (`modeling.py:598-611`),
- the query-cache observation window at each of those moments
  (`modeling.py:259-284`),
- and the actual eviction, which only fires when the *physical* cache
  length is already `>= budget` at the moment a flagged call runs
  (`r1_kv.py:46-47`).

A single bulk forward call over a truncated prefix produces a *different*
`self.length` trajectory and a *different* set of flagged/unflagged forward
boundaries than the token-by-token decode that originally produced that
prefix. Two sequences with identical final token IDs can therefore end up
with different physical cache contents, different compaction event counts,
and different surviving positions, purely because of how many forward calls
were used to reach the same point. Bulk prefill of a truncated prefix is a
different experiment, not a faster way to run the same one.

## 2. Named, checked assumption: call-shape determines `self.length` trajectory

**Assumption (to be confirmed on GPU, not yet verified — this build has no
GPU):** `self.length` (`modeling.py:560-563`) increments once per top-level
`CausalLM.forward` call, by that call's `input_ids.shape[1]`, and the
`is_newline` schedule check (`modeling.py:598-611`) runs once per call
against the resulting *post-increment* cumulative value — not once per
token. Concretely:

- **Prefill is a single multi-token call.** One forward call with
  `input_ids.shape[1] == prompt_length` increments `self.length` by the
  entire prompt length in one step, and the schedule check after that call
  sees `self.length == prompt_length` (assuming this is the very first
  call, so `self.length` was previously unset — `modeling.py:560-561`).
  Compare this to `prompt_length` individual single-token calls: those would
  cross every `divide_length` multiple *within* the prompt individually,
  each producing its own schedule check (and, per §3.3 of
  UPSTREAM_AUDIT.md, its own opportunity for prompt-token eviction if the
  running cache has already reached `budget`). A single N-token prefill
  call and N single-token calls covering the same N tokens are **not**
  guaranteed to produce the same set of schedule-check crossings, because
  `self.length % divide_length == 0` is checked only once per call, at the
  post-increment value — a multi-token jump can step *over* a multiple
  without ever seeing it land exactly on one, and land where a chain of
  single-token calls would have already fired one or more checks earlier.

- **Decode is single-token calls, by our own frozen design.** §4 of the
  build brief mandates "an explicit batch-1, token-by-token decode loop...
  Do not use `model.generate()` on the state-critical path." Under this
  loop, every decode-phase forward call has `input_ids.shape[1] == 1`, so
  `self.length` increments by exactly 1 per call and the schedule check
  runs after every single generated token. This is call-shape-consistent
  with itself, but it must also be call-shape-consistent with how the
  *original* base generation was produced, or replay silently diverges from
  what it's supposed to be reproducing.

**Consequence, stated as the actual design rule:** replay must issue
*exactly* one N-token prefill forward call for the prompt (matching how the
original base generation prefilled it), followed by *exactly* one
single-token forward call per recorded generated token (matching how the
original base generation decoded it) — in that order, with no batching,
splitting, or merging of calls relative to the original run. Matching the
**total token count** alone is not sufficient: two replays that process the
same 500 prompt tokens as one call vs. five 100-token calls can diverge in
`self.length`'s crossing pattern relative to `divide_length`, and therefore
in compaction event timing, even though every token ID is identical.

**This is the specific, load-bearing reason
`tests/integration/test_replay_gpu.py`'s comparison of compaction-event
steps (and per-layer/per-head surviving position sets at each event) is a
**hard gate**, not a diagnostic** (§6.1 of the build brief already classifies
it as a hard gate; this section is the mechanistic justification for why).
K/V tensor closeness is declared a diagnostic in that same test file
precisely because floating-point reproduction is a *weaker*, *noisier*
signal than this one: the call-shape assumption above is a structural,
discrete fact about when compaction fires, checkable exactly (same event
step, same event count, same surviving index sets) with no tolerance to
tune. If this assumption is confirmed wrong on GPU (e.g. if `self.length`
turns out to be tracked per-token internally despite the multi-token call),
this document and `kvcot.generation.replay` both need to be revised before
any Stage 0+ result is trusted — that is exactly why this is written down as
a named assumption rather than left implicit in the replay code.

**Implementation consequence in `kvcot.generation.decode`:** the base
generation function performs prefill as one call
(`forward(input_ids=full_prompt_ids, ...)`) and decode as a Python loop of
single-token calls. `kvcot.generation.replay` mirrors this exactly: one
prefill call over the recorded prompt token IDs, then one single-token call
per recorded generated token ID, teacher-forced (no re-sampling), in
original order, with no re-batching at any point — including across a probe
branch: the closing-think token sequence and the control suffix (§6 step 8)
are each fed one token at a time, never as a single multi-token call, so
they cannot introduce a call-shape divergence from the rest of the replay.

## 3. Snapshot contents

A snapshot captures, per `kvcot.generation.state.ModelState`:

| Field | Source | Why it must be captured |
|---|---|---|
| Per-layer K/V tensors | `past_key_value.key_cache[i]` / `.value_cache[i]` | The actual cache content (UPSTREAM_AUDIT.md §3.6 — accessed by attribute, not the public `Cache.update()` API, so our snapshot code must read the same attributes) |
| Per-layer query-cache tensor | `past_key_value.query_cache[i]` | The recent-query observation window (`modeling.py:259-284`); omitting it would let a restored snapshot compute different attention scores on its next compression event than the original run did |
| Per-layer, per-KV-head surviving absolute source positions | our own provenance adapter (`kvcot.generation.provenance`), not upstream (UPSTREAM_AUDIT.md H5: upstream only records at compaction events) | needed continuously, not just at events, to report prompt- vs. reasoning-token retention (§3.4) at an arbitrary snapshot fraction |
| Buffer growth since last compaction / compaction-event counter | derived from `self.length`, `divide_length`, and the physical cache length at each event | reproduces the "sawtooth" cache-length pattern described in UPSTREAM_AUDIT.md H4's refinement |
| Absolute next position (`self.length`) | `modeling.py:560-563` | §3.3: never derive `position_ids`/`cache_position` from the compressed cache length |
| Scheduling state on the patched `CausalLM` | `self.length`, `self.after_think` (if `compression_content == "think"`; not used under this repo's frozen `compression_content=all`, but the reset code must not assume its absence either — UPSTREAM_AUDIT.md §3.2) | required to resume the schedule correctly after restoring a snapshot |
| Every layer's `compress`-on-next-forward flag | `layer.self_attn.config.compression` (tri-state `None`/`True`/`False`, UPSTREAM_AUDIT.md §3.2) | must be restored per layer individually — upstream itself sets it with an explicit `for layer in self.model.layers:` loop, so a shared/aliased restore would be wrong |
| Recording/provenance metadata | `kv_cluster.kept_token_indices` etc., only when `record_kept_token_indices=True` (one `R1KV` instance per layer, UPSTREAM_AUDIT.md §3.2) | cross-checked against our own continuous provenance adapter, not relied on alone (H5) |
| Think-parse position and control-suffix position | `kvcot.probes.early_answering.ThinkSpanResult` plus the branch's own cut index | needed to know where in the generated sequence this snapshot sits |

Deep-clone strategy: every tensor field above is captured via `.clone()`
(never a view/alias) into CPU or GPU memory as configured, so branching from
the same snapshot twice (§6.1 hard gate: "restoring the same snapshot twice
and applying the same control suffix yields identical probe tokens") cannot
have one branch's teacher-forced writes bleed into another's. Python-level
bookkeeping (compaction counters, event-step lists, provenance dicts) is
deep-copied via `copy.deepcopy`, not shared references.

Dataset-scale runs do not persist full K/V tensors to disk (§6: "Do not
persist full K/V tensors for dataset runs — only for a tiny replay-
validation fixture"); the JSONL records store `snapshot_cache_hash`,
`snapshot_provenance_hash`, and `snapshot_state_hash` (SHA-256 over shapes
and, for the tiny fixture only, tensor contents) rather than raw tensors.

## 4. Structural sharing of the replay/branch path across conditions (§6)

`kvcot.generation.replay.replay_and_snapshot(...)` takes a `policy` object
(`kvcot.generation.policies.Policy`, one of `FullKVPolicy`,
`PatchedNoopPolicy`, `RKVPolicy`) as a parameter and contains no
condition-specific branching itself. FullKV probes and R-KV probes call the
exact same function with a different policy instance; there is no separate
"FullKV replay" and "R-KV replay" code path to accidentally let drift apart.
This is enforced structurally (one function, one call site per probe),
matching §6's requirement: "If FullKV probes ever branch from a bulk
prefill while R-KV probes branch from streaming replay, the comparison is
dead."

## 5. Open assumptions GPU validation must confirm or refute

Beyond §2's named call-shape assumption, restated here for a single
checklist (cross-referenced from `docs/GPU_VALIDATION_PLAN.md`):

1. **Call-shape assumption (§2 above)** — one N-token prefill call plus
   one single-token call per decoded token reproduces the same
   `self.length` trajectory, schedule-check crossings, and compaction event
   steps as the original base generation. **Hard gate**:
   `test_replay_gpu.py`'s compaction-event-step and surviving-index-set
   comparisons.
2. **FlashAttention-2 forward determinism at fixed shapes** is
   `[unverified]` (§6.1 of the build brief states this explicitly). We test
   both `flash_attention_2` and `sdpa` and report both; only exact
   token/event-step reproduction is a hard gate, K/V closeness is a
   diagnostic specifically because of this uncertainty.
3. **`DynamicCache.key_cache`/`value_cache` attribute stability** across the
   pinned `transformers==4.55.4` install — UPSTREAM_AUDIT.md §3.6 notes this
   is upstream's own internal-attribute coupling, not ours, but our snapshot
   code reads the same attributes and inherits the same fragility. If GPU
   validation shows these attributes don't exist or have a different shape
   under the installed version, that is a version-pin bug to fix before
   anything else.
4. **Prefill-time prompt eviction** (UPSTREAM_AUDIT.md §3.3): if a
   calibration/main-split prompt is ever long enough to exceed the smallest
   candidate budget (128) on its own, the very first forward call can evict
   prompt tokens before any token is generated. Our provenance adapter must
   correctly attribute this as prompt-token eviction, not reasoning-token
   eviction, from the very first snapshot (f=0) onward. Untested until GPU
   validation runs a real prompt near or over budget length.
5. **Tri-state `compression` flag reset completeness**: our reset function
   (`kvcot.generation.state.reset_patched_state`) sets every layer's
   `self_attn.config.compression` back to `None` (the documented initial
   value from `compression_config["compression"]`, UPSTREAM_AUDIT.md §3.2)
   before each independent run. Assumed but not GPU-confirmed: that no other
   upstream code path reads a stale `True`/`False` value before the first
   `CausalLM.forward` call of a fresh run completes and overwrites it again.
