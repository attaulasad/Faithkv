# B0.5-R — Protocol repair: causal false-negative discovery protocol and readiness gate

Phase B0.5-R artifact (2026-07-19). Branch `research/b0-5-protocol-repair`,
cut from `main` at `48fdf7f5020cbac6501fa45a7d1ffc38e668d4f1`. Roles applied:
strict senior research engineer; causal-inference reviewer; adversarial
ML-systems reviewer. **Documentation and pinned-source audit only.** No
GPU used, no model inference run, no Vast.ai accessed, no model weights or
MATH-500 data downloaded, no B1 harness implemented, no compression method
implemented, no file under `src/`, `tests/`, `configs/`, `results/`, or
`third_party/` modified.

This document supersedes the technical claims in
`docs/B0_5_DISCOVERY_PROTOCOL.md` and `docs/B0_5_FEASIBILITY_AUDIT.md`
listed in §5, §7, §15 below, and replaces the verdict recorded in
`docs/b0_5_decision.json`. It does **not** reopen or revise the B0 method-
novelty verdict, which is reaudited in §3 and reconfirmed unchanged.

## 1. Executive verdict

**B0.5-R VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION.**

The prior document's READY verdict did **not** survive unmodified: two of
its load-bearing technical assumptions (§5, §7 below) were checked directly
against the pinned R-KV source and found false. Both are repaired in this
document into a technically coherent, implementable design. The verdict is
carried forward only in the repaired form, scoped strictly to B1A
(CPU-side prerequisite implementation) — it does **not** authorize B1B, a
GPU, model inference, Vast.ai, the 12-example pilot, or any method claim.

## 2. Repository and source revisions inspected

```
initial branch (this session): research/b0-5-discovery-protocol
initial HEAD:                  5f1ccdc67dee07d27d044a15d688fc308643aab2
working tree:                  clean (git status --short: no output)
origin/main (after fetch):     48fdf7f5020cbac6501fa45a7d1ffc38e668d4f1
                                (f7e9dcc..48fdf7f, fetched this session)
B0.5 commit:                   5f1ccdc67dee07d27d044a15d688fc308643aab2
  — present on origin/research/b0-5-discovery-protocol, NOT on origin/main
    (git merge-base --is-ancestor 5f1ccdc origin/main → exit 0 is WRONG to
    assume; verified directly: 5f1ccdc is only reachable from
    research/b0-5-discovery-protocol / origin/research/b0-5-discovery-protocol,
    not from origin/main — main's tip f7e9dcc..48fdf7f postdates the B0
    merge (#13) but does not include the B0.5 branch, which was never
    merged. This repair branches from main, per the task's instruction,
    not from the unmerged B0.5 branch.)
pinned R-KV submodule SHA:     45eaa7d69d20b7388321f077020a610d9afb65bd
                                (git ls-tree HEAD third_party/R-KV; matches
                                configs/lock.yaml and docs/UPSTREAM_AUDIT.md)
R-KV submodule population:     populated on disk (cone sparse-checkout,
                                HuggingFace/ only) at exactly the pinned
                                commit — `git -C third_party/R-KV rev-parse
                                HEAD` → 45eaa7d69d20b7388321f077020a610d9afb65bd.
                                `git submodule status` shows a leading "-"
                                (uninitialized via `git submodule init`)
                                purely because the sparse checkout was set
                                up manually, not via `submodule update
                                --init`; the checked-out content is real and
                                at the correct commit, confirmed by direct
                                inspection, not assumed from the status flag.
```

Files read directly at the pinned commit: `third_party/R-KV/HuggingFace/rkv/compression/r1_kv.py`,
`third_party/R-KV/HuggingFace/rkv/monkeypatch.py`, `third_party/R-KV/HuggingFace/rkv/modeling.py`,
`third_party/R-KV/HuggingFace/rkv/compression/__init__.py`, `third_party/R-KV/HuggingFace/run_math.py`.
Repository docs read in full: `CLAUDE.md`, `PLAN.md`, `README.md`, `CHANGELOG.md` (tail),
`docs/UPSTREAM_AUDIT.md`, `docs/B0_5_DISCOVERY_PROTOCOL.md`, `docs/B0_5_FEASIBILITY_AUDIT.md`,
`docs/B0_5_SEARCH_LOG.md`, `docs/b0_5_decision.json`, `docs/PIVOT_PILOT_PROTOCOL.md`,
`docs/METHOD_PIVOT_SPEC.md`, `configs/lock.yaml`, `src/kvcot/generation/policies.py`,
`src/kvcot/utils/answers.py`.

## 3. What B0 and B0.5 attempted (verified)

The interpretation in the task brief is **accurate**, reconfirmed by
direct reading, not by trusting the prior verdict:

- B0 (`docs/METHOD_PIVOT_SPEC.md`) correctly rejected the original
  diagnostic (fixed-trace replay + early answering) as insufficiently
  novel against CASK (arXiv:2604.10900) and Lanham et al. (arXiv:2307.13702),
  and separately found none of three candidate compression *methods* (M1
  residual causal-utility protection, M2 interaction-aware dynamic rescue,
  M3 faithfulness-constrained allocation) clears a novelty bar: M1 PARTIAL,
  M2 KILLED, M3 PARTIAL. **METHOD PIVOT VERDICT: BLOCKED — NO NOVEL METHOD
  YET.** This repair does not touch this verdict; it is orthogonal (B0.5 is
  a *discovery* question, not a method claim, per B0's own §17).
- B0.5 (`docs/B0_5_DISCOVERY_PROTOCOL.md` §0) proposed investigating
  **causal false negatives**: cache entries with high counterfactual future
  utility that R-KV's own deployable score ranks poorly (i.e., evicts or
  would evict), at an operating point that first clears predeclared
  accuracy and compression gates. Confirmed accurate framing.
- Candidate A (`docs/B0_5_FEASIBILITY_AUDIT.md` §3) is
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` + MATH-500 + R-KV budget 1024,
  selected because the pinned R-KV README/paper (arXiv:2505.24133) reports
  this exact (model, dataset, budget) triple **lossless at a 34% retention
  ratio** — a **primary-source, externally-observed** number, not this
  repository's own reproduction.
- That external report has **not** been reproduced inside this
  repository's own batch-1, token-by-token replay engine, chat-template
  conventions, or evaluation code. Confirmed by direct search: no MATH-500
  manifest, config, or result exists in this repository for Candidate A.
  **The point is accuracy-plausible, not accuracy-preserving or validated**
  — this framing survives the repair unchanged.

B0's verdict is preserved. B0.5's research question is preserved. What
did **not** survive is B0.5's *technical* description of how the
intervention would be implemented — §5 and §7 below.

## 4. R-KV compaction semantics (ground truth, from the pinned source)

Read directly from `third_party/R-KV/HuggingFace/rkv/{modeling.py,compression/r1_kv.py,monkeypatch.py}`
at commit `45eaa7d69d20b7388321f077020a610d9afb65bd`. This repository's own
`docs/UPSTREAM_AUDIT.md` (H3, H4, H5, §3.1, §3.4, §3.6) already established
every fact below with file:line citations — the finding of this repair is
that `docs/B0_5_DISCOVERY_PROTOCOL.md` §1 **did not use its own sibling
document's findings**.

1. **`divide_length=128` is a cadence trigger, not an eviction unit.**
   `CausalLM_forward` (`modeling.py:598-611`) maintains one cumulative
   token counter `self.length` on the top-level model object and sets
   `is_newline = (self.length % divide_length == 0)` once per top-level
   forward call, then broadcasts this boolean to every layer's
   `config.compression` flag (`modeling.py:609-610`). This only decides
   **when a compaction attempt happens**, uniformly across all layers,
   simultaneously.
2. **Whether that attempt actually evicts anything is a second, separate
   condition**, inside `R1KV.update_kv` (`r1_kv.py:46-47`):
   `if kv_cache_len < self.budget: return key_states, value_states` — a
   no-op. Eviction only fires once a given layer's own cache has reached
   `budget` tokens. Schedule (when checked) and trigger (whether it fires)
   are decoupled (`docs/UPSTREAM_AUDIT.md` §3.1's own language, confirmed).
3. **Selection is independent per layer.** `self.kv_cluster = R1KV(...)` is
   constructed once per attention layer inside `*Attention_init`
   (`modeling.py:73-76`, `230-234`, `404-408` — Llama/Qwen2/Qwen3
   respectively). Each layer's `update_kv` call receives only that layer's
   own `key_states`/`value_states` and computes its own `final_score`.
   There is no cross-layer coordination of which absolute token positions
   survive.
4. **Selection is independent per KV head, within a layer.**
   `key_states`/`value_states`/`query_states` inside `update_kv` have shape
   `(bsz, num_kv_heads, seq_len, head_dim)`. `final_score` and
   `indices = final_score.topk(budget - window_size, dim=-1).indices`
   (`r1_kv.py:75-82`) preserve the `num_kv_heads` dimension and run `topk`
   along the sequence dimension **per head** — different KV heads in the
   same layer keep different absolute token positions.
5. **`final_score` is a single blended scalar per (layer, head, candidate
   position)**: `final_score = attn_cache * mix_lambda − similarity_cos *
   (1 - mix_lambda)` (`r1_kv.py:75-77`), one continuous value, not a
   two-stage filter (`docs/UPSTREAM_AUDIT.md` §7.1, reconfirmed). `topk`
   runs once directly on this blended value.
6. **`indices` are positions into the current (already possibly
   compressed) key/value tensor**, not raw generation-step indices.
   `k_past_compress = key_states[:, :, :-window_size, :].gather(dim=2,
   index=indices)` (`r1_kv.py:170-179`) — `key_states` here is whatever
   this layer's cache currently holds, which after any prior compaction is
   already a gathered subset of the true generation history.
7. **Absolute-token-position provenance across repeated compactions is
   NOT preserved by default.** `R1KV.__init__` defaults
   `record_kept_token_indices=False` (`r1_kv.py:17`); with it `False`, no
   bookkeeping of which original token a surviving slot corresponds to is
   kept anywhere — only the current relative tensor position. With it
   `True`, `r1_kv.py:88-165` computes `cur_indices` — a per-layer,
   per-head chain that maps relative post-gather positions back to
   absolute token counts by threading through `self.kept_token_indices[-1]`
   and an `evicted_token_num` running offset — confirmed
   (`docs/UPSTREAM_AUDIT.md` H5) to be **additive-only bookkeeping that
   never changes `topk`'s inputs or outputs or the real eviction gather**,
   so it is safe to enable. **This repository's own `RKVMethodConfig`
   (`src/kvcot/generation/policies.py:34`) already sets
   `record_kept_token_indices: bool = True`** — this specific load-bearing
   prerequisite is already satisfied in the codebase, not merely proposed.
   `docs/UPSTREAM_AUDIT.md` H5 additionally warns this bookkeeping updates
   only at actual compaction events (not continuously) — a future
   provenance adapter must still track surviving positions between events
   itself, not just read these lists.
8. **Prompt tokens can be evicted, including on the very first forward
   call** (`docs/UPSTREAM_AUDIT.md` §3.3, `modeling.py:288-308`) — the
   corrected decision unit below is not reasoning-token-only.

### Corrected decision unit

**The atomic unit R-KV's own eviction machinery operates over is a single
cache slot, identified by the tuple:**

```
(compaction_event_id, layer_index, kv_head_index,
 absolute_token_position, pre_compaction_storage_position)
```

`divide_length=128` names none of these — it is purely the periodic
trigger that causes every layer to *check* whether to compact next, at a
cadence independent of layer/head/position. **Every B0.5-R statement
asserting a "fixed 128-token block" as the eviction unit is corrected by
this section and formally superseded in `docs/B0_5_DISCOVERY_PROTOCOL.md`
§1 (see §15 below).**

If a future implementation groups atomic slots into a coarser "bundle" for
tractability, that grouping must be defined deterministically from the
tuple above (e.g., "all slots evicted at the same `compaction_event_id`
within one `(layer_index, kv_head_index)` pair") and must never be
described as "R-KV's native eviction unit" — it would be an analysis-time
aggregation choice, stated as such.

### Absolute-position provenance across repeated compactions

Because `record_kept_token_indices=True` is already the frozen setting in
`src/kvcot/generation/policies.py`, a B1A provenance adapter can read
`R1KV.kept_token_indices[-1]` at each compaction event per (layer, head)
and combine it with a continuously-maintained mapping this repository's
own replay layer must build (per `docs/UPSTREAM_AUDIT.md` H5's explicit
warning: the upstream lists are event-snapshots, not a continuous
position ledger). This mapping is CPU-tractable bookkeeping (integer
arrays), not a new GPU cost driver.

## 5. Failure of the 128-token-block assumption

Formally restated from §4: `docs/B0_5_DISCOVERY_PROTOCOL.md` §1 ("Unit
chosen: a fixed block of `divide_length=128` consecutive generated
tokens... Block `k`... is exactly the token span evicted or retained at
compaction checkpoint `k`") is **false**. There is no compaction
checkpoint at which a contiguous 128-token span is evicted or retained as
one object — eviction/retention is decided per (layer, kv_head,
individual slot), and different heads in the same layer keep different
positions from the same nominal 128-token span. This falsifies the
premise that a single `block_id`/`block_role` label could describe what
happened uniformly across a layer, let alone across the model. **Section
6 above is the repair.**

## 6. Failure of FullKV-shadow restoration

`docs/B0_5_DISCOVERY_PROTOCOL.md` §5 proposed recovering an evicted
block's "true KV entries" by replaying the R-KV run's own generated token
IDs, teacher-forced, through a **fresh FullKV cache from position 0**, and
inserting the result into a copy of the real R-KV cache `C_rkv,t`,
describing this as recovering the block "exactly as it would have been had
it never been evicted."

**This claim is false**, for a mechanical reason grounded in §4: token
`i`'s key/value tensors at layer `L` are `k_proj`/`v_proj` applied to that
token's hidden-state input at layer `L`, which is itself the output of
layer `L-1`'s attention computed **over whatever cache state existed at
layer `L-1` when token `i` was generated**. Under the real R-KV run, any
event sampled per this protocol's own eligibility rule (§2, "not the run's
first compaction event") occurs **after at least one earlier compaction
has already fired**, so token `i`'s real, pre-eviction K/V at every
sampled event were computed by attending over an **already-compressed**
history. A shadow-FullKV replay from position 0 never experiences any
eviction anywhere in the prefix — it computes attention over the
**entire, uncompressed** history at every step. These are two different
cache-policy trajectories. For every eligible event (by the protocol's own
"not first" rule, this is *all* of them), the shadow-FullKV reconstruction
of block `i`'s hidden state — and therefore its K/V — is **not** equal to
what R-KV's own forward pass produced for that token before eviction. The
original protocol's own §5 language ("FullKV never evicts, so this
recovers block `i`'s true KV state exactly as it would have been had it
never been evicted") is exactly the false claim Critical Issue B warned
against: it is a **policy-dependent hybrid cache entry**, not the token's
true pre-eviction R-KV state.

**Required fix — capture at the true source, not via a substitute
policy:** the only valid source of block `i`'s true pre-eviction K/V is
the live R-KV run itself, captured **immediately before** the gather that
performs eviction. Concretely, this requires a **read-only instrumentation
hook around `R1KV.update_kv`'s compaction boundary** (`r1_kv.py:82`, right
after `indices = final_score.topk(...)` and before the gather at
`r1_kv.py:168-179`), which must:

- record the full, not-yet-gathered `key_states`/`value_states` tensors
  (or the specific rows about to be dropped, identified by the complement
  of `indices`), tagged with `(layer_index, kv_head_index,
  absolute_token_position)` per §4's corrected unit;
- record `final_score`, the selected `indices`, `budget`, `window_size`
  at the same instant;
- be implemented as code living in `src/kvcot` that wraps/monkeypatches
  `R1KV.update_kv` an additional time (the same pattern
  `docs/UPSTREAM_AUDIT.md` §6 already uses to justify "instrumentation from
  this repository's own code, not a fork of the upstream source" — no
  file under `third_party/R-KV` is edited);
- be verified, by construction, to be read-only: it must not alter
  `final_score`, `indices`, or the tensors returned to
  `past_key_value.update(...)` — the same non-interference argument
  `docs/UPSTREAM_AUDIT.md` H5 already made for `record_kept_token_indices`
  applies here by the identical mechanism (additive bookkeeping computed
  from values already produced, not fed back into the decision);
- be treated as **temporary measurement evidence** captured during a
  labeling run, not part of any proposed compression method.

**If, at implementation time, this hook cannot be attached without
altering R-KV's deployed decision** (e.g., a transformers version where
the monkeypatch shape has changed, `docs/UPSTREAM_AUDIT.md` §3.5's `<4.56`
ceiling), **the intervention is blocked for that run** — captured data
must never be silently substituted with a shadow-FullKV reconstruction
again. This is a hard requirement, not a preference.

## 7. Intervention design

### 7.1 Why "mask/zero" cannot stand in for "removed"

The prior protocol's own language ("Mask/zero its KV entries") conflated
two different interventions. A zeroed key vector still occupies a slot in
the attention computation: `softmax(q·k_zero / sqrt(d))` is not
necessarily the minimum score in the row (real keys can produce negative
dot products), so a zeroed slot can receive **non-negligible** attention
probability and distort the softmax normalization for every other slot in
the same row; a zeroed value still contributes to the weighted sum
wherever the (non-zero) attention weight lands on it. Neither zeroing K
nor zeroing V is equivalent to removing the slot from the computation.
**True removal requires either (a) physically shrinking the tensor along
the sequence dimension (deleting that gather position, changing the
physical slot count) or (b) an additive `-inf` attention-mask term at
exactly that position** (numerically equivalent to removal for the
attention *output*, though it does not shrink cache memory — if used, it
must be reported as a memory-inert removal, not conflated with (a)). This
repair selects (a) for both branches below, so that "physical bytes
changed" (required by §7.2) is never ambiguous.

### 7.2 Selected primary and secondary estimands

Two structurally different perturbations were being computed on two
different populations (evicted candidates vs. retained controls) and then
pooled into one Spearman correlation and one recall statistic in the prior
`docs/B0_5_DISCOVERY_PROTOCOL.md` §7. This repair keeps them **structurally
separate, at every stage of analysis** — never combined into one pooled
utility distribution, satisfying the task brief's Design-A/Design-C
split and the "no pooling" requirement.

**Primary estimand — Design A, equal-byte add-back, evicted-candidate
population only.** For each sampled evicted candidate `i` (a single
(layer, kv_head, absolute_token_position) slot per §4's corrected unit),
construct `C_present,i` by inserting exactly one recovered slot (captured
per §6) into a copy of the real deployed cache `C_rkv,t`, at that layer
and kv_head only, at its correct sorted relative position by absolute
token position. `C_absent,i` is simply `C_rkv,t` itself, unmodified (the
real, natural eviction — nothing synthetic on this side, matching the
original protocol's correct asymmetry principle). **Every evicted
candidate's intervention adds exactly one slot at one (layer, kv_head)
pair** — added-byte cost is therefore identical across the whole evicted
population, satisfying "compare candidates only against other evicted
candidates with the same added-byte cost."

**Secondary/sensitivity estimand — Design C, retained-only physical
ablation, retained-control population only.** For each sampled retained
control `j`, construct `C_absent,j` by physically deleting slot `j`'s
gather position from a copy of `C_rkv,t` at that layer and kv_head
(tensor-slicing removal, never zeroing, per §7.1) — the cache shrinks by
exactly one slot. `C_present,j` is `C_rkv,t` itself, unmodified (the real,
natural retention). Every retained-control intervention removes exactly
one slot at one (layer, kv_head) pair — removed-byte cost is identical
across the whole retained-control population.

**These two populations' utility values are never merged into one rank
correlation, one recall statistic, or one pooled distribution.** Baseline-
signal validity (§7.5's Critical Issue A/E requirement) is checked
separately within each population. A between-population comparison (e.g.,
"is mean utility higher for evicted candidates than retained controls"),
if ever performed in a later phase, must be a separate, explicitly labeled
analysis, not a substitute for within-population statistics.

**What remains fixed during every single-slot intervention:** every other
slot's KV content at every other (layer, kv_head, position), model
weights, attention backend, decoding mode (greedy, per the frozen probe
convention), and the exact teacher-forced token sequence. Reference tokens
are always the R-KV natural run's own actually-generated continuation
(never resampled), unchanged from the original protocol's §5 — this part
survived the repair.

### 7.3 Effect definition and sign convention

One general rule, applied consistently to both populations:

```
u_state = NLL_mean(ref_tokens | state ABSENT) − NLL_mean(ref_tokens | state PRESENT)
```

evaluated as the mean per-token negative log-likelihood of the
`horizon`-length reference continuation (§8 below), under greedy/teacher-
forced decoding.

- **Evicted candidate `i`:** `state ABSENT` = `C_absent,i` = `C_rkv,t`
  (the real baseline); `state PRESENT` = `C_present,i` = `C_rkv,t + {i}`.
  `u_i = NLL(ref | C_rkv,t) − NLL(ref | C_rkv,t + {i})`. **Positive `u_i`
  means restoring block `i` reduces NLL — block `i` had positive
  counterfactual utility that R-KV's eviction discarded.**
- **Retained control `j`:** `state ABSENT` = `C_absent,j` = `C_rkv,t −
  {j}`; `state PRESENT` = `C_present,j` = `C_rkv,t` (the real baseline).
  `u_j = NLL(ref | C_rkv,t − {j}) − NLL(ref | C_rkv,t)`. **Positive `u_j`
  means block `j`'s continued presence was reducing NLL relative to its
  (counterfactual) removal** — the same interpretive direction ("positive
  = this block's presence helps prediction") even though the arithmetic
  sides are swapped, because the *real* deployed cache appears on
  opposite sides of the two formulas by construction (§7.2's asymmetry
  principle). **This sign convention is stated here once and must never
  be re-derived independently elsewhere**, mirroring this repository's
  existing discipline for `Delta_EAS` in `CLAUDE.md` §8.

## 8. Memory/byte accounting

Every single-slot intervention changes physical cache memory at exactly
one `(layer_index, kv_head_index)` pair by exactly `±1 × head_dim × 2`
(K and V) `× dtype_bytes` (BF16 = 2 bytes), i.e. one KV slot. This is
identical across the whole evicted-candidate population (all `+1` slot)
and identical across the whole retained-control population (all `−1`
slot) — no intervention ever changes total physical bytes by a different
amount than its sibling interventions in the same population, and no
intervention is ever a whole-cache or whole-layer perturbation. This
directly answers attack §17.6 ("your total cache bytes changed") — yes, by
a fixed, declared, population-uniform amount, never an undeclared one.

## 9. Provenance schema

Replaces `docs/B0_5_DISCOVERY_PROTOCOL.md` §12's block-level schema. One
JSON-lines record per **labeled slot** (not per 128-token block):

```json
{
  "schema_version": "b0_5_r.v1",
  "example_id": "string",
  "prompt_hash": "sha256 hex string",
  "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
  "model_revision": "string (HF sha, pinned before any run)",
  "tokenizer_revision": "string",
  "rkv_revision": "45eaa7d69d20b7388321f077020a610d9afb65bd",
  "config_hash": "sha256 of the resolved compression_config dict",
  "budget": 1024,
  "window_size": 8,
  "divide_length": 128,
  "compaction_event_id": "int, ordinal within the run",
  "generation_step": "int, absolute token step of this compaction",
  "layer_index": "int",
  "kv_head_index": "int",
  "absolute_token_position": "int",
  "pre_compaction_storage_position": "int (relative index inside the pre-gather tensor)",
  "retained_or_evicted": "retained | evicted",
  "rkv_final_score": "float (final_score at capture)",
  "rkv_score_components": {"attn_cache": "float", "similarity_cos": "float", "mix_lambda": 0.1},
  "recency_position": "int (distance from compaction boundary)",
  "attention_recent": "float | null",
  "key_norm": "float | null",
  "value_norm": "float | null",
  "token_id": "int (auxiliary only, never used as identity)",
  "decoded_token_display": "string (display only, never used for matching)",
  "kv_capture_artifact_id": "string (temporary artifact reference, not a method component)",
  "intervention_type": "equal_byte_add_back | retained_physical_removal",
  "physical_bytes_added": "int (0 or +1 slot equivalent)",
  "physical_bytes_removed": "int (0 or +1 slot equivalent)",
  "baseline_cache_bytes_at_layer_head": "int",
  "intervention_cache_bytes_at_layer_head": "int",
  "outcome_horizon_tokens": 48,
  "per_token_nll_absent": ["float", "..."],
  "per_token_nll_present": ["float", "..."],
  "u_primary": "float",
  "u_secondary": {
    "answer_token_nll_delta": "float | null",
    "kl_divergence_or_approx": "float",
    "kl_is_approximation": "bool",
    "top1_agreement_rate": "float",
    "target_logprob_present": "float",
    "target_logprob_absent": "float",
    "branching_marker_mass_present": "float",
    "branching_marker_mass_absent": "float"
  },
  "entropy_at_capture": "float | null",
  "logit_margin_at_capture": "float | null",
  "hidden_state_delta": "float | null",
  "recurrence_flag": "bool | null",
  "oracle_non_deployable": true,
  "thought_category": null,
  "cap_hit_flag": "bool",
  "valid_flag": "bool",
  "invalid_reason": "string | null"
}
```

`token_id`/`decoded_token_display` are explicitly marked auxiliary —
identity for matching/deduplication is always the tuple
`(layer_index, kv_head_index, absolute_token_position,
compaction_event_id)`, never decoded text. Ties in `rkv_final_score`
(possible in float32) are broken by ascending `absolute_token_position`
(deterministic, stated once here). Top-k provenance (`topk`'s own internal
tie-break, PyTorch's stable-since-CUDA-11 behavior) is not additionally
relied upon — this schema's own tie-break is independent of it.

## 10. Outcome metric

- **Reference tokens:** the real R-KV natural run's own actually-generated
  continuation (teacher-forced), never FullKV's or a resampled one —
  unchanged from the original protocol, now consistent with §6's fix
  (only the *cache construction*, not the *reference token source*,
  changed).
- **Replay prefix:** `C_rkv,t` (or its single-slot-modified copy) plus the
  ordinary decode-time forward pass mechanics already used elsewhere in
  this repository's replay engine.
- **Intervention insertion point:** exactly at the sampled compaction
  event `t`; the modified cache is used starting from the first reference
  token predicted after `t`.
- **First measured logit:** computed **after** the intervention — i.e.,
  the first scored position's next-token distribution is conditioned on
  the modified cache (`C_present`/`C_absent`), not the original
  `C_rkv,t`. This resolves the "before/after" ambiguity explicitly.
- **Horizon:** 48 teacher-forced tokens, reusing `configs/lock.yaml`'s
  frozen `probes.max_new_tokens: 48` — unchanged from the original
  protocol (no new number invented).
- **EOS:** if EOS is reached before 48 tokens, the branch is marked
  `cap_hit_flag`-equivalent for the horizon window and its per-token NLL
  series is truncated at EOS, not padded; whether such truncated branches
  are included in aggregate statistics is decided by the eligibility rule
  in §2/§9 of the original protocol (a run whose own natural continuation
  is shorter than `horizon` past the event is excluded at event-selection
  time, checked again at label-construction as a hard assertion) —
  unchanged.
- **Aggregation:** per-slot `u_state` is the mean per-token NLL
  differential over up to 48 tokens (§7.3); no further aggregation to a
  block or example level is performed for the primary statistic — the
  labeled slot is the unit of analysis, matching the corrected decision
  unit (§4).
- **Uncertainty:** by-example clustering of the rank-correlation and
  recall statistics is retained from the original protocol §7.3 (blocks
  from the same example are not independent) — unchanged, still required.
- **Branch independence:** each labeled slot's replay is an independent
  forward-pass branch off the same shared unablated baseline continuation
  for that event (one shared baseline per event, reused across all slots
  sampled at that event, per the cost model in §13) — nested within
  example and within event, never treated as fully independent across the
  whole pooled `n`.

## 11. Known-signal controls

Retained from the original protocol §7 essentially unchanged in content
(R-KV's own `final_score`/components, attention-recency, key/value norm,
recency/position, entropy, top-1/top-2 logit margin, hidden-state delta,
recurrence, the non-deployable future-attention oracle), with two
corrections:

1. **Granularity now matches §4's corrected unit** — every baseline signal
   is recorded per (layer, kv_head, absolute_token_position), not per
   128-token block, since that per-block granularity never existed.
2. **Entropy/logit-margin timing, made explicit** (per §5a of
   `docs/METHOD_PIVOT_SPEC.md`, unchanged confound): `entropy_at_capture`/
   `logit_margin_at_capture` are measured **at the moment the candidate
   token was originally generated** (pre-intervention, pre-treatment),
   under the real natural R-KV run — never a value measured after the
   single-slot intervention, and never conflated with the post-
   intervention horizon-position entropy (which is a *secondary*,
   distributionally-reported quantity, §10's `u_secondary`, not a
   baseline predictor). A treatment-induced value is never used as a
   pre-treatment predictor — restated here as a hard rule, not just a
   caveat.

arXiv:2606.00206 remains an adjacent caution about quantization-induced
overthinking, not direct evidence about R-KV eviction (that paper studies
weight/activation/KV quantization noise, not eviction) — this framing is
unchanged and already correctly hedged in `docs/METHOD_PIVOT_SPEC.md` §5a.

The "32-token occlusion methodology" remains attributed to no confirmed
primary source (`docs/B0_5_SEARCH_LOG.md` §4, unresolved) — this repair
does not resolve that attribution; it is carried forward labeled
unresolved, not cited as if pinned down, and is not relied upon for any
verdict in this document.

## 12. Exact gates

All ten categories the task requires, each with an exact value, an exact
denominator, and an explicit statement of what happens at the boundary
(equality). Ceilings reused from this repository's own frozen conventions
are cited to their source; every new threshold states its purpose and why
it is fixed now, before any inference.

| # | Gate | Exact rule | At equality | Denominator | Source/rationale |
|---|---|---|---|---|---|
| 1 | Natural-accuracy plausibility | `(FullKV_correct − RKV_correct) / 12 ≤ 0.10`, exact rational comparison (not float) | passes (`≤` is non-strict) | 12 examples | Reuses this repo's existing `pilot_accuracy_plausible` ceiling, `docs/EXPERIMENT.md` "0.10 pilot ceiling" (same number that flagged the retired GSM8K b128 point). Feasibility-only: never supports an accuracy-preserving claim. |
| 2 | Realized physical compression | `instantaneous_retention_ratio ≤ 0.70` on at least 9 of 12 natural R-KV runs (75%) | 9/12 passes | 12 examples | `0.70` reuses this repo's existing `meaningful_retention_ceiling` (`docs/EXPERIMENT.md`). `9/12` is a **new** threshold: purpose is to require a clear supermajority of runs actually compressing meaningfully (a bare 7/12 majority is too weak given natural per-example trace-length variance relative to a fixed budget; 12/12 is too strict for a first pilot). Feasibility threshold, not evidence of a distributional compression effect. |
| 3 | Minimum eligible compaction events per example | at least 3 eligible events (§2 of the original protocol, unchanged) on at least 9 of 12 examples (75%) | 9/12 passes | 12 examples | Reuses the `events_per_example=3` design value; `9/12` reuses the same 75% supermajority threshold as gate 2 for consistency, stated as a feasibility floor on population thinness, not a power calculation. |
| 4 | Cap-hit asymmetry | `\|cap_hit_rate_FullKV − cap_hit_rate_RKV\| ≤ 0.20` (both proportions out of 12) | ≤0.20 passes | 12 examples | New threshold, twice the accuracy-plausibility ceiling (gate 1): cap-hit is a length-based nuisance one step removed from the primary accuracy axis, so a looser but still bounded ceiling is used; reported regardless of pass/fail, never silently averaged (`docs/EXPERIMENT.md`'s attrition-funnel discipline). |
| 5 | Intervention/capture validity | the §6 pre-compaction hook's captured tensor shape, dtype, and `(layer_index, kv_head_index)` must exactly match the live `R1KV.update_kv` call it instruments, verified by direct object/shape identity, zero tolerance | any mismatch fails that slot only (dropped, logged, attrition-counted) | per labeled slot | Hard correctness assertion, not a statistical threshold — there is no partial pass. |
| 6 | Replay/parity error | this repository's existing replay-determinism discipline applies unchanged; no new numeric tolerance is introduced by this document | any non-bit-exact divergence beyond that existing discipline is a hard failure of that branch | per replay branch | Reuses, does not invent. |
| 7 | Missing/failed branch handling | any slot with a failed capture, cap-hit inside horizon, or failed identity check is marked `valid_flag=false` with non-null `invalid_reason`, excluded from all statistics | n/a — always excluded, always counted | per labeled slot | Never silently imputed or dropped without a count (§9 schema). |
| 8 | Runtime feasibility (B2A hard stop) | projected total wall-clock, recomputed from B2A's own measured throughput, `≤ 4.00` GPU-hours | exactly 4.00 passes | whole pilot | Reuses the 4-GPU-hour ceiling already used throughout `docs/B0_5_FEASIBILITY_AUDIT.md`; now a hard stop tied to a *measured*, not assumed, rate (§14). |
| 9 | Memory feasibility (B2A hard stop) | peak allocated CUDA memory `≤ 22 GiB` (of 24 GiB available) **and** all model parameters report `device.type == "cuda"` (no CPU/disk offload) | 22.00 GiB passes; any non-CUDA parameter fails regardless of memory | single B2A run | `22 GiB` is a new threshold: ~8% headroom below the 24 GiB card for allocator fragmentation and transient branch-copy buffers, chosen before any measurement exists. The device-placement check is boolean, not a threshold. |
| 10 | Discovery-pattern decision rule (B2B analysis, not a runnable gate in this phase) | Evicted population: discovery-supporting iff every deployable baseline signal has `\|Spearman ρ\| < 0.30` against `u_i` **and** `IQR(u_i) > 0.01` nats (a fixed noise floor, not zero, to distinguish genuine spread from float32 noise); Retained population: reported separately, same numeric rule, never combined with the evicted-population statistic into one correlation | `ρ = 0.30` exactly or `IQR = 0.01` exactly both count as **not** meeting the discovery-supporting criterion (strict `<`/`>` as stated) | per population (evicted n, retained n reported separately) | Replaces "low or near-zero," "non-trivial variance," "strong correlation," "notably above" with the exact predeclared cutoffs task-brief-required; `0.30` is the conventional weak-correlation threshold, stated as this pilot's frozen choice, not derived from these data. |

**No criterion above is marked true in any JSON record unless its exact
threshold and evaluation procedure, as stated in this table, is what was
actually applied** — this table is the single source of truth for gate
wording; `docs/b0_5_decision.json` is updated to reference it rather than
restate looser language (§15).

## 13. Model-family prerequisite (B1A scope)

`CLAUDE.md` §4 freezes the pipeline model as
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` only. Candidate A
(`deepseek-ai/DeepSeek-R1-Distill-Llama-8B`) is a **Llama** architecture.
Direct inspection of `src/kvcot/generation/policies.py` (`_PatchedPolicyBase.load`,
lines 116-125) confirms it **unconditionally imports and calls
`rkv.monkeypatch.replace_qwen2`** — there is no architecture branch, and
`replace_llama` (which exists in the pinned `rkv/monkeypatch.py:15-21`) is
never imported or referenced anywhere in this repository's source.

**This is a silent-corruption risk, not merely a missing feature.**
`replace_qwen2` monkeypatches `transformers.models.qwen2.modeling_qwen2.Qwen2Attention`/
`Qwen2ForCausalLM` at the class level. Loading a Llama checkpoint
instantiates `transformers.models.llama.modeling_llama.LlamaAttention`/
`LlamaForCausalLM` instead — classes the Qwen2 monkeypatch never touches.
The result would **not** crash: it would silently run ordinary,
unmodified HF Llama attention (stock FullKV-equivalent behavior, no R-KV
compression ever engaged, no compaction ever fires), while every artifact
downstream would still label the condition `rkv_b1024` — a mislabeled,
silently-incorrect run, exactly the class of defect this repository's own
`docs/UPSTREAM_AUDIT.md` §3.2 warns about for process-global monkeypatches.

**B1A must implement**, before any GPU use:
- explicit architecture detection (e.g., from `AutoConfig.model_type`) that
  selects `replace_llama` vs. `replace_qwen2` vs. `replace_qwen3` — no
  fallback to a default;
- a hard failure (not a silent default) if the detected architecture has
  no corresponding `replace_*` entry;
- deterministic state reset between examples/policies, extended to
  whichever `*Attention`/`*ForCausalLM` classes are actually patched for
  the run's architecture (the existing `docs/UPSTREAM_AUDIT.md` §3.2 reset
  list is Qwen2-specific and must be generalized);
- FullKV/R-KV construction-parity tests for the Llama code path,
  mirroring the existing Qwen2 parity tests;
- an explicit assertion that no CPU offload occurred (`device_map="auto"`
  can silently place layers on CPU under VRAM pressure — this must be
  checked, not assumed, per gate 9 in §12).

This is identified as a concrete, scoped B1A prerequisite — not a blocker
to this document's own verdict, and not something B0.5-R implements.

## 14. MATH-500 verifier prerequisite (B1A scope)

`src/kvcot/utils/answers.py` implements boxed-answer extraction and
**numeric-only** normalization (`normalize_numeric_string`, lines 69-98):
it parses plain integers and decimals, and otherwise returns the boxed
content as a lightly-cleaned literal string (`_normalize_boxed_content`,
lines 101-111). `answers_match`/`answers_match_or_none` (lines 165-188)
perform **exact string equality on already-normalized values only**.

**Confirmed gap:** there is no symbolic-equivalence checking anywhere in
this module. `\boxed{1/2}`, `\boxed{0.5}`, and `\boxed{\frac{1}{2}}` would
normalize to three different literal strings and would not match each
other, despite being the same value — MATH-500 answers routinely include
fractions, algebraic expressions, signs, tuples, sets, and intervals, all
of which this extractor treats as opaque literal text. This is adequate
for the frozen GSM8K/1.5B pipeline (numeric answers only) and is **not**
adequate for MATH-500 grading under Candidate A.

**B1A must implement**, before any GPU use: reliable MATH-500 answer
extraction feeding into a symbolic-equivalence verifier (a suitable
established library/approach — not specified further here, since
selecting and implementing it is B1A's own scoped work, not B0.5-R's);
coverage for fractions, equivalent algebraic expressions, signs, tuples,
sets, intervals, and multiple boxed expressions; explicit malformed-output
handling; timeout and verifier-error handling; unit tests for accepted and
rejected equivalence pairs; and a hard rule that no fallback silently
marks an unverifiable answer correct.

## 15. Corrected cost model

The existing `docs/B0_5_FEASIBILITY_AUDIT.md` §5 cost model is **not**
found to overstate measured throughput as fact — it already states, in
its own words, that the 35-50 tok/s RTX 3090 range is "a conservative
estimate... not a measurement" and explicitly flags the A100→3090 scaling
as "general hardware-comparison knowledge, not a source specific to this
model" (already recorded in `docs/b0_5_decision.json` `uncertainties`).
That caveat survives this repair unchanged and is not the defect.

What the repair adds is a **separation of cost components** the original
model bundled together, and a **mandatory measured-throughput gate**
before any full-pilot GPU time is spent (B2A, below):

| Component | Distinguished because |
|---|---|
| Model load + tokenizer/dataset setup | fixed, one-time, per process |
| FullKV natural generation (prefill + decode) | prefill is compute-bound and parallel across the prompt; decode is memory-bandwidth-bound and sequential — the two are not the same cost-per-token |
| R-KV natural generation (prefill + decode) | additionally pays the periodic `update_kv` compaction cost (topk/gather every `divide_length` steps), not present in FullKV |
| Fixed-trace teacher-forced replay (§10) | shorter, bounded (`horizon=48`), but pays a fresh prefill-equivalent cost to reach the branch point each time unless the shared-prefix baseline (§10, one shared baseline per event) is reused, which this repair keeps from the original design |
| Pre-compaction cache snapshot/copy (§6's hook) | new cost this repair introduces relative to the original design: one extra tensor copy per sampled compaction event, at exactly the `(layer, kv_head)` pairs sampled — bounded by `events_per_example × evicted_per_event` |
| Single-slot intervention-branch construction (§7) | O(1) tensor slice/insert per branch, not O(cache length) |
| CPU transfer/artifact serialization | I/O-bound, not GPU-bound; must not be charged against GPU-hours in the projection |
| Grading (MATH-500 verifier, §14) | CPU-bound, not GPU-bound |
| Retries/failures | covered by the existing ×3 safety factor, unchanged |

The original model's **conservative** (35 tok/s) and **optimistic**
(50 tok/s) estimates (2.4–3.4 GPU-hours, ×3 safety factor already applied)
are retained as the **planning** projection — they are not measurements
and must not be reported as such. **B2A (§16) replaces this planning
number with a measured one before B2B is authorized to spend GPU time at
the full 12-example scale.**

## 16. B2A GPU calibration gate — mandatory, not optional

Formalized as its own phase (the original protocol's §13 "stopping rule 3"
already gestured at this but did not make it a separate, hard-gated
phase). B2A must, on **one example, one eligible compaction event, the
minimal intervention set** (one evicted-candidate add-back, one
retained-control removal):

1. run a correctness/parity smoke test (FullKV/R-KV construction parity,
   per §13's B1A prerequisite);
2. measure real prefill rate and real decode rate separately (not
   conflated, per §15);
3. measure the real single-slot branch-construction and replay cost;
4. measure peak allocated and peak reserved VRAM;
5. assert all model parameters report `device.type == "cuda"` (gate 9,
   §12) — no CPU/disk offload;
6. recompute a projected complete-12-example-pilot duration from these
   measured rates (not the planning estimate);
7. **automatically stop** — do not proceed to B2B — if the recomputed
   projection exceeds 4.00 GPU-hours (gate 8, §12) or if the memory gate
   (gate 9, §12) fails.

**B2A is mandatory before the 12-example discovery pilot (B2B) is
authorized**, independent of how confident the planning estimate in §15
looks. A 12-example pilot is never authorized on the planning estimate
alone.

## 17. Repaired roadmap

- **B0.5-R (this document)** — documentation and pinned-source audit
  only; no inference; produces an exact intervention and exact gates.
- **B1A — Prerequisite implementation** (this document's own authorization
  boundary): MATH-500 verifier (§14); architecture-aware R-KV dispatch
  (§13); decision/provenance schema (§9); read-only pre-compaction
  instrumentation hook (§6); CPU unit and integration tests; **no cloud
  GPU inference.**
- **B1B — Intervention harness implementation** (not authorized by this
  document): the exact captured-KV intervention (§7); deterministic
  replay; the coherent Design-A/Design-C comparison, never pooled; CPU
  tests and mocked/tiny synthetic tests; GPU test commands defined but not
  executed.
- **B2A — One-example GPU calibration** (not authorized by this
  document): §16, in full, with its hard stop.
- **B2B — Bounded discovery pilot** (not authorized by this document):
  only if B2A passes; exactly the frozen 12/3/2/2 sample size (§8 of the
  original protocol, unchanged); no method claim; no accuracy-preserving
  claim.
- **C0 — Method design and fresh novelty gate** (not authorized by this
  document): only if B2B finds robust false negatives unexplained by
  known signals per gate 10 (§12); a genuinely new selection/compression
  technique design; a new adversarial literature check before any
  implementation.

**This document does not skip from B0.5-R to GPU execution.** It
authorizes B1A only.

## 18. Remaining blockers

- `CLAUDE.md` §4's model freeze still names only
  `DeepSeek-R1-Distill-Qwen-1.5B`. A dated `CLAUDE.md`/`CHANGELOG.md`
  amendment, scoped exactly to the validated future GPU phase, is required
  **before** any GPU run of B1B/B2A/B2B — this document does not perform
  that amendment and does not have the authority to grant it as a side
  effect of a documentation repair.
- The pre-compaction capture hook (§6) is a design, not yet code — its
  first real implementation and CPU-side unit tests are B1A's own scoped
  work.
- Candidate A's architecture dispatch (§13) and MATH-500 verifier (§14)
  do not exist in `src/` yet — identified as exact B1A deliverables, not
  implemented here.
- The "32-token occlusion methodology" attribution remains unresolved
  (§11, `docs/B0_5_SEARCH_LOG.md` §4) — not relied upon for this
  document's verdict.
- B2A's measured throughput/VRAM numbers do not exist yet — the planning
  estimate in §15 remains a planning estimate until B2A actually runs.

## 19. Final B0.5-R verdict

**B0.5-R VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION**

This authorizes only B1A: CPU-side prerequisite implementation (MATH-500
verifier, architecture-aware R-KV dispatch, decision/provenance schema,
read-only compaction instrumentation, CPU unit/integration tests). It does
**not** authorize B1B, model inference, a GPU, Vast.ai, the 12-example
pilot, or any method claim.

## 20. Adversarial self-review (§17 of the task)

1. *"R-KV never evicts 128-token blocks. Your unit does not exist."* —
   Correct, confirmed at `r1_kv.py`/`modeling.py` (§4). Repaired: the
   corrected unit is `(compaction_event_id, layer_index, kv_head_index,
   absolute_token_position, pre_compaction_storage_position)`.
2. *"Your rescued tensors came from FullKV, so this is a hybrid-state
   intervention."* — Correct (§6). Repaired: source is now a read-only
   hook at the true R-KV pre-gather boundary; if unattachable, the
   intervention is blocked, not silently substituted.
3. *"Adding an evicted entry and zeroing a retained entry are different
   treatments."* — Correct. Repaired: Design A (add-back) and Design C
   (physical removal) are never pooled (§7.2); "zeroing" is replaced by
   physical tensor-slice removal everywhere (§7.1).
4. *"Zero values remain in the attention softmax."* — Correct (§7.1);
   this is exactly why zeroing was rejected in favor of physical removal.
5. *"You pooled scores from different layers and heads."* — Not in the
   repaired design: every schema record and every statistic is keyed by
   `(layer_index, kv_head_index)` (§9); baseline-signal correlation is
   computed within, not across, the evicted/retained populations, and the
   task does not require further stratification by layer/head for the
   discovery-pattern statistic — if a future reviewer requires that too,
   it is a straightforward extension of §9's schema, not a redesign.
6. *"Your total cache bytes changed."* — By a fixed, declared,
   population-uniform amount (§8: always exactly ±1 slot at one
   (layer, kv_head) pair), never an undeclared one.
7. *"Your outcome threshold was selected after seeing the data."* — All
   thresholds in §12 are stated now, before any B1A code exists and long
   before any GPU result exists; §12 explicitly distinguishes feasibility
   thresholds (gates 1-9) from the discovery-pattern evidentiary rule
   (gate 10), and gate 10 is itself frozen now, not fit post hoc.
8. *"The chosen model cannot use the repository's Qwen-only monkeypatch."*
   — Correct, confirmed directly in `policies.py` (§13); this is a named,
   scoped B1A prerequisite, not silently ignored, and B0.5-R's verdict is
   explicitly scoped to not authorize any run that depends on it existing.
9. *"The repository cannot grade general MATH-500 answers."* — Correct,
   confirmed directly in `answers.py` (§14); named as a B1A prerequisite,
   same treatment as #8.
10. *"Your four-hour estimate was never measured on the target GPU."* —
    Correct, and stated as such in the original audit already; repaired by
    making B2A a mandatory, separately-gated phase (§16) whose measured
    numbers — not the planning estimate — gate B2B.

No answer above is incomplete in a way that would require withholding
READY-for-B1A; every open item is either fully repaired in this document
or explicitly scoped as B1A/B1B/B2A future work, never silently assumed.

## 21. Remaining uncertainties

- Exact source paper for the "32-token/stride-16 counterfactual occlusion"
  methodology remains unpinned (`docs/B0_5_SEARCH_LOG.md` §4) — not relied
  upon here.
- The R-KV paper's own "lossless at 34%/1024 tokens" claim for Candidate A
  remains unreproduced inside this repository's own replay engine — gate 1
  (§12) is exactly the check that would catch a failure to reproduce it,
  and it has not been run.
- B2A has not been run; the 4-GPU-hour/22-GiB projection in §12/§16 is a
  planning number until it is.
- Whether a future transformers version shift breaks the §6 hook's
  attachability is unknown until B1A actually writes and tests it against
  the pinned `<4.56` ceiling (`docs/UPSTREAM_AUDIT.md` §3.5).
- No systematic forward-citation-database sweep has been run for the
  discovery *methodology* question (same gap `docs/B0_5_SEARCH_LOG.md` §6
  already discloses; not changed by this repair).

## 22. Cross-reference

Superseded passages, and their exact replacements, are marked inline in
`docs/B0_5_DISCOVERY_PROTOCOL.md` and `docs/B0_5_FEASIBILITY_AUDIT.md`
with `**[SUPERSEDED — see docs/B0_5_PROTOCOL_REPAIR.md §N]**` banners; no
historical text was deleted. `docs/b0_5_decision.json` retains every
original field for provenance and adds a `superseded_by`/`b0_5_r_verdict`
block pointing here.
