# B0.5-R2 — Dense-cache representability and capture-strategy repair

**[FURTHER SUPERSEDED — see `docs/B0_5_R2_1_FINAL_PROTOCOL.md` (B0.5-R2.1,
2026-07-19), the current authorized design and verdict.** B0.5-R2.1 found
an off-by-one timing defect in §14's estimand (the first post-event
reference token was treated as scoreable when it cannot be, since its
logits are produced before the swap is applied), an under-specified
sampling rule in §9.2/§10 (an unrestricted layer/head hash that does not
actually guarantee depth coverage, and a candidate/donor tie-break that is
systematically edge-biased rather than a genuine random sample), and a
pooled (rather than per-example-nested) association test in §16's Gate 10.
All three are repaired with exact, frozen algorithms and an expanded
three-outcome decision rule (DISCOVERY-SUPPORTING / NOT
DISCOVERY-SUPPORTING / NOT ADJUDICABLE) including a mandatory no-op
control. This document's fixed-shape within-head swap design (§5-§6), its
rotary/per-slot-metadata argument (§6.1), its capture-strategy wrapper
(§8), and its aggregation hierarchy (§13) are all unaffected and remain
current. See `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §15 for the full
cross-reference. Nothing below is deleted.]**

Phase B0.5-R2 artifact (2026-07-19). Branch
`research/b0-5-r2-dense-cache-repair`, cut from `main` at
`d472f0514cd1396774b557dc27ec19900a11c1eb` (tip of `origin/main` at session
start, merge commit "Merge pull request #15 from
attaulasad/research/b0-5-protocol-repair", itself containing B0.5-R commits
`0c9a6153431a27f8d3289d928e29a350daf8ed20` and
`86bbe86bd470e0366d3b84c6ba850481e57ccf2f`). Roles applied: strict senior
ML-systems engineer; causal-inference reviewer; adversarial research
reviewer. **Documentation and pinned-source audit only.** No GPU used, no
model inference run, no Vast.ai accessed, no model weights or datasets
downloaded, no file under `src/`, `tests/`, `configs/`, `results/`, or
`third_party/` modified.

This document audits `docs/B0_5_PROTOCOL_REPAIR.md` (B0.5-R) against the
pinned R-KV source and the pinned `transformers` source and finds its
selected intervention (§7-§8 of that document: "equal-byte add-back" /
"retained-only physical ablation", both described as changing cache size
"at that layer and kv_head only") **physically impossible** in the dense
KV-cache tensor representation both R-KV and `transformers.DynamicCache`
actually use. It repairs the intervention to a fixed-memory **within-head
swap**, repairs the capture-strategy claim (B0.5-R §6's "immediately after
topk and before gather" hook), and repairs the discovery gate, sampling
plan, and cost model accordingly. It does **not** reopen B0's method-pivot
verdict or B0.5's research question, which are unaffected by a
representability fix to the intervention mechanics.

## 1. Executive verdict

**B0.5-R2 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION.**

B0.5-R's READY verdict does **not** survive unmodified: its selected
intervention (Design A / Design C, §7-§8) is not representable as
specified, and its capture-strategy claim (§6, "a hook right after
`indices = final_score.topk(...)` and before the gather") assumes a
mechanism no supported Python wrapper actually provides. Both are repaired
below into a technically coherent, implementable design — a fixed-shape
**within-head fixed-byte swap**, captured via a before/after instance-level
wrapper plus independent, exact score recomputation, verified for parity
against R-KV's own real bookkeeping, never against an invented internal
hook. The verdict is carried forward only in this repaired form, scoped
strictly to B1A (CPU-side prerequisite implementation) — it does **not**
authorize B1B, a GPU, model inference, Vast.ai, the discovery pilot, or any
method claim.

## 2. Git and source revisions inspected

```
session start:
  branch                    research/b0-5-protocol-repair
  HEAD                      86bbe86bd470e0366d3b84c6ba850481e57ccf2f
  git status --short        (clean)
  git diff --check          (clean)
origin (fetched this session):
  origin/main               d472f0514cd1396774b557dc27ec19900a11c1eb
                            ("Merge pull request #15 from
                            attaulasad/research/b0-5-protocol-repair")
  git log origin/main -5    d472f05, 86bbe86, 0c9a615, 48fdf7f, 5f1ccdc
                            — matches the task brief's expected state exactly.
this session's working branch:
  git switch main && git pull --ff-only   (fast-forwarded e0b6c68..d472f05,
                                            14 files, all doc/decision-JSON
                                            content added by PR #15)
  git switch -c research/b0-5-r2-dense-cache-repair
  (created cleanly from up-to-date main; no uncommitted work found to
  preserve, no separation-safety concern)
rkv submodule:
  git submodule status       -45eaa7d69d20b7388321f077020a610d9afb65bd
                              third_party/R-KV (leading "-": not
                              `submodule init`-registered, but populated on
                              disk via manual cone sparse-checkout, per
                              docs/UPSTREAM_AUDIT.md §0 — content verified
                              directly below, not assumed from the status flag)
transformers actually installed on this machine (verified, not assumed):
  python -c "import transformers; print(transformers.__version__)"
  -> 4.55.4  (the exact version docs/UPSTREAM_AUDIT.md H6 cites as upstream's
              own validated version; installed at
              .venv/lib/site-packages/transformers, readable directly)
```

Files read directly at the pinned R-KV commit (from the populated
sparse-checkout, not from memory or the README): `third_party/R-KV/HuggingFace/rkv/compression/r1_kv.py`
(full file), `third_party/R-KV/HuggingFace/rkv/utils.py` (full file),
`third_party/R-KV/HuggingFace/rkv/monkeypatch.py` (full file). Files read
directly from the installed `transformers==4.55.4` package: `cache_utils.py`
(`DynamicLayer`, `DynamicCache`, `KeyValuesWrapper`), `models/qwen2/modeling_qwen2.py`
(rotary application and cache-update call site). Repository files read in
full: `CLAUDE.md`, `docs/B0_5_PROTOCOL_REPAIR.md`, `docs/B0_5_DISCOVERY_PROTOCOL.md`,
`docs/B0_5_FEASIBILITY_AUDIT.md`, `docs/b0_5_decision.json`, `docs/UPSTREAM_AUDIT.md`,
`docs/REPLAY_DESIGN.md`, `docs/PIVOT_PILOT_PROTOCOL.md`, `src/kvcot/generation/policies.py`,
`src/kvcot/generation/replay.py`, `src/kvcot/generation/state.py`,
`src/kvcot/generation/provenance.py`, `PLAN.md`, `README.md`, `CHANGELOG.md` (head entries).

## 3. Previous B0.5-R verdict — what survives, what does not

Verified accurate and **not** reopened: the §4 corrected decision unit
(per-layer, per-KV-head, per-slot eviction — `divide_length` is a cadence,
never an eviction unit), the "mask/zero is not removal" argument (§7.1),
the exact numeric gate table's *feasibility* gates 1-4/8/9 (unchanged
below), the B1A-1 (architecture dispatch) and B1A-2 (MATH-500 verifier)
prerequisite findings (`src/kvcot/generation/policies.py:122-125` still
imports and calls only `replace_qwen2` unconditionally; `src/kvcot/utils/answers.py`
remains numeric-only — re-verified by direct read this session, unchanged
lines), and B0's own method-pivot verdict (untouched, out of scope here).

**Found false this session, confirmed by direct source inspection, not
carried forward:**

1. **§7-§8's "equal-byte add-back" / "retained-only physical ablation"
   design** ("insert exactly one recovered slot... at that layer and
   kv_head only", "physically deleting slot `j`'s gather position... at
   that layer and head only") is **not representable** in the dense KV
   tensor R-KV and `transformers.DynamicCache` actually use — §4-§5 below.
2. **§6's capture-hook claim** ("a read-only instrumentation hook... right
   after `indices = final_score.topk(...)` and before the gather") assumes
   a wrapper can observe a function's internal local variables from
   outside. No supported Python mechanism does this without either editing
   `third_party/R-KV` (prohibited) or a fragile frame/bytecode hack (never
   proposed and not endorsed here either) — §8-§9 below repair this to an
   implementable before/after wrapper plus independent recomputation.
3. **A previously unflagged, independently discovered defect**: even if a
   hook *could* read R-KV's own persisted "kept final score" bookkeeping
   (`kept_final_scores`, populated only when `record_kept_token_indices=True`),
   that bookkeeping is computed by a **different formula** than the one
   that actually drove the real eviction decision — §7 below. B0.5-R's
   schema field `rkv_final_score` would have silently sourced the wrong
   quantity had this not been caught here.

## 4. Dense-cache shape — ground truth, from the pinned and installed source

**`transformers==4.55.4` (installed at `.venv/lib/site-packages/transformers`,
read directly, matching the version `docs/UPSTREAM_AUDIT.md` H6 already
confirmed as upstream's own validated version):**

`cache_utils.py:68-104` (`DynamicLayer`, the concrete backing store for
`DynamicCache`'s default layer type):

```python
class DynamicLayer(CacheLayerMixin):
    """... stores the Key and Value states as tensors with shape
    `[batch_size, num_heads, seq_len, head_dim]`."""
    def update(self, key_states, value_states, cache_kwargs=None):
        if self.keys is None:
            self.keys = key_states
            self.values = value_states
        else:
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values
```

**One tensor per layer, per K/V. `num_heads` and `seq_len` are two
different tensor axes; `seq_len` (dim=-2) is a single shared dimension read
by every head's slice.** There is no per-head length field, no jagged/ragged
storage, and no third structure that could let head 3 have length 1020 while
head 5 has length 1021.

**Pinned R-KV (`third_party/R-KV/HuggingFace/rkv/compression/r1_kv.py:75-179`,
read directly this session):**

```python
final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
# shape: (bsz, num_kv_heads, budget - window_size)
indices = final_score.topk(self.budget - self.window_size, dim=-1).indices
...
indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
k_past_compress = key_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
v_past_compress = value_states[:, :, : -self.window_size, :].gather(dim=2, index=indices)
k_cur = key_states[:, :, -self.window_size :, :]
v_cur = value_states[:, :, -self.window_size :, :]
key_states = torch.cat([k_past_compress, k_cur], dim=2)
value_states = torch.cat([v_past_compress, v_cur], dim=2)
return key_states, value_states
```

`topk(self.budget - self.window_size, dim=-1)` requests the **same count**
`budget - window_size` for every KV head simultaneously (`dim=-1` is the
sequence axis of a `(bsz, num_kv_heads, seq_len)` score tensor — `topk`
returns that many indices *per head*, but always the same number of them).
Different heads select **different absolute positions**, but always the
**same count**. After `gather` + `cat` with the always-kept `window_size`
recent slots, `key_states`/`value_states` are shape `(bsz, num_kv_heads,
budget, head_dim)` — **every head has exactly `budget` slots, always,
after every compaction.** This is confirmed independently by this
repository's own code: `src/kvcot/generation/replay.py:83` and
`:215/482/522` all read `cache.key_cache[layer_idx].shape[-2]` as **one
scalar length for the whole layer**, never per-head, and
`src/kvcot/generation/provenance.py:48`'s own `LayerProvenance.positions`
is declared `shape (num_kv_heads, current_cache_len)` — a rectangular
(dense), not ragged, array, by this repository's own existing design.

**Per-slot position metadata:** `models/qwen2/modeling_qwen2.py:156-161`
(installed source, read directly):

```python
query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
if past_key_value is not None:
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
```

Rotary position embedding is applied to `key_states` **before** it reaches
`past_key_value.update(...)`. The stored key vector's rotary phase is baked
in from the absolute position it was generated at, at cache-write time.
`DynamicLayer` (§ above) stores only `.keys`/`.values` — no separate
per-slot position array exists anywhere in this cache implementation. A
physical slot's index in the tensor carries no positional meaning of its
own; the content already encodes it.

## 5. Why per-head insertion/deletion is impossible — adjudicating Critical Issue A

**Confirmed. B0.5-R §7-§8's claim ("insert/delete exactly one slot at that
layer and kv_head only, leaving other heads unchanged") is false.**

A dense tensor `(bsz, num_kv_heads, seq_len, head_dim)` has exactly one
`seq_len` value, shared by every head's slice along dim=1. To give head `h`
one extra (or one fewer) slot while every other head at that layer keeps
its current slot count, the tensor would need a **different seq_len per
head** — which this representation cannot express. The only two
representable ways to change dim=2 by one element are:

1. **Change dim=2 for the whole layer tensor** — every head at that layer
   gains (or loses) one slot, not just `h`. B0.5-R's own byte-accounting
   language ("±1 slot at one (layer, kv_head) pair") is therefore either
   describing something that changes **every** head at that layer (an
   undeclared, much larger perturbation than stated, and one that requires
   deciding what to put in the new slot for every *other* head — the exact
   padding/duplication/masking alternatives the task brief already forbids
   and B0.5-R §7.1 already rejected for a related reason), or is simply
   not achievable as literally described.
2. **Replace content at a fixed slot without changing dim=2 at all** — a
   pure substitution, shape-preserving, single-head-scoped. This is
   representable (§6 below) and is what this repair adopts.

There is no third option. B0.5-R §7's "Design A — equal-byte add-back...
at that layer and kv_head only" and §7's "Design C — retained-only
physical ablation... at that layer and head only" (tensor-slicing removal
of one slot, one head, other heads untouched) are both **superseded** —
neither can leave "every other head unchanged" while changing dim=2 for
only one head. `docs/B0_5_PROTOCOL_REPAIR.md` §7, §8 are marked superseded
in place (§14 below); nothing is deleted.

## 6. Selected representable intervention — within-head fixed-byte swap

**Design:** for one real R-KV compaction event `t`, layer `L`, KV head `h`:
pick an **evicted candidate** `e` (an absolute token position R-KV's real
topk at event `t` did **not** select, drawn from the same non-recent,
score-compared candidate pool) and a **retained donor** `r` (an absolute
token position R-KV's real topk at event `t` **did** select, occupying
physical post-compaction storage slot `r_slot` at `(L, h)`). Starting from
the real, deployed, post-compaction R-KV cache `C_rkv,t`:

```
key_cache[L][0, h, r_slot, :]   = captured_key_e     # e's real pre-gather K
value_cache[L][0, h, r_slot, :] = captured_value_e   # e's real pre-gather V
```

leaving the tensor's shape, every other layer, every other head, and every
other slot at `(L, h)` byte-for-byte unchanged. This is a **within-head
fixed-byte swap**, not an add-back and not an ablation — no dimension is
resized, so §5's impossibility does not apply.

**Representable through ordinary tensor indexing, confirmed by three
independent sources:** (a) `DynamicLayer.update` (§4) never forbids
in-place indexed writes to `.keys`/`.values` once populated — it is a plain
tensor; (b) R-KV's own upstream code already writes to cache internals
directly by index (`docs/UPSTREAM_AUDIT.md` §3.6: `past_key_value.key_cache[self.layer_idx]
= key_states_compress`, whole-tensor reassignment after `.update()` has
already populated the layer); (c) this repository's own `kvcot.generation.replay`/`state`
modules already clone and reassign whole per-layer `key_cache[i]`/`value_cache[i]`
tensors for snapshot/restore (`replay.py:288-289`, `_populate_fresh_cache`) —
a single-slot indexed write is a strict subset of operations this codebase
already performs on the same objects.

**Net physical cache bytes: exactly zero change.** No slot is added, no
slot is removed, at `(L, h)` or anywhere else. This is simpler and stronger
than B0.5-R §8's "±1 slot, population-uniform" accounting — it does not
require a population-level uniformity argument at all, because every
single swap, individually, changes zero bytes.

**What remains fixed:** every other `(layer, kv_head, slot)`'s content,
model weights, attention backend, decoding mode (greedy, per the frozen
probe convention), and the exact teacher-forced reference token sequence
(the real R-KV run's own actually-generated continuation, never
resampled — unchanged principle from every prior version of this
document).

**Baseline side:** `C_rkv,t` itself, unmodified — the real, natural,
deployed cache. There is exactly one synthetic side per swap (the
substituted slot), and it is a content substitution, never a shape change,
never a zeroing (B0.5-R §7.1's zeroing objection is moot here — the
slot is never emptied, only its content is replaced with another real,
captured value).

### 6.1 Rotary encoding and per-slot metadata (adversarial questions 7-8, answered directly)

**Does the swapped key retain its correct rotary encoding?** Yes, by
construction, with no extra step required. Per §4, rotary position
embedding is applied to a token's key vector **at the moment it is
projected and cached**, using that token's own absolute position — not
recomputed from, or dependent on, the physical slot index it is later
stored at. `e`'s captured pre-gather key already carries `e`'s own rotary
phase from the real forward pass that produced it; writing it into slot
`r_slot` (which previously held `r`'s content) does not require
re-deriving or reapplying any rotary transform, because the slot index
itself was never part of the positional encoding to begin with.

**Does `DynamicCache` store additional per-slot position metadata that
must also be restored?** No — confirmed directly from the installed
`transformers==4.55.4` source (§4): `DynamicLayer` holds only `.keys` and
`.values`. There is no parallel per-slot position-id array, timestamp, or
index inside `DynamicLayer` itself. (`cache_position` is a call-time
argument used to route the *current* forward call's writes; it is not
stored per historical slot.) Nothing beyond the K and V tensors needs to be
touched for this swap.

## 7. A previously unflagged defect: R-KV's own persisted score is not the real decision score

Independently discovered this session while auditing the capture strategy
(§8-§9 of the task), not previously flagged in any prior version of this
document. `r1_kv.py:49-82` (the real, decision-driving computation):

```python
attn_weights = compute_attention_scores(query_states, key_states)
attn_weights_sum = softmax(attn_weights[:, :, -window_size:, :-window_size], dim=-1).mean(dim=-2)
attn_cache = max_pool1d(attn_weights_sum, ...)
similarity_cos = cal_similarity(key_states, ...)[:, :, :-window_size]
final_score = attn_cache * mix_lambda - similarity_cos * (1 - mix_lambda)
indices = final_score.topk(budget - window_size, dim=-1).indices   # THE REAL DECISION
```

`r1_kv.py:88-165` (the bookkeeping path, only entered when
`record_kept_token_indices=True` — this repository's own frozen setting,
`src/kvcot/generation/policies.py:34`):

```python
similarity_cos_analysis = cal_similarity(key_states, ...)             # UNSLICED, full seq_len
attn_weights_sum_analysis = softmax(attn_weights, dim=-1).mean(dim=-2)  # UNSLICED, full seq_len, no window restriction
attn_cache_analysis = max_pool1d(attn_weights_sum_analysis, ...)
final_score_analysis = attn_cache_analysis * mix_lambda - similarity_cos_analysis * (1 - mix_lambda)
...
self.kept_final_scores.append(kept_final)   # derived from final_score_analysis, NOT final_score
```

`attn_weights_sum` (real decision) slices `attn_weights[:, :, -window_size:,
:-window_size]` — attention **from only the most recent `window_size`
queries, onto only the non-recent keys** — before softmax-normalizing and
averaging. `attn_weights_sum_analysis` (persisted bookkeeping) softmaxes
and averages the **entire, unsliced** attention matrix instead. These are
two different tensors, with different shapes and different normalization,
not the same quantity computed twice. **`kept_final_scores` (and
`kept_attention_scores`, `kept_similarity_scores`) — the only place R-KV
itself persists anything resembling a final score — is therefore not the
value that drove the real `topk` at line 82.** A capture design that
sourced `rkv_final_score` from `kv_cluster.kept_final_scores` (as a naive
reading of "R-KV's own internal joint score" in `docs/B0_5_DISCOVERY_PROTOCOL.md`
§3, or an under-specified reading of B0.5-R §9's schema field, might have
done) would silently log the wrong quantity. §9 below's recomputation must
replicate `r1_kv.py:49-77` exactly (the sliced/windowed formula), never
`r1_kv.py:91-116` (the analysis formula).

## 8. Capture-strategy audit (task §8) — the internal-hook claim does not hold

**Audited directly: a Python wrapper around a function call can observe
that function's inputs before the call and its return value / the target
object's mutated attributes after the call. It cannot read a local
variable (`final_score`, `indices`) at an arbitrary line inside the
function's body** without one of: editing the function's source
(`third_party/R-KV`, prohibited by task and by `CLAUDE.md`), attaching a
`sys.settrace`/frame-inspection hook at that exact line (fragile,
version-coupled to bytecode/line numbers, not proposed), or a debugger
attach (not a production instrumentation mechanism). B0.5-R §6's claim of
a hook "right after `indices = final_score.topk(...)` and before the
gather" is **not** implementable as literally stated and is retracted.

### 8.1 Strategies evaluated

**A — Before/after wrapper plus exact recomputation (SELECTED).**
`R1KV.update_kv` is an **instance method** — `self.kv_cluster = R1KV(...)`
is constructed once per attention layer (`docs/UPSTREAM_AUDIT.md` H1-H2;
confirmed again directly in `r1_kv.py:8-36`'s `__init__`). This means the
wrapper can be attached **per instance** (e.g. `kv_cluster.update_kv =
types.MethodType(_wrapped_update_kv, kv_cluster)`, capturing the original
bound method in a closure), never a process-global class patch and never a
global `torch.topk`/`Tensor.topk` interception. The wrapper:

1. clones the pre-call `key_states`/`query_states`/`value_states` inputs
   (read-only; the clones are new tensors, never fed back);
2. calls the **original, unmodified** `update_kv` and keeps its real return
   value;
3. independently **recomputes** `attn_weights`, `attn_weights_sum`,
   `attn_cache`, `similarity_cos`, and `final_score` from the cloned
   pre-call inputs, replicating `r1_kv.py:49-77` **exactly** (the real,
   windowed formula — never the analysis formula, per §7) using the
   pinned `compute_attention_scores`/`cal_similarity` helpers imported from
   `rkv.compression`/`rkv.utils` themselves (reused, not reimplemented, so
   there is only one source of truth for the arithmetic);
4. recomputes `indices = final_score.topk(budget - window_size, dim=-1).indices`
   from the recomputed `final_score`;
5. **checks agreement** between the recomputed selection and the real,
   observed selection via two independent signals: (i) when
   `record_kept_token_indices=True` (frozen), `kv_cluster.kept_token_indices[-1]`
   is R-KV's own real, remapped, absolute-position ground truth for this
   event (confirmed unaltered by enabling the flag, `docs/UPSTREAM_AUDIT.md`
   H5); (ii) gathering the **original** (uncloned, real) `key_states` at
   the recomputed indices must reproduce the real returned
   `k_past_compress` bit-for-bit;
6. **invalidates** the event/layer/head for swap-candidate purposes on any
   mismatch — never falls back to the recomputed selection as ground
   truth, and never feeds the recomputation back into R-KV's own state or
   return value.

The recomputed `final_score` is used **only** as a covariate value
(`score_e`, `score_r` in the schema, §12) and as this internal
self-consistency check — never as the source of *which* tokens are
candidates/donors. That identity question is answered entirely from real,
observed R-KV behavior (§9).

**B — FaithKV replacement copy of `update_kv`.** Rejected as B0.5-R's task
brief already disfavors it: duplicating upstream's decision code creates an
ongoing parity burden against every future upstream change, for no benefit
Strategy A does not already provide (Strategy A reuses the pinned helper
functions directly, so there is no duplicated arithmetic to keep in sync).

**C — Editing `third_party/R-KV`.** Prohibited, unconditionally, by
`CLAUDE.md` and this task.

**D — Global interception of `torch.topk`/`Tensor.topk`.** Rejected. A
global patch cannot be scoped to only this one call site without
inspecting the call stack (fragile), risks interacting with any other
`topk` call in the same forward pass (attention top-k sampling, if any is
ever added; none exists in the frozen pipeline today, but a global patch
is a standing risk regardless), and is not exception-safe across a
multi-layer, multi-call generation loop in the way a per-instance method
wrap is. Strategy A already solves the identical problem (observing a
selection made by an instance method) without this risk.

### 8.2 Exact parity assertions (task requirement)

- Input shapes, dtype, device match between the clone and the real call
  (assert equality on capture, not assumed).
- `budget`, `window_size` used in recomputation are read from the **same**
  `kv_cluster` instance being wrapped (`kv_cluster.budget`,
  `kv_cluster.window_size`), never a second, independently-configured copy.
- Recomputed selected pre-storage positions (top-k indices, mapped through
  the wrapper's own absolute-position bookkeeping) equal the observed
  selected positions from `kept_token_indices[-1]`, per-head, set equality
  (not just count equality).
- Gathering the real, original `key_states`/`value_states` at the
  recomputed indices reproduces the real returned compressed tensors
  bit-for-bit (`torch.equal`, not an approximate/tolerance comparison,
  since both sides are the identical deterministic computation on the same
  inputs — a real mismatch here means the recomputation formula itself is
  wrong, not a numerical-precision artifact).
- The wrapper mutates **no** input tensor, **no** `kv_cluster` configuration
  attribute (`budget`, `window_size`, `mix_lambda`, `retain_ratio`,
  `retain_direction`), and **no** returned tensor — verified in B1A's CPU
  unit tests by comparing a hook-off run against a hook-on run on
  controlled synthetic tensors and asserting bit-exact equality of the
  returned compressed K/V in both cases.
- **Object identity is explicitly not required** for the cloned evidence
  tensors (clones are, by design, different storage from the originals);
  what is required is exact shape/dtype/device/value equality against the
  real source at the moment of capture.
- Any parity failure invalidates that event/layer/head pair — logged with
  a non-null `invalid_reason`, excluded from all statistics, never
  silently imputed or retried with a relaxed tolerance.

## 9. Candidate and donor eligibility (frozen)

For a compaction event `t`, layer `L`, KV head `h` (selected per §11):

1. `e` (evicted candidate) and `r` (retained donor) are from the **same**
   `t`, `L`, `h` — never cross-event, cross-layer, or cross-head.
2. Both are drawn from the **non-recent, score-compared pool**: the
   `kv_cache_len - window_size` candidates that entered `attn_weights_sum`/
   `similarity_cos`/`final_score`/`topk` at event `t` (`r1_kv.py:51-82`).
   The most-recent `window_size` slots (`k_cur`/`v_cur`, `r1_kv.py:176-177`)
   are **never** scored and **never** compete in `topk` — they are always
   kept verbatim by construction. **A protected recent-window slot is
   never used as `r`.**
3. `e` must be **actually evicted**: an absolute position present in the
   pre-call non-recent pool (from the wrapper's cloned pre-call tensor,
   §8) but **absent** from the real post-call `kept_token_indices[-1]` for
   `(L, h)` at event `t`.
4. `r` must be **actually selected**: an absolute position present in both
   the pre-call pool and `kept_token_indices[-1]` for `(L, h)` at event
   `t`, and not part of the always-kept recent window (rule 2).
5. Both identities are resolved through **exact absolute-position
   provenance** — this repository's own existing `kvcot.generation.provenance.LayerProvenance`
   ledger (`positions[h, j]` = absolute source position of KV-head `h`'s
   physical slot `j`, continuously maintained, `provenance.py:41-78`) plus
   upstream's own `kept_token_indices[-1]` at the event boundary — never a
   fresh, ad hoc position computation invented for this document.
6. `e`'s **pre-compaction storage position** is its index in the pre-call
   (cloned) non-recent tensor. `r`'s **post-compaction storage position**
   (`r_slot`, §6) is its index in the post-call compressed tensor for
   `(L, h)`, resolvable directly from `LayerProvenance.positions[h]` after
   the event (find the column `j` with `positions[h, j] == r`'s absolute
   position).
7. Candidate and donor can never refer to the same absolute token (true by
   construction — rules 3-4 place them in disjoint evicted/retained sets —
   restated here as an explicit invalidity check the implementation must
   assert, not merely assume).
8. Invalid or ambiguous mappings (a provenance lookup that fails, returns
   more than one slot, or disagrees with the §8.2 parity check) **fail the
   pair**: excluded from statistics, logged, counted in the attrition
   report — never silently dropped or imputed.
9. **No FullKV tensor is ever substituted for anything.** `e`'s captured
   K/V comes from the same live, real R-KV forward pass at event `t` — a
   strictly stronger guarantee than B0.5-R §6 provided (that document still
   needed to reject a shadow-FullKV *reconstruction*; this design never
   introduces a FullKV code path into the intervention at all).
10. No cross-layer or cross-head pair is ever constructed (rule 1).

### 9.1 Multiple donors per candidate

One evicted candidate may be paired with multiple retained donors at the
same event/layer/head, and vice versa. If the design retains 2 evicted
candidates and 2 retained donors per selected event/layer/head pair (§11),
all four cross-product swaps `(e1,r1), (e1,r2), (e2,r1), (e2,r2)` are
constructed and evaluated. **These four observations are nested within one
event and are never treated as four independent samples** — enforced by
the aggregation hierarchy (§13).

### 9.2 Deterministic selection and tie-breaking

`kept_token_indices`, as upstream actually records it, is the ground truth
for *which* absolute positions were selected — no tie-break is needed to
determine real selection, since upstream's own `topk` call already
resolved any ties one specific way and that result is read, not
re-derived. `torch.topk`'s own tie-break order is **not** relied upon as
stable for anything: the real, observed `kept_token_indices` output is
used as-is. Where this document's **own** analysis needs a deterministic
order over a set of eligible candidates/donors (e.g., "pick 2 evicted
candidates out of N eligible ones"), the tie-break is **ascending absolute
token position**, stated once here, matching the existing convention
`docs/B0_5_PROTOCOL_REPAIR.md` §9 already used for `rkv_final_score` ties.

## 10. Event, layer, and head sampling (frozen)

**[SUPERSEDED IN PART — see `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §4-§6
(2026-07-19).** This section's layer/head hash rule (an unrestricted
`SHA256(...) % num_hidden_layers` draw, independently per event) does
**not** guarantee one early-, middle-, and late-third layer per example —
three independent uniform draws over the full range can collide in the
same third with nonzero probability; B0.5-R2.1 §5 restricts each selected
event's draw to its own `[lo, hi)` third of the depth range to make the
guarantee real. Candidate/donor selection (§9.2 below) used a plain
ascending-position tie-break, which is a systematically edge-biased
sample, not a representative one; B0.5-R2.1 §6.2 replaces it with genuine
SHA-256-seeded `random.Random.sample` over each pool, using two
independent seed streams (`"b05r21_evicted"`/`"b05r21_donor"`). Event
selection itself (§2 of `docs/B0_5_DISCOVERY_PROTOCOL.md`,
reused/unchanged in substance here) is additionally given an exact,
frozen algorithm and UTF-8 serialization spec in B0.5-R2.1 §4.2-§4.3 that
was previously left as prose. Text below preserved as the historical
(under-specified) proposal.]**

Bounded design, reusing the original protocol's total-cost envelope
(12 × 3 × 2 × 2 = 144 planned branches, unchanged in count from
`docs/B0_5_DISCOVERY_PROTOCOL.md` §8 and `docs/B0_5_PROTOCOL_REPAIR.md` §12
gate 3's population target) but now **explicit about layer/head selection**,
which no prior version of this repository's documents froze:

| Quantity | Value | Rationale |
|---|---|---|
| Examples | 12 | Unchanged from B0.5/B0.5-R — reuses the already-justified cost envelope. |
| Eligible compaction events sampled per example | 3 | Unchanged (B0.5-R §12 gate 3). |
| **(Layer, KV-head) pairs per event** | **exactly 1, deterministically selected** | New: sampling all layers × all heads per event would multiply cost by the model's full layer count × KV-head count (tens to low hundreds of times), far outside the 4-GPU-hour ceiling. Restricting to one deterministically-chosen pair per event keeps the total branch count identical to the original protocol's budget while still distributing coverage across model depth (below). |
| Evicted candidates per selected (event, layer, head) | 2 | Unchanged from the original protocol's `evicted_per_event=2`. |
| Retained donors per selected (event, layer, head) | 2 | Unchanged (`control_per_event=2`). |
| Cross-product swaps per event | 4 | `2 evicted × 2 donors` (§9.1). |
| **Total planned branches** | **12 × 3 × 4 = 144** | Identical total to the pre-repair protocol's `n_examples × events_per_example × (evicted_per_event × control_per_event)` — the swap design changes *what* each branch is, not the sample-size envelope. |

**Deterministic (layer, KV-head) selection rule**, fixed before any GPU
result exists: for a given `(example_id, event_id)`,

```
h_bytes  = SHA256(f"{example_id}|{event_id}|{model_revision}|{rkv_revision}|layer")
layer_index   = int.from_bytes(h_bytes[:8], "big") % num_hidden_layers
k_bytes  = SHA256(f"{example_id}|{event_id}|{model_revision}|{rkv_revision}|head")
kv_head_index = int.from_bytes(k_bytes[:8], "big") % num_key_value_heads
```

— the same SHA-256-derived-seed family this repository already uses
(`kvcot.utils.seeding.derive_seed`, `docs/B0_5_DISCOVERY_PROTOCOL.md` §2's
literal-suffix-extension convention), with disambiguating literal suffixes
`"layer"`/`"head"` so the two draws are independent of each other and of
every other sub-seed already in use. Because the hash input spans
`example_id` and `event_id`, selected layers/heads are **not** silently
concentrated on one layer across the whole 12-example pilot — a uniform
hash distributes draws across `[0, num_hidden_layers)` and
`[0, num_key_value_heads)` independently per event. `model_revision`/`rkv_revision`
are included so the draw is reproducible and re-derivable from the
pinned commit identity, not from an arbitrary constant.

**144 pair branches are never treated as 144 independent examples** — see
§13.

## 11. Two-pass capture plan (frozen)

**[NOTE — see `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §7 (2026-07-19).** The
two-pass structure below is unaffected and not redesigned. B0.5-R2.1 fixes
where the eligibility rule's future-token count is measured: the "≥48
future reference tokens" clause a few lines below is corrected to "≥49" —
one bridge token plus 48 scored tokens, `docs/B0_5_R2_1_FINAL_PROTOCOL.md`
§3.2 — and clarifies that the bridge-token/scored-window mechanics (§14
below) attach to a third step, branch evaluation, strictly after Pass 2
completes for a given target — never inside Pass 1 or Pass 2
themselves.]**

**The protocol cannot truthfully claim to capture only preselected events
during a single natural-generation pass.** Event eligibility (not first,
not last, ≥48 future reference tokens, no cap-hit within the horizon)
depends on the **complete** natural trajectory length, which is only known
after natural generation finishes. Capturing "only selected events" during
that same run would require deciding selection before the run completes —
impossible — or retaining full pre-compaction tensors for **every**
compaction event just in case (unbounded memory, exactly what this
document must not silently assume away).

**Pass 1 — natural R-KV generation, no heavy capture:**

- Run the real R-KV natural generation (`record_kept_token_indices=True`,
  already frozen), producing the frozen reference token IDs.
- Record compaction event positions and per-layer/per-head surviving
  index sets from `kept_token_indices` growth alone (cheap: integer
  bookkeeping already produced by upstream, no additional tensor capture).
- From this bookkeeping alone (no GPU-heavy work), deterministically
  compute: eligible event IDs (§9 of `docs/B0_5_DISCOVERY_PROTOCOL.md`,
  unchanged), the one (layer, KV-head) pair per eligible event (§10), and
  the 2 evicted candidates / 2 retained donors at that pair (§9.2's
  deterministic tie-break over the real evicted/retained sets at that
  event).
- **Do not retain full pre-compaction K/V tensors for every event during
  Pass 1** — only the cheap index bookkeeping above.

**Pass 2 — instrumented, token-identical R-KV replay:**

- Reset all mutable state (`kvcot.generation.state.reset_patched_state`,
  already exists and is reused unchanged).
- Replay the **exact** Pass-1 token IDs, teacher-forced, using this
  repository's existing call-shape discipline (one N-token prefill call,
  then one single-token call per decoded token, `docs/REPLAY_DESIGN.md`
  §2 — never a bulk prefill of a truncated prefix, for the same
  call-shape-determines-`self.length`-trajectory reason already
  documented there).
- At every forward call, require **event-by-event parity** against Pass 1:
  same compaction-event step positions, same per-layer/per-head
  `kept_token_indices` surviving-set — this repository's replay engine
  already performs exactly this class of cross-check
  (`kvcot.generation.replay._sync_layer_after_call`, reused, not
  reinvented). **Any mismatch invalidates the whole example immediately**
  — trajectory divergence, event-count divergence, or selection-set
  divergence are all hard failures, never partially trusted.
- Attach the §8 capture wrapper **only** at the preselected
  `(event_id, layer_index, kv_head_index)` targets from Pass 1 — capturing
  the pre-call K/V for the 2 preselected evicted candidates at each such
  target, plus running the §8 recomputation/parity check there. Not at
  every event, not at every layer, not at every head.
- Pass 2 may stop as soon as the **last** preselected target's capture is
  complete (the same early-stopping principle `replay_and_snapshot`
  already applies for snapshot fractions, `replay.py:230-241`) — it does
  not need to replay all the way to the end of the trace if every
  preselected event occurs earlier, but it is still a **second,
  real, separate decode pass** up to that point, not a free side effect of
  Pass 1.
- **Never substitute a FullKV replay for Pass 2.** Pass 2 must be the same
  R-KV policy, replayed token-identically (§9 rule 9's guarantee depends on
  this).

**Pass 2 is a real, separate, costed component of the runtime model — see
§12.**

## 12. Corrected cost model

B0.5-R §15's cost model assumed shadow-FullKV prefix reconstructions as
its heaviest per-event line item and never included a second full replay
pass, because its intervention design (per-block add-back) did not require
knowing event eligibility before capturing. This repair's two-pass design
(§11) does require it, so the cost model is corrected to include it
explicitly rather than silently absorbing it into the existing "safety
factor."

| Component | Distinguished because |
|---|---|
| Model load + tokenizer/dataset setup | fixed, one-time, per process — unchanged. |
| Pass 1: natural R-KV generation (prefill + decode) | unchanged in kind from prior cost models; still pays the periodic `update_kv` cost. |
| **Pass 2: instrumented, token-identical R-KV replay** | **new, explicit line item** this repair adds: a second decode pass up to the last preselected event per example, not a free side effect of Pass 1 — see §11. Previously entirely absent from B0.5-R §15's cost table. |
| Score recomputation (§8) | cheap, CPU/GPU tensor ops over one layer/head's data per preselected target — bounded, not a cache-length-scaling cost. |
| Targeted K/V capture (§8, §9) | one clone of one head's pre-call K/V per preselected evicted candidate — `O(kv_cache_len_at_event × head_dim)` for that one head, not a full-layer or full-model clone. |
| Post-compaction snapshot (needed for the swap's baseline) | this repository's existing `capture_snapshot`/full per-layer clone (`replay.py:263-304`) — reused, not reinvented, but not free: a full per-layer K/V clone, at each event actually used for a swap branch. |
| **Complete-cache branch cloning/restoration** | **new, explicit line item**: each of the 144 swap branches needs its own independent, restorable cache clone (`restore_snapshot`, already deep-clones per §6.1 hard-gate discipline) so branches never share mutable storage — this is the same per-branch clone cost this repository's existing snapshot/branch machinery already pays elsewhere, now counted explicitly for 144 branches rather than assumed negligible. |
| Fixed-shape swap construction (§6) | `O(head_dim)` per branch — one indexed tensor write, negligible relative to the clone above. |
| 48-token teacher-forced replay per branch | unchanged in kind from prior cost models (`horizon=48`, `configs/lock.yaml`). |
| CPU transfer / artifact serialization | I/O-bound, not GPU-bound — unchanged. |
| Grading (MATH-500 verifier) | CPU-bound — unchanged (and itself a B1A prerequisite, §16). |
| Retries/failures | covered by the existing ×3 safety factor — unchanged. |

**No specific GPU-hour number is asserted here.** The prior conservative/optimistic
planning range (`docs/B0_5_FEASIBILITY_AUDIT.md` §5, 2.4-3.4 GPU-hours) was
built on a cost model that did not include Pass 2 or per-branch full-cache
cloning at their corrected weight, so that number is **not** reused,
re-derived, or re-asserted as a new estimate here — doing so from this
desk audit alone, without a real measurement, would repeat exactly the
mistake B2A (§16, unchanged) exists to prevent. B2A must measure every
line item above directly.

## 13. Aggregation hierarchy (frozen)

1. **Pair-level**: `swap_gain(e, r)` for each of the ≤4 cross-product
   swaps at a selected (event, layer, head) — §9.1.
2. **Event-level**: for the one selected (layer, head) pair at that event,
   report (a) whether **any** of its ≤4 valid pairs has
   `swap_gain > noise_floor` (Boolean, "this event shows a reversal") and
   (b) the maximum `swap_gain` among its valid pairs (descriptive).
   Invalid pairs (§9 rule 8) are excluded from both, and their count is
   reported separately.
3. **Example-level**: fraction of that example's eligible, valid events
   showing a reversal (from 2), plus that example's maximum event-level
   `swap_gain`. **Exactly one number per example for the primary discovery
   statistic** — never pool the ≤3 events' pair branches as if they were
   3 (or 12) independent example-level observations.
4. **12-example discovery summary**: count/fraction of examples with
   ≥1 reversal-event (from 3), reported alongside the distribution of
   per-example maximum `swap_gain` and a by-example clustered/bootstrap
   uncertainty statement (this repository's existing discipline for not
   pretending within-example or within-event observations are iid,
   `docs/B0_5_DISCOVERY_PROTOCOL.md` §7.3, reused unchanged in spirit).

Pairs from the same event are nested and are never reported as if `n=144`
independent samples at any stage above.

## 14. Corrected pairwise causal estimand and claim boundary

**[SUPERSEDED — see `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §3 (2026-07-19).**
The estimand below has an off-by-one timing defect: it treats the first
post-event reference token as already scoreable, but that token's logits
are produced by the very forward call during which the swap fires and
therefore cannot be affected by the swap. B0.5-R2.1 §3.1 corrects this: one
unscored "bridge" token (the real, already-generated next token) must be
fed identically into both branches first; the 48-token scored window
starts one token later than stated below. The "under greedy decoding"
phrase is also corrected — no decoding decision is ever made during branch
evaluation; this is teacher-forced NLL evaluation of fixed reference
tokens, restated precisely in B0.5-R2.1 §3.3. The sign convention
(`swap_gain > 0` = pairwise ranking reversal, not standalone utility) is
unaffected. Text below preserved as the historical (off-by-one) proposal.]**

Replacing B0.5-R §7.3's `u_state` (defined over the now-superseded
add-back/ablation pair) with the swap estimand:

```
swap_gain(e, r) = mean_NLL(reference | original_RKV_cache)
                  - mean_NLL(reference | cache_with_e_replacing_r)
```

evaluated over the `horizon=48`-token teacher-forced reference
continuation (unchanged number, `configs/lock.yaml` `probes.max_new_tokens`),
under greedy decoding, starting immediately after the swap at event `t`.

- `swap_gain > 0`: replacing R-KV's selected donor `r` with the rejected
  candidate `e` **improves** the reference continuation's likelihood — a
  **fixed-memory pairwise selection-regret / ranking-reversal**, at this
  specific (event, layer, head).
- `swap_gain < 0`: R-KV's real selection performs better for this
  counterfactual than the substitution would have.
- `swap_gain = 0`: no measured difference at the working precision (never
  counted as a reversal, §15).

**This is not:** the absolute standalone causal utility of `e`; the
isolated effect of adding capacity; a pure ablation of `r`; proof that `e`
should always be retained; a new compression method; an
accuracy-preserving claim; or evidence about internal cognition/faithfulness
(`CLAUDE.md` §1's claim-boundary discipline applies with equal force here,
even though this is a different diagnostic from the frozen primary
pipeline). **It is:** a fixed-memory pairwise selection-regret / ranking-
reversal diagnostic, at exactly the memory cost R-KV's own deployed policy
already pays, nothing more and nothing less.

## 15. Repaired schema (supersedes `docs/B0_5_PROTOCOL_REPAIR.md` §9)

**[SUPERSEDED — see `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §10 (2026-07-19),
the current schema.** Every field below is retained there; four new fields
are added for the §14 timing correction (`bridge_token_absolute_position`,
`first_scored_absolute_position`, `first_affected_logit_absolute_position`,
`reference_horizon_sha256`), plus per-signal difference fields for the
final Gate 10's mandatory-signal set and an `is_noop_control` flag for the
new §9.C control. Text below preserved as the historical (pre-timing-fix)
schema.]**

One JSON-lines record per swap-branch (pair):

```json
{
  "schema_version": "b0_5_r2.v1",
  "example_id": "string",
  "model_revision": "string",
  "rkv_revision": "45eaa7d69d20b7388321f077020a610d9afb65bd",
  "compaction_event_id": "int",
  "layer_index": "int",
  "kv_head_index": "int",
  "evicted_absolute_token_position": "int",
  "evicted_pre_storage_position": "int",
  "retained_absolute_token_position": "int",
  "retained_pre_storage_position": "int",
  "retained_post_storage_position": "int",
  "score_e": "float (recomputed, r1_kv.py:49-77 formula, never the analysis formula)",
  "score_r": "float (same formula)",
  "score_margin_e_minus_r": "float",
  "parity_check_passed": "bool",
  "parity_failure_reason": "string | null",
  "swap_gain": "float",
  "baseline_per_token_nll": ["float", "..."],
  "swapped_per_token_nll": ["float", "..."],
  "net_physical_bytes_changed": "int (must equal 0)",
  "recency_position_e": "int",
  "recency_position_r": "int",
  "key_norm_e": "float",
  "key_norm_r": "float",
  "value_norm_e": "float",
  "value_norm_r": "float",
  "entropy_at_capture": "float | null",
  "logit_margin_at_capture": "float | null",
  "oracle_non_deployable": true,
  "cap_hit_flag": "bool",
  "valid_flag": "bool",
  "invalid_reason": "string | null"
}
```

Identity for matching/deduplication is always
`(compaction_event_id, layer_index, kv_head_index, evicted_absolute_token_position,
retained_absolute_token_position)`, never decoded text. `net_physical_bytes_changed`
is asserted `== 0` for every valid record — any nonzero value is itself a
hard bug signal, not a reportable finding.

## 16. Controls and decision rule (repairs gate 10)

**[SUPERSEDED — see `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §9 (2026-07-19),
the current, final Gate 10.** The rule below (part (b)) pools Spearman
correlation across all examples' pairs into one statistic — the task
brief's own adversarial review flags this as the wrong-granularity
decision statistic, since it can show a strong apparent association driven
entirely by between-example variance even with no real within-example
association, or the reverse. B0.5-R2.1 §9.B replaces it with a
per-example-nested statistic (Spearman rho computed separately within each
example, absolute value, median across examples), an explicit 8-evaluable-
example floor per mandatory signal (below which the whole gate is NOT
ADJUDICABLE, not silently passed or failed), and a mandatory no-op control
(§9.C) that must pass before either (a) or (b) is interpreted at all. The
possible-outcomes set is also expanded from a two-way pass/fail to three
outcomes: DISCOVERY-SUPPORTING, NOT DISCOVERY-SUPPORTING, NOT ADJUDICABLE.
Text below preserved as the historical (pooled, two-outcome) proposal.]**

Controls (reused from B0.5-R §11, restated per-pair rather than per-block):
`score_margin_e_minus_r`, recency-position difference, key-norm difference,
value-norm difference, entropy/logit-margin at the moment each token was
**originally generated** (pre-treatment, never a post-swap value used as a
predictor — restated hard rule, unchanged from B0.5-R §11 point 2), layer
depth, KV-head identity, event position.

**Gate 10 is repaired.** B0.5-R's rule (`|Spearman ρ| < 0.30` AND
`IQR(u) > 0.01`) is insufficient here for the same reason the task brief
flags: it does not require any **actual positive reversal** to exist — a
population of entirely negative or zero `swap_gain` values with modest
spread could still pass it, which would not motivate any discovery claim.
**Frozen discovery-support rule:**

> Discovery-supporting iff **both**:
> (a) at least **4 of the 12 examples** (≥1/3) show at least one valid
>     pair with `swap_gain > 0.01` nats (the same fixed positive noise
>     floor B0.5-R already used, reused here — not re-derived from any
>     result, since none exists — justified as a float32 mean-per-token-NLL
>     noise floor over a 48-token horizon, the same magnitude and rationale
>     as before); **and**
> (b) `|Spearman ρ(swap_gain, score_margin_e_minus_r)| < 0.30` over all
>     valid pairs (pooled across examples, clustered-by-example spread
>     reported alongside) — i.e., R-KV's own score margin does not already
>     fully explain which swaps improve likelihood.

Exact-zero `swap_gain` never counts toward (a). Invalid pairs (§9 rule 8)
are excluded from both (a) and (b) and their count is reported separately,
never imputed. Uncertainty for (a) is reported as the by-example proportion
with at least one qualifying reversal, with a percentile-bootstrap interval
over examples (same 10,000-resample/fixed-seed convention `CLAUDE.md` §8
already uses for `Delta_EAS`, reused here for consistency, not
re-derived). **This is a bounded discovery criterion for a 12-example
pilot, not a confirmatory significance test** — stated explicitly, per the
task's requirement, and matching this repository's existing discipline
for every other small-`n` pilot gate in this document chain.

## 17. Repaired B1A scope

**B1A-1 — Model-family dispatch.** Unchanged from `docs/B0_5_PROTOCOL_REPAIR.md`
§13: explicit `AutoConfig.model_type`-based dispatch to `replace_llama`/`replace_qwen2`/`replace_qwen3`,
hard failure on unknown architectures, generalized per-layer state reset,
FullKV/R-KV construction-parity tests for the Llama path, explicit
no-CPU-offload assertion. No GPU inference.

**B1A-2 — MATH-500 verification.** Unchanged from `docs/B0_5_PROTOCOL_REPAIR.md`
§14: symbolic-equivalence verifier covering fractions, algebraic
equivalence, signs, tuples, sets, intervals; malformed-answer and
verifier-timeout handling; no unverifiable-equals-correct fallback; CPU
unit tests only.

**B1A-3 — Pairwise provenance schema.** **Repaired** from B0.5-R §9 to this
document's §15: candidate/donor absolute and physical-slot identity,
per-event/layer/head score provenance (recomputed via the §8.1 Strategy-A
formula, never the analysis formula), deterministic serialization and
validation, `net_physical_bytes_changed == 0` assertion. No model
inference.

**B1A-4 — Read-only capture wrapper.** **Repaired** from B0.5-R §6/§14 to
this document's §8: a per-instance `R1KV.update_kv` before/after wrapper
(never a class-level or global patch), exact recomputation of the real,
windowed `final_score` formula (`r1_kv.py:49-77`, never the analysis
formula at `r1_kv.py:91-116`), the §8.2 parity assertions, pre-compaction
K/V capture for the preselected evicted candidates only, no upstream
modification, controlled CPU tensor tests (synthetic small tensors,
`torch` available without a GPU) verifying bit-exact hook-off/hook-on
agreement. B1B remains blocked until these parity tests pass.

B1A performs no GPU inference and no Vast.ai work, unchanged.

## 18. B2A hard stop (unchanged in kind, corrected in content)

B2A remains mandatory before any 12-example pilot, per
`docs/B0_5_PROTOCOL_REPAIR.md` §16, with its measurement list corrected to
match §12 above: it must separately measure Pass-1 generation rate, **Pass-2
replay rate** (new), score-recomputation cost, targeted-capture cost,
per-branch full-cache clone/restore cost (new, explicit), single-swap
construction cost, 48-token branch-replay cost, peak allocated/reserved
VRAM, and the all-parameters-on-CUDA assertion — then recompute a projected
complete-pilot runtime from these measured rates and **automatically stop**
if it exceeds 4.00 GPU-hours or 22 GiB peak allocated memory (gates 8-9,
`docs/B0_5_PROTOCOL_REPAIR.md` §12, unchanged numeric values, reused not
re-derived).

## 19. Remaining uncertainties

- B2A has not been run; no GPU-hour number in this document (§12
  deliberately gives no figure) is a measurement.
- Whether a future `transformers` version shift changes `DynamicLayer`'s
  internal representation (and therefore this repair's §6 indexed-write
  mechanism) is unknown until B1A's CPU tests run against the pinned
  `<4.56` ceiling; the `<4.56` ceiling itself is unchanged from
  `docs/UPSTREAM_AUDIT.md` §3.5/H6.
- The exact wall-clock weight of Pass 2 relative to Pass 1 is unmeasured;
  §12 states it as a real, separate component but does not estimate its
  magnitude.
- `CLAUDE.md` §4's model freeze (`DeepSeek-R1-Distill-Qwen-1.5B` only)
  still requires its own separate, dated amendment before any GPU run of a
  later phase using Candidate A — unchanged blocker, not resolved here.
- The "32-token occlusion methodology" attribution remains unresolved
  (`docs/B0_5_SEARCH_LOG.md` §4) — not relied upon for this document's
  verdict.

## 20. Adversarial self-review

1. *How can one dense tensor have different sequence lengths per KV
   head?* — It cannot; confirmed directly from `transformers.DynamicLayer`
   (§4) and R-KV's own uniform-count `topk` call (§4) — this is exactly
   why §5 supersedes B0.5-R's Design A/C.
2. *Does your intervention change tensor shape?* — No (§6): the swap is a
   fixed-shape indexed write.
3. *Does it change total physical cache bytes?* — No, exactly zero net
   change, for every valid swap (§6, §15's `net_physical_bytes_changed == 0`
   assertion).
4. *Does it modify another KV head?* — No; only `(L, h, r_slot)` is
   written; every other head/slot/layer is untouched by construction
   (ordinary indexed assignment, §6).
5. *Is the donor from R-KV's genuinely score-selected pool?* — Yes,
   verified against real `kept_token_indices` (§9 rules 2, 4), never the
   protected recent window.
6. *Are candidate and donor from the same selection decision?* — Yes, same
   `(compaction_event_id, layer_index, kv_head_index)` by eligibility rule
   1 (§9).
7. *Does the candidate key retain its correct rotary positional
   encoding?* — Yes, confirmed from installed `transformers` source: RoPE
   is applied at cache-write time, baked into the vector, independent of
   physical slot index (§6.1).
8. *Does `DynamicCache` store additional per-slot position metadata that
   must change?* — No, confirmed directly from source: `DynamicLayer`
   holds only `.keys`/`.values` (§6.1).
9. *Can a wrapper really access `final_score` and `indices` inside
   `update_kv`?* — No, not as internal locals; retracted and repaired to
   an implementable before/after wrapper plus independent recomputation
   (§8).
10. *How does recomputed top-k prove parity with upstream's actual
    selection?* — Two independent checks: agreement with real
    `kept_token_indices[-1]`, and bit-exact reproduction of the real
    returned compressed K/V when gathering at the recomputed indices
    (§8.2). Any disagreement invalidates the pair.
11. *When does the system learn which event to capture?* — Only after
    Pass 1's natural generation completes (eligibility depends on total
    trace length) — hence the mandatory two-pass design (§11).
12. *Did the cost model include the second replay pass?* — Yes, added as
    an explicit, previously-absent line item (§12), alongside per-branch
    full-cache clone/restore cost.
13. *Are four pair swaps from one event incorrectly treated as four
    independent examples?* — No; explicit pair→event→example→12-example
    aggregation hierarchy (§13), with nesting stated at every level.
14. *Does positive swap gain mean standalone utility, or only pairwise
    replacement benefit?* — Only pairwise replacement benefit at fixed
    memory (§14) — explicitly bounded, never generalized.
15. *Does the discovery rule require actual positive ranking reversals?*
    — Yes: repaired gate 10 (§16) requires a predeclared fraction of
    examples to exhibit at least one pair exceeding a positive noise
    floor, not merely low correlation plus nonzero spread.

No answer above is incomplete in a way that would require withholding a
READY verdict; every open item is either fully repaired in this document
or explicitly scoped to B1A/B1B/B2A (§17-§19), never silently assumed.

## 21. Final B0.5-R2 verdict

**[SUPERSEDED — see `docs/B0_5_R2_1_FINAL_PROTOCOL.md` §14 for the
current authorized verdict. This section's verdict was carried forward
from a timing definition (§14), sampling specification (§9.2, §10), and
Gate 10 (§16) since found to contain an off-by-one defect, an
under-specified (edge-biased/unrestricted-hash) sampling rule, and a
pooled rather than per-example-nested association test, respectively; all
three are repaired in `docs/B0_5_R2_1_FINAL_PROTOCOL.md`. Preserved here as
the historical record, not as an active authorization.]**

**B0.5-R2 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION**

This authorizes only B1A: CPU-side prerequisite implementation (MATH-500
verifier, architecture-aware R-KV dispatch, the repaired pairwise
provenance schema, the repaired per-instance read-only capture wrapper
with exact score recomputation and parity assertions, CPU unit/integration
tests). It does **not** authorize B1B, model inference, a GPU, Vast.ai, the
discovery pilot, or any method claim. `CLAUDE.md` §4's model-freeze
amendment remains required, separately, before any GPU run of a later
phase, and is not granted by this document.

## 22. Cross-reference

Superseded passages, and their exact replacements, are marked inline in
`docs/B0_5_PROTOCOL_REPAIR.md` (§7, §8, §9, §12 gate 10, §14) with
`**[SUPERSEDED — see docs/B0_5_R2_DENSE_CACHE_REPAIR.md §N]**` banners; no
historical text is deleted. `docs/b0_5_decision.json` retains every prior
field for provenance and adds a `b0_5_r2_verdict`/`superseded_by` block
pointing here.
