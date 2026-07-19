# B0.5-R2.1 — Final timing, sampling and signal-control correction

Phase B0.5-R2.1 artifact (2026-07-19). Branch
`research/b0-5-r2-dense-cache-repair`, expected and confirmed HEAD at
session start `9d04ecd7268656894815fedb7d080f0d27c7fad3` ("Repair
dense-cache intervention and capture strategy", the B0.5-R2 commit).
Roles applied: strict senior ML-systems engineer; causal-inference
reviewer; adversarial research reviewer. **Documentation and protocol
correction only.** No GPU used, no model inference run, no Vast.ai
accessed, no model weights or datasets downloaded, no file under `src/`,
`tests/`, `configs/`, `results/`, or `third_party/` modified.

This document is the final B0.5 protocol correction: it fixes an
off-by-one timing defect in how the intervention's effect window was
defined relative to R-KV's own forward-pass execution order, freezes the
exact deterministic algorithms for event/layer/head/candidate/donor
selection (previously under-specified as "a deterministic tie-break" or
left as prose), and repairs Gate 10 to require per-example-nested
association testing with an explicit adjudicability floor and a mandatory
no-op control. It does **not** reopen or redesign the fixed-shape
within-head swap itself (`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §6) — that
design remains exactly as repaired: same event, layer and KV head;
evicted candidate `e` replaces retained donor `r`; K and V replaced at
`r`'s existing physical slot; cache shape unchanged; net physical bytes
changed = 0. It also does not reopen B0's method-pivot verdict or B0.5's
research question.

## 1. Executive verdict

**B0.5-R2.1 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION.**

B0.5-R2's READY verdict does **not** survive unmodified: its branch-timing
definition contained an off-by-one error (§3 below), its event/layer/head/
candidate/donor sampling was specified with a tie-break rule rather than a
true randomized, reproducible sampling algorithm (§4-§6 below), and its
Gate 10 (`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16) pooled Spearman
correlation across examples rather than nesting it per-example and did not
define an adjudicability floor or a no-op sanity control (§9 below). All
four are repaired here into an exact, implementable, no-longer-ambiguous
specification. The verdict is carried forward only in this repaired form,
scoped strictly to B1A (CPU-side prerequisite implementation) — it does
**not** authorize B1B, a GPU, model inference, Vast.ai, the discovery
pilot, or any method claim.

## 2. Git and repository state verified this session

```
git status --short        -> (clean)
git branch --show-current -> research/b0-5-r2-dense-cache-repair
git rev-parse HEAD        -> 9d04ecd7268656894815fedb7d080f0d27c7fad3
git fetch origin          -> (no new refs affecting this branch)
git diff --check          -> (clean, no whitespace errors)
```

HEAD matches the task's expected commit exactly before any edit in this
session. Nothing was reset, rebased, or amended to reach this state.

## 3. Fixing the off-by-one timing defect

### 3.1 What was wrong

`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §14 defined `swap_gain` as evaluated
"starting immediately after the swap at event `t`," and
`docs/B0_5_DISCOVERY_PROTOCOL.md` §10 (an earlier, still-cross-referenced
document) stated "the first scored position's next-token distribution is
conditioned on the modified cache" — i.e. both treated the very first
post-event logits as already affected by the intervention. This is false,
verified against R-KV's own pinned execution order
(`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §4, unchanged and reused here, not
re-derived):

1. The forward call that consumes token `x_t` (call it `FC(t)`) is the
   call **during which** compaction fires — the `topk`/`gather` that
   produces the post-event cache runs inside `FC(t)`.
2. `FC(t)` **already produces its own output logits** — the distribution
   over the token at position `t+1` — as an ordinary side effect of a
   causal-LM forward call over input `x_t`. Those logits are computed and
   returned before the swap is ever conceptually applied to the resulting
   cache: the swap is a **post-hoc modification of `FC(t)`'s cache
   output**, never a change to `FC(t)`'s own already-computed logits.
3. Therefore **swapping the post-event cache cannot change the logits
   that predict `x_{t+1}`.** Scoring `x_{t+1}` as if it were sensitive to
   the swap (as the superseded language above did) attributes zero real
   effect to a nonzero measured quantity — a real off-by-one, not a
   rounding nicety.
4. The real R-KV run's own next token, `x_{t+1}`, is a **given** — the
   fixed, already-generated reference token, identical on both branches.
   It must be fed into the cache (baseline and swapped, identically) as
   one **bridge** forward call, `FC(t+1)`, so that both branches' caches
   are advanced to the same absolute position before anything is scored.
   `FC(t+1)`'s *input* is unscored; what matters is that it is the first
   forward call whose **input state** (the KV cache it reads from) has
   actually diverged between the baseline and swapped branches.
5. `FC(t+1)`'s **output** logits — the distribution over the token at
   position `t+2` — are computed by reading from the now-diverged cache
   (baseline vs. swapped). **These are the first logits that can actually
   differ between the two branches.** `x_{t+2}` is therefore the first
   token whose teacher-forced NLL is a real measurement of the swap's
   effect.
6. From `x_{t+2}` onward, score exactly 48 further real, fixed reference
   tokens (`x_{t+2}` through `x_{t+49}`), one teacher-forced step at a
   time, identically on both branches.

### 3.2 Frozen constants

```
bridge_tokens                       = 1
scored_horizon                      = 48
minimum_future_tokens_after_event   = 49
```

`minimum_future_tokens_after_event = 49` is `bridge_tokens + scored_horizon`
— the count of real, already-generated tokens that must exist strictly
after `x_t` (i.e. `x_{t+1}` through `x_{t+49}`) for an event to be
eligible at all. **Events with fewer than 49 real future tokens are
ineligible.** The scored window is never padded (with a synthetic token)
and never shortened (scoring fewer than 48 tokens) — an event that cannot
support the full window is dropped at eligibility time, not truncated at
scoring time. This supersedes every prior document's looser "at least
`horizon=48` tokens remain" eligibility language
(`docs/B0_5_DISCOVERY_PROTOCOL.md` §2 rule 3,
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §11's implicit reuse of it) with an
exact, one-token-larger threshold that accounts for the bridge token.

Because the horizon is never truncated under this rule, the older
"per-token NLL series is truncated at EOS... marked `cap_hit_flag`-
equivalent" language (`docs/B0_5_PROTOCOL_REPAIR.md` §10, itself already
marked historical) does not apply to scored branches under R2.1: an event
that would require such truncation is simply ineligible before selection,
never truncated after selection.

### 3.3 Corrected estimand

```
swap_gain(e, r) = mean_{k=1..48} NLL(x_{t+1+k} | original_RKV_cache, bridge = x_{t+1})
                - mean_{k=1..48} NLL(x_{t+1+k} | swapped_cache,       bridge = x_{t+1})
```

evaluated by feeding `x_{t+1}` identically into both branches as one
unscored bridge step, then feeding `x_{t+2}, x_{t+3}, ..., x_{t+49}` one
at a time, teacher-forced, identically on both branches, reading each
step's `-log P(next real reference token)` from that branch's own
evolving cache. **This is teacher-forced NLL evaluation, not greedy
decoding** — no argmax or sampling decision is ever made or consulted
during branch evaluation; the reference tokens are always the real,
already-generated tokens from the natural R-KV run. This corrects
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §14's "under greedy decoding"
phrasing, which was ambiguous at best (nothing is actually decoded during
scoring) and wrong at worst (it never meant to imply resampling, but the
words said "decoding"). The sign convention is otherwise unchanged from
§14: `swap_gain > 0` means replacing donor `r` with rejected candidate `e`
**reduces** mean NLL over the scored window — a pairwise, fixed-memory
ranking reversal at this specific `(event, layer, head)`. It remains not
standalone utility, not a method, and not an accuracy claim.

### 3.4 New schema fields

Four fields, as required, added to the per-swap-branch record (§10 below
gives the complete repaired schema):

| Field | Meaning |
|---|---|
| `bridge_token_absolute_position` | Absolute position of the one unscored bridge token, `t+1`. |
| `first_scored_absolute_position` | Absolute position of the first *scored* reference token, `t+2`. |
| `reference_horizon_sha256` | `sha256(",".join(str(tok_id) for tok_id in [x_{t+2}, ..., x_{t+49}]).encode("utf-8")).hexdigest()` — a hash of the exact 48 scored reference token IDs, comma-separated decimal, UTF-8, no padding. Recorded so a later audit can confirm the scored window was the real, unmodified reference continuation without re-storing all 48 token IDs verbatim in every record. |
| `first_affected_logit_absolute_position` | Absolute position of the forward-call **input** token that is the first to read from the diverged (baseline-vs-swapped) cache — `t+1`, the bridge token's own position. Its *output* logits (predicting `t+2`) are the first that can differ between branches. |

**Validation invariant, asserted at write time, not merely descriptive:**
`first_scored_absolute_position == first_affected_logit_absolute_position + 1`,
always, by construction (§3.1 point 5). A record where this does not hold
is a hard bug signal, `valid_flag=false`, `invalid_reason` set — never
silently accepted.

### 3.5 What is unaffected

The fixed-shape within-head swap mechanism itself
(`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §6), rotary-encoding argument (§6.1),
capture-strategy wrapper (§8), candidate/donor eligibility rules apart
from their sampling algorithm (§9, repaired in §6 below), and the
aggregation hierarchy (§13) are all unaffected by this timing correction —
only *which tokens are scored, and from which forward call's output* was
wrong; *what is swapped* was already correct.

## 4. Frozen event selection

### 4.1 Eligibility (corrected)

A compaction event `t` is eligible iff all of B0.5-R2 §9's existing rules
(same-event/layer/head-only pairing, non-recent score-compared pools, not
the run's first or last event) **and** the §3.2 rule: at least
`minimum_future_tokens_after_event = 49` real, already-generated tokens
exist strictly after `x_t`.

### 4.2 Deterministic seed derivation (one canonical helper, reused for every draw in this document)

```python
import hashlib

def sha256_seed(*parts: object) -> int:
    canonical = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
```

`str(p)` for every part is Python's ordinary decimal integer-to-string
conversion for `int` parts (no leading zeros, no sign for non-negative
values, e.g. `str(7)` -> `"7"`) and the literal string itself for string
parts (`model_revision`, `rkv_revision`, dataset name, and the literal
disambiguating suffixes below) — never a re-encoding, hex form, or
zero-padded form. Parts are joined with a single ASCII pipe `"|"`
separator, matching the existing convention in
`kvcot.utils.seeding.derive_seed` and
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §10's layer/head hash. The joined
string is UTF-8 encoded, SHA-256 hashed, and the first 8 digest bytes are
interpreted as a big-endian **unsigned** 64-bit integer — this is the
seed. `random.Random` accepts arbitrary-size Python ints directly, so no
additional masking (unlike `derive_seed`'s 63-bit clamp, which exists only
for downstream JSON/other-language interoperability, not needed here since
these seeds are consumed immediately by `random.Random` and never
serialized as a seed themselves).

### 4.3 Event sampling algorithm (frozen)

```python
event_seed = sha256_seed(
    global_seed, dataset_name, problem_index,
    model_revision, rkv_revision, "b05r21_event",
)
rng = random.Random(event_seed)
eligible_sorted = sorted(eligible_event_ids)   # chronological, ascending compaction_event_id
if len(eligible_sorted) < 3:
    # example fails -- not merely "this event skipped"
    return EXAMPLE_INELIGIBLE
selected = rng.sample(eligible_sorted, 3)      # exactly 3, without replacement
selected_sorted = sorted(selected)
ordinal = {event_id: k for k, event_id in enumerate(selected_sorted)}  # 0, 1, 2
```

`random.Random.sample` draws without replacement deterministically given
the seed and the exact input sequence order; because `eligible_sorted` is
always constructed by sorting `eligible_event_ids` ascending before
sampling, the draw is reproducible independent of any incidental
iteration/dict order upstream. **Fewer than 3 eligible events fails that
example entirely** (excluded from the pilot, counted in the attrition
report, §9 Gate D) — it is not treated as "run with 1 or 2 events." The
three selected events are sorted chronologically and labeled ordinal `k
in {0, 1, 2}` — this ordinal, not the raw `compaction_event_id`, is what
feeds the layer-depth-coverage rule (§5). **Never resample after parity or
outcome inspection**: this draw happens once, from Pass 1's bookkeeping
alone (per `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §11), before Pass 2 runs
and long before any branch is evaluated.

## 5. Guaranteed layer-depth coverage (corrected)

`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §10's layer-selection rule
(`layer_index = SHA256(...) % num_hidden_layers`) was an **unrestricted**
hash over the full layer range for every event, independently. That does
**not** guarantee one early-, middle-, and late-third layer per example —
three independent uniform draws over `[0, num_hidden_layers)` can, and
with nonzero probability will, land in the same third. **This document
does not claim an unrestricted random hash guarantees coverage** — it
corrects the rule to actually guarantee it, by restricting each ordinal's
draw to its own third of the depth range:

```python
def layer_and_head(k: int, event_id: int, num_hidden_layers: int, num_key_value_heads: int) -> tuple[int, int]:
    lo = (k * num_hidden_layers) // 3
    hi = ((k + 1) * num_hidden_layers) // 3
    layer_seed = sha256_seed(
        global_seed, dataset_name, problem_index,
        model_revision, rkv_revision, event_id, k, "b05r21_layer",
    )
    layer_index = lo + (layer_seed % (hi - lo))

    head_seed = sha256_seed(
        global_seed, dataset_name, problem_index,
        model_revision, rkv_revision, event_id, k, "b05r21_head",
    )
    kv_head_index = head_seed % num_key_value_heads
    return layer_index, kv_head_index
```

For selected-event ordinal `k in {0, 1, 2}`, `lo = floor(k *
num_hidden_layers / 3)`, `hi = floor((k+1) * num_hidden_layers / 3)`
(integer floor division, matching Python's `//` for non-negative
operands). The layer is drawn uniformly **within** `[lo, hi)` via
`layer_seed % (hi - lo)`, offset by `lo`. Because `k` ranges over exactly
`{0, 1, 2}` (one per selected event, per §4.3) and the three `[lo, hi)`
intervals partition `[0, num_hidden_layers)` into (as close to) equal
thirds, **every example that reaches this step draws exactly one layer
from the early third, one from the middle third, and one from the late
third** — a guarantee, not a probabilistic tendency. The KV head is drawn
independently (a separate SHA-256 seed with a distinct `"b05r21_head"`
suffix) uniformly over the full `[0, num_key_value_heads)` range — head
selection has no depth-coverage requirement, so it is not similarly
restricted.

This repairs `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §10's layer/head rule in
place; §10's "144 pair branches are never treated as 144 independent
examples" framing and the surrounding sampling-cost table are unaffected
(the total branch count, `12 x 3 x 4 = 144`, does not change).

## 6. Frozen candidate and donor sampling (corrected)

### 6.1 Pool construction (unchanged in substance from `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §9, restated for completeness)

For the selected `(event_id, layer_index, kv_head_index)`:

- **`evicted_pool`**: absolute token positions that were actually evicted
  at this event/layer/head (present in the pre-call non-recent
  score-compared candidate set, absent from post-call
  `kept_token_indices[-1]` for this `(layer, head)`), sorted ascending by
  absolute token position.
- **`retained_pool`**: absolute token positions that were actually
  selected by the real top-k at this event/layer/head (present in both
  the pre-call pool and post-call `kept_token_indices[-1]`), **excluding**
  the always-kept protected recent window, sorted ascending by absolute
  token position.

Both pools are restricted to the non-recent, score-compared candidates
only — the protected recent window never competes in `topk` and is never
eligible as `e` or `r` (`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §9 rule 2,
unchanged).

### 6.2 Sampling algorithm (frozen — replaces §9.2's deterministic tie-break with true seeded sampling)

`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §9.2 selected candidates/donors by a
plain ascending-position tie-break when a document-level choice among
eligible entries was needed. That produces a **systematically
edge-biased** sample (always the lowest-position entries) rather than a
representative one. This is corrected to genuine SHA-256-seeded random
sampling, using two **separate** `random.Random` streams so the evicted
draw and the donor draw are independent of each other:

```python
def evicted_and_donor_candidates(event_id, layer_index, kv_head_index, evicted_pool, retained_pool):
    if len(evicted_pool) < 2 or len(retained_pool) < 2:
        return EVENT_INVALID   # invalidate the whole event, never partially sample

    evicted_seed = sha256_seed(
        global_seed, dataset_name, problem_index, model_revision, rkv_revision,
        event_id, layer_index, kv_head_index, "b05r21_evicted",
    )
    evicted_rng = random.Random(evicted_seed)
    evicted_selected = evicted_rng.sample(evicted_pool, 2)   # evicted_pool sorted ascending, see 6.1

    donor_seed = sha256_seed(
        global_seed, dataset_name, problem_index, model_revision, rkv_revision,
        event_id, layer_index, kv_head_index, "b05r21_donor",
    )
    donor_rng = random.Random(donor_seed)
    donor_selected = donor_rng.sample(retained_pool, 2)      # retained_pool sorted ascending, see 6.1

    return [(e, r) for e in evicted_selected for r in donor_selected]  # 4 cross-product swaps
```

**If either pool has fewer than 2 entries, the event is invalidated in
full** (not "sample what's available") — logged, counted in attrition
(§9 Gate D), never resampled after viewing branch results, and selection
never depends on `swap_gain` (both draws happen from Pass 1 bookkeeping,
before any branch is constructed or scored, per
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §11). All four cross-product swaps
`(e1,r1), (e1,r2), (e2,r1), (e2,r2)` are constructed and evaluated, nested
within this one event per the existing aggregation hierarchy
(`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §13, unaffected).

Observed R-KV kept/evicted indices (`kept_token_indices`) remain identity
ground truth throughout — the recomputed top-k from the §8 capture wrapper
is only a parity check and a score source (`docs/B0_5_R2_DENSE_CACHE_REPAIR.md`
§8.1-§8.2, unaffected).

## 7. Two-pass capture — retained, timing bridged in

The two-pass plan (`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §11) is **not**
redesigned here:

- **Pass 1** — natural R-KV run; freeze reference token IDs; record
  events and observed survivor provenance; deterministically compute
  eligible events (now using the §3.2/§4.1 corrected 49-token rule),
  select events/layers/heads/candidates/donors per §4-§6 above.
- **Pass 2** — reset all mutable state; replay the exact Pass-1 token IDs
  under R-KV; require identical event positions and per-layer/head
  survivor sets; attach only a per-instance before/after `update_kv`
  wrapper (`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §8, unchanged — **the
  wrapper still cannot read internal function locals; this is restated,
  not redesigned**); clone required pre-call tensors; call the original
  `update_kv` unchanged; recompute the real windowed decision score;
  require selected-set and bit-exact gather parity; invalidate the entire
  example on any trajectory/event/selection mismatch; never use FullKV
  reconstruction; never modify `third_party/R-KV`.

**What is new here is only where the §3 timing correction attaches**: the
bridge token and the 48-scored-token window are **not** part of Pass 1 or
Pass 2 — they belong to a third, already-costed step
(`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §12's "48-token teacher-forced
replay per branch" line item), branch evaluation, which happens strictly
after Pass 2's capture completes for a given target. Branch evaluation
now feeds the one bridge token (`x_{t+1}`) identically into the baseline
and swapped cache copies before scoring the 48 real reference tokens that
follow, per §3.3. This adds exactly one extra forward-call token per
branch relative to the pre-R2.1 cost model (144 branches x 1 bridge token
= 144 additional token-forwards total) — negligible relative to the
existing per-branch full-cache clone/restore cost
(`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §12), noted for completeness, not
re-derived as a new GPU-hour estimate.

## 8. Mandatory cheap signals

### 8.1 Signals recorded for every `(e, r)` pair, as pre-treatment differences

For every evaluated `(e, r)` swap pair, record the pre-treatment
(natural-run, never post-swap) difference `value(e) - value(r)` for:

1. **Real windowed R-KV final score** — the §8.1-strategy recomputed
   `final_score` (`r1_kv.py:49-77`'s formula, never the differently
   windowed `kept_final_scores` bookkeeping formula,
   `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §7).
2. **Attention component** — the recomputed `attn_cache` term.
3. **Similarity/redundancy component** — the recomputed `similarity_cos`
   term.
4. **Recency** — distance from the compaction boundary at event `t`.
5. **Key norm** — `||k_e|| - ||k_r||` at capture.
6. **Value norm** — `||v_e|| - ||v_r||` at capture.

### 8.2 Classification of the remaining candidate signals (frozen before any GPU result exists)

| Signal | Classification | Rationale |
|---|---|---|
| Entropy at capture | **Mandatory** | `docs/METHOD_PIVOT_SPEC.md` §5a's confound control (arXiv:2606.00206-adjacent caution) is explicitly "not optional" in every prior version of this document chain — an unexplained-reversal claim that does not rule out a locally uncertain decision point as the real cause is not adjudicable. |
| Logit margin at capture | **Mandatory** | Same rationale as entropy — the paired confound-control signal, same precedent. |
| Hidden-state delta | **Optional** | A real EpiKV-family signal worth recording, but it has no prior explicit "not optional" precedent in this document chain, is a higher-variance derived quantity, and its absence does not by itself make a reversal unexplainable the way missing entropy/margin would (those two specifically target the "was this just a hard decision point" confound; hidden-state delta targets a different, non-blocking question). Recorded when available; its absence does not block adjudicability. |
| Recurrence flag (LazyEviction-style) | **Context** | Descriptive bookkeeping (does the token's content reappear in the retained-attention pattern shortly after eviction) — useful for qualitative interpretation of a discovery-supporting result, not itself an association-tested gate input. |
| Future-attention oracle | **Non-deployable oracle** | Requires future context unavailable at real decision time, by construction (`docs/B0_5_DISCOVERY_PROTOCOL.md` §7, `docs/B0_5_PROTOCOL_REPAIR.md` §11, unchanged framing) — recorded only as a diagnostic upper bound, `oracle_non_deployable: true`, never compared to any deployable signal as if it were a fair baseline, and never counted among the "mandatory deployable signals" Gate 10B tests. |

Combining §8.1's six always-recorded differences with the two signals
classified **mandatory** here (entropy, logit margin) gives **eight
mandatory deployable signals** total, each independently subject to Gate
10B (§9) below.

### 8.3 Hard rule

**A missing mandatory signal makes the "unexplained reversal" decision NOT
ADJUDICABLE for that signal, never simply passed over.** This is enforced
structurally at Gate 10B (§9): fewer than 8 evaluable examples for *any*
one mandatory signal makes the *entire* Gate 10 result NOT ADJUDICABLE,
not just that one signal's row. Post-swap or treatment-induced values are
never used as predictors — every value in §8.1 and every mandatory signal
in §8.2 is measured at the moment the candidate token was **originally
generated or evicted** during the natural run, never after the synthetic
swap is applied (restated hard rule, unchanged in substance from
`docs/B0_5_PROTOCOL_REPAIR.md` §11 point 2).

## 9. Final Gate 10 (repairs `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16)

Discovery-supporting requires **all** of A, B, and C below; D is always
reported regardless of outcome.

### 9.A — Existence of actual reversals

At least **4 of the 12 examples** have at least one valid pair with

```
swap_gain > 0.01 mean nats per scored token
```

Equality (`swap_gain == 0.01` exactly) does **not** pass — strict `>`
only. This is unchanged in numeric value from
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16(a); restated here as part of the
single, final Gate 10 specification.

### 9.B — Weak association with every mandatory deployable signal (repaired: per-example nesting, not pooled)

For **each** of the eight mandatory deployable signals (§8.1's six plus
entropy and logit margin, §8.2):

1. **Calculate Spearman rho separately within each evaluable example** —
   i.e., across that example's own valid swap pairs only (up to 12 pairs
   per example: 3 events x up to 4 pairs), never pooled across examples
   into one correlation. This directly repairs
   `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §16(b)'s "pooled across examples"
   language, which the task brief flags as the wrong decision statistic —
   pair, event, and example nesting are kept explicit at every step, per
   this repository's existing discipline
   (`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §13).
2. **Use absolute rho**: `|rho|` for each example's within-example
   correlation.
3. **Report median absolute rho across examples** (the median of the
   per-example `|rho|` values, not a mean, and not a pooled statistic).
4. **Require median absolute rho < 0.30.** Equality (`median |rho| ==
   0.30` exactly) does **not** pass.
5. **Constant or undefined cases are not evaluable.** An example whose
   valid-pair `swap_gain` values are all identical, or whose signal values
   are all identical (Spearman rho undefined — zero-variance input), is
   **not evaluable** for that signal — excluded from the median, but
   **counted** (not silently dropped) in that signal's evaluable-example
   tally and in the attrition report (§9.D).
6. **Require at least 8 evaluable examples per mandatory signal.** Fewer
   than 8 evaluable examples for even one mandatory signal makes the
   **entire** Gate 10 result **NOT ADJUDICABLE** — not merely that one
   signal's row skipped, and not downgraded to "not discovery-supporting"
   (a genuinely unassociated signal and an unmeasurable one are different
   claims; conflating them would misreport a data-thinness problem as a
   negative finding).
7. Gate 10B **passes** only if **every** one of the eight mandatory
   signals independently satisfies `median |rho| < 0.30` with `>= 8`
   evaluable examples.

**Pooled pair-level Spearman across all examples is never used as the
decision statistic** — restated as the task brief requires, since it is
exactly the wrong-granularity statistic §16(b) previously specified.

### 9.C — Mandatory no-op control

Replacing donor `r` with its **own** captured K/V (i.e., `e := r`, a
degenerate "swap" that writes back exactly what was already there) must
produce **exactly zero change** in deterministic CPU tests — a bit-exact
(`torch.equal`) assertion on synthetic small tensors, implementable and
testable now, at B1A, with no GPU. This is the direct empirical check that
the swap mechanism itself introduces no incidental side effect (a stray
in-place mutation, an aliasing bug, an off-by-one in slot indexing) beyond
the intended content substitution.

A second, **future** check — not runnable now, explicitly scoped to a
later GPU phase (B2A or later, never this document) — is that repeated
real no-op swaps (`e := r` on real captured tensors, real forward passes)
must show `swap_gain` variation below `0.01` nats, bounding numerical
noise (BF16 accumulation order, kernel nondeterminism) separately from any
genuine effect.

**If the CPU no-op test fails, stop before interpreting any discovery
result** — a failing no-op control means the swap mechanism itself is
broken, and no `swap_gain` value produced by it can be trusted regardless
of what Gate 10A/10B say. This maps directly to the **NOT ADJUDICABLE**
outcome (§9.5 below), never to "not discovery-supporting" (that would
misreport a mechanism bug as a substantive negative finding).

### 9.D — Attrition and invalid branches (always counted)

Every filter applied anywhere in this document — event ineligibility
(§4.1), fewer-than-3-eligible-events example failure (§4.3), pool-size
event invalidation (§6.2), parity/trajectory mismatch (§7,
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §8.2), and Gate-10B non-evaluable
examples per signal (§9.B point 5) — is reported in a single attrition
accounting, reusing this repository's existing attrition-funnel discipline
(`CLAUDE.md` §8.4, `kvcot.analysis.summaries.build_attrition_funnel_table`'s
existing pattern, extended in spirit to this pilot's own filter stages,
not literally reused code since this pilot's schema differs from the
primary EAS pipeline's). No filter's loss count is ever silently absorbed
into a smaller reported denominator.

### 9.E — Decision table and possible outcomes

Possible outcomes are exactly **DISCOVERY-SUPPORTING**, **NOT
DISCOVERY-SUPPORTING**, or **NOT ADJUDICABLE** — no other label is used.

| Condition | Outcome |
|---|---|
| 9.C (no-op control) fails | **NOT ADJUDICABLE** — stop before interpreting A/B. |
| Any mandatory signal has `< 8` evaluable examples under 9.B | **NOT ADJUDICABLE**. |
| 9.C passes, all 8 signals have `>= 8` evaluable examples, and both 9.A and 9.B pass | **DISCOVERY-SUPPORTING**. |
| 9.C passes, all 8 signals have `>= 8` evaluable examples, and 9.A fails (fewer than 4/12 examples show a qualifying reversal) | **NOT DISCOVERY-SUPPORTING**. |
| 9.C passes, all 8 signals have `>= 8` evaluable examples, 9.A passes, but at least one mandatory signal fails `median |rho| < 0.30` under 9.B | **NOT DISCOVERY-SUPPORTING** (that signal already substantially explains the reversal pattern — not "unexplained"). |

9.D is reported alongside every outcome above, unconditionally — it is
never itself a pass/fail gate, only a mandatory disclosure.

This remains a **small discovery pilot, not a significance test, method
claim, accuracy-preserving claim, or faithfulness result.**
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §14's claim-boundary language is
otherwise unchanged and still applies in full.

## 10. Repaired schema (supersedes `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §15)

One JSON-lines record per swap-branch (pair). Every field from the R2
schema is retained; the four §3.4 timing fields are added; nothing is
removed.

```json
{
  "schema_version": "b0_5_r2_1.v1",
  "example_id": "string",
  "model_revision": "string",
  "rkv_revision": "45eaa7d69d20b7388321f077020a610d9afb65bd",
  "compaction_event_id": "int",
  "event_ordinal": "int (0, 1, or 2 -- from section 4.3)",
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
  "attention_component_diff": "float",
  "similarity_component_diff": "float",
  "recency_diff": "int",
  "key_norm_diff": "float",
  "value_norm_diff": "float",
  "entropy_diff": "float | null",
  "logit_margin_diff": "float | null",
  "parity_check_passed": "bool",
  "parity_failure_reason": "string | null",
  "bridge_token_absolute_position": "int",
  "first_scored_absolute_position": "int",
  "first_affected_logit_absolute_position": "int",
  "reference_horizon_sha256": "string (sha256 hex of the 48 scored reference token ids, comma-joined decimal, utf-8)",
  "swap_gain": "float",
  "baseline_per_token_nll": ["float", "... (exactly 48 entries)"],
  "swapped_per_token_nll": ["float", "... (exactly 48 entries)"],
  "net_physical_bytes_changed": "int (must equal 0)",
  "hidden_state_delta": "float | null",
  "recurrence_flag": "bool | null",
  "future_attention_oracle": "float | null",
  "oracle_non_deployable": true,
  "is_noop_control": "bool (true iff e == r, section 9.C)",
  "cap_hit_flag": "bool",
  "valid_flag": "bool",
  "invalid_reason": "string | null"
}
```

Identity for matching/deduplication remains
`(compaction_event_id, layer_index, kv_head_index, evicted_absolute_token_position,
retained_absolute_token_position)`, unchanged. `net_physical_bytes_changed`
is still asserted `== 0` for every valid record. `baseline_per_token_nll`/
`swapped_per_token_nll` now unambiguously start at
`first_scored_absolute_position`, never at the bridge position — resolves
the exact off-by-one this document exists to fix. The validation invariant
from §3.4 (`first_scored_absolute_position ==
first_affected_logit_absolute_position + 1`) is checked at write time.

## 11. B1A scope — unchanged

Nothing in this document expands or narrows B1A's scope beyond
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §17: model-family dispatch (B1A-1),
MATH-500 verification (B1A-2), the pairwise provenance schema — now this
document's §10 rather than R2's §15 — (B1A-3), and the read-only capture
wrapper (B1A-4). B1A additionally now covers implementing and CPU-testing
the exact deterministic sampling algorithms in §4-§6 (previously specified
only as a tie-break rule) and the §9.C no-op control's CPU unit test. B1A
performs no GPU inference and no Vast.ai work, unchanged. B2A remains the
mandatory GPU-calibration hard stop before any 12-example pilot
(`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §18, unaffected by this document —
its measurement list should additionally record the one extra bridge-token
forward per branch, §7, but that is a negligible addition to an already
itemized cost model, not a new line item requiring re-derivation).

## 12. Remaining uncertainties

- B2A has not been run; no GPU-hour or memory figure is asserted or
  re-derived by this document.
- Whether `hi - lo == 0` can occur in §5's layer-depth partition (i.e.
  `num_hidden_layers < 3`) is not a concern for the frozen model
  (`DeepSeek-R1-Distill-Qwen-1.5B` has considerably more than 3 layers,
  `configs/lock.yaml`/`CLAUDE.md` §4), but a future model swap must check
  this before reusing §5's formula unmodified — noted, not resolved here,
  since no model change is authorized by this document.
- The exact wall-clock and memory cost of the §9.C CPU no-op unit test and
  the §4-§6 sampling algorithms' implementation is unmeasured (B1A work,
  not yet done).
- Whether entropy/logit-margin values will in practice be available at
  every candidate/donor position (e.g., numerical edge cases at very low
  or very high confidence) is unverified until B1A's capture wrapper is
  implemented and exercised against controlled synthetic tensors.
- `CLAUDE.md` §4's model freeze still requires its own separate, dated
  amendment before any GPU run of a later phase — unchanged, unresolved
  blocker, not addressed by this document.
- The "32-token occlusion methodology" attribution remains unresolved
  (`docs/B0_5_SEARCH_LOG.md` §4) — not relied upon for this document's
  verdict.

## 13. Adversarial self-review

1. *Can the swap change the logits that predict `x_{t+1}`?* — No (§3.1
   points 2-3): those logits are already computed by `FC(t)` before the
   swap is applied to `FC(t)`'s cache output. This is exactly why
   `x_{t+1}` must be an unscored bridge token, never a scored one.
2. *Why feed `x_{t+1}` at all, if it isn't scored?* — Because it is the
   forward call whose **input cache state** is the first to actually
   differ between the baseline and swapped branches (§3.1 point 4); its
   *output* logits (predicting `x_{t+2}`) are the first real measurement
   of the swap's effect, so both branches must reach that forward call
   identically before scoring begins.
3. *Does "teacher-forced NLL evaluation" mean anything is actually
   decoded?* — No; restated explicitly in §3.3 to correct the prior
   "greedy decoding" phrasing, which invited exactly this confusion. No
   argmax or sampling occurs; only real, fixed reference tokens are
   scored.
4. *Does an unrestricted `hash % num_hidden_layers` guarantee one
   layer per third?* — No (§5); three independent uniform draws over the
   full range can collide in the same third with nonzero probability.
   This is why §5 restricts each ordinal's draw to its own `[lo, hi)`
   interval rather than the full range.
5. *Does ascending-position tie-break produce a representative candidate/
   donor sample?* — No (§6.2); it is systematically biased toward the
   lowest absolute positions in each pool. Genuine SHA-256-seeded
   `random.Random.sample` over the sorted pool corrects this while
   remaining fully deterministic and reproducible.
6. *Are the evicted-candidate and donor draws independent of each
   other?* — Yes, by construction: two separate SHA-256 seeds with
   distinct literal suffixes (`"b05r21_evicted"`/`"b05r21_donor"`), each
   feeding its own `random.Random` instance (§6.2).
7. *Can candidate/donor selection be influenced by how a swap turns
   out?* — No; both draws happen from Pass-1-derived bookkeeping alone,
   before Pass 2's capture and before any branch is constructed or scored
   (§6.2, §7) — selection never depends on `swap_gain`.
8. *Does pooling Spearman correlation across all examples answer the same
   question as computing it per-example and taking the median?* — No;
   pooling can produce a strong apparent correlation driven entirely by
   between-example variance (e.g. easier examples having both higher
   scores and higher `swap_gain`) even when no within-example association
   exists, or vice versa. Gate 10B is explicit that the per-example,
   median-of-`|rho|` statistic is the decision statistic, never the pooled
   one (§9.B).
9. *What happens if a mandatory signal simply cannot be measured on
   enough examples?* — The whole Gate 10 result is NOT ADJUDICABLE (§9.B
   point 6, §9.E), not silently downgraded to a negative finding — a
   data-thinness problem and a genuine null association are different
   claims and must not be conflated.
10. *What if the swap mechanism itself is subtly broken (a stray mutation,
    an off-by-one slot write)?* — The §9.C no-op control is designed to
    catch exactly this class of defect before any Gate 10A/10B result is
    trusted; a no-op-control failure forces NOT ADJUDICABLE regardless of
    what the other gates would otherwise say.
11. *Does this document reopen the within-head swap's representability
    argument?* — No (§1, §3.5); `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §5-§6
    is unaffected and reused as-is.
12. *Does fixing the timing change the total branch count or sample
    size?* — No (§4.3, §6.2, §7); still `12 x 3 x 4 = 144` planned
    branches, one extra bridge token per branch (negligible cost, §7),
    never a change to `n_examples`, `events_per_example`, or the
    evicted/donor counts.

No answer above is incomplete in a way that would require withholding a
READY verdict; every open item is either fully repaired in this document
or explicitly scoped to B1A/B1B/B2A/B2B (§11-§12), never silently assumed.

## 14. Final B0.5-R2.1 verdict

**B0.5-R2.1 VERDICT: READY FOR B1A PREREQUISITE IMPLEMENTATION**

This authorizes only B1A: CPU-side prerequisite implementation (MATH-500
verifier, architecture-aware R-KV dispatch, this document's §10 pairwise
provenance schema, the `docs/B0_5_R2_DENSE_CACHE_REPAIR.md` §8 per-instance
read-only capture wrapper with exact score recomputation and parity
assertions, this document's §4-§6 deterministic sampling algorithms, the
§9.C no-op control's CPU unit test, and CPU unit/integration tests
generally). It does **not** authorize B1B, model inference, a GPU,
Vast.ai, the discovery pilot, or any method claim. `CLAUDE.md` §4's
model-freeze amendment remains required, separately, before any GPU run of
a later phase, and is not granted by this document.

## 15. Cross-reference

Superseded passages, and their exact replacements, are marked inline in
`docs/B0_5_R2_DENSE_CACHE_REPAIR.md` (§10 sampling rule, §11 timing
attachment point, §14 estimand/sign convention, §15 schema, §16 Gate 10,
§21 verdict) with `**[SUPERSEDED — see docs/B0_5_R2_1_FINAL_PROTOCOL.md
§N]**` banners; no historical text is deleted. Top-of-document
"further superseded" banners are added to `docs/B0_5_PROTOCOL_REPAIR.md`,
`docs/B0_5_DISCOVERY_PROTOCOL.md`, and `docs/B0_5_FEASIBILITY_AUDIT.md`,
consistent with the existing chain of such banners each of those documents
already carries for B0.5-R2. `docs/b0_5_decision.json` retains every prior
field for provenance and adds a `superseded_by_r2_1`/`b0_5_r2_1_*` block
pointing here.
