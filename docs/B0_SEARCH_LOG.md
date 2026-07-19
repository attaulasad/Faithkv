# B0 search log — method-pivot novelty gate

Phase B0 artifact. Search date: 2026-07-19. Literature cutoff: 2026-07-19.
This log covers the B0 method-novelty search only; the Phase A3 diagnostic
search has its own frozen log (`docs/A3_SEARCH_LOG.md`), which was reused
but not re-run.

## Engines and databases

- General web search (US results), 6 queries (exact strings below).
- Direct arXiv fetches (abstract pages, HTML full texts, one PDF), 21
  fetch attempts, 20 successful.
- One official GitHub repository fetch (CASK).
- Local primary sources: pinned R-KV submodule
  (`third_party/R-KV`, commit 45eaa7d) and `docs/UPSTREAM_AUDIT.md`.
- Reuse of frozen A3 records (`docs/related_work_matrix.json`) for R-KV,
  KIVI, ShotKV, LazyEviction, VaSE, CASK-evaluation-code facts.

**Evidence limitation (applies to every fetch):** fetched pages were
interrogated through a fetch-and-summarize tool that returns targeted
answers with verbatim quotes. "Full text inspected" in B0 artifacts means
the full HTML/PDF was fetched and interrogated with specific method
questions and quotes captured — not a human cover-to-cover read. Web
search result summaries were treated as discovery tools only; every
verdict-bearing fact was re-grounded in a direct fetch of the primary
source, except where a record is explicitly marked
`search_summary_only` and no verdict depends on it.

## Exact web-search queries

1. `Locret learned retaining heads KV cache eviction training causal importance`
2. `KV cache eviction residual score calibration correct heuristic errors learned rescue evicted tokens`
3. `"KV cache" eviction recompute importance after eviction interaction-aware conditional dynamic rescoring 2026`
4. `faithfulness-aware KV cache compression reasoning dependence eviction constrained allocation 2026`
5. `counterfactual KV cache ablation answer likelihood supervision eviction reasoning "thought block" masking`
6. `KV cache "second chance" OR "rescue" OR restoration rehydration evicted tokens reasoning model 2025 2026`

These six queries jointly cover the task-mandated families: learned causal
KV importance / causal KV token protection (q1, q5); counterfactual KV
ablation (q5); residual token utility / compression-score calibration /
false-negative KV rescue (q2); interaction-aware eviction / dynamic
rescoring after eviction / cache-state-conditional importance (q3);
learned retaining heads / Locret (q1); KV restoration-rehydration (q6);
future-utility KV prediction (q1, q3 results); constrained KV allocation /
faithfulness-aware KV compression / reasoning-dependence-aware eviction
(q4); answer-likelihood KV eviction and thought-block KV masking (q5).
Additional mandated families (causal reasoning-state compression, merge
scratch tokens, thought-type classification) were covered by the mandatory
papers themselves (CASK, ThinKV) rather than separate queries.

## Direct fetches (primary sources)

Abstract pages: 2605.22106 (ArborKV), 2510.01290 (ThinKV), 2602.10238
(Learning to Evict), 2601.03066 (functional importance), 2606.26875
(InfoKV), 2606.26472 (EpiKV), 2605.18053 (Protection), 2606.03928 (VaSE),
2506.19143 (Thought Anchors), 2504.14051 (CAOTE), 2604.10900 (CASK),
2603.10899 (LookaheadKV), 2505.20334 (Lookahead Q-Cache).

Full texts: arxiv.org/html/2510.01290v2 (ThinKV v2 — fetched after the
PDF failed, see below), arxiv.org/html/2605.22106v1 (ArborKV),
arxiv.org/html/2606.09916v1 (IntentKV), arxiv.org/html/2602.03203v2
(ForesightKV), arxiv.org/html/2607.13205 (Adaptive Filtering),
arxiv.org/pdf/2606.11164 (ReasonAlloc — summary-level yield),
arxiv.org/pdf/2604.18002 (Neural GC — summary-level yield).

Code repositories fetched this session: github.com/Skyline-23/CASK
(README + file listing). Code repositories *located but not inspected*:
github.com/apple/ml-learning-to-evict, github.com/RUCAIBox/ForesightKV,
github.com/terarachang/VaSE, github.com/gpgabriel25/KVCacheBoundaryProtection,
github.com/Halo-949/LazyEviction (URL carried from A3),
github.com/Zefan-Cai/R-KV (inspected locally as the pinned submodule, not
re-fetched). thought-anchors.com noted, not inspected.

Failed fetch: arxiv.org/pdf/2510.01290v2 (ThinKV PDF) — exceeded the fetch
tool's content-size limit; the HTML full text was used instead.

## Screening tiers (honest counts)

- **Screened (title/snippet in search results):** ~40 distinct works
  surfaced across the six queries (approximate — search pages do not give
  stable totals; this count is the number of distinct titles actually read
  in the returned result lists, not a fabricated database hit count).
  Includes, beyond the reviewed set: Make Each Token Count (2605.09649),
  KVpop (2607.05061), IndexMem (2605.25475), Judge Q (2509.10798),
  MomentKV (2606.01563), KVSlimmer (2603.00907), PagedEviction (2509.04377),
  LKV (2605.06676), SkipKV (2512.07993), KVzip (id not captured), CacheFlow
  (2604.25080), HCache/LMCache/KVcached (systems tier), a KV-optimization
  survey (2607.08057) and an awesome-list, Fixed-Contract (2605.08234, A3),
  and the A3-known set.
- **Abstract reviewed (direct fetch of abstract page):** 13 papers (list
  above).
- **Full text inspected (via fetch tool, quotes captured):** 7 papers —
  ThinKV v2, ArborKV, IntentKV, ForesightKV, Adaptive Filtering,
  ReasonAlloc (partial yield), Neural GC (partial yield).
- **Code inspected:** CASK README/structure (this session; its
  `replay_reference_fidelity.py` was inspected line-level in A3); R-KV
  pinned submodule (local, audited in `docs/UPSTREAM_AUDIT.md`).

## Citation chaining

- **Backward:** the mandatory-paper briefs and fetched texts cite H2O,
  SnapKV, StreamingLLM, Ada-KV, QUEST, KIVI, R-KV, Lanham — all already in
  the frozen A3 matrix; no backward chain surfaced an uncovered
  method-level threat.
- **Forward:** approximated by the 2026-scoped queries (q2–q4, q6), which
  surfaced the post-2025 descendants (ForesightKV, IntentKV, IndexMem,
  LookaheadKV, ReasonAlloc, Adaptive Filtering, KVpop, MomentKV). No
  citation-database (Semantic Scholar / Google Scholar) forward pass was
  run; this is a logged gap, mitigated by the fact that every additional
  discovery in this space could only add overlaps to an already-negative
  verdict.

## Unavailable / partial sources

- ThinKV official code: none found in v2 text.
- ArborKV code: none found.
- Adaptive Filtering: author list not captured from fetched HTML; no code.
- LookaheadKV: abstract only; supervision details UNKNOWN.
- Locret: search-level only this session (pre-2025 paper; treated as a
  category representative for learned retention heads).
- LKV, IndexMem, KVpop, MomentKV, Judge Q, KVzip, SkipKV: screened only.
- ReasonAlloc and Neural GC PDFs yielded summary-level (not
  verbatim-dense) fetch results; their fields are marked accordingly.

## Version uncertainties

- ThinKV reviewed at v2 (2026-05-07) per the task's latest-version
  requirement.
- CAOTE at v6 (2025-10-05). ForesightKV at v2 (2026-06-01). EpiKV at v2
  (2026-06-28). Thought Anchors at v4 (2025-10-27). Functional importance
  at v3 (2026-04-21). Others at the versions listed in the JSON.
- Any revision published after 2026-07-19 is uncovered by construction.

## Saturation status

- **M1: saturated for a negative verdict.** Three independent HIGH-overlap
  kills from three different directions (ArborKV: ablation supervision;
  IntentKV: residual architecture; ThinKV: counterfactual-to-policy
  pipeline) plus two corroborating (ForesightKV, Adaptive Filtering).
  Further search can only strengthen the negative. Saturation for a
  *positive* verdict was therefore not required and was not claimed.
- **M2: saturated.** Killed by directly quoted primitives (ForesightKV MDP
  state definition; Neural GC conditionality; R-KV's own per-compaction
  rescoring, verified locally). Repeated queries (q3) stopped yielding new
  operation classes — later results were allocation/system papers.
- **M3: saturated for the machinery, checked-absent for the metric.**
  ReasonAlloc + LKV + Ada-KV lineage close the allocation machinery; no
  paper found using a causal-dependence constraint (checked absence across
  q4 and the reviewed set) — but the verdict does not depend on that
  absence, since a new constraint metric is insufficient by the
  predeclared standard.

## What was NOT searched

- No Semantic Scholar / OpenReview / ACL Anthology API sweeps; no
  systematic forward-citation pass (logged above).
- No non-English literature.
- No re-run of the A3 diagnostic-novelty queries (out of B0 scope; A3 log
  frozen).
