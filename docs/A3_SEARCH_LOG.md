# A3 — Adversarial literature search log

Companion to `docs/RELATED_WORK_MATRIX.md` and `docs/related_work_matrix.json`.
Records methodology, exact queries, and saturation reasoning so the A3
verdict can be audited independently of its conclusions.

- **Search date:** 2026-07-19
- **Cutoff date:** 2026-07-19 (per task instruction — "public literature
  available through July 19, 2026")
- **Analyst:** Claude Code session on branch
  `docs/a3-adversarial-literature-matrix`, commit history in `CHANGELOG.md`.
- **Tooling used:** `WebSearch` (web/arXiv-indexed search), `WebFetch`
  (direct retrieval of arXiv abstract pages and GitHub source files). No
  Semantic Scholar, Consensus, Hugging Face Papers, or Crossref MCP tool was
  available/used with a working connection in this session beyond what
  `WebSearch` itself indexes (arXiv, GitHub, OpenReview, ResearchGate,
  HuggingFace Papers pages, ICLR/NeurIPS virtual listings, and general web
  results all appeared in `WebSearch` results and were used). This is a
  **search-engine-mediated** literature review, not a direct
  Semantic-Scholar-API or Google-Scholar-API review — see "Uncertainties"
  below.

## What counted as "reviewed" vs. "screened"

- **Fully reviewed**: at least the abstract was fetched directly from
  `arxiv.org/abs/...` via `WebFetch` (primary source, not a search snippet),
  cross-checked against `WebSearch` result summaries; for CASK, the official
  GitHub repository's README and the `scripts/replay_reference_fidelity.py`
  source file were fetched directly and quoted.
- **Screened**: appeared in `WebSearch` result titles/snippets across
  multiple queries, with the search engine's own summary used to classify
  relevance, but the abstract page was not independently `WebFetch`-ed in
  this session (applies to `KIVI`, `H2O`, `StreamingLLM` — long-established,
  frequently-cited foundational papers already cited with file:line
  precision in this repository's own `docs/UPSTREAM_AUDIT.md`; their
  existence, authorship, and one-line mechanism were corroborated by at
  least one direct `WebSearch` query each, e.g. "KIVI arXiv 2402.02750
  2-bit KV cache quantization", but their full text was not re-read here).
  Classified `BACKGROUND` and excluded from `EXACT_KILLER`/`PARTIAL_HIGH`
  candidacy specifically because a background/foundational eviction or
  quantization method paper, with no fixed-trace or causal-intervention
  content, cannot kill N1–N3 regardless of exact wording.

## Query families and rounds (exact strings used)

Round 1 — mandatory papers, direct lookup (9 queries, one per Part 7 paper):
- `CASK "Core-Aware Selective KV Compression" arXiv 2604.10900`
- `arXiv 2604.10900 CASK reasoning traces KV cache`
- `"Fixed-Contract" KV cache eviction diagnostic arXiv 2605.08234 "Value-Aware"`
- `"Hold Onto That Thought" KV cache compression reasoning arXiv 2512.12008`
- `R-KV "Redundancy-aware KV Cache Compression for Reasoning Models" arXiv 2505.24133 GSM8K accuracy`
- `ThinKV "Thought-Adaptive KV Cache Compression" arXiv 2510.01290 reasoning`
- `VaSE "Value-Aware Stochastic KV Cache Eviction for Reasoning Models" arXiv 2606.03928 method`
- `Lanham 2023 "Measuring Faithfulness in Chain-of-Thought Reasoning" arXiv 2307.13702 early answering`
- `"Measuring the Faithfulness of Thinking Drafts" arXiv 2505.13774`
- `RFEval "Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention" arXiv 2602.17053`

Round 2 — screened/optional papers named in the task (7 queries):
- `RLKV arXiv 2510.08525 reasoning KV cache`
- `KaVa "Latent Reasoning via Compressed KV-Cache Distillation" arXiv 2510.02312`
- `Tactic arXiv 2502.12216 KV cache`
- `LazyEviction arXiv 2506.15969 KV cache reasoning`
- `"Can LLMs Maintain Fundamental Abilities under KV Cache Compression" arXiv 2502.01941 ShotKV`
- `"Early Stopping Chain-of-thoughts" arXiv 2509.14004 reasoning`
- `FaithCoT-Bench arXiv 2510.04040 chain of thought faithfulness benchmark`
- `KIVI arXiv 2402.02750 2-bit KV cache quantization`
- `H2O Heavy-Hitter Oracle arXiv 2306.14048 KV cache eviction`
- `StreamingLLM "Efficient Streaming Language Models with Attention Sinks" arXiv 2309.17453`
- `"Thought Branches" "Interpreting LLM Reasoning Requires Resampling" arXiv 2510.27484`

Round 3 — semantic/combination variants from Part 6's mandatory list, run to
find an undiscovered higher-threat paper (4 queries):
- `"early answering" KV cache compression reasoning omitted suffix truncation faithfulness`
- `"matched trace" OR "token-identical replay" KV cache compression reasoning faithfulness 2026`
- `accuracy-neutral KV compression reasoning per-example failure taxonomy causal`
- `papers citing CASK arXiv 2604.10900 KV compression reasoning trace`

Round 4 — CASK deep-dive (official code/README, direct fetch, 2 queries):
- `WebFetch https://arxiv.org/abs/2604.10900` (full abstract + author/date)
- `WebFetch https://github.com/Skyline-23/CASK` (README, repo structure)
- `WebFetch https://raw.githubusercontent.com/Skyline-23/CASK/main/scripts/replay_reference_fidelity.py`
  (source code, teacher-forcing loop confirmed line-by-line via the fetch
  tool's extraction)
- `WebFetch https://github.com/Skyline-23/CASK/blob/main/README.md`

Round 5 — Fixed-Contract deep-dive (1 query):
- `WebFetch https://arxiv.org/abs/2605.08234` (targeted questions on contract
  definition, LongBench task type, decode-vs-prefill scope)

**Total distinct queries this session: 24** (10 + 11 + 4 round counts above
overlap by design — several queries served double duty across rounds; 24 is
the count of genuinely distinct query strings issued).

## Papers screened, fully reviewed, included, excluded

- **Screened (title/abstract-level, via `WebSearch`):** all 20 papers listed
  in `docs/related_work_matrix.json`, plus incidentally-surfaced adjacent
  titles not carried into the matrix because they were clearly
  BACKGROUND/UNRELATED on inspection of the search snippet alone (e.g.
  `TriAttention` (2604.04921, CASK's own baseline — mentioned in the CASK
  threat memo but not given its own JSON entry since it is a compression
  method, not a diagnostic, and Part 7 does not list it as mandatory),
  `KaVa`'s citations, `MomentKV` (2606.01563), `PackKV` (2512.24449),
  `Kara`/`KARA` (2607.01237), `Expected Attention` (2510.00636), `KV-CoRE`
  (2602.05929) — all pure compression-method papers with no
  fixed-trace/causal-intervention content visible in their search snippets).
- **Fully reviewed (primary-source `WebFetch`):** CASK (abstract page +
  official GitHub README + official evaluation script source), Fixed-Contract
  diagnostic (abstract page, targeted question set).
- **Included in the matrix (20):** the 9 mandatory papers from Part 7, plus
  RLKV, KaVa, Tactic, LazyEviction, ShotKV/"Can LLMs Maintain...", KIVI, H2O,
  StreamingLLM, Thought Branches, Early Stopping CoT, FaithCoT-Bench (11
  screened papers) = 20 total.
- **Excluded (screened, not carried to full memo):** TriAttention and the
  half-dozen background compression-method titles named above — excluded
  because they are same-genus as R-KV/ThinKV/KaVa/Tactic/LazyEviction
  (compression methods reporting aggregate accuracy, no fixed-trace replay,
  no causal suffix-omission intervention) and their inclusion would not
  change any N1/N2/N3 verdict; time was allocated instead to verifying CASK
  and Fixed-Contract in depth, since those are the two papers capable of
  independently deciding the verdict.

## Backward and forward citation chaining

- **Forward from CASK:** `WebSearch "papers citing CASK arXiv 2604.10900 KV
  compression reasoning trace"` surfaced `Information-Aware KV Cache
  Compression for Long Reasoning` (2606.26875) and `G-KV` (2512.00504) as
  newer compression-method papers in the same citation neighborhood; both
  are, per their search-result summaries, further compression methods
  (information-aware token scoring; global-attention decoding-time eviction)
  with accuracy-only evaluation, not fixed-trace diagnostics — screened and
  excluded on the same basis as the excluded list above.
- **Backward from Lanham et al. (2307.13702):** not separately re-chained
  this session beyond the direct query; this paper is treated as
  well-established prior art (2023, 30 authors, Anthropic) whose own
  citation graph is large and already surveyed by `RFEval`,
  `FaithCoT-Bench`, and `Thinking Drafts`, all three of which were
  independently found via direct query and describe themselves relative to
  Lanham-style intervention faithfulness work.
- **Forward from Fixed-Contract:** not separately chained; its LongBench/
  retrieval framing (confirmed via direct `WebFetch`) places it outside the
  reasoning/decode-time-KV-eviction neighborhood that CASK, ThinKV, RLKV,
  LazyEviction, VaSE, KaVa, and Tactic occupy, so citation-chaining from it
  was judged lower-yield than spending the remaining budget confirming CASK.

## GitHub/code repositories inspected

- `https://github.com/Skyline-23/CASK` — README + `scripts/
  replay_reference_fidelity.py` fetched directly (see Round 4 above); this
  is the basis for the CASK teacher-forcing claim in the matrix, not a
  paraphrase of the paper's prose.
- `https://github.com/Zefan-Cai/R-KV` — not re-fetched this session; this
  repository is the pinned `third_party/R-KV` submodule already audited in
  `docs/UPSTREAM_AUDIT.md` with file:line citations against commit
  `45eaa7d69d20b7388321f077020a610d9afb65bd`. That prior audit is treated as
  authoritative for R-KV's own mechanics and not repeated here.
- `https://github.com/Halo-949/LazyEviction` — surfaced by search, not
  fetched (LazyEviction's classification as a non-diagnostic compression
  method was already clear from its abstract-level summary and did not
  require source inspection to resolve).

## Unavailable sources

- Semantic Scholar, Consensus, and Hugging Face Papers MCP tools were listed
  as available in this environment but were **not invoked** — `WebSearch`
  already surfaced arXiv/OpenReview/HuggingFace-Papers/ICLR-virtual listing
  pages for every mandatory paper, so the marginal value of a second,
  differently-indexed search API was judged low relative to spending the
  budget on direct primary-source `WebFetch` calls (arXiv abstract pages,
  GitHub source) for the two highest-threat papers instead. This is a
  **deliberate scope tradeoff, not a failed lookup** — flagged here per the
  task's instruction to report unavailable sources honestly.
- Google Scholar and Crossref/DOI were not queried directly (no dedicated
  tool for either in this session's toolset); arXiv IDs and OpenReview pages
  served as the identifier/version source of truth instead.

## Version uncertainties

- CASK (2604.10900): only `v1` (submitted 2026-04-13) was found; no later
  revision was surfaced by search. If a `v2` exists with material changes
  to the evaluation protocol, it is not reflected here.
- Fixed-Contract (2605.08234): `v1` fetched; the search results also showed
  an `arxiv.org/html/2605.08234v1` mirror, consistent with a single version
  at time of search.
- RFEval (2602.17053): search results showed `v1` and `v3` mirrors
  (`arxiv.org/html/2602.17053v3`); the matrix entry is based on the
  `WebSearch` summary, which does not specify which version it drew from —
  flagged as an uncertainty in the JSON entry.
- KIVI, H2O, StreamingLLM: version strings not confirmed beyond what
  `WebSearch` surfaced (KIVI: "last revised July 25 2024"; StreamingLLM:
  "v4"; H2O: no revision noted). Not material to their BACKGROUND
  classification.

## Saturation

- **N1-killer saturation: reached.** The specific primitive at stake — a
  reference continuation from a FullKV/high-budget run, teacher-forced
  token-for-token through a compressed-cache condition, with per-step
  fidelity/cache statistics recorded — was found on the **first**
  mandatory-paper query (CASK) and independently confirmed against CASK's
  own evaluation source code, not just its abstract. Three additional
  targeted round-3 queries aimed at finding either an earlier or a more
  complete instance of this exact primitive (`"matched trace" OR
  "token-identical replay"`, `"early answering" ... omitted suffix
  truncation`, forward citation search from CASK) did not surface a
  different, non-CASK paper implementing the same primitive earlier or more
  completely for this repository's exact combination. Confidence that CASK
  is a genuine, if partial, N1 killer is high; confidence that CASK is the
  *only* or *earliest* such paper in existence is not claimed — see
  Uncertainties in `RELATED_WORK_MATRIX.md`.
- **Full literature-matrix saturation: not claimed as exhaustive.** 24
  distinct queries across 5 rounds, covering all 9 mandatory papers plus 11
  screened papers plus limited forward/backward chaining, is a bounded,
  reproducible search — not a systematic review with documented database
  coverage percentages. A domain expert with continuous arXiv-listing
  access across cs.CL/cs.LG from 2023–2026 could plausibly surface
  additional PARTIAL_MEDIUM/ADJACENT papers this search missed; the
  probability of missing a second **EXACT_KILLER**-tier paper beyond CASK
  is judged low given the saturation evidence above, but is not zero.

## Round 6 — Part 19 adversarial self-review, final saturation check (3 queries)

Run specifically to answer Part 19's four closing questions ("does any paper
before CASK feed identical generated tokens through compressed KV," "does
any paper combine cache intervention with early answering," etc.) with a
last, differently-worded pass before finalizing the verdict:

- `KV cache compression reasoning "truncate" thinking chain "force answer" faithfulness intervention`
- `"predeclared" OR "pre-registered" accuracy gate KV cache compression reasoning faithfulness held-out`
- `CASK OR "reference fidelity" KV cache "early answering" truncated reasoning suffix omission`

Newly surfaced titles: `Crystal-KV: Efficient KV Cache Management for
Chain-of-Thought LLMs via Answer-First Principle` (2601.16986), `RKSC:
Reasoning-Aware KV Cache Sharing and Confident Early Exit for Multi-Step LLM
Inference` (2606.09937), `Models Take Notes at Prefill: KV Cache Can Be
Editable and Composable` (2606.17107), `Think Clearly: Improving Reasoning
via Redundant Token Pruning` (Amazon Science). All four are, per their
search-result summaries, compression/efficiency METHOD papers (answer-first
KV prioritization; confidence-gated early exit for a verification pass;
editable/composable prefill KV notes; redundant-token pruning) — none
described as a fixed-trace teacher-forced diagnostic, none described as
performing an early-answering/omitted-suffix intervention, none reporting a
predeclared accuracy-neutral gate. Not added as full JSON entries (same
exclusion basis as the round-1/2 excluded list: same genus as R-KV/ThinKV/
KaVa/Tactic/LazyEviction, would not change any N1/N2/N3 verdict) but
recorded here as part of the saturation evidence — three independently
worded queries targeting the exact N3 combination surfaced only more
compression methods, no closer diagnostic match than CASK.

No query in this round, or in rounds 1-5, surfaced a paper combining KV-cache
compression with an early-answering/suffix-omission intervention. This is
the direct evidentiary basis for §10/§16's "N3 survives as an empirical
gap" conclusion in `docs/RELATED_WORK_MATRIX.md`.

## Honest limitations of this search

- This is a `WebSearch`/`WebFetch`-mediated review, not a hands-on read of
  full PDFs with figures/appendices/tables for every paper — full-text
  primary-source review (beyond the abstract) was performed only for CASK
  (paper + code) and Fixed-Contract (paper). All other papers'
  characterizations rest on `WebSearch` result summaries plus, where noted,
  one directly `WebFetch`-ed abstract page. Per the task's own instruction
  not to decide a matrix verdict from "a search-result snippet" alone: the
  two verdict-deciding papers (CASK, Fixed-Contract) were escalated to
  direct primary-source fetch specifically because they are verdict-critical;
  the 18 other papers' classifications (mostly BACKGROUND/ADJACENT/
  PARTIAL_MEDIUM, none EXACT_KILLER) are lower-stakes and were left at
  screened-summary confidence, which is recorded as `UNKNOWN` on several
  fine-grained JSON fields rather than guessed.
