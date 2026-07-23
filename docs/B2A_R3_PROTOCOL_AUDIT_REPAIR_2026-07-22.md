# B2A-R3 Protocol Audit-Repair Ledger (dated 2026-07-22)

This is a concise finding ledger for the independent audit of
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` (first frozen at
commit `93b6ba869eb5e555684704a6d1f2250f16884768`). It is not a second copy
of the protocol — every repair's authoritative rule lives in the protocol
document itself; this ledger records what was wrong, what evidence was
used to fix it, and exactly where the fix now lives.

This ledger now covers two repair rounds. **Step 2A** (findings
R3-AUDIT-01 through R3-AUDIT-18, immediately below) repaired the original
18 defects found by the first independent audit. **Step 2B** (findings
R3-AUDIT-19 through R3-AUDIT-25, appended after R3-AUDIT-18) repairs seven
further implementation-level ambiguities found by a second independent
re-audit of the Step 2A repair commit (`81e11cb57202e0d4b434aabb
347963ae3c34b80b`). Both rounds are documentation-only repairs to the
same protocol document; neither authorizes any CPU implementation, GPU
activity, or method work by itself.

```text
STATUS: STEP 2A AND STEP 2B AUDIT FINDINGS (25 TOTAL) REPAIRED IN THIS
LEDGER AND IN THE PROTOCOL DOCUMENT.
The repairing author does not self-certify this protocol.
Step 3 CPU implementation remains blocked until a separate independent
re-audit verifies the Step 2B commit.
Stage B FullKV qualification, Stage C B2A-R3 execution, all GPU/CUDA
activity, B2B, and FaithKV method implementation remain prohibited.
```

---

## R3-AUDIT-01 — Missing §22

**Original defect:** §1 of the first-frozen protocol referenced "this
document's own internal self-audit, §22 below" — no §22 existed.

**Evidence inspected:** `docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md`
at commit `93b6ba8` (20 sections total, no §22).

**Exact repair:** Added §22, "Independent protocol audit and repair
disposition," containing a summary table of all 18 findings and the
required disposition text ("INDEPENDENT AUDIT FINDINGS REPAIRED IN THIS
COMMIT... STEP 3 CPU IMPLEMENTATION REMAINS BLOCKED...").

**Files changed:** `docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md`
(§22, new).

**Remaining authorization state:** Step 3 CPU implementation blocked
pending independent re-audit (unchanged by this finding alone).

---

## R3-AUDIT-02 — Contradictory authorization status

**Original defect:** The active status block read "PROTOCOL FROZEN — CPU
IMPLEMENTATION AUTHORIZED — GPU EXECUTION PROHIBITED" as a bare
declarative line, while the paragraph immediately below it said CPU
implementation becomes authorized only after an independent audit that
had not occurred. A reader skimming the status block alone could
reasonably conclude Stage A was already live.

**Evidence inspected:** `docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md`
§1; `PLAN.md`, `README.md`, `CHANGELOG.md`, `CLAUDE.md` §1e (all echoed the
same status block or a paraphrase of it).

**Exact repair:** Replaced the status block everywhere with:

```text
B2A-R3 STATUS:
PROTOCOL REPAIRED — INDEPENDENT RE-AUDIT PENDING
CPU IMPLEMENTATION BLOCKED
GPU EXECUTION PROHIBITED
```

and added explicit prose in §1 stating the original line was itself an
audit finding, with no self-certification implied.

**Files changed:** `docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md`
§1; `CLAUDE.md` (new §1f); `PLAN.md` (current-status section); `README.md`
(current-status section); `CHANGELOG.md` (new entry).

**Remaining authorization state:** CPU implementation blocked; GPU
prohibited; independent re-audit is the next required action.

---

## R3-AUDIT-03 — Rounded runtime constants

**Original defect:** The predictor section used `10.25` (a rounded
display value) and `~19.0` (an approximate setup-time value) as if they
were exact constants — insufficient precision for a byte-identical CPU
predictor implementation or deterministic tests asserting an exact
pass/fail integer boundary.

**Evidence inspected:**
`docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`, object
`runtime_and_memory`: `per_example_total_wall_seconds:
1378.3004406290129`, `per_real_pair_seconds: 10.247917714063078`,
`projected_total_seconds: 18034.603590369457`. Cross-checked against
`docs/B2A_R2_RESULT_2026-07-22.md` §4's rounded display values (`1378.30`,
`10.25`, `18034.6`) and against `src/kvcot/discovery/execution_measurement.py`
(`conservative_pair = max(b2a_real_pair_seconds)`,
`build_runtime_projection` defaults `example_count=12`,
`real_pair_count=144`).

**Exact repair:** Froze `REFERENCE_TOTAL_TOKENS = 4931`,
`REFERENCE_EXAMPLE_SECONDS = 1378.3004406290129`,
`REFERENCE_PAIR_SECONDS = 10.247917714063078`,
`REFERENCE_PROJECTED_TOTAL_SECONDS = 18034.603590369457`, and derived
`REFERENCE_SETUP_SECONDS = 19.298151996218968` algebraically (recomputed
and verified during this repair, exact agreement to double precision — see
the Step 2A final report §D). Froze the continuous token ceiling
(`2775.0857674895859...`) and the integer test boundary
(2775 passes, 2776 fails), both independently recomputed and verified.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §7.1, §7.3, §7.5.

**Remaining authorization state:** Unchanged — documentation only, no
predictor code implemented.

---

## R3-AUDIT-04 — Mixed-level ordering ambiguity

**Original defect:** The protocol specified "8 level-4 and 8 level-5"
candidates but never defined how the two subsets combine into one ordered
16-row manifest (globally re-sorted? concatenated? interleaved?).

**Evidence inspected:** `src/kvcot/discovery/b2a_r2_candidates.py`
(`build_candidate_manifest`'s existing single-level, globally-sorted
design, which does not need to answer this question because B2A-R2 used
only one level); `src/kvcot/discovery/manifest_prepare.py`
(`EXPECTED_MATH500_COLUMNS`, confirming `level` is a bare digit string).

**Exact repair:** Froze the exact 12-step construction: filter, exclude,
sort level-4 and level-5 **independently** by `(ordering_hash ascending,
unique_id ascending)`, take the first 8 of each, interleave
level-4-rank-i/level-5-rank-i strictly alternating starting with level-4,
assign `candidate_ordinal` only after interleaving. Froze the resulting
counts (16 total, 8/8 by level, 4/4 within the first 8 qualifiable
candidates, level 4 first).

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §8.2.

**Remaining authorization state:** Unchanged — no candidate generator
implemented, no manifest generated.

---

## R3-AUDIT-05 — Unfrozen exclusion set

**Original defect:** The protocol described the exclusion *sources* (B2A-R1's
row, B2A-R2's candidate manifest) but never froze the literal 13-row list,
and implied it should be reconstructed by production code running `git
show` dynamically.

**Evidence inspected:** `git show
9fe27a2:configs/discovery/b2a_one_example_manifest.json` (B2A-R1's row:
`test/precalculus/807.json`); `configs/discovery/b2a_r2_candidate_manifest.json`
(all 12 `unique_id` values and their ordinals, `canonical_sha256 =
ac2dcc4550a89f2cfa701acd608a8087b4a1ebaa0ea05eb15d8f71e3434ee0ec`).

**Exact repair:** Built a 13-row table (`unique_id`, `source_attempt`,
`source_manifest_or_commit`, `source_ordinal`), verified exactly 13
unique, non-duplicate IDs, sorted them lexicographically, and computed
`EXCLUSION_SET_SHA256 = 0c46510c79a22d08e8fd610104a527e867f821a540063a5059b51a660d25bc69`
over the canonical newline-terminated UTF-8 payload (recomputed and
verified during this repair — Step 2A final report §D). Also corrected an
imprecise justification in the first-frozen text (it said all 12 were
excluded "because qualification inference already ran against three of
them," which is not why the other 9 are excluded).

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §8.1.

**Remaining authorization state:** Unchanged — no exclusion logic
implemented; the frozen list/hash exists only as documentation.

---

## R3-AUDIT-06 / R3-AUDIT-07 — Undefined thinking-span validity / trace completeness

**Original defect:** §10.1 said "thinking span parses correctly" and
"trace is complete" without any exact, machine-checkable definition.

**Evidence inspected:** `src/kvcot/probes/early_answering.py`,
`find_think_span`/`ThinkSpanResult` — the four possible
`think_parse_status` literal values in source (`"ok"`,
`"generation_prompt_preopened_ok"`, `"no_open_marker"`,
`"no_close_marker"`).

**Exact repair:** Froze `THINK_PARSE_SUCCESS_STATUSES = {"ok",
"generation_prompt_preopened_ok"}` (the two success values found in
source) and the exact boolean formulas for `thinking_span_valid` and
`trace_complete`, kept `fullkv_answer_correct` as a separate condition so
a structurally-complete-but-wrong trace fails correctness, not structure.
Froze that Step 3 must reuse `find_think_span` via the canonical FullKV
worker, never a second independent parser, and may extend the worker
result schema in a backward-compatible, versioned way without touching
historical B2A-R1/R2 artifacts.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §10.1.

**Remaining authorization state:** Unchanged — no worker schema extension
implemented.

---

## R3-AUDIT-08 — Self-referential hash ambiguity

**Original defect:** Conceptual schemas used `manifest_hash`/`artifact_hash`
without defining whether the payload being hashed includes its own hash
field (a self-referential ambiguity that would make the hash
unreproducible).

**Evidence inspected:** `src/kvcot/utils/hashing.py`, `sha256_json`
(`json.dumps(obj, sort_keys=True, separators=(",", ":"),
ensure_ascii=True)` then SHA-256 hex digest) — the repository's one
existing canonical-hashing helper, already used by
`b2a_r2_candidates.build_candidate_manifest`'s
`canonical_sha256 = sha256_json(manifest)`.

**Exact repair:** Standardized the field name to `canonical_sha256`
everywhere, froze `canonical_sha256 = sha256_json(payload with
canonical_sha256 omitted)`, and froze the five-step verification
procedure (parse, require exactly one 64-hex field, remove it, recompute,
require exact string equality). Applied uniformly to the candidate
manifest, qualification artifact, selection-provenance artifact, and
authorization-claim artifact.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.1.

**Remaining authorization state:** Unchanged — no hashing code
implemented.

---

## R3-AUDIT-09 — Nondeterministic candidate-manifest timestamp

**Original defect:** The candidate-manifest conceptual schema included
`created_at`, which would make the committed manifest non-reproducible
byte-for-byte from the same inputs (a build run five minutes later would
hash differently for no scientific reason).

**Evidence inspected:**
`src/kvcot/discovery/b2a_r2_candidates.py`'s existing `CandidateRow`/
manifest fields (no `created_at`-equivalent field present in the existing,
working B2A-R2 convention — confirming its omission is consistent with
existing practice, not a new restriction).

**Exact repair:** Removed `created_at` (and, explicitly, any random ID,
filesystem path, or network-fetch timestamp) from the candidate
manifest's canonical, hashed payload. Clarified that the qualification
artifact and authorization claim, by contrast, legitimately carry
operational timestamps in their canonical payload because they each
describe one real, non-repeatable execution event, not a
static/reproducible manifest.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.3, §12.6,
§12.8.

**Remaining authorization state:** Unchanged.

---

## R3-AUDIT-10 — Unfrozen artifact paths

**Original defect:** Every artifact path was qualified as "recommended,"
leaving Step 3 free to choose a different path and breaking
reproducibility of cross-references between documents.

**Evidence inspected:**
`configs/discovery/b2a_r2_candidate_manifest.json`,
`results/decisions/b2a_r2_qualification.json`,
`configs/discovery/b2a_one_example_manifest.json`,
`docs/evidence/B2A_R2_SELECTION_PROVENANCE_2026-07-22.json`,
`results/decisions/b2a_attempt_20260722T101253300941Z_.../` (existing
B2A-R2 path conventions).

**Exact repair:** Froze one exact path for each artifact type, each
mirroring B2A-R2's existing directory convention with `b2a_r2_` replaced
by `b2a_r3_` where a direct analogue exists (candidate manifest,
qualification artifact, attempt directory root), reused the exact same
filename for the selected one-example manifest (since B2A-R2's freezer
overwrites that file in place rather than creating a new one), and froze
a `docs/B2A_R3_STAGE_{B,C}_..._AUTHORIZATION_<date>.md` naming pattern for
future dated authorization documents (date left as a pattern placeholder
since it does not exist yet — freezing a fabricated future date was
rejected as inventing an unverifiable fact).

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.2; consumed by
§8.1, §12.3-12.8, §13, §14.4.

**Remaining authorization state:** Unchanged — no artifact created at any
of these paths.

---

## R3-AUDIT-11 — Undefined phase-wide wall-time behavior

**Original defect:** §11 simultaneously said `VERIFY BEFORE FREEZE` for
the wall-time limit and required Step 3 to "calculate and display the
protocol-derived wall-time limit," without ever freezing a formula that
could produce such a number — a direct internal contradiction.

**Evidence inspected:** `src/kvcot/discovery/constants.py`,
`B2A_WORKER_TIMEOUT_SECONDS = 7200` (the existing per-worker subprocess
timeout); `docs/B2A_R2_RESULT_2026-07-22.md` §2 (B2A-R2's own
qualification pass took ~9.9 minutes across 3 candidates, a single data
point, not a worst-case bound for 8 candidates).

**Exact repair:** Froze `PER_CANDIDATE_WORKER_TIMEOUT_SECONDS = 7200`,
`QUALIFICATION_CANDIDATE_LIMIT = 8`, and the purely arithmetic
`ABSOLUTE_TIMEOUT_ENVELOPE_SECONDS = 57600` (`8 × 7200`), explicitly
labeled as a mathematical envelope, never an authorized GPU budget or the
actual Stage B limit. Froze the required dry-run display fields
(`qualification_phase_wall_time_limit = null`,
`gpu_qualification_authorized = false`,
`wall_time_authorization_required = true`) and removed the instruction to
compute a nonexistent "protocol-derived" limit — a real, stricter,
phase-wide limit is now explicitly deferred to a future Stage B
authorization document, which does not block Stage A (CPU) work.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §11.

**Remaining authorization state:** No wall-time limit is authorized;
Stage B remains blocked pending its own future authorization.

---

## R3-AUDIT-12 — Undefined attempt-consumption lifecycle

**Original defect:** The protocol required "an attempt-consumption guard"
without ever defining what it tracks, where it lives, or when
authorization is considered spent.

**Evidence inspected:**
`results/decisions/b2a_attempt_20260722T101253300941Z_fb6f5081d47f45f4b4f9258c25e6883d/`
(B2A-R2's existing attempt-directory convention, confirmed via
`docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`'s `file_inventory`
and `provenance_git` blocks, including its `dirty`/`staged_paths`/
`unstaged_paths` clean-worktree accounting).

**Exact repair:** Froze a two-part model: (1) an immutable, committed,
dated authorization document per grant (schema frozen, §14.4), and (2) an
atomic `authorization_claim.json` written inside a new immutable attempt
directory before any CUDA/model/GPU activity — the existence of one valid
claim is permanent consumption regardless of the run's later outcome
(crash, timeout, gate failure, or success), with no retry. Froze that
CPU-only dry-run paths must never create an attempt directory or claim.
Froze that the claim lives under the attempt-artifact root (never
`configs/`) so it is handled as an authorized artifact-root path by
existing provenance/clean-worktree accounting, not as an unexplained
modification.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §14.4, §12.8.

**Remaining authorization state:** No authorization document exists yet;
no claim has ever been made; this section is documentation only.

---

## R3-AUDIT-13 — Historical branch-constant conflict

**Original defect:** `src/kvcot/discovery/attempt_verification.py`
contains a module-level `REQUIRED_BRANCH = "research/b1b-r4-final-b2a-closure"`
used in historical B2A-R1/R2 verification. Naively "updating" this global
for B2A-R3 (which runs on a different branch) would silently reinterpret
historical verification of prior, already-consumed attempts.

**Evidence inspected:** `src/kvcot/discovery/attempt_verification.py:134,
158-159` (the constant's definition and its one call site, comparing it
against `git_evidence.get("branch")`).

**Exact repair:** Froze a requirement for a new, separately-constructed
`AttemptProvenancePolicy` object carrying B2A-R3's own required
repository/branch/commit/ancestors/R-KV SHA/authorization identity, sourced
from B2A-R3's own future dated authorization document — explicitly never
by editing or reusing `REQUIRED_BRANCH`. Froze nine rules preserving
historical verification unchanged while making B2A-R3 verification
independent of any single global "current branch" constant.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §14.5.

**Remaining authorization state:** Unchanged — `attempt_verification.py`
was inspected only, never modified, per this task's restrictions.

---

## R3-AUDIT-14 — Dynamic final-gate inheritance

**Original defect:** §16 said B2A-R3 "inherits whatever
`FINAL_MANDATORY_GATE_CONDITIONS` exists at Step 3 implementation time" —
explicitly a dynamic, unfrozen inheritance rule, not a frozen protocol
value, and vulnerable to silent drift if the source tuple changes between
this freeze and Step 3's eventual implementation.

**Evidence inspected:**
`src/kvcot/discovery/final_contract.py` at commit `93b6ba869eb5e555684704a6d1f2250f16884768`
(`git hash-object` → blob `0b4063023b4da5cf33b5a1e419fc9577db363b6b`), 30
conditions; `src/kvcot/discovery/b2a_contract.py` at the same commit
(`git hash-object` → blob `4d63a83a81855cfcbc7defc74ef332946870f112`), 29
conditions.

**Exact repair:** Copied both exact ordered tuples verbatim into the
protocol, each with its source path, source commit, blob SHA-256, and
exact count. Froze that any future addition, removal, renaming, or
semantic change to either tuple requires a dated protocol amendment and
its own independent audit — no silent inheritance from a later source
version, ever. Noted explicitly that B2A-R2's own historical "27 of 28"
count reflects the tuple's membership as of B2A-R2's execution date, not
these frozen counts — the two are expected to differ and this document
does not paper over that.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §16.1, §16.2.

**Remaining authorization state:** Unchanged — the source tuples
themselves were inspected only, never modified.

---

## R3-AUDIT-15 — No-op exactness/tolerance wording

**Original defect:** The protocol described the no-op gate as both
"exact equality" and `_close(...)`-with-`1e-9`-tolerance in the same
sentence, without distinguishing which quantity each description applied
to.

**Evidence inspected:** `src/kvcot/discovery/schemas.py:250-274`
(`_noop_invariants`) and `:51,54` (`_FLOAT_DIFF_TOLERANCE = 1e-9`,
`_close`).

**Exact repair:** Froze the precise distinction: the NLL arrays
(`baseline_per_token_nll`/`swapped_per_token_nll`) are checked by literal
Python list `!=` (exact, zero-tolerance, elementwise), while the derived
`swap_gain` is checked against `0.0` via `_close`, which uses the
existing, unchanged `1e-9` absolute tolerance. Froze exact wording for
every future reference to this gate that keeps the two checks
distinguished.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §16.3 (table row),
§16.4 (new subsection), §17 (cross-reference updated).

**Remaining authorization state:** Unchanged — the tolerance itself is
not modified by this repair.

---

## R3-AUDIT-16 — Inconsistent schema field names

**Original defect:** §§12-14 used inconsistent names for the same
concept across different sub-schemas: `candidate_index` vs.
`candidate_ordinal`; `manifest_hash` vs. `canonical_sha256` vs.
`artifact_hash`; `selected_row_id` vs. `selected_unique_id`;
`dataset_row_id` vs. `unique_id`.

**Evidence inspected:**
`src/kvcot/discovery/b2a_r2_candidates.py` (`CandidateRow.candidate_ordinal`,
`.unique_id`, `.problem_sha256`); `src/kvcot/discovery/b2a_qualification.py`
(`CandidateQualificationOutcome.candidate_ordinal`,
`.prompt_token_ids_sha256`, `.generated_token_ids_sha256`).

**Exact repair:** Standardized on the existing source-code field names
wherever a direct analogue already exists (`candidate_ordinal`,
`unique_id`, `problem_sha256`, `prompt_token_ids_sha256`,
`generated_token_ids_sha256`), and on `canonical_sha256` for every
self-hash field (R3-AUDIT-08). Built six explicit schema tables (candidate
manifest, candidate row, qualification outcome, qualification artifact,
selection provenance, authorization claim), each field annotated with
type, nullability, meaning, hash inclusion, and source.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.3-12.8.

**Remaining authorization state:** Unchanged — no schema implemented in
code.

---

## R3-AUDIT-17 — Real qualification artifact incorrectly implied in CPU scope

**Original defect:** Stage A's allowed/forbidden lists did not clearly
distinguish "generating the deterministic candidate manifest" (CPU-only,
no model involved) from "generating a qualification artifact" (which
describes the outcome of real FullKV inference) — a careless reading
could conclude Stage A authorizes a real qualification run.

**Evidence inspected:** `src/kvcot/discovery/b2a_qualification.py`
(qualification's `run_qualification_execute` path requires loading and
running the real FullKV worker — not CPU-only).

**Exact repair:** Reworded Stage A explicitly: the candidate manifest MAY
be real and committed under Stage A (it needs no model, CUDA, or GPU —
only the pinned dataset's public content and pure Python hashing); the
qualification *artifact* may only ever be synthetic/injected under Stage
A, because producing a real one requires FullKV inference, which Stage A
never authorizes regardless of audit outcome.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §14.1.

**Remaining authorization state:** Stage A remains blocked pending
independent re-audit; even once granted, Stage A will not authorize a
real qualification artifact.

---

## R3-AUDIT-18 — Repository identity mismatch

**Original defect:** §21 of the first-frozen protocol treated
`attaulasad/Faithkv` (from an earlier task's own wording) as an unresolved
discrepancy requiring ongoing attention.

**Evidence inspected:** `git remote -v` at this repair's starting commit
→ `https://github.com/asad073-ui/Faithkv.git` (fetch and push both).

**Exact repair:** Replaced the discrepancy framing with a resolved
identity statement (`repository = asad073-ui/Faithkv`), keeping the
`attaulasad/Faithkv` wording only as a historical note about a past task
prompt's own documentation mismatch, explicitly not an active ambiguity.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §21.

**Remaining authorization state:** Unchanged — this is a documentation
clarification with no bearing on any authorization gate.

---

# Step 2B — second independent re-audit repair (dated 2026-07-22)

The Step 2A repair above (commit `81e11cb57202e0d4b434aabb347963ae3c34b80b`)
was independently re-audited. The first-frozen protocol's 18 defects were
materially repaired, but the re-audit found seven remaining
implementation-level ambiguities that would still have let Step 3 invent
behavior the protocol did not actually pin down. All seven are repaired
below and in the protocol document. This closes Step 2B; it does not
authorize Step 3 — a separate, further independent re-audit of this Step
2B commit is still required first (§1g of `CLAUDE.md`; §1/§22 of the
protocol document).

```text
STATUS: STEP 2B FINDINGS REPAIRED IN THIS COMMIT.
The repairing author does not self-certify this protocol.
Step 3 CPU implementation remains blocked until a separate independent
re-audit verifies this Step 2B commit.
```

---

## R3-AUDIT-19 — Qualification-condition tuple not frozen

**Original defect:** §10 (§10.1-§10.4) described qualification gates as
prose bullets and separate predicate definitions, but never froze one
single, exact, ordered tuple of condition names Step 3 must implement —
leaving Step 3 free to invent its own condition names, omit a condition,
or order `failed_conditions` differently from another implementation.
This mirrors the exact defect R3-AUDIT-14 already repaired for the
*final* mechanical gate tuple (§16.1), but §10's *qualification* gate had
never received the same treatment.

**Evidence inspected:** `src/kvcot/discovery/b2a_qualification.py`
(`QUALIFICATION_CONDITIONS`, a 10-name frozen tuple for B2A-R2's simpler
qualifier, and `evaluate_candidate_qualification`/`build_candidate_outcome`,
which compute `conditions`/`qualified`/`failed_conditions` from it — the
existing pattern this repair generalizes, not invents);
`src/kvcot/discovery/pass1.py` (`eligible_event_positions`, the exact
eligible-event rule reused unchanged); `src/kvcot/probes/early_answering.py`
(`find_think_span`/`ThinkSpanResult`, reused unchanged per R3-AUDIT-06/07);
`src/kvcot/discovery/strict_device.py`
(`verify_placement_from_raw_evidence`/`verify_device_gate_from_raw_evidence`,
confirming that stricter raw `ParameterPlacementEvidence` fields
— `requested_device`, `every_parameter_on_cuda`, `no_offload_verified`,
`parameter_count`, `unique_device_types`, `unique_devices`, `hf_device_map`
— already exist as a canonical source, so the frozen conditions must
derive from these raw fields, never only a legacy summary boolean);
`src/kvcot/utils/hashing.py` (`sha256_json`, `sha256_file`); `src/kvcot/
config.py` (`config_identity = sha256_file`); `configs/discovery/
llama8b_math500_b1024.yaml` (the exact `generation:` block, confirmed
byte-for-byte identical to the frozen generation-configuration payload
below).

**Exact repair:** Added protocol §10.5, freezing
`B2A_R3_QUALIFICATION_CONDITIONS`, an exact 27-name ordered tuple, its
exact count (27), the `qualified`/`failed_conditions` derivation formulas
(mirroring the existing `b2a_qualification.py` pattern exactly), and a
complete, unambiguous boolean definition for every one of the 27 names —
reusing already-frozen predicates where they exist (§10.1's
`thinking_span_valid`/`trace_complete`, §7.4's runtime formula, §9's
ordering hash) rather than redefining them a second time. This B2A-R3-only
tuple is explicitly distinct from, and never confused with, B2A-R2's
historical 10-name `QUALIFICATION_CONDITIONS` in `b2a_qualification.py`,
which is not modified.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` new §10.5
(cross-referenced from §10, §12.5, §12.6, §15).

**Remaining authorization state:** Unchanged — no evaluator implemented in
code; this is a frozen specification only.

---

## R3-AUDIT-20 — Artifact provenance schemas incomplete

**Original defect:** §12.3-§12.8's schema tables (frozen by R3-AUDIT-08/
09/10/16) named fields but omitted several that later sections implicitly
required (e.g. `config_sha256`/`generation_config_sha256` binding, a
runtime-source-artifact path/hash pair, per-candidate worker identity
echoes, expected-vs-observed prompt-token-hash pairs, placement/timing
evidence) — a Step 3 implementer following §12 alone could not actually
reproduce the identity/hash bindings §4/§7/§9 elsewhere required.

**Evidence inspected:** `src/kvcot/discovery/b2a_qualification.py`
(`CandidateQualificationOutcome`, `QualificationArtifact` — the existing
B2A-R2 field set this repair extends, never duplicates independently);
`src/kvcot/discovery/b2a_r2_candidates.py` (`CandidateRow`, `build_
candidate_manifest` — existing manifest-level field set); `src/kvcot/
config.py` (`config_identity`); `docs/evidence/
B2A_R2_ATTEMPT_INDEX_2026-07-22.json` (the file the runtime-source-artifact
hash binds to); `configs/discovery/llama8b_math500_b1024.yaml` (the exact
identity fields — dataset repo/config/split/revision, model/tokenizer
name/revision, budget — every schema below must echo back).

**Exact repair:** Replaced §12.3 (candidate manifest), §12.5 (per-candidate
qualification outcome), and §12.6 (qualification artifact) with complete,
exact field lists (protocol §12.3, §12.5, §12.6) — every field the
protocol's own text elsewhere requires is now named, typed, and marked
required/nullable in the schema it belongs to. No field may be added
outside a future versioned schema revision.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.3, §12.5, §12.6.

**Remaining authorization state:** Unchanged — no schema implemented in
code; documentation only.

---

## R3-AUDIT-21 — Selected-manifest hash semantics unresolved

**Original defect:** §12.8's authorization-claim schema (frozen by
R3-AUDIT-08/16) referenced `selected_manifest_canonical_sha256` with the
parenthetical caveat "if that file gains one" — the historical
`B2AOneExampleManifest` (`configs/discovery/b2a_one_example_manifest.json`)
has never carried a `canonical_sha256` self-hash field, and this repair
left it ambiguous whether Step 3 was expected to add one (which would
silently reinterpret B2A-R1/R2's historical schema) or use some other
mechanism.

**Evidence inspected:** `src/kvcot/discovery/manifest.py`
(`B2AOneExampleManifest.manifest_hash()` = `sha256_json(self.model_dump
(mode="json"))` — an existing, already-implemented EXTERNAL hash method
over the whole manifest, with no self-referential field inside the hashed
payload; used today by `manifest_prepare.prepare_manifest`'s `--force`
old/new hash printout). This is a hash the repository already computes,
not a new field on the manifest schema.

**Exact repair:** Froze `selected_manifest_sha256 =
B2AOneExampleManifest.manifest_hash()` (an external hash, computed the
same way the existing code already computes it — `sha256_json(model_dump
(mode="json"))` — never a new field added to `B2AOneExampleManifest`
itself), and `selected_manifest_hash_algorithm =
"B2AOneExampleManifest.manifest_hash-v1"` as its named algorithm-version
string. Renamed every reference from `selected_manifest_canonical_sha256`
to `selected_manifest_sha256` throughout §12.7 (selection provenance) and
§12.8/§M (authorization claim), with an explicit note that the
`canonical_sha256` self-hash rule (§12.1) applies only to artifacts that
actually contain a `canonical_sha256` field — the selected manifest does
not, and this repair does not add one to it.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.1 (scope note),
§12.7, §12.8, §13.

**Remaining authorization state:** Unchanged — `manifest.py` was
inspected only, never modified (out of scope per Task B).

---

## R3-AUDIT-22 — Candidate rows insufficient for deterministic freezing

**Original defect:** §12.4's candidate-row schema table (frozen by
R3-AUDIT-16) listed identity/hash fields but did not explicitly freeze
whether the complete pinned dataset row is embedded in the manifest or
must be refetched at freeze time — leaving "embed or refetch" as an
implementation choice, which would make freezing either non-deterministic
(refetch could observe a since-changed live source) or silently dependent
on an unstated convention.

**Evidence inspected:** `src/kvcot/discovery/b2a_r2_candidates.py`
(`CandidateRow.row: dict[str, Any]` — the existing B2A-R2 convention
already embeds the complete row; `build_candidate_manifest` computes
`raw_row_sha256=sha256_json(row)`, `problem_sha256=sha256_text(row
["problem"])`, `gold_answer_sha256=sha256_text(row["answer"])` from the
embedded row directly); `src/kvcot/discovery/b2a_r2_freeze.py`
(`freeze_qualified_row` reads `candidate_row["row"]`, recomputes
`sha256_json(row)`, and compares it against the stored `raw_row_sha256` —
confirming the existing freezer already uses the embedded row and never
refetches); `src/kvcot/discovery/manifest_prepare.py`
(`EXPECTED_MATH500_COLUMNS = ("problem", "solution", "answer", "subject",
"level", "unique_id")`).

**Exact repair:** Froze the exact candidate-row schema (protocol §12.4)
including the `row` field (the complete embedded pinned row, exact columns
`("problem", "solution", "answer", "subject", "level", "unique_id")`), the
three verification formulas (`sha256_json(row) == raw_row_sha256`,
`sha256_text(row["problem"]) == problem_sha256`, `sha256_text(row
["answer"]) == gold_answer_sha256`), the three identity cross-checks
(`row["unique_id"] == unique_id`, `row["subject"] == subject`, `int(row
["level"]) == level`), and an explicit, non-optional rule: the future
freezer (§13) must use the embedded `row` and must NEVER refetch the
dataset during selected-row freezing — mirroring
`kvcot.discovery.b2a_r2_freeze.freeze_qualified_row`'s existing behavior
exactly, not a new design.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.4, §13.

**Remaining authorization state:** Unchanged — no candidate generator or
freezer implemented in code; documentation only.

---

## R3-AUDIT-23 — Markdown authorization hash conflicts with JSON self-hash rule

**Original defect:** §12.1's canonical-hashing rule
(`canonical_sha256 = sha256_json(payload with canonical_sha256 omitted)`)
was written as applying to "any authorization document or authorization-
claim artifact" — but the future dated authorization documents (§12.2)
are Markdown files, not JSON objects, so `sha256_json`'s self-referential-
field-omission procedure cannot apply to them at all (there is no JSON
object to parse, no field to omit). The protocol never actually froze how
a Markdown document's hash is computed, leaving Step 3 to guess.

**Evidence inspected:** `docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_
<date>.md`/`docs/B2A_R3_STAGE_C_EXECUTION_AUTHORIZATION_<date>.md`'s naming
pattern (§12.2, already frozen as Markdown, never JSON); `src/kvcot/utils/
hashing.py` (`sha256_file`, the repository's one existing whole-file byte
hash helper, already used for config-identity hashing via `kvcot.config
.config_identity`).

**Exact repair:** Froze `authorization_document_sha256 = sha256_file(the
exact committed Markdown authorization document)` — a plain whole-file
SHA-256 over committed bytes, with no self-referential-field procedure
involved because a Markdown document has no such field. Clarified
explicitly, in §12.1 and everywhere §12.1 was previously cross-referenced
(§12.8, §14.4, §14.5, §M), that the `canonical_sha256`/`sha256_json`
self-hash rule applies only to JSON artifacts that actually contain a
`canonical_sha256` field (candidate manifest, qualification artifact,
selection-provenance artifact, authorization-claim artifact) — never to
the Markdown authorization documents themselves, which use
`authorization_document_sha256`/`sha256_file` instead.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §12.1, §14.4, §M
(authorization claim schema).

**Remaining authorization state:** Unchanged — no authorization document
has been written yet; this is a hashing-rule clarification only.

---

## R3-AUDIT-24 — Authorization claim consumption not globally atomic

**Original defect:** §14.4's claim lifecycle (frozen by R3-AUDIT-12)
required Step 3 to "search the attempt root for any existing valid claim
using the same `authorization_id`" before writing a new claim inside a
freshly created attempt directory — a scan-then-write pattern that is
inherently racy: two concurrent processes could each complete the scan
(finding nothing) before either writes its claim, both then proceeding to
launch CUDA/model activity under the same "consumed once" authorization.
The consumption event was never tied to one single atomic filesystem
operation with a fixed, precomputable path.

**Evidence inspected:** `src/kvcot/discovery/attempt_verification.py`
(the existing attempt-directory/provenance conventions this repair must
stay compatible with — clean-worktree accounting, `dirty`/`staged_paths`/
`unstaged_paths`); `docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json`
(the existing `results/decisions/b2a_attempt_.../` directory-naming
convention the new global-claim design must sit alongside without
disturbing).

**Exact repair:** Replaced the scan-then-write design entirely with a
globally exclusive claim at one deterministic path,
`results/decisions/b2a_r3_authorization_claims/<authorization_id>.json`,
created via `os.open(claim_path, O_WRONLY | O_CREAT | O_EXCL, mode)` (or an
equivalent atomic-across-processes primitive) — **creation of the
filesystem entry, not a successful subsequent write, is the consumption
event**. Froze: an authorization-ID validation pattern
(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, rejecting path separators, `..`,
whitespace, control characters); the exact seven-step claim sequence
(pre-claim verification, path derivation, exclusive creation, payload
write+flush+fsync, attempt-directory creation only after the global claim
succeeds, an immutable copy/reference of the claim inside the attempt
directory, execution only after both exist); explicit crash semantics
(any filesystem entry at the deterministic path — complete, partial,
empty, or corrupt — means permanently consumed, never available for
retry, no repair or deletion restores it); a mandatory future CPU
concurrency test (two concurrent claim attempts for one authorization ID
must produce exactly one successful exclusive creation and one refusal);
explicit dry-run behavior (no claim directory, no claim file, no attempt
directory, `authorization_claim_created=false`, `authorization_consumed=
false`); and worktree/provenance rules (clean worktree required before
claim creation, only the exact active global claim path and active
attempt root recognized as expected post-claim artifacts, never placed in
`configs/`, historical B2A-R1/R2 clean-worktree verification never
weakened).

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` §14.4 (replaced),
§12.2 (new global claim path), §M (authorization claim schema).

**Remaining authorization state:** Unchanged — no claim mechanism
implemented in code; this is a frozen specification only.

---

## R3-AUDIT-25 — Protocol-identity fields conflated across schemas

**Original defect:** Several schema tables reused a single, ambiguous
`protocol_version` field name for what are actually four distinct
concepts (candidate ordering convention, qualification-artifact schema
convention, runtime-predictor formula/constant version, selection
convention) — a Step 3 implementer could not tell, from the field name
alone, which of the four a given `protocol_version` string on a given
artifact was actually versioning, or whether bumping one should ever bump
another.

**Evidence inspected:** `src/kvcot/discovery/b2a_r2_candidates.py`
(`CANDIDATE_MANIFEST_PROTOCOL_VERSION = "faithkv-b2a-r2-row-order-v1"`,
used both as the manifest's top-level `protocol_version` field AND inside
`_ordering_hash`'s payload — confirming the historical B2A-R2 schema
itself already conflates "artifact schema" and "ordering convention" under
one name, a pattern this repair does not carry forward into new B2A-R3
schemas); `src/kvcot/discovery/b2a_qualification.py`
(`QUALIFICATION_PROTOCOL_VERSION = "faithkv-b2a-r2-qualification-v1"`, the
qualification artifact's own separate, historical version string).

**Exact repair:** Froze four separate, independently-versioned identity
fields (protocol §12.9, and per-schema in §12.3/§12.6/§12.7/§M):
`artifact_schema_version` + `candidate_order_protocol_version` for the
candidate manifest; `artifact_schema_version` + `qualification_protocol_
version` + `runtime_predictor_version` for the qualification artifact;
`artifact_schema_version` + `selection_protocol_version` for the selection
provenance; `artifact_schema_version` alone for the authorization claim
(one concept, no second sub-version needed). Froze the exact string value
for each. Historical B2A-R1/R2 artifacts keep their historical field names
and meanings unchanged — this standardization applies only to new B2A-R3
artifacts, and a bare `protocol_version` never appears in any new B2A-R3
schema table.

**Files changed:**
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` new §12.9; §12.3,
§12.6, §12.7, §M updated to use the separated fields.

**Remaining authorization state:** Unchanged — no schema implemented in
code; documentation only.

---

```text
STEP 2A AND STEP 2B FINDINGS (25 TOTAL) REPAIRED IN THIS LEDGER AND IN
THE PROTOCOL DOCUMENT.

The repairing author does not self-certify this protocol.

STEP 3 CPU IMPLEMENTATION REMAINS BLOCKED UNTIL A SEPARATE INDEPENDENT
RE-AUDIT VERIFIES THE STEP 2B COMMIT.

STAGE B FULLKV QUALIFICATION, STAGE C B2A-R3 EXECUTION, ALL GPU/CUDA
ACTIVITY, B2B, AND FAITHKV METHOD IMPLEMENTATION REMAIN PROHIBITED.
```
