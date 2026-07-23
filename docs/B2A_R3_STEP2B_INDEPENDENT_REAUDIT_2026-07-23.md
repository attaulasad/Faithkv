# B2A-R3 Step 2B independent re-audit (dated 2026-07-23)

```text
INDEPENDENT STEP 2B RE-AUDIT: PASS
STEP 3 STAGE-A CPU IMPLEMENTATION AUTHORIZED
STAGE B FULLKV QUALIFICATION PROHIBITED
STAGE C B2A-R3 EXECUTION PROHIBITED
GPU / CUDA / MODEL INFERENCE PROHIBITED
```

This document records the outcome of the independent re-audit of the Step
2B repair commit (`81e11cb57202e0d4b434aabb347963ae3c34b80b`'s successor,
the Step 2B repair itself, both folded into
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` and
`docs/B2A_R3_PROTOCOL_AUDIT_REPAIR_2026-07-22.md` at this document's
starting commit, `382de26fe68206ac195ba85a56cc7f5f7696be11`). It is the
separate, genuinely independent re-audit that `CLAUDE.md` §1g, the
protocol document's §1/§22, and the repair ledger's Step 2B section all
stated was still required before Stage A (CPU implementation) could begin.
This document does not repeat the protocol's own content — it records the
audit disposition, the two independently reproduced hashes central to the
frozen predictor and qualification contract, and the four narrow
implementation clarifications Step 3 must follow. The protocol document
itself is not rewritten by this pass, except for the one narrow
explanatory correction §1 below identifies (binary64 representation of
`1.20`), which corrects prose only, not the frozen gate.

## 1. Disposition of the seven Step 2B findings (R3-AUDIT-19 .. R3-AUDIT-25)

| Finding | Original ambiguity | Repair location | Audit disposition |
|---|---|---|---|
| R3-AUDIT-19 | Qualification-condition tuple not frozen | protocol §10.5 | PASS — exact 27-name ordered tuple, full boolean definitions, rejection rules all present and internally consistent |
| R3-AUDIT-20 | Candidate manifest / outcome / qualification-artifact schemas incomplete | protocol §12.3, §12.5, §12.6 | PASS — every field referenced elsewhere in the protocol (identity echoes, hash pairs, placement/timing evidence) is now named, typed, and required/nullable |
| R3-AUDIT-21 | Selected-manifest hash semantics unresolved | protocol §12.1, §12.7, §13 | PASS — `selected_manifest_sha256 = B2AOneExampleManifest.manifest_hash()`, never a new field on the historical manifest, never `canonical_sha256` |
| R3-AUDIT-22 | Candidate-row schema insufficient for deterministic freezing | protocol §12.4, §13 | PASS — embed-not-refetch rule frozen, three verification formulas frozen, exact embedded-row columns frozen |
| R3-AUDIT-23 | Markdown authorization-document hash vs. JSON self-hash rule | protocol §12.1, §12.8, §14.4 | PASS, with one narrow correction (§2 below) |
| R3-AUDIT-24 | Authorization-claim consumption was scan-then-write | protocol §14.4 | PASS — replaced with a globally exclusive, atomically created claim at one deterministic path; creation IS consumption |
| R3-AUDIT-25 | Protocol-identity fields conflated under one `protocol_version` | protocol §12.9, and per-schema | PASS — four (five, counting the in-memory `AttemptProvenancePolicy`) separately named, separately versioned identity fields, no bare `protocol_version` in any new schema |

All seven findings are independently confirmed repaired. No eighth defect
was found that would block Stage A. Stage B, Stage C, GPU/CUDA activity,
B2B, and FaithKV method implementation remain independently confirmed
prohibited by the protocol text itself (§14, §19) — this audit changes
none of those boundaries.

## 2. One narrow explanatory correction (not a gate change)

The protocol's §10.5.4 `safety_multiplier_exact` prose states that `1.20`
"is exactly representable in IEEE-754 binary64 as literal `1.2`." That
sentence is imprecise: `1.2` (and `1.20`, the same value) is a *repeating*
binary fraction and is **not** exactly representable in binary64 — what
*is* exactly reproducible is comparing the literal Python float `1.2`
against itself (`==`), because both sides round to the identical nearest
representable double. The gate itself (`safety_multiplier == 1.20`) is
unaffected and is not changed by this correction — Step 3 must implement
it as exact equality against the same frozen Python float literal
(`SAFETY_MULTIPLIER = 1.20`), never `math.isclose` or an epsilon
tolerance. Only the explanatory sentence is corrected; see
`kvcot.discovery.b2a_r3_contract`'s module docstring for the corrected
wording actually shipped in code.

## 3. Independently reproduced hashes

Both hashes central to the frozen predictor/qualification contract were
independently recomputed against the actual committed repository files at
this document's starting commit, using the repository's own
`kvcot.utils.hashing.sha256_file`/`sha256_json` (never a second,
independently-defined hash function):

```text
runtime_source_artifact_sha256
    file: docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json
    sha256_file(...) = f9eff6c1785df69406309157a6921a6fa7729dfb4fb5bfe43fa3401e0691443e
    MATCHES the value frozen in protocol §12.6.

generation_config_sha256
    payload: the exact frozen object in protocol §10.5 (`generation_mode`,
    `do_sample`, `temperature`, `top_p`, `batch_size`, `max_new_tokens`,
    `framework_seed`, `attention_backend`, `cache_implementation`,
    `no_offload_required`)
    sha256_json(...) = b67ed818bb94e0a674e01c1400caf2e005c41c12203d6a29b0f0f5c69eef01a3
    MATCHES the value frozen in protocol §10.5 and echoed in §12.3/§12.6.

exclusion_set_sha256
    payload: the exact frozen 13-line UTF-8 payload in protocol §8.1
    sha256_text(...) = 0c46510c79a22d08e8fd610104a527e867f821a540063a5059b51a660d25bc69
    MATCHES EXCLUSION_SET_SHA256 frozen in protocol §8.1.

runtime predictor integer boundary (protocol §7.5)
    projected_gpu_hours(candidate_total_tokens=2775) = 3.5999041059674...   -> PASS
    projected_gpu_hours(candidate_total_tokens=2776) = 3.6010221756819636  -> FAIL
    both independently recomputed from §7.3's exact constants and MATCH
    the frozen expectation to full double-precision agreement.
```

None of these four values were taken on faith from the protocol document;
each was independently recomputed from either the committed repository
file or the frozen constant table and compared byte-for-byte / to full
double precision.

## 4. Four mandatory implementation clarifications

These four clarifications are binding on the Step 3 Stage-A implementation
and are restated (not redefined) from the task authorization; the
authoritative rule in each case remains the protocol document, this
section only makes the resolution explicit and points at where the code
implements it.

1. **Python float `1.20`.** See §2 above. Gate implemented as exact
   `==` against the module constant `SAFETY_MULTIPLIER`, never a
   tolerance comparison.
2. **One cap-hit source.** The qualification evidence and outcome schemas
   store `cap_hit` only. No second, independently-named
   `generation_cap_hit` boolean field exists anywhere in the new schemas;
   where the protocol's prose uses that name (§10.1's `trace_complete`
   formula), the implementation binds it to the same `cap_hit` value via a
   local alias only, never a second stored field.
3. **Authorization-document hash.** The (not-yet-created) dated Markdown
   authorization documents hash via
   `authorization_document_sha256 = sha256_file(path)` — a plain
   whole-file byte hash. The JSON `canonical_sha256`/`sha256_json`
   self-hash rule is never applied to a Markdown file.
4. **Post-claim worktree allowlist.** Once a (future, not-yet-created)
   authorization claim exists, the only two worktree paths a provenance
   verifier may treat as expected are the exact active global claim path
   and the exact active attempt-directory root — implemented and tested
   only against synthetic fixtures in this Stage-A pass; no real claim is
   ever created by this pass.

## 5. Scope of this authorization

This document authorizes exactly what CLAUDE.md §1e/§1f/§1g and the
protocol's §14.1 already described as Stage A:

- CPU-only Stage-A implementation: strict contracts/schemas, the
  deterministic candidate generator (including one real, committed
  candidate manifest built from the pinned MATH-500 dataset's already-
  public content), the pure runtime predictor, the pure qualification
  evaluator (exercised only against synthetic/injected evidence), artifact
  verifiers, a synthetic-only selected-row freezer, the atomic
  authorization-claim mechanism (exercised only against synthetic
  fixtures, no real claim created), a separately-constructed
  `AttemptProvenancePolicy`, and CPU-only dry-run/verification CLI
  commands.
- CPU unit and integration tests for all of the above.

This document does **not** authorize, and no line item in it may be read
as authorizing: Stage B (FullKV qualification against real weights),
Stage C (B2A-R3 execution), any CUDA initialization, any R-KV import
outside its historical B2A-R1/R2 role, a real qualification artifact, a
real selected-manifest replacement, a real authorization claim, B2B, or
any FaithKV method implementation.

Remote CI for the documentation-only starting commit
(`382de26fe68206ac195ba85a56cc7f5f7696be11`) was **not** independently
observed as part of this audit — this document makes no claim about a
green or red remote CI run at that commit; it is not evidence this audit
relies on.

```text
This document authorizes CPU implementation only.
It is not a Stage-B authorization.
It is not a Stage-C authorization.
No GPU authorization exists.
```
