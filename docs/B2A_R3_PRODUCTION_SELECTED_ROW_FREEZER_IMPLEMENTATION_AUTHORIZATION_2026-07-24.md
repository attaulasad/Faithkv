# B2A-R3 Production Selected-Row Freezer Implementation Authorization (dated 2026-07-24)

## 1. Status

```text
B2A-R3 PRODUCTION SELECTED-ROW FREEZER IMPLEMENTATION AUTHORIZED —
CPU-ONLY IMPLEMENTATION AND CPU TESTS

FREEZER EXECUTION: PROHIBITED
STAGE C: PROHIBITED
FULLKV / R-KV / B2B: PROHIBITED
MODEL LOADING / CUDA: PROHIBITED (tokenizer-only, CPU, local snapshot)
```

## 2. Binding preconditions (independently verified before this document was written)

- Repository: `asad073-ui/Faithkv`.
- Branch: `research/b2a-r3-runtime-qualified-calibration`.
- Starting `HEAD` (local and remote, verified equal):
  `87b995c90a317863b2d3b44bbc345018ae9356b6`
  ("Repair Stage-B evidence-acceptance test-assumption CI failure").
- Exact-SHA CI run `30076024867`: `headSha` equals the starting SHA above,
  `status=completed`, `conclusion=success`.
- R-KV submodule pin: `45eaa7d69d20b7388321f077020a610d9afb65bd`.
- Accepted Stage-B authorization ID: `stage-b-2026-07-24-r2-final`.
- Accepted candidate-manifest canonical SHA-256:
  `b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`.
- Accepted consumed-claim canonical SHA-256:
  `68d055876a2260b179681fb276b79c37b6d1f987ae1899658fc969fcd05af975`.
- Accepted qualification-artifact canonical SHA-256:
  `4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d`.
- Selected candidate ordinal: `1`.
- Selected unique ID: `test/number_theory/631.json`.
- Historical production selected-manifest row (unchanged as of this
  document): `test/number_theory/820.json`
  (`configs/discovery/b2a_one_example_manifest.json`).
- Selection provenance (`results/decisions/b2a_r3_selection_provenance.json`)
  confirmed absent.
- `kvcot verify-b2a-r3-candidates`, `kvcot verify-b2a-r3-qualification`, and
  `kvcot freeze-b2a-r3-selected-row --dry-run` all pass with
  `selected_ordinal=1`, `selected_unique_id=test/number_theory/631.json`,
  `would_freeze=True`, `would_load_tokenizer_for_execution=False`,
  `would_write_selected_manifest=False`,
  `would_write_selection_provenance=False`.
- Frozen model/tokenizer identity:
  `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` at revision
  `6a6f4aa4197940add57724a7707d069478df56b1` (unchanged from
  `CLAUDE.md` §4c; this authorization touches only the tokenizer side of
  this identity, never the model).
- Fixed production paths (unchanged):
  - candidate manifest: `configs/discovery/b2a_r3_candidate_manifest.json`
  - qualification artifact: `results/decisions/b2a_r3_qualification.json`
  - selected manifest: `configs/discovery/b2a_one_example_manifest.json`
  - selection provenance:
    `results/decisions/b2a_r3_selection_provenance.json`

Any of the above differing from the live repository state at the moment
this document's binding is checked invalidates this authorization; the
authorization verifier must fail closed rather than proceed.

## 3. Scope authorized

CPU-only implementation, against the accepted evidence bound in §2, of:

- A production tokenizer renderer that resolves the exact local tokenizer
  snapshot via the existing `resolve_local_snapshot` boundary
  (`asset_type="tokenizer"`, `local_files_only=True`, exact revision
  match, no network fallback) and renders the selected row's prompt using
  the existing `render_base_user_message` / chat-template convention —
  never a second prompt-rendering convention.
- Fixed-path production freeze-plan construction: an in-memory, typed plan
  binding repository root, current Git SHA, the four fixed production
  paths and their hashes/identities, the selected ordinal/unique ID, the
  historical selected-manifest identity, the resolved tokenizer snapshot,
  and the expected new manifest/provenance payloads and hashes —
  constructed and internally verified in full before any filesystem
  write.
- A guarded publication state machine (States A-E, §7 of the governing
  task) that classifies the live production output state before writing
  anything, performs the normal A publication order (manifest, then
  no-clobber provenance, then full-chain verification) only from State A,
  recovers States C and D without ever double-writing a verified partial,
  treats State B as a no-op success, and refuses State E without deleting
  or overwriting any target.
- Production CLI `--execute` wiring for the existing
  `kvcot freeze-b2a-r3-selected-row` command, exclusively exactly one of
  `--dry-run`/`--execute`, fixed production paths only, no
  `--output`/`--manifest-output`/`--provenance-output`/`--force` flags.
- Git/worktree safety guards required before any execute-mode publication
  (clean tracked state, unchanged candidate manifest/qualification
  artifact/consumed claim/R-KV pin, no unrelated untracked files in
  production output directories).
- CPU tests for all of the above (tokenizer boundary, path security,
  evidence-chain rejection, historical guard, state-machine recovery and
  refusal, CLI mode handling, and a read-only integration test against the
  real committed artifacts).
- Documentation of this implementation.

## 4. Scope prohibited

- Executing the production freezer (`--execute`) in this phase.
- Modifying the production selected manifest
  (`configs/discovery/b2a_one_example_manifest.json`).
- Creating `results/decisions/b2a_r3_selection_provenance.json`.
- Loading a real model, `AutoModel`, or any model class.
- Any direct `torch` import, CUDA inspection, or CUDA initialization.
- Dataset fetching or network access beyond GitHub push/CI inspection.
- Stage C B2A-R3 execution, FullKV, R-KV, or B2B execution.
- Any change to a scientific setting: model/tokenizer identity, dataset
  identity, budgets, thresholds, seeds, or any value frozen by `CLAUDE.md`
  §4/§4a-§4c or `configs/lock.yaml`.
- Any change to `configs/discovery/b2a_r3_candidate_manifest.json`,
  `results/decisions/b2a_r3_qualification.json`,
  `results/decisions/b2a_r3_authorization_claims/stage-b-2026-07-24-r2-final.json`,
  or the R-KV gitlink/pin.
- Merging this branch.

## 5. Implementation-authorization document lifecycle

Unlike a Stage-B authorization claim, this document does not gate a
single-use external claim — it is a scope boundary for a CPU
implementation-and-test round, committed separately from that round's own
code changes so the round's diff can be audited against an exact starting
SHA. The bound starting SHA for the implementation round is this
document's own commit SHA once committed (recorded in `CHANGELOG.md` and
`PLAN.md` as `FREEZER_AUTH_SHA`). Implementation must not begin against
production code before exact-SHA CPU CI is green for that commit.

## 6. Next required action

```text
Push this authorization commit. Confirm exact-SHA CPU CI success for
FREEZER_AUTH_SHA. Only then implement the production selected-row
freezer exactly within the scope of §3, never executing it, against the
fixed production paths and accepted evidence bound in §2.
```
