# B2A-R3 Production Selected-Row Freezer Implementation (dated 2026-07-24)

## 1. Status

```text
B2A-R3 PRODUCTION SELECTED-ROW FREEZER IMPLEMENTATION COMPLETE --
CPU-ONLY, CPU TESTS GREEN;
PRODUCTION FREEZER NOT EXECUTED;
SELECTED MANIFEST NOT UPDATED;
SELECTION PROVENANCE NOT CREATED;
STAGE C REMAINS BLOCKED
```

Implements, under
`docs/B2A_R3_PRODUCTION_SELECTED_ROW_FREEZER_IMPLEMENTATION_AUTHORIZATION_2026-07-24.md`,
the production selected-row freezer against the accepted Stage-B evidence
(authorization ID `stage-b-2026-07-24-r2-final`; candidate-manifest
canonical SHA-256
`b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42`;
consumed-claim canonical SHA-256
`68d055876a2260b179681fb276b79c37b6d1f987ae1899658fc969fcd05af975`;
qualification-artifact canonical SHA-256
`4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d`;
selected candidate ordinal 1, unique ID `test/number_theory/631.json`).
This is an implementation-and-test round only -- the freezer is never
executed by this work, the production selected manifest still contains
the historical row (`test/number_theory/820.json`), and no selection
provenance file exists.

## 2. Exact local tokenizer boundary

New module: `src/kvcot/discovery/b2a_r3_production_tokenizer.py`.
Deliberately kept out of `b2a_r3_freeze.py` -- that module's own AST-level
import-safety test statically forbids `transformers`/`torch` anywhere in
its source, so the one place a real tokenizer is loaded had to live
elsewhere.

- `resolve_production_tokenizer_snapshot()` calls
  `kvcot.discovery.snapshot_boundary.resolve_local_snapshot(repository_id=
  TOKENIZER_NAME, revision=TOKENIZER_REVISION, asset_type="tokenizer")` and
  requires `requested_revision == resolved_revision == TOKENIZER_REVISION`,
  `asset_type == "tokenizer"`, and `local_files_only is True` -- any
  `SnapshotBoundaryError` (including the resolver's own "network/floating
  fallback is forbidden" refusal) is re-raised as
  `ProductionTokenizerResolutionRefused`, never silently retried over the
  network.
- `render_production_prompt(row, snapshot=...)` loads
  `AutoTokenizer.from_pretrained(snapshot.local_path, local_files_only=True,
  use_fast=True)`, requires a non-empty `chat_template`, and renders via
  `kvcot.discovery.manifest_prepare.render_with_loaded_tokenizer` -- the
  SAME frozen chat-template call every other B2A prompt-identity path
  already uses (`{"role": "user", ...}`, `tokenize=True`,
  `add_generation_prompt=True`), never a second, independently-invented
  convention. No `AutoModel` import, no direct `torch` import, no CUDA
  inspection, no dataset fetch (the row is always the caller's already-
  resolved embedded row).
- `build_production_tokenizer_renderer()` resolves the snapshot once,
  eagerly, and returns a closure implementing
  `kvcot.discovery.b2a_r3_freeze.TokenizerRenderer` -- injected into
  `construct_selected_manifest_and_provenance` exactly like a test's fake
  renderer.

## 3. Fixed production paths

Unchanged, exactly as authorized:

- candidate manifest: `configs/discovery/b2a_r3_candidate_manifest.json`
- qualification artifact: `results/decisions/b2a_r3_qualification.json`
- selected manifest: `configs/discovery/b2a_one_example_manifest.json`
- selection provenance:
  `results/decisions/b2a_r3_selection_provenance.json`

`kvcot freeze-b2a-r3-selected-row --execute` never accepts
`--candidates`/`--artifact`/`--config` (rejected outright, even if the
supplied value equals the fixed path) and has no
`--output`/`--manifest-output`/`--provenance-output`/`--force` flag.
`--dry-run` is unaffected and keeps its existing read-only overrides.

## 4. Production freeze-plan construction

`kvcot.discovery.b2a_r3_freeze.construct_production_freeze_plan` builds a
frozen `ProductionFreezePlan` dataclass entirely in memory, in the
required order:

1. Verify persisted Stage-B authorization binding
   (`verify_persisted_stage_b_authorization_binding`).
2. Verify candidate manifest / 3. Verify qualification artifact / 4.
   Replay selected-row chain -- all three delegated to the existing
   `construct_selected_manifest_and_provenance` (unchanged Stage-A code),
   so no second, independently-written verification path exists.
5. Resolve exact local tokenizer -- performed by the caller
   (`resolve_production_tokenizer_snapshot`) before construction begins;
   the resolved identity is bound directly onto the plan.
6. Render exact prompt -- via the injected `tokenizer_renderer`, inside
   step 7/8's call.
7. Construct selected manifest / 8. Construct provenance -- same call.
9. Verify the expected complete chain in memory
   (`verify_selection_provenance`, called BEFORE any write).

The plan binds: repository root, current exact Git SHA, both input paths
and their canonical hashes, both output paths, selected ordinal/unique
ID, the historical selected-manifest's exact Git blob SHA and committed
byte hash (read from git history at the current commit, never from
whatever happens to be on disk -- fixed during implementation, see §8),
the expected new manifest/provenance bytes and hashes, the tokenizer
repository/revision/resolved-snapshot identity, and the publication state
classified at construction time. No filesystem publication happens before
all nine steps succeed.

## 5. Publication state machine (States A-E)

`classify_publication_state` reads only the plan's frozen evidence and the
CURRENT bytes at the two fixed output paths:

- **State A (initial):** live manifest == committed historical bytes,
  provenance absent.
- **State B (complete):** manifest == expected new bytes, provenance ==
  expected bytes. `publish_production_freeze` returns success without
  rewriting either file (verified idempotent: unchanged mtimes and byte
  content on a second call).
- **State C (provenance-first partial, recoverable but not the normal
  order):** manifest still historical, provenance already == expected.
  Publishes only the manifest, then runs full verification.
- **State D (manifest-first partial, recoverable):** manifest == expected
  new bytes, provenance absent. Publishes only the provenance, then runs
  full verification.
- **State E (invalid/ambiguous):** anything else -- unknown manifest
  bytes, unknown provenance bytes, a missing manifest, etc. Refused
  without deleting or overwriting any target.

Normal order (State A): atomically replace the historical manifest with
the expected new manifest (verified byte-for-byte before and after),
publish the provenance no-clobber, then run `verify_selection_provenance`
against the freshly reloaded files. Full-chain verification is required
once, only after both outputs are confirmed present -- never demanded
while intentionally in a one-file partial state.

## 6. Publication mechanics

- Every write: temp file in the target directory, write, flush, fsync,
  byte-verify the temp file, publish, fsync the parent directory where
  supported, reload, require exact payload equality.
- Historical-manifest replacement (`_atomic_replace_verified_historical`)
  uses `os.replace` -- but ONLY after confirming the live bytes on disk
  exactly equal the expected committed historical bytes; otherwise it
  refuses without touching the file.
- Provenance publication (`_exclusive_publish_no_clobber`) is genuinely
  no-clobber: `os.link(temp_path, provenance_path)`, `FileExistsError` is
  a hard refusal (never caught-and-retried), the temp path is always
  cleaned up on any exception path. `os.replace` is never used for
  provenance.
- A synchronous failure mid-write (tested by forcing `os.fsync` to raise)
  leaves both targets byte-identical to their pre-call state and leaves no
  leftover temp file.

## 7. Git/worktree safety

`verify_git_worktree_safety_for_freeze` runs before any write: exact
repository identity, exact branch (bound to the Stage-B claim's own
`authorized_branch`, never a second hard-coded constant), exact current
commit SHA unchanged since plan construction, exact R-KV pin (bound to the
claim's `required_rkv_sha`), and an allowlist over `git status` dirty
paths containing only the two fixed output paths themselves -- any other
tracked or untracked change (an altered candidate manifest, qualification
artifact, unrelated file, wrong R-KV gitlink, etc.) is rejected as an
"unrelated worktree change."

## 8. A construction bug found and fixed during implementation

The first draft read `historical_selected_manifest_bytes` from whatever
was currently on disk at `selected_manifest_path`. After a successful
freeze, rebuilding the plan (e.g. for the idempotent State-B check, or a
recovery attempt) then saw the ALREADY-NEW manifest bytes and
misclassified State D as State A. Fixed to always read the historical
bytes from git history at the current commit
(`git_state.file_text_at_commit(SELECTED_MANIFEST_PATH, current_git_sha)`)
-- stable regardless of what the freezer itself has already written to
disk, since publication never commits its own writes. Caught by an
end-to-end smoke test before the formal test suite was written; the fixed
behavior is covered by
`test_publish_production_freeze_state_b_is_idempotent_no_rewrite` and the
State-C/D recovery tests.

## 9. CLI

`kvcot freeze-b2a-r3-selected-row` requires exactly one of `--dry-run` /
`--execute` (both or neither is a hard `SystemExit`). `--dry-run` is
unchanged (no snapshot resolution, no tokenizer load, no write). `--execute`
resolves the tokenizer snapshot, constructs the plan, classifies and
publishes, and prints: `selected_unique_id`, `selected_ordinal`,
`selected_manifest_path`, `selected_manifest_sha256`,
`selection_provenance_path`, `selection_provenance_canonical_sha256`,
`tokenizer_snapshot_revision`, `publication_state_before`,
`publication_state_after`, `already_frozen`, `verification_passed`.
Returns non-zero if verification did not pass.

## 10. Tests

Extended `tests/unit/discovery/test_b2a_r3_freeze.py` (50 new Phase-2
tests: plan construction, all five publication states including two
distinct State-E shapes, no-clobber/atomic-replace unit tests, a
synchronous-failure-leaves-targets-unchanged test, Git/worktree safety
(allowed paths, unrelated file, altered candidate manifest, wrong R-KV
pin), evidence-chain tamper rejections, mandatory full-chain verification,
and a tokenizer-boundary section that runs against the real local
snapshot when cached and skips cleanly (verified against an empty cache
directory) when it is not -- CI runners have no model cache) and
`tests/unit/test_cli_b2a_r3.py` (mode-exclusivity, alternate-path
rejection including the "even if it equals the fixed path" case, no
`--output`/`--force`-shaped flags, delegation with monkeypatched
production functions, non-zero return on verification failure, and a
read-only integration test against the real committed candidate
manifest/qualification artifact confirming ordinal 1 /
`test/number_theory/631.json` / `would_freeze=True` with no production
writes as a side effect).

All state-machine and plan-construction tests run against a real,
temporary, throwaway git repository (never the actual repository) built
the same way `test_b2a_r3_authorization.py`'s own end-to-end persisted-
binding test builds its fixture -- real committed config bytes, the real
committed candidate manifest content, a coordinator-produced qualification
artifact driven by a fake `fullkv_worker_runner` (no torch, no CUDA, no
real FullKV inference).

## 11. What remains

Stage C, FullKV, R-KV, and B2B remain fully blocked. `--execute` was not
invoked against the real repository at any point during this
implementation round -- the production selected manifest still contains
`test/number_theory/820.json`, and
`results/decisions/b2a_r3_selection_provenance.json` does not exist. The
next required action is an independent audit of the exact implementation
SHA; only after that audit passes may a separate task run
`kvcot freeze-b2a-r3-selected-row --execute`.
