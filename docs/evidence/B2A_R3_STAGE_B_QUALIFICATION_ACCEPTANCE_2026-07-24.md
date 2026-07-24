# B2A-R3 Stage-B Qualification Evidence Acceptance — 2026-07-24

This document records the acceptance of the audited B2A-R3 Stage-B FullKV
qualification evidence into the repository. It is a persistence record only:
no Stage-C activity, no R-KV execution, and no production selected-row
freezer execution is authorized, described, or implied by this document.

## Authorization identity

```text
authorization ID:                stage-b-2026-07-24-r2-final
audited Stage-B code SHA:        4117baea139f745ceeff85039258445639e85049
Stage-B authorization/execution
SHA (observed_execution_commit_sha
on the consumed claim, and the
commit at which this acceptance
work is performed):              16d01ebe5c0659330bd78ccff96b9e64aea787ac
authorization document path:     docs/B2A_R3_STAGE_B_QUALIFICATION_AUTHORIZATION_2026-07-24.md
authorization document SHA-256:  e523e310abb4d2b561d32cd8f085b96574b7da87a403776ddd4bc08e2e72cfe0
```

## Consumed claim

```text
path:               results/decisions/b2a_r3_authorization_claims/stage-b-2026-07-24-r2-final.json
byte SHA-256:        cc691c4dc00fff8556d297fb4f21db42af0dc64d40f47ffe3f9129f4a109e233
canonical SHA-256:   68d055876a2260b179681fb276b79c37b6d1f987ae1899658fc969fcd05af975
attempt_id:          stage-b-2026-07-24-r2-final
attempt_directory_path:
  results/decisions/b2a_r3_attempt_20260724T063418952560Z_stage-b-2026-07-24-r2-final
authorized/observed branch:       research/b2a-r3-runtime-qualified-calibration
authorized/observed repository:   asad073-ui/Faithkv
required/observed R-KV SHA:       45eaa7d69d20b7388321f077020a610d9afb65bd
candidate_manifest_canonical_sha256:
  b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42
```

`verify-b2a-r3-candidates` and the claim's `canonical_sha256` field (verified via
`verify_canonical_sha256`) both PASS against this claim.

## Qualification artifact

```text
path:                 results/decisions/b2a_r3_qualification.json
byte SHA-256:          79480516df73aadd8239d1d927292ac357d8cb0bafc284b22e40f5ce9528d878
canonical SHA-256:     4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d
authorization_claim_canonical_sha256:
  68d055876a2260b179681fb276b79c37b6d1f987ae1899658fc969fcd05af975
candidate_manifest_canonical_sha256:
  b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42
```

`kvcot verify-b2a-r3-qualification --artifact results/decisions/b2a_r3_qualification.json`
PASSES:

```text
verify-b2a-r3-qualification: verification PASSED for results/decisions/b2a_r3_qualification.json
  selection_status=selected selected_unique_id=test/number_theory/631.json
```

## Candidate manifest binding

```text
path:                configs/discovery/b2a_r3_candidate_manifest.json
canonical SHA-256:    b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42
candidate_count:      16
level_mixture:        {"level_4": 8, "level_5": 8}
```

`kvcot verify-b2a-r3-candidates` PASSES against the committed manifest.

## Selected row

```text
selected candidate ordinal:     1
selected unique ID:             test/number_theory/631.json
qualification_stopped_reason:   first_pass
attempted_candidate_count:      2
authorized_maximum_candidates:  8
extracted answer:               36
answer_verification_status:     correct
eligible compaction events:     7
predicted_event_count:          9
projected B2B runtime:          2.7948939114811786 GPU-hours (~2.7949)
peak_cuda_allocated_bytes:      16338395648 (~15.22 GiB)
peak_cuda_reserved_bytes:       16540237824
budget:                         1024
prompt_token_count:             119
generated_token_count:          1936
fullkv_wall_seconds:            102.09455542100477
```

All 26 qualification conditions in the artifact's `attempted[1].conditions`
object evaluate to `true`, including `peak_memory_within_limit`,
`projected_runtime_within_qualification_target`, and
`runtime_predictor_version_match`.

## Frozen model / data identities (unchanged, reasserted here for binding)

```text
model:               deepseek-ai/DeepSeek-R1-Distill-Llama-8B
model revision:      6a6f4aa4197940add57724a7707d069478df56b1
tokenizer:           deepseek-ai/DeepSeek-R1-Distill-Llama-8B
tokenizer revision:  6a6f4aa4197940add57724a7707d069478df56b1
dataset:             HuggingFaceH4/MATH-500
dataset config:      default
dataset split:       test
dataset revision:    6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be
```

## CI

```text
CI run:      30072509903 ("Authorize repaired B2A-R3 Stage-B qualification", CPU tests)
head SHA:    16d01ebe5c0659330bd78ccff96b9e64aea787ac
conclusion:  success
```

## External evidence archive and independent audit

```text
Stage-B evidence archive:  /workspace/faithkv-stage-b-2026-07-24-r2-final-evidence.tar.gz
archive SHA-256:           ba3acfdb04087d540936f71abaf00c0a5ea17a03326bf503c26964af87af5c55
(verified via sha256sum -c against
 /workspace/faithkv-stage-b-2026-07-24-r2-final-evidence.tar.gz.sha256 — OK)

independent audit report:  /workspace/faithkv-stage-b-2026-07-24-r2-final-independent-audit.md
audit report SHA-256:      091c6befcc5bbb54f73d7e54ef24165922f68c2dcb8017654c3776cb75fd44ee
(verified via sha256sum -c against
 /workspace/faithkv-stage-b-2026-07-24-r2-final-independent-audit.md.sha256 — OK)
```

Accepted independent verdict:

```text
INDEPENDENT STAGE-B EVIDENCE AUDIT PASS —
B2A-R3 FULLKV QUALIFICATION ARTIFACT ACCEPTED;

SELECTED ROW:
test/number_theory/631.json

QUALIFICATION ARTIFACT CANONICAL SHA-256:
4349edc97a273819d4f5a3e75812af80437971f584071b66b25c858ffa02ff1d

READY FOR CPU-ONLY PRODUCTION SELECTED-ROW FREEZER IMPLEMENTATION;
STAGE C REMAINS BLOCKED
```

## Mandatory correction on claim timestamps

`claimed_at_utc` (`2026-07-24T06:34:18.952560+00:00` on the consumed claim) is
the external claim payload timestamp. It is **not** the canonical
atomic-consumption timestamp. The exact atomic-consumption time is not
represented by any protocol field. Filesystem timestamps on the claim or
attempt-directory paths are auxiliary evidence only and must not be treated
as authoritative for consumption ordering or exclusivity.

## Explicit non-claims

* No R-KV execution has occurred under this authorization. This document
  records FullKV-only Stage-B qualification evidence.
* Stage C has not been authorized, planned in a binding way, or executed.
* The production selected-row freezer (`kvcot freeze-b2a-r3-selected-row
  --execute`) has not been implemented against this accepted evidence and has
  not been run. The production selected manifest
  (`configs/discovery/b2a_one_example_manifest.json`) still contains the
  historical row (`test/number_theory/820.json`) and no selection provenance
  file exists at `results/decisions/b2a_r3_selection_provenance.json`.
