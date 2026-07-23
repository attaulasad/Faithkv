# B2A-R3 Stage-B Qualification Authorization (dated 2026-07-23)

This document authorizes exactly one B2A-R3 Stage-B FullKV qualification
claim at the clean execution commit that contains this document.

Authority is limited to the committed B2A-R3 Stage-B FullKV qualification
path. This document does not authorize Stage-C B2A-R3 execution, B2B
execution, any R-KV worker execution, CUDA activity outside the Stage-B
FullKV qualification path, any model/config/dataset/manifest change, any
claim JSON committed into the repository, or any FaithKV method
implementation.

The audited CPU implementation commit is:

```text
d6bf377c3f694feb7cd012f9f3522615740bcddd
```

The authorization verifier must reject use unless the current clean `HEAD`
is the later execution commit that contains this exact document, the audited
code commit above is its ancestor, and the diff from that code commit to
the execution commit contains only this file.

<!-- BEGIN B2A-R3 AUTHORIZATION JSON -->
```json
{
  "authorization_document_schema_version": "faithkv-b2a-r3-stage-authorization-document-v2",
  "authorization_id": "stage-b-2026-07-23",
  "authorization_stage": "fullkv_qualification",
  "authorized_repository": "asad073-ui/Faithkv",
  "authorized_branch": "research/b2a-r3-runtime-qualified-calibration",
  "authorized_code_commit_sha": "d6bf377c3f694feb7cd012f9f3522615740bcddd",
  "required_ancestor_shas": [],
  "required_rkv_sha": "45eaa7d69d20b7388321f077020a610d9afb65bd",
  "candidate_manifest_canonical_sha256": "b8148647698ca5ab5335ea28dc1416109b26f73dd05b87eed2fe9eca4b25ff42",
  "maximum_candidates": 8,
  "phase_wall_time_limit_seconds": 57600,
  "qualification_artifact_canonical_sha256": null,
  "selected_manifest_sha256": null,
  "selected_manifest_hash_algorithm": null,
  "created_at_utc": "2026-07-23T12:55:00+00:00"
}
```
<!-- END B2A-R3 AUTHORIZATION JSON -->
