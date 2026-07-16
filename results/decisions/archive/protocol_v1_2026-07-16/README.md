# Archived protocol-v1 fixed-trace decisions (2026-07-16)

These four files were produced by the **fixed-trace protocol v1** GPU run
(`git_commit bb1917a`, `schema_version "1.1.0"`) and moved here from
`results/decisions/` on 2026-07-16, alongside the protocol v2 fix
(`CHANGELOG.md`'s 2026-07-16 "Fixed-trace protocol v2" entry).

**Do not use these as evidence for or against the G1 hypothesis.** Protocol
v1 had two independent defects — an empty fixed-trace suffix that made the
f=1 anchor almost always fall back to a noisy mid-sentence number, and an
eligibility gate keyed on a recorded compaction *event count* rather than
realized (physical-cache) compression — that together produced `n_eligible
= 0` at every budget tested. That is zero scientific information about the
hypothesis, not a negative result.

They are kept (not deleted) purely as a provenance record of what protocol
v1 actually output, and because `early_gap_b512_accuracy_compaction.json`'s
raw per-example retention/compaction numbers were the basis for choosing
`configs/early_gap_v2_b128.yaml`'s budget under protocol v2 (see that
config's own header comment).

`kvcot.analysis.fixed_trace.run_fixed_trace_analysis` now validates every
input record's schema (`schema_version` is a Pydantic `Literal`, currently
`"1.3.0"` — see `docs/SCHEMA.md`'s Versioning section for the current value
and full history) and (config, model, upstream-commit) identity before
analyzing anything, so these files — or any other protocol-v1/mismatched-run
data — cannot be silently re-read as if they were current protocol-v2
output.
