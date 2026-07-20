"""The real, one-example B2A GPU calibration run (B1B-R4 §8-§12/§14/§16/§21,
superseding B1B-R3's `docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md`).

**Never invoked in this pass.** `kvcot.cli.cmd_b2a_calibrate`'s `--execute`
mode is the only caller, and it hard-stops on CPU-checkable preconditions
(CUDA required) before `run_b2a_calibration` is ever reached.

## Coordinator / worker split (B1B-R3 §11, worker bodies moved to
## `kvcot.discovery.b2a_workers` by B1B-R4 §19)

`run_b2a_calibration` (the coordinator, called by the CLI) never loads a
model itself -- it verifies the prompt identity, then delegates to
`kvcot.discovery.b2a_workers.run_both_workers_via_subprocess`, which
launches the FullKV and R-KV workers as SEPARATE OS processes. The two
canonical worker bodies (`run_fullkv_worker`/`run_rkv_worker`) now live
entirely in `kvcot.discovery.b2a_workers` (B1B-R4 §19: one canonical
worker API, removing the B1B-R3 split where `run_rkv_worker_body` lived
here while `b2a_workers.run_rkv_worker` was a misleading
`NotImplementedError` stub).

## B1B-R4 repair summary (this coordinator's responsibilities)

- Every gate-evidence field the coordinator builds comes from an
  INDEPENDENTLY-reported worker field -- never a literal `True`, never
  `rkv.example_valid` reused for five different conditions (§8/§9/§10).
- Timing/projection uses the frozen §12 formula: `per_example_total =
  fullkv.wall_seconds + rkv.wall_seconds_pass1 + rkv.wall_seconds_pass2`,
  `per_real_pair_seconds = max(rkv.real_pair_wall_seconds)` -- never an
  aggregate branch-time bucket multiplied by 144.
- VRAM gates on `max(peak_allocated, peak_reserved)` across BOTH workers
  (§14).
- Partial FullKV evidence survives an R-KV failure into the fail artifact
  (§16) -- `WorkerFailedError.partial_fullkv_result` is folded in, never
  discarded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class B2AExecutionRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class B2ACalibrationArtifact:
    config_hash: str
    manifest_hash: str
    gate_result: Any  # kvcot.discovery.b2a_contract.B2AGateResult
    artifact_path: Path


def _verify_resolved_prompt_identity(config, manifest) -> None:
    """B1B-R3 §6: re-render, re-tokenize, and re-hash the frozen prompt
    from the manifest's own dataset row and the config's pinned tokenizer,
    and compare EVERY stored identity field before any model is loaded.
    Reuses `kvcot.discovery.manifest_prepare`'s own fetch/render/tokenize
    functions directly -- never a second, independently-written
    verification path that could silently drift from what
    `prepare-b2a-manifest` itself computed. B1B-R4 §15: this call path
    never inspects or rejects pre-existing model weights -- the weight-cache
    guard lives only around `manifest_prepare.resolve_prompt_identity`'s own
    call site, never here."""
    from kvcot.discovery.manifest_prepare import _fetch_pinned_dataset_row, _render_and_tokenize, _verify_row_schema
    from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text

    if not manifest.prompt_identity_is_resolved:
        raise B2AExecutionRefused(
            "manifest prompt-token identity is unresolved -- run `kvcot prepare-b2a-manifest --execute` first."
        )
    if config.dataset.revision != manifest.dataset_revision:
        raise B2AExecutionRefused("config dataset.revision does not match manifest.dataset_revision.")
    if config.model.tokenizer_revision != manifest.tokenizer_revision_used_for_prompt_hash:
        raise B2AExecutionRefused(
            f"config.model.tokenizer_revision ({config.model.tokenizer_revision!r}) does not match "
            f"manifest.tokenizer_revision_used_for_prompt_hash ({manifest.tokenizer_revision_used_for_prompt_hash!r})."
        )

    fetched = _fetch_pinned_dataset_row(manifest.dataset_repo, manifest.dataset_revision, manifest.example_index)
    _verify_row_schema(fetched.row)
    if fetched.raw_content_hash != manifest.raw_content_hash:
        raise B2AExecutionRefused(
            f"re-fetched dataset row hash {fetched.raw_content_hash!r} does not match "
            f"manifest.raw_content_hash {manifest.raw_content_hash!r}."
        )
    if fetched.row["unique_id"] != manifest.unique_id:
        raise B2AExecutionRefused(
            f"re-fetched row unique_id {fetched.row['unique_id']!r} does not match manifest.unique_id "
            f"{manifest.unique_id!r}."
        )
    if fetched.row["answer"] != manifest.gold_answer:
        raise B2AExecutionRefused("re-fetched row answer does not match manifest.gold_answer.")

    tokenizer, user_message, messages, token_ids = _render_and_tokenize(
        fetched.row, config.model.tokenizer_name, config.model.tokenizer_revision
    )

    checks = (
        ("rendered_user_message_sha256", sha256_text(user_message), manifest.rendered_user_message_sha256),
        ("chat_template_source_sha256", sha256_text(tokenizer.chat_template), manifest.chat_template_source_sha256),
        ("chat_message_payload_sha256", sha256_json(messages), manifest.chat_message_payload_sha256),
        ("prompt_token_ids_sha256", sha256_int_ids(token_ids), manifest.prompt_token_ids_sha256),
        ("prompt_token_count", len(token_ids), manifest.prompt_token_count),
    )
    for field_name, recomputed, frozen in checks:
        if recomputed != frozen:
            raise B2AExecutionRefused(f"prompt identity mismatch on {field_name!r}: recomputed={recomputed!r} frozen={frozen!r}")
    if manifest.prompt_token_ids is not None and list(token_ids) != list(manifest.prompt_token_ids):
        raise B2AExecutionRefused("recomputed prompt_token_ids array does not match manifest.prompt_token_ids.")


def _build_fail_artifact_payload(
    config,
    manifest,
    config_hash: str,
    manifest_hash: str,
    exc: Exception,
    *,
    partial_fullkv_result: Any = None,
) -> dict:
    """B1B-R4 §16: if a partial FullKV result survived the failure (the
    R-KV worker failed AFTER FullKV already succeeded), it is folded into
    the fail artifact -- never discarded just because the overall gate
    could not be evaluated."""
    import traceback

    payload = {
        "passed": False,
        "config_hash": config_hash,
        "manifest_hash": manifest_hash,
        "failure_reason": f"{type(exc).__name__}: {exc}",
        "failure_traceback": traceback.format_exc(),
        "model": {"name": config.model.name, "revision": config.model.revision},
        "dataset": {"repo": manifest.dataset_repo, "revision": manifest.dataset_revision},
        "one_example_only": True,
        "timed_out": bool(getattr(exc, "timed_out", False)),
    }
    if partial_fullkv_result is not None:
        payload["partial_fullkv_worker"] = partial_fullkv_result.model_dump(mode="json")
    return payload


def run_b2a_calibration(
    config,
    manifest,
    *,
    config_path: str,
    manifest_path: str,
    python_executable: str | None = None,
    subprocess_runner=None,
) -> B2ACalibrationArtifact:
    """The coordinator (B1B-R3 §11, repaired B1B-R4 §8-§12/§14/§16/§21) --
    called by `kvcot.cli.cmd_b2a_calibrate --execute` (which has already
    re-checked CUDA availability and every CPU-checkable precondition
    before calling this). Never loads a model itself: verifies the prompt
    identity, then launches the FullKV and R-KV workers as separate
    subprocesses via `kvcot.discovery.b2a_workers
    .run_both_workers_via_subprocess`, combines their results into REAL
    (never literal-`True`) gate evidence, evaluates the gate, and writes an
    immutable artifact -- ALWAYS, whether the gate passes, fails, or a
    worker/verification step raises before either worker even runs."""
    import subprocess as subprocess_module

    from kvcot.discovery.b2a_artifact import build_and_write_b2a_artifact
    from kvcot.discovery.b2a_contract import (
        B2AOneExampleMeasurement,
        build_gate_evidence_from_measurement,
        evaluate_b2a_gate,
    )
    from kvcot.discovery.b2a_evidence import (
        derive_meaningful_compression_observed,
        per_real_pair_projection_seconds,
        project_complete_pilot_gpu_hours,
    )
    from kvcot.discovery.b2a_workers import WorkerFailedError, run_both_workers_via_subprocess
    from kvcot.discovery.constants import (
        B2A_NOOP_PAIR_EVALUATIONS_TOTAL,
        B2A_REAL_PAIR_EVALUATIONS_TOTAL,
        B2A_SELECTED_EVENTS,
    )
    from kvcot.discovery.discovery_config import canonical_config_hash

    subprocess_runner = subprocess_runner or subprocess_module.run
    config_hash = canonical_config_hash(config)
    manifest_hash = manifest.manifest_hash()

    try:
        _verify_resolved_prompt_identity(config, manifest)

        coordination = run_both_workers_via_subprocess(
            config_path, manifest_path, python_executable=python_executable, subprocess_runner=subprocess_runner,
        )

        fullkv = coordination.fullkv
        rkv = coordination.rkv

        per_example_total = fullkv.wall_seconds + rkv.wall_seconds_pass1 + rkv.wall_seconds_pass2
        per_real_pair_seconds = per_real_pair_projection_seconds(tuple(rkv.real_pair_wall_seconds))
        projected_gpu_hours = project_complete_pilot_gpu_hours(
            per_example_total_seconds=per_example_total, per_real_pair_seconds=per_real_pair_seconds,
        )

        measurement = B2AOneExampleMeasurement(
            fullkv_natural_generation_wall_seconds=fullkv.wall_seconds,
            rkv_pass1_wall_seconds=rkv.wall_seconds_pass1,
            rkv_pass2_wall_seconds=rkv.wall_seconds_pass2,
            targeted_capture_wall_seconds=rkv.wall_seconds_targeted_capture,
            per_example_total_wall_seconds=per_example_total,
            real_pair_wall_seconds=list(rkv.real_pair_wall_seconds),
            no_op_pair_wall_seconds=list(rkv.no_op_pair_wall_seconds),
            per_real_pair_seconds=per_real_pair_seconds,
            peak_cuda_allocated_bytes=max(rkv.peak_cuda_allocated_bytes, fullkv.peak_cuda_allocated_bytes),
            peak_cuda_reserved_bytes=max(rkv.peak_cuda_reserved_bytes, fullkv.peak_cuda_reserved_bytes),
            # B1B-R4 §11/§21 self-review finding: `no_offload_verified` must
            # use the STRONGER `parameter_placement.no_offload_verified`
            # check (which also inspects `hf_device_map` for cpu/disk/meta
            # entries), never the weaker top-level `every_parameter_on_cuda`
            # (a per-parameter `.device.type` walk alone, which a
            # `device_map="auto"` load can satisfy while STILL having an
            # offloaded entry -- `kvcot.discovery.runtime_evidence
            # .derive_parameter_placement`'s own docstring/tests).
            every_parameter_on_cuda=(
                fullkv.parameter_placement["no_offload_verified"] and rkv.parameter_placement["no_offload_verified"]
            ),
            observed_retention_ratio=rkv.observed_retention_ratio,
            event_count=rkv.selected_compaction_events,
            projected_complete_pilot_gpu_hours=projected_gpu_hours,
        )

        # B1B-R4 §9: independent identity comparison -- FullKV vs expected,
        # R-KV vs expected, THEN FullKV vs R-KV (`shared_identity_ok`) --
        # three separate checks per condition, never collapsed into one.
        fullkv_identity_ok = fullkv.runtime_identity["model_revision_match"] and fullkv.runtime_identity["tokenizer_revision_match"]
        rkv_identity_ok = rkv.runtime_identity["model_revision_match"] and rkv.runtime_identity["tokenizer_revision_match"]

        def _field_matches_manifest_and_each_other(field_name: str, expected: Any) -> bool:
            """(1) FullKV's reported field matches the coordinator's own
            expected (manifest-derived) value, (2) R-KV's does too, (3) the
            two workers agree with each other (`shared_identity_ok`) -- all
            three, never only the third."""
            return (
                getattr(fullkv, field_name) == expected
                and getattr(rkv, field_name) == expected
                and coordination.shared_identity_ok
            )

        dataset_revision_match = _field_matches_manifest_and_each_other("dataset_revision", manifest.dataset_revision)
        dataset_row_identity_match = (
            fullkv.dataset_repo == manifest.dataset_repo
            and rkv.dataset_repo == manifest.dataset_repo
            and coordination.shared_identity_ok
        )
        manifest_hash_match = _field_matches_manifest_and_each_other("manifest_hash", manifest.manifest_hash())
        prompt_token_hash_match = _field_matches_manifest_and_each_other(
            "prompt_token_ids_sha256", manifest.prompt_token_ids_sha256
        )

        # B1B-R4 §11: `one_example_only` derived from the manifest's own
        # scope (a single `example_index`/prompt) matching what BOTH workers
        # independently observed -- never a bare literal.
        one_example_only = (
            fullkv.prompt_token_count == manifest.prompt_token_count
            and rkv.prompt_token_count == manifest.prompt_token_count
        )

        evidence = build_gate_evidence_from_measurement(
            measurement,
            token_identical_replay=rkv.token_identical_replay,
            prefill_decode_boundary_parity=rkv.prefill_decode_boundary_parity,
            compaction_position_equality=rkv.compaction_position_equality,
            capture_gather_parity=rkv.capture_gather_parity,
            absolute_position_parity=rkv.absolute_position_parity,
            no_op_numerical_parity=rkv.no_op_numerical_parity,
            dataset_revision_match=dataset_revision_match,
            dataset_row_identity_match=dataset_row_identity_match,
            manifest_hash_match=manifest_hash_match,
            prompt_token_hash_match=prompt_token_hash_match,
            model_revision_match=fullkv_identity_ok and rkv_identity_ok,
            tokenizer_revision_match=fullkv_identity_ok and rkv_identity_ok,
            generation_config_hash_match=(fullkv.runtime_generation_config_hash == rkv.runtime_generation_config_hash),
            rkv_config_hash_match=rkv.rkv_config_hash_match,
            batch_size_verified=(fullkv.batch_size == 1 and rkv.batch_size == 1),
            one_example_only=one_example_only,
            meaningful_compression_observed=derive_meaningful_compression_observed(
                selected_event_count=rkv.selected_compaction_events, observed_retention_ratio=rkv.observed_retention_ratio,
            ),
            sufficient_eligible_events=(rkv.eligible_compaction_events >= B2A_SELECTED_EVENTS),
            selected_event_count_exact=(rkv.selected_compaction_events == B2A_SELECTED_EVENTS),
            real_pair_count_exact=(
                rkv.attempted_real_pair_count == B2A_REAL_PAIR_EVALUATIONS_TOTAL
                and rkv.completed_real_pair_count == B2A_REAL_PAIR_EVALUATIONS_TOTAL
            ),
            no_op_count_exact=(
                rkv.attempted_no_op_pair_count == B2A_NOOP_PAIR_EVALUATIONS_TOTAL
                and rkv.completed_no_op_pair_count == B2A_NOOP_PAIR_EVALUATIONS_TOTAL
            ),
            all_required_pair_evaluations_completed=rkv.all_required_pair_evaluations_completed,
        )
        gate_result = evaluate_b2a_gate(evidence)

        payload = {
            "passed": gate_result.passed,
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "b2a_selected_events": B2A_SELECTED_EVENTS,
            "b2a_real_pair_evaluations_total": B2A_REAL_PAIR_EVALUATIONS_TOTAL,
            "b2a_noop_pair_evaluations_total": B2A_NOOP_PAIR_EVALUATIONS_TOTAL,
            "fullkv_worker": fullkv.model_dump(mode="json"),
            "rkv_worker": rkv.model_dump(mode="json"),
            "shared_identity_ok": coordination.shared_identity_ok,
            "shared_identity_mismatches": list(coordination.shared_identity_mismatches),
            "measurement": measurement.model_dump(mode="json"),
            "gate_result": {k: v for k, v in gate_result.__dict__.items()},
        }
        artifact_path = build_and_write_b2a_artifact(payload, config_hash, manifest_hash)
        return B2ACalibrationArtifact(
            config_hash=config_hash, manifest_hash=manifest_hash, gate_result=gate_result, artifact_path=artifact_path,
        )
    except WorkerFailedError as exc:
        # B1B-R4 §16: preserve partial FullKV evidence (if FullKV succeeded
        # before R-KV failed) into the fail artifact -- never discarded.
        partial = getattr(exc, "partial_fullkv_result", None)
        fail_payload = _build_fail_artifact_payload(
            config, manifest, config_hash, manifest_hash, exc, partial_fullkv_result=partial,
        )
        build_and_write_b2a_artifact(fail_payload, config_hash, manifest_hash)
        raise
    except Exception as exc:
        # Catch-all: prompt-identity refusal, a pydantic validation error
        # while building measurement/evidence, or any other exception --
        # every one of these must still write exactly one fail artifact
        # (B1B-R4 §16/§17), matching this coordinator's original guarantee.
        fail_payload = _build_fail_artifact_payload(config, manifest, config_hash, manifest_hash, exc)
        build_and_write_b2a_artifact(fail_payload, config_hash, manifest_hash)
        raise
