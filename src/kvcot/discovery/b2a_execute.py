"""The real, one-example B2A GPU calibration run (B1B-R4 §8-§12/§14/§16/§21,
superseding B1B-R3's `docs/B1B_R3_EXECUTABLE_STATE_CLOSURE.md`).

`kvcot.cli.cmd_b2a_calibrate`'s explicit `--execute` mode is the only
production caller.  The CPU closure does not invoke that GPU path; the
CLI first enforces all CPU-checkable preconditions and CUDA availability.

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


class B2AFinalWriteError(RuntimeError):
    """F5: raised when every pre-final artifact was written and verified but
    the terminal `final.json` write itself failed -- the attempt is NOT a
    completed successful attempt, every pre-final artifact is preserved,
    and `completion.json` is never overwritten."""


class B2AFinalVerificationError(RuntimeError):
    """R4 (residual independent-audit repair): raised when `final.json` was
    written successfully but `verify_final_reference_manifest` then finds it
    does not genuinely, correctly reference every pre-final artifact --
    e.g. a torn write, a filesystem-level corruption, or a race that
    mutated a pre-final artifact after its content was already hashed into
    the reference manifest. `final.json` and every pre-final artifact are
    preserved as-is (never overwritten); the attempt is NOT reported as a
    completed successful attempt."""


@dataclass(frozen=True)
class B2ACalibrationArtifact:
    config_hash: str
    manifest_hash: str
    gate_result: Any  # kvcot.discovery.b2a_contract.B2AGateResult
    artifact_path: Path
    final_gate_result: Any | None = None


def _verify_resolved_prompt_identity(config, manifest) -> dict[str, Any]:
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

    # Independent-audit Gate H4.5 repair: this verification used to call
    # `_render_and_tokenize` with only `tokenizer_name`/`tokenizer_revision`
    # -- resolved through `huggingface_hub`'s ordinary (potentially
    # network-touching) cache lookup, never proven local-only, despite this
    # whole function's purpose being to verify a STRICT local-snapshot
    # boundary. The exact local tokenizer snapshot is now resolved first
    # (the SAME function/boundary the workers use,
    # `kvcot.discovery.snapshot_boundary.resolve_local_snapshot`) -- this
    # raises `SnapshotBoundaryError` (never silently falls back to network)
    # if the exact pinned revision is not already verified-local, and its
    # resolved path is what the tokenizer is actually loaded from, with
    # `local_files_only=True`.
    from kvcot.discovery.snapshot_boundary import SnapshotBoundaryError, resolve_local_snapshot

    try:
        tokenizer_snapshot = resolve_local_snapshot(
            config.model.tokenizer_name, config.model.tokenizer_revision, "tokenizer"
        )
    except SnapshotBoundaryError as exc:
        raise B2AExecutionRefused(
            f"exact local tokenizer snapshot unavailable for prompt verification: {exc}"
        ) from exc

    tokenizer, user_message, messages, token_ids = _render_and_tokenize(
        fetched.row, config.model.tokenizer_name, config.model.tokenizer_revision,
        local_only_path=tokenizer_snapshot.local_path,
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
    problem = fetched.row.get("problem")
    if not isinstance(problem, str):
        raise B2AExecutionRefused("re-fetched dataset row has no string problem field")
    return {
        "verified": True,
        "dataset_repo": manifest.dataset_repo,
        "dataset_config": manifest.dataset_config,
        "dataset_split": manifest.dataset_split,
        "dataset_revision": manifest.dataset_revision,
        "example_index": manifest.example_index,
        "unique_id": manifest.unique_id,
        "raw_row_sha256": fetched.raw_content_hash,
        "question_sha256": sha256_text(problem),
        "gold_answer_sha256": sha256_text(fetched.row["answer"]),
        "manifest_canonical_sha256": manifest.manifest_hash(),
        "rendered_message_sha256": sha256_text(user_message),
        "chat_template_sha256": sha256_text(tokenizer.chat_template),
        "prompt_token_sha256": sha256_int_ids(token_ids),
        "prompt_token_count": len(token_ids),
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
    }


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
        "partial_success": partial_fullkv_result is not None,
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
    attempt_directory=None,
    cli_device_preflight: dict[str, Any] | None = None,
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
    from kvcot.discovery.b2a_evidence import derive_meaningful_compression_observed
    from kvcot.discovery.b2a_workers import WorkerFailedError, run_both_workers_via_subprocess
    from kvcot.discovery.constants import (
        B2A_NOOP_PAIR_EVALUATIONS_TOTAL,
        B2A_REAL_PAIR_EVALUATIONS_TOTAL,
        B2A_SELECTED_EVENTS,
    )
    from kvcot.discovery.discovery_config import canonical_config_hash
    from kvcot.discovery.execution_measurement import build_runtime_projection
    from kvcot.discovery.final_contract import (
        evaluate_final_gates,
        expected_generation_record,
        memory_contract_satisfied,
        timing_contract_satisfied,
    )

    subprocess_runner = subprocess_runner or subprocess_module.run
    config_hash = canonical_config_hash(config)
    manifest_hash = manifest.manifest_hash()

    try:
        prompt_verification = _verify_resolved_prompt_identity(config, manifest)
        if isinstance(prompt_verification, dict):
            from kvcot.discovery.attempt_artifacts import sha256_file

            prompt_verification = {
                **prompt_verification,
                "manifest_file_byte_sha256": sha256_file(Path(manifest_path)),
            }

        coordination = run_both_workers_via_subprocess(
            config_path, manifest_path, python_executable=python_executable, subprocess_runner=subprocess_runner,
            attempt_directory=attempt_directory,
        )

        fullkv = coordination.fullkv
        rkv = coordination.rkv

        def required_phase_seconds(records, phase):
            matches = [record for record in records if record.get("phase") == phase and record.get("completed") is True]
            if len(matches) != 1:
                raise B2AExecutionRefused(f"required timing phase {phase!r} is missing or duplicated")
            value = matches[0].get("duration_seconds")
            if not isinstance(value, (int, float)) or value <= 0:
                raise B2AExecutionRefused(f"required timing phase {phase!r} is not a positive completed duration")
            return float(value)

        # Independent-audit Gate H2 repair: this projection used to sum only
        # `{role}_worker_startup` + `model_load`, silently omitting snapshot
        # resolution, tokenizer loading, and post-load runtime/no-offload
        # validation -- all genuinely one-time cost incurred before
        # measured inference, all already real, synchronized, non-
        # overlapping `timer.measure()` calls in `run_fullkv_worker`/
        # `run_rkv_worker` (`kvcot.discovery.b2a_workers`): `{role}
        # _worker_startup` -> `snapshot_tokenizer_resolution` ->
        # `tokenizer_load` -> `model_load` -> `post_load_validation`, each a
        # separate top-level `timer.measure()` call in strict sequence, so
        # summing all five is exact, never double-counted. Every phase
        # summed is listed here explicitly, per the audit's "list every
        # included phase; prove they do not overlap; prove none is
        # omitted" requirement.
        _ONE_TIME_SETUP_PHASES = (
            "{role}_worker_startup", "snapshot_tokenizer_resolution", "tokenizer_load", "model_load",
            "post_load_validation",
        )

        def _startup_and_load_seconds(records, role: str) -> float:
            return sum(
                required_phase_seconds(records, phase.format(role=role)) for phase in _ONE_TIME_SETUP_PHASES
            )

        fullkv_startup_and_load = _startup_and_load_seconds(fullkv.timing_evidence, "fullkv")
        rkv_startup_and_load = _startup_and_load_seconds(rkv.timing_evidence, "rkv")
        runtime_projection = build_runtime_projection(
            fullkv_startup_and_model_load_seconds=fullkv_startup_and_load,
            rkv_startup_and_model_load_seconds=rkv_startup_and_load,
            fullkv_natural_generation_seconds=fullkv.wall_seconds,
            rkv_pass1_seconds=rkv.wall_seconds_pass1,
            rkv_pass2_seconds=rkv.wall_seconds_pass2,
            b2a_real_pair_seconds=list(rkv.real_pair_wall_seconds),
        )
        per_example_total = runtime_projection.per_example_inference_seconds
        per_real_pair_seconds = runtime_projection.conservative_real_pair_seconds
        projected_gpu_hours = runtime_projection.projected_total_seconds / 3600.0

        # Independent-audit Gate H2.5: the Python worker subprocess's own
        # launch/import overhead occurs OUTSIDE its internal
        # `SynchronizedTimer` (that timer's first measurement only starts
        # once the worker process is already running and has imported
        # everything it needs) -- measured separately here, at the
        # coordinator, and exported as its own diagnostic. Never folded
        # into `runtime_projection` (which stays exactly the frozen §12
        # formula) -- this is purely an honesty-preserving export so
        # process-launch overhead is visible rather than silently absent.
        worker_internal_startup_and_load = {"fullkv": fullkv_startup_and_load, "rkv": rkv_startup_and_load}
        worker_internal_inference = {"fullkv": fullkv.wall_seconds, "rkv": rkv.wall_seconds_pass1 + rkv.wall_seconds_pass2}
        observed_process_seconds = coordination.coordinator_observed_process_seconds or {}
        process_overhead_diagnostic = {
            role: {
                "coordinator_observed_process_seconds": observed_process_seconds.get(role),
                "worker_internal_startup_and_load_seconds": worker_internal_startup_and_load[role],
                "worker_internal_inference_seconds": worker_internal_inference[role],
                "unattributed_process_overhead_seconds": (
                    None if observed_process_seconds.get(role) is None
                    else observed_process_seconds[role]
                    - worker_internal_startup_and_load[role] - worker_internal_inference[role]
                ),
            }
            for role in ("fullkv", "rkv")
        }

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
        expected_row_identity = {
            "dataset_repo": manifest.dataset_repo,
            "dataset_revision": manifest.dataset_revision,
            "example_index": manifest.example_index,
            "unique_id": manifest.unique_id,
            "raw_content_hash": manifest.raw_content_hash,
            "manifest_canonical_hash": manifest.manifest_hash(),
            "rendered_user_message_sha256": manifest.rendered_user_message_sha256,
            "chat_template_source_sha256": manifest.chat_template_source_sha256,
            "prompt_token_ids_sha256": manifest.prompt_token_ids_sha256,
            "prompt_token_count": manifest.prompt_token_count,
        }
        dataset_row_identity_match = (
            fullkv.dataset_row_identity == expected_row_identity
            and rkv.dataset_row_identity == expected_row_identity
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
            # B1B-R4.1 §18/§30: a dedicated, explicitly-named condition --
            # a pair whose real record was constructed but whose semantic
            # swap failed to update provenance/kept-index bookkeeping is
            # recorded as a `semantic_swap_parity_failure` pair-failure
            # detail (`kvcot.discovery.pipeline.build_swap_pair_record`,
            # `kvcot.discovery.orchestrator.run_example`) -- never allowed
            # to pass silently under the coarser
            # `all_required_pair_evaluations_completed` umbrella alone.
            #
            # B1 execution-boundary closure §12: derived from POSITIVE
            # counts (`checks_attempted == checks_required == 12` AND
            # `checks_failed == 0`) rather than absence-of-a-failure-record
            # -- a worker that never actually reached the semantic-swap
            # check for any pair (e.g. every pair failed earlier, at
            # candidate/donor pool lookup, so `pair_failure_details` would
            # contain zero `semantic_swap_parity_failure` entries despite
            # the check never having run at all) must fail this condition,
            # not vacuously pass it.
            semantic_swap_parity=(
                rkv.semantic_swap_checks_required == B2A_REAL_PAIR_EVALUATIONS_TOTAL
                and rkv.semantic_swap_checks_attempted == B2A_REAL_PAIR_EVALUATIONS_TOTAL
                and rkv.semantic_swap_checks_passed == B2A_REAL_PAIR_EVALUATIONS_TOTAL
                and rkv.semantic_swap_checks_failed == 0
            ),
            # B1 execution-boundary closure §13: exact, duplicate-detecting
            # pair-identity conditions (`kvcot.discovery.b2a_evidence
            # .PairIdentityEvidence`) -- distinct from `real_pair_count_exact`
            # above, which only ever compares a raw COUNT and cannot detect
            # the same (event, layer, head, candidate, donor) identity
            # recorded more than once.
            unique_real_pair_count_exact=(
                len({tuple(sorted(identity.items())) for identity in rkv.completed_pair_identities
                     if identity.get("pair_kind") == "real"}) == B2A_REAL_PAIR_EVALUATIONS_TOTAL
            ),
            events_with_four_unique_pairs_exact=(
                len({item["compaction_event_id"] for item in rkv.selected_event_evidence}) == B2A_SELECTED_EVENTS
                and all(
                    len({tuple(sorted(identity.items())) for identity in rkv.completed_pair_identities
                         if identity.get("pair_kind") == "real"
                         and identity.get("compaction_event_id") == event_id}) == 4
                    for event_id in {item["compaction_event_id"] for item in rkv.selected_event_evidence}
                )
            ),
            no_duplicate_pair_identity=(
                len(rkv.attempted_pair_identities)
                == len({tuple(sorted(identity.items())) for identity in rkv.attempted_pair_identities})
                and sum(1 for identity in rkv.attempted_pair_identities if identity.get("pair_kind") == "no_op") == 1
            ),
            dataset_revision_match=dataset_revision_match,
            dataset_row_identity_match=dataset_row_identity_match,
            manifest_hash_match=manifest_hash_match,
            prompt_token_hash_match=prompt_token_hash_match,
            model_revision_match=fullkv_identity_ok and rkv_identity_ok,
            tokenizer_revision_match=fullkv_identity_ok and rkv_identity_ok,
            generation_config_hash_match=(fullkv.runtime_generation_config_hash == rkv.runtime_generation_config_hash),
            rkv_config_hash_match=rkv.rkv_config_hash_match,
            batch_size_verified=(
                fullkv.actual_batch_size_verified
                and rkv.actual_batch_size_verified
                and bool(fullkv.actual_call_evidence)
                and bool(rkv.actual_call_evidence)
                and all(event.get("batch_size") == 1 for event in fullkv.actual_call_evidence)
                and all(event.get("batch_size") == 1 for event in rkv.actual_call_evidence)
            ),
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

        # The expected EOS identity comes from the coordinator's independent
        # pinned-tokenizer resolution, never from either worker's observation.
        expected_eos_token_id = (
            prompt_verification.get("tokenizer_eos_token_id")
            if isinstance(prompt_verification, dict)
            else None
        )
        expected_full_generation = expected_generation_record(config, manifest, expected_eos_token_id)
        expected_rkv_generation = expected_generation_record(config, manifest, expected_eos_token_id)
        fullkv_generation_matches_expected = fullkv.runtime_generation == expected_full_generation
        rkv_generation_matches_expected = rkv.runtime_generation == expected_rkv_generation
        workers_generation_match = (
            fullkv.runtime_generation == rkv.runtime_generation
            and fullkv.runtime_generation_config_hash == rkv.runtime_generation_config_hash
        )

        attempted_real = [item for item in rkv.attempted_pair_identities if item.get("pair_kind") == "real"]
        completed_real = [item for item in rkv.completed_pair_identities if item.get("pair_kind") == "real"]
        attempted_noop = [item for item in rkv.attempted_pair_identities if item.get("pair_kind") == "no_op"]
        completed_noop = [item for item in rkv.completed_pair_identities if item.get("pair_kind") == "no_op"]
        real_keys = [tuple(sorted(item.items())) for item in attempted_real]
        completed_real_keys = {tuple(sorted(item.items())) for item in completed_real}
        selected_ids = [item.get("compaction_event_id") for item in rkv.selected_event_evidence]
        unique_selected_ids = set(selected_ids)
        events_four = (
            len(unique_selected_ids) == B2A_SELECTED_EVENTS
            and all(
                len({key for key in completed_real_keys if dict(key).get("compaction_event_id") == event_id}) == 4
                for event_id in unique_selected_ids
            )
        )

        replay = rkv.replay_evidence
        complete_token_trace_match = (
            bool(replay)
            and replay.get("pass1_token_ids") == replay.get("pass2_fed_token_ids")
            and replay.get("pass1_token_sha256") == replay.get("pass2_token_sha256")
            and replay.get("token_first_mismatch") is None
        )
        complete_call_trace_match = (
            rkv.prefill_decode_boundary_parity
            and rkv.pass1_call_boundary == rkv.pass2_call_boundary
            and rkv.pass1_call_boundary.get("prefill_call_count") == 1
            and replay.get("pass1_calls") == replay.get("pass2_calls")
            and replay.get("pass1_call_sha256") == replay.get("pass2_call_sha256")
            and replay.get("call_first_mismatch") is None
            and replay.get("pass1_actual_calls") == replay.get("pass2_actual_calls")
            and replay.get("pass1_actual_call_sha256") == replay.get("pass2_actual_call_sha256")
            and replay.get("actual_call_first_mismatch") is None
        )
        complete_compaction_trace_match = (
            replay.get("pass1_compaction_positions") == replay.get("pass2_compaction_positions")
            and replay.get("pass1_compaction_sha256") == replay.get("pass2_compaction_sha256")
            and replay.get("compaction_first_mismatch") is None
            and replay.get("complete_compaction_trace_match") is True
        )

        no_op = rkv.no_op_evidence
        no_op_exact_parity = (
            len(attempted_noop) == len(completed_noop) == 1
            and rkv.no_op_identity == attempted_noop[0] == completed_noop[0]
            and attempted_noop[0].get("candidate_absolute_position")
            == attempted_noop[0].get("donor_absolute_position")
            and no_op.get("baseline_nll") == no_op.get("no_op_nll")
            and no_op.get("baseline_nll_sha256") == no_op.get("no_op_nll_sha256")
            and no_op.get("mean_difference") == 0.0
            and no_op.get("maximum_absolute_per_token_difference") == 0.0
            and bool(no_op.get("starting_snapshot_sha256"))
            and no_op.get("physical_byte_delta") == 0
            and bool(no_op.get("provenance_before_sha256"))
            and no_op.get("provenance_before_sha256") == no_op.get("provenance_after_sha256")
            and bool(no_op.get("kept_index_before_sha256"))
            and no_op.get("kept_index_before_sha256") == no_op.get("kept_index_after_sha256")
        )

        semantic_real_reports = [
            report for report in rkv.semantic_mutation_reports
            if report.get("pair_identity", {}).get("pair_kind") == "real"
        ]
        positive_semantic_swap_parity = (
            rkv.semantic_swap_checks_required == B2A_REAL_PAIR_EVALUATIONS_TOTAL
            and rkv.semantic_swap_checks_attempted == B2A_REAL_PAIR_EVALUATIONS_TOTAL
            and rkv.semantic_swap_checks_passed == B2A_REAL_PAIR_EVALUATIONS_TOTAL
            and rkv.semantic_swap_checks_failed == 0
            and len(semantic_real_reports) == B2A_REAL_PAIR_EVALUATIONS_TOTAL
            and all(report.get("attempted") is True and report.get("passed") is True for report in semantic_real_reports)
        )

        attempt_files_verified = False
        attempt_verification_reasons: tuple[str, ...] = ()
        worker_envelopes_verified = False
        git_clean_verified = False
        rkv_submodule_match = False
        if attempt_directory is not None:
            import json
            import sys as sys_module

            from kvcot.discovery.attempt_verification import verify_attempt_artifacts, verify_worker_envelopes

            # Independent-audit Gate H6 repair, extended by F4: the ONE
            # authoritative attempt verifier -- parses and cross-validates
            # every pre-final artifact's content (top-level artifacts,
            # exact worker command identity, saved-result-vs-coordinator
            # agreement, typed envelope/result validation, coordinator
            # process outcome, and full progress-journal validation).
            attempt_files_verified, attempt_verification_reasons = verify_attempt_artifacts(
                attempt_directory, fullkv_result=fullkv.model_dump(mode="json"), rkv_result=rkv.model_dump(mode="json"),
                expected_config_hash=config_hash, expected_manifest_hash=manifest_hash,
                python_executable=python_executable or sys_module.executable, typed_results=True,
            )
            worker_envelopes_verified = verify_worker_envelopes(attempt_directory)
            provenance_path = attempt_directory / "provenance.json"
            if provenance_path.is_file():
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                git_clean_verified = provenance.get("git", {}).get("dirty") is False
                rkv_submodule_match = provenance.get("git", {}).get("rkv_submodule_match") is True

        from kvcot.discovery.snapshot_boundary import (
            revalidate_snapshot_evidence_against_directory,
            verify_snapshot_evidence_raw,
        )

        def snapshot_verified(worker, asset: str) -> bool:
            # Independent-audit Gate H4.4/H4.6 repair: no longer trusts a
            # bare `snapshot_evidence["verified"] is True` plus a single
            # `resolved_revision` field comparison -- fully re-validates
            # the reported `VerifiedLocalSnapshot` content (repository
            # identity, asset type, exact-SHA revision agreement,
            # `local_files_only`, non-empty file inventory, no incomplete/
            # lock files, and the required config/tokenizer/weight files
            # actually present) from the RAW worker-reported dict.
            expected_repository_id = config.model.name if asset == "model" else config.model.tokenizer_name
            expected_revision = config.model.revision if asset == "model" else config.model.tokenizer_revision
            return (
                worker.snapshot_evidence.get("verified") is True
                and verify_snapshot_evidence_raw(
                    worker.snapshot_evidence.get(asset),
                    expected_repository_id=expected_repository_id,
                    expected_revision=expected_revision,
                    asset_type=asset,
                )
                # F8: when the worker-reported local snapshot directory is
                # readable from the coordinator, recompute the inventory,
                # sizes, hash, and index/shard accounting from disk too.
                and revalidate_snapshot_evidence_against_directory(worker.snapshot_evidence.get(asset))
            )
        actual_batch_size_verified = (
            fullkv.actual_batch_size_verified and rkv.actual_batch_size_verified
            and bool(fullkv.actual_call_evidence) and bool(rkv.actual_call_evidence)
            and all(call.get("batch_size") == 1 for call in fullkv.actual_call_evidence + rkv.actual_call_evidence)
        )
        from kvcot.discovery.strict_device import (
            verify_device_gate_from_raw_evidence,
            verify_placement_from_raw_evidence,
        )

        final_gate_result = evaluate_final_gates({
            "git_clean_verified": git_clean_verified,
            "rkv_submodule_match": rkv_submodule_match,
            # Independent-audit Gate H4.1/H4.2 repair: recomputed from raw
            # device-evidence fields on BOTH workers (visible GPU count,
            # device index, GPU name, VRAM range, driver/CUDA/cuDNN
            # presence) and their mutual agreement -- never a bare
            # worker-reported `verified=True` boolean.
            "single_rtx3090_verified": verify_device_gate_from_raw_evidence(
                fullkv.device_evidence, rkv.device_evidence, cli_device_preflight
            ),
            "local_model_snapshot_verified": snapshot_verified(fullkv, "model") and snapshot_verified(rkv, "model"),
            "local_tokenizer_snapshot_verified": (
                snapshot_verified(fullkv, "tokenizer") and snapshot_verified(rkv, "tokenizer")
            ),
            "dataset_row_identity_verified": (
                isinstance(prompt_verification, dict)
                and prompt_verification.get("verified") is True
                and prompt_verification.get("dataset_repo") == manifest.dataset_repo
                and prompt_verification.get("dataset_config") == manifest.dataset_config
                and prompt_verification.get("dataset_split") == manifest.dataset_split
                and prompt_verification.get("dataset_revision") == manifest.dataset_revision
                and prompt_verification.get("example_index") == manifest.example_index
                and prompt_verification.get("unique_id") == manifest.unique_id
                and all(
                    isinstance(prompt_verification.get(name), str)
                    and len(prompt_verification[name]) == 64
                    for name in (
                        "raw_row_sha256", "question_sha256", "gold_answer_sha256",
                        "manifest_canonical_sha256", "manifest_file_byte_sha256",
                    )
                )
            ),
            "prompt_identity_verified": (
                isinstance(prompt_verification, dict)
                and prompt_verification.get("prompt_token_sha256") == manifest.prompt_token_ids_sha256
                and prompt_verification.get("prompt_token_count") == manifest.prompt_token_count
                and prompt_verification.get("rendered_message_sha256") == manifest.rendered_user_message_sha256
                and prompt_verification.get("chat_template_sha256") == manifest.chat_template_source_sha256
            ),
            "fullkv_generation_matches_expected": fullkv_generation_matches_expected,
            "rkv_generation_matches_expected": rkv_generation_matches_expected,
            "workers_generation_match": workers_generation_match,
            "actual_batch_size_verified": actual_batch_size_verified,
            "complete_token_trace_match": complete_token_trace_match,
            "complete_call_trace_match": complete_call_trace_match,
            "complete_compaction_trace_match": complete_compaction_trace_match,
            "capture_gather_parity": rkv.capture_gather_parity,
            "absolute_position_parity": rkv.absolute_position_parity,
            "selected_event_ids_exact": len(selected_ids) == B2A_SELECTED_EVENTS and len(unique_selected_ids) == B2A_SELECTED_EVENTS,
            "unique_real_pair_count_exact": len(real_keys) == len(set(real_keys)) == B2A_REAL_PAIR_EVALUATIONS_TOTAL,
            "events_with_four_unique_pairs_exact": events_four,
            "no_duplicate_pair_identity": len(rkv.attempted_pair_identities) == len({tuple(sorted(item.items())) for item in rkv.attempted_pair_identities}),
            "authorized_no_op_identity_exact": len(attempted_noop) == len(completed_noop) == 1 and rkv.no_op_identity == attempted_noop[0],
            "positive_semantic_swap_parity": positive_semantic_swap_parity,
            "no_op_exact_parity": no_op_exact_parity,
            # F7: complete requested-device + CPU/disk/meta/offload placement
            # boundary from BOTH workers' raw parameter-placement evidence.
            "no_offload_and_placement_verified": verify_placement_from_raw_evidence(
                fullkv.parameter_placement, rkv.parameter_placement
            ),
            # F9: exact multiplicities, with per-call decode/prefill counts
            # cross-checked against raw actual model-call evidence.
            "all_required_timings_present": timing_contract_satisfied(
                fullkv.timing_evidence, rkv.timing_evidence,
                fullkv_actual_calls=fullkv.actual_call_evidence, rkv_actual_calls=rkv.actual_call_evidence,
            ),
            "all_required_memory_phases_present": memory_contract_satisfied(fullkv.memory_phase_evidence, rkv.memory_phase_evidence),
            "runtime_within_limit": gate_result.runtime_within_limit,
            "peak_vram_within_limit": gate_result.peak_vram_within_limit,
            "worker_envelopes_verified": worker_envelopes_verified,
            "attempt_artifacts_verified": attempt_files_verified,
        })

        payload = {
            "passed": gate_result.passed and (final_gate_result.passed if attempt_directory is not None else True),
            "legacy_gate_passed": gate_result.passed,
            "config_hash": config_hash,
            "manifest_hash": manifest_hash,
            "b2a_selected_events": B2A_SELECTED_EVENTS,
            "b2a_real_pair_evaluations_total": B2A_REAL_PAIR_EVALUATIONS_TOTAL,
            "b2a_noop_pair_evaluations_total": B2A_NOOP_PAIR_EVALUATIONS_TOTAL,
            "fullkv_worker": fullkv.model_dump(mode="json"),
            "rkv_worker": rkv.model_dump(mode="json"),
            "shared_identity_ok": coordination.shared_identity_ok,
            "shared_identity_mismatches": list(coordination.shared_identity_mismatches),
            "worker_processes": {
                "return_codes": coordination.return_codes,
                "timeout_state": coordination.timeout_state,
                "partial_success": coordination.partial_success,
                "coordinator_observed_process_seconds": coordination.coordinator_observed_process_seconds,
                "process_overhead_diagnostic": process_overhead_diagnostic,
            },
            "cli_device_preflight": cli_device_preflight,
            "measurement": measurement.model_dump(mode="json"),
            "runtime_projection": runtime_projection.__dict__,
            "gate_result": {k: v for k, v in gate_result.__dict__.items()},
            "final_gate_result": {
                "passed": final_gate_result.passed,
                "conditions": final_gate_result.conditions,
                "failed_conditions": list(final_gate_result.failed_conditions),
            },
            "attempt_verification_reasons": list(attempt_verification_reasons),
            "dataset_row_verification": prompt_verification,
        }
        if attempt_directory is not None:
            # F5: the coherent successful ordering -- gates are already
            # calculated above; `completion.json` is written BEFORE the
            # reference manifest is built, so the manifest can include it;
            # `final.json` is written LAST and is the only artifact
            # excluded from its own reference set.
            from datetime import datetime, timezone

            from kvcot.discovery.attempt_artifacts import atomic_write_json, build_attempt_references, AttemptDirectory
            from kvcot.discovery.attempt_verification import verify_attempt_artifacts as _verify_prefinal

            attempt = AttemptDirectory(attempt_id=attempt_directory.name.rsplit("_", 1)[-1], path=attempt_directory)
            overall_passed = bool(gate_result.passed and final_gate_result.passed)
            atomic_write_json(attempt_directory / "completion.json", {
                "attempt_id": attempt.attempt_id,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "outcome": "gate_passed" if overall_passed else "gate_failed",
                "exit_code": 0 if overall_passed else 2,
                "gate_passed": overall_passed,
                "intended_final_relative_path": "final.json",
                "artifact_path": str(attempt_directory / "final.json"),
                "config_hash": config_hash,
                "manifest_hash": manifest_hash,
            })
            import sys as sys_module

            prefinal_ok, prefinal_reasons = _verify_prefinal(
                attempt_directory, fullkv_result=fullkv.model_dump(mode="json"), rkv_result=rkv.model_dump(mode="json"),
                expected_config_hash=config_hash, expected_manifest_hash=manifest_hash,
                python_executable=python_executable or sys_module.executable, typed_results=True,
            )
            payload["pre_final_verification"] = {"verified": prefinal_ok, "reasons": list(prefinal_reasons)}
            if overall_passed and not prefinal_ok:
                raise B2AExecutionRefused(
                    f"pre-final artifact verification failed after completion was recorded: {list(prefinal_reasons)}"
                )
            payload["attempt_artifacts"] = build_attempt_references(attempt, exclude=("final.json",))
            try:
                artifact_path = atomic_write_json(attempt_directory / "final.json", payload)
            except BaseException as final_exc:
                # F5: every pre-final artifact is preserved; completion.json
                # is never overwritten; the attempt is NOT reported as a
                # completed successful attempt.
                try:
                    atomic_write_json(attempt_directory / "final_write_failure.json", {
                        "attempt_id": attempt.attempt_id,
                        "failure_type": type(final_exc).__name__,
                        "failure_message": str(final_exc),
                        "intended_final_relative_path": "final.json",
                    })
                except Exception:  # noqa: BLE001 -- best-effort only, never masks the real failure
                    pass
                raise B2AFinalWriteError(
                    f"final.json write failed: {type(final_exc).__name__}: {final_exc}"
                ) from final_exc

            # R4: production must verify the COMPLETED final manifest, not
            # merely trust that the write succeeded -- `verify_final_
            # reference_manifest` independently recomputes every reference
            # hash in the just-written `final.json` against what is
            # actually on disk. `final.json` and every pre-final artifact
            # are left exactly as written either way; only a terminal
            # failure record is added on top.
            from kvcot.discovery.attempt_verification import verify_final_reference_manifest

            final_verified, final_reasons = verify_final_reference_manifest(attempt_directory)
            if not final_verified:
                try:
                    atomic_write_json(attempt_directory / "final_verification_failure.json", {
                        "attempt_id": attempt.attempt_id,
                        "reasons": list(final_reasons),
                        "intended_final_relative_path": "final.json",
                    })
                except Exception:  # noqa: BLE001 -- best-effort only, never masks the real failure
                    pass
                raise B2AFinalVerificationError(
                    f"final.json failed post-write verification: {list(final_reasons)}"
                )
        else:
            artifact_path = build_and_write_b2a_artifact(payload, config_hash, manifest_hash)
        return B2ACalibrationArtifact(
            config_hash=config_hash, manifest_hash=manifest_hash, gate_result=gate_result, artifact_path=artifact_path,
            final_gate_result=final_gate_result,
        )
    except WorkerFailedError as exc:
        # B1B-R4 §16: preserve partial FullKV evidence (if FullKV succeeded
        # before R-KV failed) into the fail artifact -- never discarded.
        partial = getattr(exc, "partial_fullkv_result", None)
        fail_payload = _build_fail_artifact_payload(
            config, manifest, config_hash, manifest_hash, exc, partial_fullkv_result=partial,
        )
        if attempt_directory is not None:
            from kvcot.discovery.attempt_artifacts import atomic_write_json
            atomic_write_json(attempt_directory / "failure.json", fail_payload)
        else:
            build_and_write_b2a_artifact(fail_payload, config_hash, manifest_hash)
        raise
    except Exception as exc:
        # Catch-all: prompt-identity refusal, a pydantic validation error
        # while building measurement/evidence, or any other exception --
        # every one of these must still write exactly one fail artifact
        # (B1B-R4 §16/§17), matching this coordinator's original guarantee.
        fail_payload = _build_fail_artifact_payload(config, manifest, config_hash, manifest_hash, exc)
        if attempt_directory is not None:
            from kvcot.discovery.attempt_artifacts import atomic_write_json
            atomic_write_json(attempt_directory / "failure.json", fail_payload)
        else:
            build_and_write_b2a_artifact(fail_payload, config_hash, manifest_hash)
        raise
