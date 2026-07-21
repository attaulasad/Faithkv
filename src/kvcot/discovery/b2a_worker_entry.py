"""Subprocess entry point for one B2A worker (B1B-R4 §16/§19, superseding
B1B-R3's version of this module). Invoked ONLY as `python -m kvcot
.discovery.b2a_worker_entry --role {fullkv|rkv} --config ... --manifest ...
--output ...` by `kvcot.discovery.b2a_workers.run_both_workers_via_subprocess`
-- never imported for its side effects, and this module's `main()` is never
called by any CPU test that also launches a real subprocess (every
coordinator test exercises `run_both_workers_via_subprocess` via an
injected fake `subprocess_runner`; `tests/unit/discovery
/test_b2a_worker_entry.py` exercises `main()` directly, in-process, with
injected fake worker functions -- B1B-R4 §20).

## B1B-R4 §19: one canonical worker API

Calls `kvcot.discovery.b2a_workers.run_fullkv_worker`/`run_rkv_worker`
ONLY -- the misleading B1B-R3 split (a `NotImplementedError` stub in
`b2a_workers.run_rkv_worker` while this module called
`kvcot.discovery.b2a_execute.run_rkv_worker_body` directly for the "rkv"
role) no longer exists; both roles go through the same module's two
canonical, symmetric functions.

## B1B-R4 §16: durable worker-attempt envelope

Writes exactly one JSON RESULT file (the corresponding `FullKVWorkerResult`/
`RKVWorkerResult`, schema-validated) and exits 0 ONLY on success -- on any
failure, no result file is written and this process exits nonzero with a
message on stderr, exactly as before. NEW in this pass: an ENVELOPE file
(`<output>.envelope.json`) is ALWAYS written, success or failure -- the
coordinator (`run_both_workers_via_subprocess`) still treats a nonzero exit
or a missing RESULT file as `WorkerFailedError` (unchanged contract), but
the envelope preserves attempt metadata (identities, partial measurements,
error type/message/traceback) even when the result file itself was never
written, for post-mortem debugging that does not depend on captured
stdout/stderr alone.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kvcot.discovery.b2a_worker_entry")
    parser.add_argument("--role", required=True, choices=["fullkv", "rkv"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempt-id")
    args = parser.parse_args(argv)

    from kvcot.discovery.discovery_config import load_discovery_config
    from kvcot.discovery.manifest import load_b2a_one_example_manifest
    from kvcot.discovery.worker_envelope import (
        build_failure_envelope,
        build_success_envelope,
        default_hardware_metadata,
        new_attempt_id,
        now_iso,
        write_worker_envelope,
    )

    attempt_id = args.attempt_id or new_attempt_id()
    started_at = now_iso()
    output_path = Path(args.output)
    requested_identities = {"config_path": args.config, "manifest_path": args.manifest, "role": args.role}

    progress_path = output_path.parent / "progress.jsonl" if args.attempt_id else None
    previous_progress_environment = {
        name: os.environ.get(name)
        for name in ("KVCOT_B2A_ATTEMPT_ID", "KVCOT_B2A_PROGRESS_PATH")
    }

    def restore_progress_environment() -> None:
        for name, old_value in previous_progress_environment.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value

    if progress_path is not None:
        # The canonical production calls below keep their exact two-argument
        # API.  The body discovers only this attempt-local journal context.
        os.environ["KVCOT_B2A_ATTEMPT_ID"] = attempt_id
        os.environ["KVCOT_B2A_PROGRESS_PATH"] = str(progress_path)

    def progress(stage: str, status: str, counters=None) -> None:
        if progress_path is None:
            return
        from kvcot.discovery.attempt_artifacts import append_progress

        append_progress(
            progress_path, attempt_id=attempt_id, worker_role=args.role, stage=stage,
            status=status, counters=counters,
        )

    try:
        progress("startup", "completed")
        config = load_discovery_config(args.config)
        progress("config validation", "completed")
        manifest = load_b2a_one_example_manifest(args.manifest)
        progress("manifest validation", "completed")
        requested_identities.update(
            {
                "model_revision": config.model.revision,
                "tokenizer_revision": config.model.tokenizer_revision,
                "dataset_revision": manifest.dataset_revision,
                "manifest_hash": manifest.manifest_hash(),
            }
        )
        if args.role == "fullkv":
            from kvcot.discovery.b2a_workers import run_fullkv_worker

            result_dict = run_fullkv_worker(config, manifest)
        else:
            from kvcot.discovery.b2a_workers import run_rkv_worker

            result_dict = run_rkv_worker(config, manifest)
    except Exception as exc:  # noqa: BLE001 -- this process's only job is to report failure to the coordinator
        from kvcot.discovery.worker_partial_evidence import WorkerBodyFailure

        # Gate H1 repair: a `WorkerBodyFailure` carries real partial evidence
        # (`.evidence`, a `PartialWorkerEvidence`) accumulated by the worker
        # body before it failed -- threaded into the envelope instead of the
        # bare `partial_measurements=None, determinism_policy=None` this
        # branch used unconditionally before this repair. Any OTHER
        # exception (e.g. `config`/`manifest` failing to load, before any
        # worker body even started) genuinely has no partial evidence to
        # report, and correctly falls through to the `None`/empty defaults.
        partial_measurements: dict[str, Any] | None = None
        determinism_policy: dict[str, Any] | None = None
        failure_stage: str | None = None
        last_completed_stage: str | None = None
        is_oom = False
        is_timeout = False
        reported_exc: BaseException = exc
        if isinstance(exc, WorkerBodyFailure):
            evidence = exc.evidence
            partial_measurements = evidence.model_dump(mode="json")
            determinism_policy = evidence.determinism_policy
            failure_stage = evidence.failing_stage
            last_completed_stage = evidence.last_completed_stage
            is_oom = evidence.is_oom
            is_timeout = evidence.is_timeout
            # Report the ORIGINAL failure (the real cause `WorkerBodyFailure`
            # chains via `__cause__`), never the wrapper's own composed
            # message, as `error_type`/`error_message` -- `traceback_module
            # .format_exc()` inside `build_failure_envelope` still captures
            # the full chained traceback (both exceptions), so nothing about
            # the wrapper itself is lost either.
            reported_exc = exc.__cause__ or exc
        envelope = build_failure_envelope(
            role=args.role, attempt_id=attempt_id, started_at=started_at,
            requested_identities=requested_identities, resolved_identities={},
            partial_measurements=partial_measurements, determinism_policy=determinism_policy,
            software_versions=_software_versions(), hardware_metadata=default_hardware_metadata(),
            exc=reported_exc, failure_stage=failure_stage, last_completed_stage=last_completed_stage,
            is_oom=is_oom, is_timeout=is_timeout,
        )
        # H1.4: the durable failure journal event is appended, and the
        # atomic failure envelope written, BEFORE environment variables are
        # restored -- `progress()`/`write_worker_envelope` do not read the
        # env vars this process set (they close over already-resolved local
        # `progress_path`/`attempt_id` values), so this ordering costs
        # nothing and matches the required sequence exactly.
        progress(
            "failed", "failed",
            {"failure_stage": failure_stage, "is_oom": is_oom, "error_type": type(reported_exc).__name__},
        )
        try:
            write_worker_envelope(envelope, output_path)
        except Exception:  # noqa: BLE001 -- the envelope write itself must never mask the real failure
            pass
        restore_progress_environment()
        print(f"b2a_worker_entry role={args.role} failed: {type(reported_exc).__name__}: {reported_exc}", file=sys.stderr)
        return 1

    restore_progress_environment()

    resolved_identities = {
        "resolved_model_revision": result_dict.get("runtime_identity", {}).get("resolved_model_revision"),
        "resolved_tokenizer_revision": result_dict.get("runtime_identity", {}).get("resolved_tokenizer_revision"),
    }

    # Independent-audit Gate H6.4 repair: this used to re-materialize a
    # "completed" progress event for every named phase/pair by replaying
    # `result_dict["timing_evidence"]` after the body returned -- but the
    # body ALREADY appends each of these exact same (stage, "completed")
    # events LIVE, as work completes, via `_production_progress_callback`
    # (`kvcot.discovery.b2a_workers.measured`, wired through
    # `KVCOT_B2A_ATTEMPT_ID`/`KVCOT_B2A_PROGRESS_PATH`, using the identical
    # `_progress_stage_for_phase` name mapping this block duplicated).
    # Replaying the final timing list here produced a SECOND, redundant
    # "completed" event for every one of those phases -- progress is now
    # written live only, exactly once per stage, never materialized twice
    # from two different sources.

    from kvcot.discovery.attempt_artifacts import atomic_write_json

    atomic_write_json(output_path, result_dict)
    progress("result construction", "completed")

    envelope = build_success_envelope(
        role=args.role, attempt_id=attempt_id, started_at=started_at, requested_identities=requested_identities,
        resolved_identities=resolved_identities, result_payload=result_dict,
        determinism_policy=result_dict.get("determinism_policy"), software_versions=_software_versions(),
        hardware_metadata={**default_hardware_metadata(), **result_dict.get("device_evidence", {})},
    )
    try:
        write_worker_envelope(envelope, output_path)
        progress("envelope construction", "completed")
    except Exception as exc:  # a result without a valid atomic envelope is not success
        print(f"b2a_worker_entry role={args.role} envelope write failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


def _software_versions() -> dict[str, str]:
    import importlib.metadata
    import platform

    versions = {"python": platform.python_version()}
    for package in (
        "torch", "transformers", "accelerate", "flash-attn", "datasets",
        "huggingface-hub", "pydantic", "numpy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


if __name__ == "__main__":
    raise SystemExit(main())
