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
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m kvcot.discovery.b2a_worker_entry")
    parser.add_argument("--role", required=True, choices=["fullkv", "rkv"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
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

    attempt_id = new_attempt_id()
    started_at = now_iso()
    output_path = Path(args.output)
    requested_identities = {"config_path": args.config, "manifest_path": args.manifest, "role": args.role}

    config = load_discovery_config(args.config)
    manifest = load_b2a_one_example_manifest(args.manifest)
    requested_identities.update(
        {
            "model_revision": config.model.revision,
            "tokenizer_revision": config.model.tokenizer_revision,
            "dataset_revision": manifest.dataset_revision,
            "manifest_hash": manifest.manifest_hash(),
        }
    )

    try:
        if args.role == "fullkv":
            from kvcot.discovery.b2a_workers import run_fullkv_worker

            result_dict = run_fullkv_worker(config, manifest)
        else:
            from kvcot.discovery.b2a_workers import run_rkv_worker

            result_dict = run_rkv_worker(config, manifest)
    except Exception as exc:  # noqa: BLE001 -- this process's only job is to report failure to the coordinator
        envelope = build_failure_envelope(
            role=args.role, attempt_id=attempt_id, started_at=started_at,
            requested_identities=requested_identities, resolved_identities={}, partial_measurements=None,
            determinism_policy=None, software_versions=_software_versions(), hardware_metadata=default_hardware_metadata(),
            exc=exc,
        )
        try:
            write_worker_envelope(envelope, output_path)
        except Exception:  # noqa: BLE001 -- the envelope write itself must never mask the real failure
            pass
        print(f"b2a_worker_entry role={args.role} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    resolved_identities = {
        "resolved_model_revision": result_dict.get("runtime_identity", {}).get("resolved_model_revision"),
        "resolved_tokenizer_revision": result_dict.get("runtime_identity", {}).get("resolved_tokenizer_revision"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_dict, indent=2, sort_keys=True), encoding="utf-8")

    envelope = build_success_envelope(
        role=args.role, attempt_id=attempt_id, started_at=started_at, requested_identities=requested_identities,
        resolved_identities=resolved_identities, result_payload=result_dict,
        determinism_policy=result_dict.get("determinism_policy"), software_versions=_software_versions(),
        hardware_metadata=default_hardware_metadata(),
    )
    try:
        write_worker_envelope(envelope, output_path)
    except Exception:  # noqa: BLE001 -- the envelope write itself must never fail a successful attempt
        pass

    return 0


def _software_versions() -> dict[str, str]:
    try:
        import torch

        return {"torch": torch.__version__}
    except ImportError:
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
