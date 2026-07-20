"""Subprocess entry point for one B2A worker (B1B-R3 §11). Invoked ONLY as
`python -m kvcot.discovery.b2a_worker_entry --role {fullkv|rkv} --config
... --manifest ... --output ...` by
`kvcot.discovery.b2a_workers.run_both_workers_via_subprocess` -- never
imported for its side effects, and this module's `main()` is never called
by any CPU test (every test exercises the coordinator via an injected fake
`subprocess_runner`, never a real subprocess launch of this file).

Writes exactly one JSON output file (the corresponding `FullKVWorkerResult`/
`RKVWorkerResult`, schema-validated) and exits 0 on success. On any
failure, writes nothing and exits nonzero with a message on stderr -- the
coordinator (`run_both_workers_via_subprocess`) treats a nonzero exit or a
missing output file as `WorkerFailedError`, never a silently-empty result.
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

    config = load_discovery_config(args.config)
    manifest = load_b2a_one_example_manifest(args.manifest)

    try:
        if args.role == "fullkv":
            from kvcot.discovery.b2a_workers import run_fullkv_worker

            result_dict = run_fullkv_worker(config, manifest)
        else:
            from kvcot.discovery.b2a_execute import run_rkv_worker_body

            result_dict = run_rkv_worker_body(config, manifest)
    except Exception as exc:  # noqa: BLE001 -- this process's only job is to report failure to the coordinator
        print(f"b2a_worker_entry role={args.role} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_dict, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
