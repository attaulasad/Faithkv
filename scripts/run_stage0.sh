#!/usr/bin/env bash
# Stage 0 — correctness smoke (§10). Run only after
# scripts/verify_environment.sh and the CPU + GPU unit/integration tests
# pass (see docs/GPU_VALIDATION_PLAN.md for the required order).
set -euo pipefail
CONFIG="configs/stage0_smoke.yaml"
echo "== Stage 0 smoke: config=$CONFIG =="

for condition in full patched_noop rkv_b96; do
    echo "-- generate: condition=$condition --"
    kvcot generate --config "$CONFIG" --condition "$condition" --resume "$@"
done

for condition in patched_noop rkv_b96; do
    echo "-- replay-probe: condition=$condition --"
    kvcot replay-probe --config "$CONFIG" --condition "$condition" --resume "$@"
done

echo "-- analyze --"
kvcot analyze --config "$CONFIG"

echo "== Stage 0 complete. Review results/decisions/ and the throughput"
echo "   extrapolation this stage prints before authorizing Stage 2. =="
