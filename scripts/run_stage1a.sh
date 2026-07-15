#!/usr/bin/env bash
# Stage 1A — baseline measurability (§10). FullKV-only, calibration split,
# seed 42. Run only after Stage 0 passes.
set -euo pipefail
CONFIG="configs/stage1a_measurability.yaml"
echo "== Stage 1A measurability: config=$CONFIG =="

kvcot generate --config "$CONFIG" --condition full --resume "$@"
kvcot replay-probe --config "$CONFIG" --condition full --resume "$@"
kvcot analyze --config "$CONFIG"

echo "== Stage 1A complete. Read results/decisions/stage1a_baseline_measurability.json"
echo "   'recommendation' field before proceeding to Stage 1B. =="
