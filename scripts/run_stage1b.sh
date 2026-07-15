#!/usr/bin/env bash
# Stage 1B — fixed-budget calibration (§10). Runs all four candidate
# budgets {128,256,512,1024} on the calibration split, seed 42, then
# recommends (but does not silently apply) an operating point. Run only
# after Stage 1A recommends proceeding on GSM8K.
set -euo pipefail

for budget in 128 256 512 1024; do
    CONFIG="configs/stage1b_budget_${budget}.yaml"
    echo "== Stage 1B: budget=$budget config=$CONFIG =="
    kvcot generate --config "$CONFIG" --condition full --resume "$@"
    kvcot generate --config "$CONFIG" --condition "rkv_b${budget}" --resume "$@"
    kvcot replay-probe --config "$CONFIG" --condition "rkv_b${budget}" --resume "$@"
    kvcot analyze --config "$CONFIG"
done

echo "== computing calibration decision across all four budgets =="
kvcot calibrate-budget --config-dir configs

echo "== Stage 1B complete. Review the printed recommendation, then manually"
echo "   copy configs/selected_operating_point.yaml.example to"
echo "   configs/selected_operating_point.yaml and fill it in from the real"
echo "   decision JSON before running Stage 2 (§10: never silently choose"
echo "   the closest-to-10% budget). =="
