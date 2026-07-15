#!/usr/bin/env bash
# Stage 2 — main exploratory pilot (§10). Refuses to start (via
# kvcot.runtime.require_operating_point, invoked by every `kvcot generate`
# call against stage2_main.yaml) unless
# configs/selected_operating_point.yaml exists and is filled in.
set -euo pipefail
CONFIG="configs/stage2_main.yaml"
echo "== Stage 2 main pilot: config=$CONFIG =="

if [ ! -f configs/selected_operating_point.yaml ]; then
    echo "FATAL: configs/selected_operating_point.yaml does not exist." >&2
    echo "Run scripts/run_stage1b.sh first and fill in the operating point from its decision." >&2
    exit 1
fi

for condition in full rkv_selected; do
    echo "-- generate: condition=$condition --"
    kvcot generate --config "$CONFIG" --condition "$condition" --resume "$@"
    echo "-- replay-probe: condition=$condition --"
    kvcot replay-probe --config "$CONFIG" --condition "$condition" --resume "$@"
done

echo "-- analyze --"
kvcot analyze --config "$CONFIG"

echo "== Stage 2 complete. Review results/tables/attrition_funnel.csv and"
echo "   results/decisions/ before drawing any conclusion. =="
