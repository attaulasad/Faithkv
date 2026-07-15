#!/usr/bin/env bash
set -euo pipefail
echo "== kvcot freeze-manifests =="
echo "config: configs/lock.yaml"
kvcot freeze-manifests --config configs/lock.yaml "$@"
