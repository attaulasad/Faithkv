#!/usr/bin/env bash
# Verifies the GPU host environment before anything else runs. First step
# of docs/GPU_VALIDATION_PLAN.md. Never switches attn implementation
# silently — fails loudly if flash_attention_2 or BF16 support is missing.
set -euo pipefail

echo "== kvcot environment verification =="
echo "python: $(python --version 2>&1)"

python - <<'PYEOF'
import sys
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FATAL: no CUDA device available", file=sys.stderr)
    sys.exit(1)
print(f"cuda device: {torch.cuda.get_device_name(0)}")
print(f"cuda version (torch build): {torch.version.cuda}")

if not torch.cuda.is_bf16_supported():
    print("FATAL: BF16 not supported on this device", file=sys.stderr)
    sys.exit(1)
print("bf16: supported")

try:
    import flash_attn
    print(f"flash_attn: {getattr(flash_attn, '__version__', 'unknown')}")
except ImportError:
    print("FATAL: flash_attn not importable", file=sys.stderr)
    sys.exit(1)

import transformers
print(f"transformers: {transformers.__version__}")
if not (transformers.__version__.startswith("4.55") ):
    print(
        f"WARNING: transformers=={transformers.__version__} is not the "
        "upstream-validated 4.55.4 — see docs/UPSTREAM_AUDIT.md H6.",
        file=sys.stderr,
    )
PYEOF

echo "== checking pinned submodule =="
EXPECTED_COMMIT=$(python -c "import yaml; print(yaml.safe_load(open('configs/lock.yaml'))['upstream']['commit'])")
ACTUAL_COMMIT=$(git -C third_party/R-KV rev-parse HEAD)
if [ "$EXPECTED_COMMIT" != "$ACTUAL_COMMIT" ]; then
    echo "FATAL: third_party/R-KV is at $ACTUAL_COMMIT, expected $EXPECTED_COMMIT" >&2
    exit 1
fi
echo "submodule pinned correctly: $ACTUAL_COMMIT"

if [ -f requirements-lock.txt ] && ! grep -q "^# PLACEHOLDER" requirements-lock.txt; then
    echo "== checking requirements-lock.txt against installed versions =="
    python - <<'PYEOF'
import re
import sys
import subprocess

installed = {
    line.split("==")[0].lower(): line.split("==")[1].strip()
    for line in subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout.splitlines()
    if "==" in line
}
state_critical = ["torch", "transformers", "flash-attn", "flash_attn"]
with open("requirements-lock.txt") as f:
    locked = {
        line.split("==")[0].lower(): line.split("==")[1].strip()
        for line in f if "==" in line and not line.startswith("#")
    }
mismatches = []
for pkg in state_critical:
    if pkg in locked and pkg in installed and locked[pkg] != installed[pkg]:
        mismatches.append((pkg, locked[pkg], installed[pkg]))
if mismatches:
    for pkg, want, have in mismatches:
        print(f"FATAL: {pkg} locked={want} installed={have}", file=sys.stderr)
    sys.exit(1)
print("requirements-lock.txt matches installed state-critical packages")
PYEOF
else
    echo "requirements-lock.txt is still the placeholder — generate it after this script passes (see its header)."
fi

echo "== environment verification passed =="
