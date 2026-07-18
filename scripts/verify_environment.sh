#!/usr/bin/env bash
# Verifies the GPU host environment before anything else runs. First step
# of docs/GPU_VALIDATION_PLAN.md. Never switches attn implementation
# silently — fails loudly if flash_attention_2 or BF16 support is missing.
#
# Strengthened 2026-07-18 (external review): the previous version PRINTED
# most versions without enforcing them (transformers drift was only a
# WARNING; python/torch/cuda/flash-attn versions and the CXX11 ABI were
# never checked at all; no FlashAttention kernel was ever actually
# executed; and the requirements-lock checker's `pkg==version` parser
# silently skipped the flash_attn direct-URL line entirely). Every pin
# below is now FATAL on mismatch. Pins mirror requirements-lock.txt
# (recorded from the validated RTX 3090 host, commit ef9bb1e) and
# docs/UPSTREAM_AUDIT.md H6.
set -euo pipefail

EXPECTED_PYTHON_PREFIX="3.12."
EXPECTED_TORCH="2.6.0+cu124"
EXPECTED_TORCH_CUDA="12.4"
EXPECTED_TRANSFORMERS="4.55.4"
EXPECTED_FLASH_ATTN="2.7.4.post1"

echo "== kvcot environment verification =="
echo "python: $(python --version 2>&1)"

EXPECTED_PYTHON_PREFIX="$EXPECTED_PYTHON_PREFIX" \
EXPECTED_TORCH="$EXPECTED_TORCH" \
EXPECTED_TORCH_CUDA="$EXPECTED_TORCH_CUDA" \
EXPECTED_TRANSFORMERS="$EXPECTED_TRANSFORMERS" \
EXPECTED_FLASH_ATTN="$EXPECTED_FLASH_ATTN" \
python - <<'PYEOF'
import os
import platform
import sys

failures = []

def check(name, actual, expected):
    if actual != expected:
        failures.append(f"{name}: expected {expected!r}, found {actual!r}")
    else:
        print(f"{name}: {actual} (ok)")

python_version = platform.python_version()
if not python_version.startswith(os.environ["EXPECTED_PYTHON_PREFIX"]):
    failures.append(
        f"python: expected {os.environ['EXPECTED_PYTHON_PREFIX']}x, found {python_version}"
    )
else:
    print(f"python: {python_version} (ok)")

import torch

check("torch", torch.__version__, os.environ["EXPECTED_TORCH"])
check("torch cuda build", torch.version.cuda, os.environ["EXPECTED_TORCH_CUDA"])

if not torch.cuda.is_available():
    failures.append("cuda: no CUDA device available")
else:
    print(f"cuda device: {torch.cuda.get_device_name(0)} (ok)")
    if not torch.cuda.is_bf16_supported():
        failures.append("bf16: not supported on this device")
    else:
        print("bf16: supported (ok)")

# The pinned flash-attn wheel is the cxx11abiFALSE build — a torch compiled
# with the other ABI would import but crash/corrupt at kernel call time.
cxx11_abi = torch.compiled_with_cxx11_abi()
if cxx11_abi is not False:
    failures.append(f"torch CXX11 ABI: expected False (cxx11abiFALSE wheel), found {cxx11_abi}")
else:
    print("torch CXX11 ABI: False (ok)")

import transformers

check("transformers", transformers.__version__, os.environ["EXPECTED_TRANSFORMERS"])

try:
    import flash_attn
except ImportError:
    failures.append("flash_attn: not importable")
    flash_attn = None
if flash_attn is not None:
    check("flash_attn", getattr(flash_attn, "__version__", None), os.environ["EXPECTED_FLASH_ATTN"])

# Execute a real BF16 FlashAttention CUDA kernel (never trust importability
# alone): correct output shape, all values finite, causal=True path works.
if flash_attn is not None and torch.cuda.is_available() and not failures:
    from flash_attn import flash_attn_func

    batch, seqlen, nheads, headdim = 2, 128, 4, 64
    q = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(batch, seqlen, nheads, headdim, device="cuda", dtype=torch.bfloat16)
    out = flash_attn_func(q, k, v, causal=True)
    torch.cuda.synchronize()
    if tuple(out.shape) != (batch, seqlen, nheads, headdim):
        failures.append(
            f"flash-attn kernel: wrong output shape {tuple(out.shape)}, "
            f"expected {(batch, seqlen, nheads, headdim)}"
        )
    elif not torch.isfinite(out.float()).all().item():
        failures.append("flash-attn kernel: non-finite values in BF16 causal output")
    else:
        print("flash-attn BF16 causal=True kernel: executed, shape and finiteness ok")

if failures:
    for f in failures:
        print(f"FATAL: {f}", file=sys.stderr)
    sys.exit(1)
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
from importlib.metadata import PackageNotFoundError, version

# importlib.metadata, not `pip freeze` parsing: freeze reports a direct-URL
# install (the pinned flash_attn wheel) as `pkg @ url`, which a `==`-split
# would silently drop from the installed map.
state_critical = ["torch", "transformers", "flash_attn"]
installed = {}
for pkg in state_critical:
    try:
        installed[pkg] = version(pkg)
    except PackageNotFoundError:
        pass
locked = {}
with open("requirements-lock.txt") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-e "):
            continue
        if "==" in line and " @ " not in line:
            name, version = line.split("==", 1)
            locked[name.lower().replace("-", "_")] = version.strip()
        elif " @ " in line:
            # Direct-URL requirement (e.g. the pinned flash_attn wheel).
            # The old `==`-only parser skipped these lines entirely, so the
            # single most state-critical pin in the file was never checked.
            name, url = line.split(" @ ", 1)
            name = name.strip().lower().replace("-", "_")
            m = re.search(rf"{name}-([0-9][A-Za-z0-9.!]*?)[-+]", url.rsplit("/", 1)[-1])
            if m:
                locked[name] = m.group(1)
            else:
                print(f"FATAL: could not parse a version out of the direct-URL pin: {line}", file=sys.stderr)
                sys.exit(1)
mismatches = []
for pkg in state_critical:
    if pkg not in locked:
        mismatches.append((pkg, "<missing from requirements-lock.txt>", installed.get(pkg, "<not installed>")))
        continue
    installed_version = installed.get(pkg, "<not installed>")
    # torch's lock entry may carry the +cu124 local tag; compare prefix-free.
    if installed_version.split("+")[0] != locked[pkg].split("+")[0] or installed_version == "<not installed>":
        mismatches.append((pkg, locked[pkg], installed_version))
if mismatches:
    for pkg, want, have in mismatches:
        print(f"FATAL: {pkg} locked={want} installed={have}", file=sys.stderr)
    sys.exit(1)
print("requirements-lock.txt matches installed state-critical packages (direct-URL pins included)")
PYEOF
else
    echo "requirements-lock.txt is still the placeholder — generate it after this script passes (see its header)."
fi

echo "== environment verification passed =="
