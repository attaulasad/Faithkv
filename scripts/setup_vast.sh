#!/usr/bin/env bash
# One-time GPU host setup (e.g. a rented Vast.ai RTX 3090 instance). Never
# run on this build machine — it installs torch/transformers/flash-attn and
# downloads model weights, both explicitly out of scope for the build here.
set -euo pipefail

: "${HF_HOME:?Set HF_HOME to a configurable large-disk path before running this script, e.g. export HF_HOME=/workspace/hf_cache}"

echo "== kvcot GPU host setup =="
echo "HF_HOME=$HF_HOME"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip

echo "== installing torch (CUDA build) =="
# Pin matches docs/UPSTREAM_AUDIT.md H6 (upstream-validated torch==2.6.0,
# transformers==4.55.4). Adjust the cu12x index URL to match the host's
# actual CUDA driver if this ever needs to run on a different image.
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "== installing the rest of requirements.txt EXCEPT flash-attn (see below for why it's separate) =="
grep -v '^flash-attn' requirements.txt > /tmp/kvcot-requirements-no-flash-attn.txt
pip install -r /tmp/kvcot-requirements-no-flash-attn.txt

echo "== installing flash-attn separately, --no-build-isolation =="
# flash-attn's setup.py does `import torch` at build-requirements-gathering
# time to detect the CUDA/torch ABI to compile against. By default `pip
# install -r requirements.txt` builds each package's wheel in a FRESH,
# ISOLATED build environment (PEP 517) that does NOT include the already-
# installed torch from this venv -- even though torch was installed above,
# in an EARLIER, separate pip invocation. That isolated build env has no
# torch at all, so flash-attn's build step fails with `ModuleNotFoundError:
# No module named 'torch'` (this exact failure is preserved verbatim in
# logs/setup.log from the run that hit it). `--no-build-isolation` makes
# pip build the wheel using THIS environment's already-installed torch
# instead of a fresh isolated one -- the same fix that made the later
# manual install succeed (logs/flash_attn_install.log).
FLASH_ATTN_SPEC=$(grep '^flash-attn' requirements.txt)
pip install --no-build-isolation "$FLASH_ATTN_SPEC"

echo "== installing this package =="
pip install -e .

echo "== initializing the pinned R-KV submodule (full checkout — no sparse-checkout needed on Linux) =="
git submodule update --init --recursive

echo "== installing the pinned upstream HuggingFace R-KV implementation, editable =="
pip install -e third_party/R-KV/HuggingFace

echo "== verifying GPU BF16 + FlashAttention-2 availability =="
bash scripts/verify_environment.sh

echo "== generating requirements-lock.txt for real (see its header for why this couldn't be done on the build machine) =="
pip freeze > requirements-lock.txt

echo "== setup complete. Next: docs/GPU_VALIDATION_PLAN.md =="
