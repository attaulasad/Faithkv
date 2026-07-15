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

echo "== installing the rest of requirements.txt (includes transformers==4.55.4, flash-attn) =="
pip install -r requirements.txt

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
