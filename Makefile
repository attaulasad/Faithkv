.PHONY: install install-cpu-tools test test-gpu dry-run freeze-manifests clean

# CPU-only install: core deps + this package, no torch/transformers/GPU deps.
install:
	pip install -e .

# Adds tokenizer/dataset tooling (transformers tokenizer-only, datasets,
# huggingface_hub, jinja2) needed for freeze-manifests and PROBE_PROTOCOL.md
# regeneration. Still no torch.
install-cpu-tools:
	pip install -e ".[cpu-tools,dev]"

# Full GPU-host install (torch, transformers w/ torch backend, flash-attn) —
# only meaningful on a CUDA machine. See scripts/setup_vast.sh.
install-gpu:
	pip install -r requirements.txt
	git submodule update --init --recursive
	pip install -e third_party/R-KV/HuggingFace

# CPU-only test suite; GPU tests auto-skip via tests/conftest.py.
test:
	pytest -m "not gpu" tests/

# Explicit GPU test run — only meaningful on the rented GPU host.
test-gpu:
	pytest -m gpu tests/integration/

dry-run:
	kvcot generate --config configs/stage0_smoke.yaml --condition full --dry-run
	kvcot generate --config configs/stage0_smoke.yaml --condition rkv --dry-run
	kvcot replay-probe --config configs/stage0_smoke.yaml --condition rkv --dry-run
	kvcot analyze --config configs/stage1a_measurability.yaml --dry-run
	kvcot calibrate-budget --config-dir configs --dry-run
	kvcot generate --config configs/stage2_main.yaml --condition full --dry-run

freeze-manifests:
	kvcot freeze-manifests --config configs/lock.yaml

clean:
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache
