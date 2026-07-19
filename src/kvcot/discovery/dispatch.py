"""Architecture-aware R-KV monkeypatch dispatch (Part V.9 of
`docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md`).

`src/kvcot/generation/policies.py` previously called
`rkv.monkeypatch.replace_qwen2` unconditionally — loading a non-Qwen2
checkpoint (e.g. the discovery track's Llama-8B) would silently leave its
attention/CausalLM classes unpatched (stock, FullKV-equivalent behavior,
compression never fires) while still being labeled `rkv_b{budget}`: a
silent mislabeling defect, not a crash.

Verified directly against the pinned submodule
(`third_party/R-KV/HuggingFace/rkv/monkeypatch.py`, commit
`45eaa7d69d20b7388321f077020a610d9afb65bd`): it exports exactly
`replace_llama`, `replace_qwen2`, and `replace_qwen3` — no other
architecture. This module only maps to those three verified names; it never
invents an import.

Pure Python (no torch/transformers import at module scope) — the actual
`rkv.monkeypatch` import happens inside `resolve_patcher`, at call time,
exactly like every other real-`rkv`-package touchpoint in this repository
(`kvcot.generation.policies`).
"""
from __future__ import annotations

from typing import Callable

# Verified against third_party/R-KV/HuggingFace/rkv/monkeypatch.py -- do not
# add an entry here without confirming the pinned submodule actually
# exports and supports that architecture (never claim support merely
# because a document mentions it).
MODEL_TYPE_TO_PATCHER_NAME: dict[str, str] = {
    "qwen2": "replace_qwen2",
    "llama": "replace_llama",
    "qwen3": "replace_qwen3",
}


class UnsupportedArchitectureError(RuntimeError):
    pass


def resolve_patcher_name(model_type: str) -> str:
    """Pure lookup, no import -- testable without `rkv`/`transformers`
    installed at all. Raises before any model construction ever happens for
    an unknown architecture; there is no default/fallback patcher."""
    try:
        return MODEL_TYPE_TO_PATCHER_NAME[model_type]
    except KeyError:
        raise UnsupportedArchitectureError(
            f"no verified R-KV monkeypatch for model_type={model_type!r}. "
            f"Supported model_type values: {sorted(MODEL_TYPE_TO_PATCHER_NAME)}. "
            "There is no default/fallback patcher -- add explicit support only "
            "after verifying the pinned third_party/R-KV submodule actually "
            "exports and supports the architecture."
        ) from None


def resolve_patcher(model_type: str, compression_config: dict) -> Callable[[], None]:
    """Resolve and apply the correct process-global R-KV patcher for
    `model_type`, returning nothing (the patch is applied as a side
    effect, matching upstream's own `replace_*(compression_config)` shape)
    -- callers must invoke this strictly BEFORE constructing
    `AutoModelForCausalLM`, matching the required dispatch order:
    1. load/read AutoConfig, 2. determine model_type, 3. resolve+invoke the
    patcher, 4. only then construct the model.
    """
    patcher_name = resolve_patcher_name(model_type)
    from rkv import monkeypatch as rkv_monkeypatch  # pinned submodule; GPU host only

    patcher = getattr(rkv_monkeypatch, patcher_name, None)
    if patcher is None:
        raise UnsupportedArchitectureError(
            f"rkv.monkeypatch has no function named {patcher_name!r} -- the pinned "
            "R-KV submodule may have changed since this dispatch table was verified; "
            "do not invent a fallback."
        )
    patcher(compression_config)
