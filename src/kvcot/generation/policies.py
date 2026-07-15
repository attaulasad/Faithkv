"""Condition policies: FullKV (stock), patched-noop (diagnostic), R-KV
(§3.2, §10). One Policy subtype's model may ever be loaded per OS process —
enforced via kvcot.generation.state.declare_process_mode, since the R-KV
monkeypatch is a process-global class patch with no per-instance undo
(docs/UPSTREAM_AUDIT.md H1).

Imports torch/transformers/rkv at call time inside `.load()`, not at module
scope for the `rkv` package specifically (it lives in the pinned submodule
and is only installed on the GPU host) — torch/transformers themselves are
imported at module scope here since this whole module is only ever imported
from a real generation code path (see pyproject.toml's deferred-import
note), never from `--dry-run`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class RKVMethodConfig:
    budget: int
    window_size: int = 8
    mix_lambda: float = 0.1
    retain_ratio: float = 0.2
    retain_direction: str = "last"
    divide_method: str = "step_length"
    divide_length: int = 128
    compression_content: str = "all"
    # Our provenance adapter needs this; docs/UPSTREAM_AUDIT.md H5 confirms
    # enabling it does not alter R-KV's scores or selected indices.
    record_kept_token_indices: bool = True
    update_kv: bool = True


class Policy:
    """Base interface. Subclasses load a model in a fresh process and know
    how to build the compression_config dict the monkeypatch consumes."""

    condition_name: str

    def load(self, model_name: str, revision: str, dtype: torch.dtype, attn_implementation: str) -> Any:
        raise NotImplementedError

    def fresh_cache(self):
        from transformers.cache_utils import DynamicCache

        return DynamicCache()


def _set_static_token_id_attrs(model, tokenizer) -> None:
    """model.newline_token_ids / model.after_think_token_ids — set once,
    static (run_math.py:271-282, UPSTREAM_AUDIT.md §3.2). Our frozen
    divide_method=step_length never reads newline_token_ids and
    compression_content=all never reads after_think_token_ids
    (modeling.py:593-596 only fires when compression_content=="think"), but
    both are set anyway for parity with upstream's own setup and so a
    future CHANGELOG.md-gated config change to compression_content=think
    does not hit a missing-attribute error."""
    model.newline_token_ids = [
        tokenizer.encode("\n")[-1],
        tokenizer.encode(".\n")[-1],
        tokenizer.encode(")\n")[-1],
        tokenizer.encode("\n\n")[-1],
        tokenizer.encode(".\n\n")[-1],
        tokenizer.encode(")\n\n")[-1],
    ]
    model.after_think_token_ids = [tokenizer.encode("</think>")[-1]]


class FullKVPolicy(Policy):
    condition_name = "full"

    def load(self, model_name, revision, dtype, attn_implementation):
        from kvcot.generation.state import declare_process_mode

        declare_process_mode("stock")
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            attn_implementation=attn_implementation,
        )
        model.eval()
        return model


class _PatchedPolicyBase(Policy):
    def __init__(self, method_config: RKVMethodConfig):
        self.method_config = method_config

    def _compression_config(self) -> dict:
        mc = self.method_config
        return {
            "method": "rkv",
            "method_config": {
                "budget": mc.budget,
                "window_size": mc.window_size,
                "mix_lambda": mc.mix_lambda,
                "retain_ratio": mc.retain_ratio,
                "retain_direction": mc.retain_direction,
                "first_tokens": 4,  # accepted by upstream's CLI shape; unused by R1KV.__init__ (absorbed by **kwargs)
                "record_kept_token_indices": mc.record_kept_token_indices,
            },
            "compression": None,  # initial tri-state value, modeling.py:230-234 / run_math.py:230
            "update_kv": mc.update_kv,
        }

    def load(self, model_name, revision, dtype, attn_implementation):
        from kvcot.generation.state import declare_process_mode

        declare_process_mode("patched")
        # `rkv` is the pinned submodule package (third_party/R-KV/HuggingFace),
        # installed editable on the GPU host only (scripts/setup_vast.sh).
        from rkv.monkeypatch import replace_qwen2
        from transformers import AutoModelForCausalLM, AutoTokenizer

        replace_qwen2(self._compression_config())

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            use_cache=True,
            attn_implementation=attn_implementation,
        )
        model.eval()
        model.config.update(
            {
                "divide_method": self.method_config.divide_method,
                "divide_length": self.method_config.divide_length,
                "compression_content": self.method_config.compression_content,
            }
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True)
        _set_static_token_id_attrs(model, tokenizer)
        return model


class PatchedNoopPolicy(_PatchedPolicyBase):
    """Diagnostic condition (§3.2): R-KV patch enabled, budget larger than
    the entire smoke trace, so compression never fires. Used only to
    validate patched-vs-stock parity (tests/integration/
    test_patched_noop_parity_gpu.py); never a real experimental condition."""

    condition_name = "patched_noop"

    def __init__(self, divide_method: str = "step_length", divide_length: int = 128, compression_content: str = "all", noop_budget: int = 1_000_000):
        super().__init__(
            RKVMethodConfig(
                budget=noop_budget,
                divide_method=divide_method,
                divide_length=divide_length,
                compression_content=compression_content,
            )
        )


class RKVPolicy(_PatchedPolicyBase):
    def __init__(
        self,
        budget: int,
        window_size: int = 8,
        mix_lambda: float = 0.1,
        retain_ratio: float = 0.2,
        retain_direction: str = "last",
        divide_method: str = "step_length",
        divide_length: int = 128,
        compression_content: str = "all",
    ):
        super().__init__(
            RKVMethodConfig(
                budget=budget,
                window_size=window_size,
                mix_lambda=mix_lambda,
                retain_ratio=retain_ratio,
                retain_direction=retain_direction,
                divide_method=divide_method,
                divide_length=divide_length,
                compression_content=compression_content,
            )
        )
        self.condition_name = f"rkv_b{budget}"


def build_policy(condition: str, lock) -> Policy:
    """Construct the right Policy from a condition string ("full",
    "patched_noop", or "rkv_b{budget}") plus the frozen lock settings
    (kvcot.config.FrozenSettings). This is the only place condition-name
    strings are parsed back into policy objects — keep it that way so a
    naming drift shows up in one place, not scattered call sites.
    """
    if condition == "full":
        return FullKVPolicy()
    if condition == "patched_noop":
        return PatchedNoopPolicy(
            divide_method=lock.rkv.divide_method,
            divide_length=lock.rkv.divide_length,
            compression_content=lock.rkv.compression_content,
        )
    if condition.startswith("rkv_b"):
        budget = int(condition[len("rkv_b") :])
        return RKVPolicy(
            budget=budget,
            window_size=lock.rkv.window_size,
            mix_lambda=lock.rkv.mix_lambda,
            retain_ratio=lock.rkv.retain_ratio,
            retain_direction=lock.rkv.retain_direction,
            divide_method=lock.rkv.divide_method,
            divide_length=lock.rkv.divide_length,
            compression_content=lock.rkv.compression_content,
        )
    raise ValueError(f"unrecognized condition: {condition!r}")
