"""Frozen framework-seed application and determinism-policy recording
(B1B-R3 §15). Applied once, before either worker (FullKV / R-KV) runs.

Two seeds are deliberately distinguished, never conflated:

- The **selection seed** (`kvcot.discovery.sampling.IdentitySeedParts`) --
  drives event/depth/layer/head/candidate/donor selection deterministically
  from `(global_seed, dataset_name, problem_index, model_revision,
  rkv_revision)`, unrelated to framework RNG state.
- The **framework execution seed**
  (`config.generation.framework_seed`, frozen at `13`,
  `kvcot.discovery.discovery_config.DiscoveryGenerationLock`) -- applied
  here to Python's and PyTorch's own RNGs, for reproducibility of any
  framework-level nondeterminism (dict/set iteration order, etc.), even
  though token selection itself is greedy/argmax (no sampling).

This module records exactly what determinism guarantee was actually
requested -- it never claims full bitwise determinism where FlashAttention
or CUDA kernels cannot provide it (B1B-R3 §15: "Do not claim full bitwise
determinism where FlashAttention or CUDA kernels cannot guarantee it")."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeterminismPolicy:
    framework_seed: int
    python_random_seeded: bool
    torch_cpu_seeded: bool
    torch_cuda_seeded: bool
    cudnn_deterministic_requested: bool
    attention_backend: str
    bitwise_determinism_guaranteed: bool
    tolerance_note: str
    # B1B-R4.1 §28 repair: `random.seed()` does NOT control Python's hash
    # randomization seed (dict/set iteration order for str/bytes/datetime
    # keys) -- that is fixed once, at interpreter STARTUP, from the
    # `PYTHONHASHSEED` environment variable, and cannot be changed by any
    # code running inside the process after it has already started. This
    # field is a genuine RUNTIME OBSERVATION (`os.environ.get`), reflecting
    # whatever the process launcher (`kvcot.discovery.b2a_workers
    # ._launch_worker`, which must set it BEFORE the worker subprocess
    # starts, never inside this already-running process) actually set --
    # never a claim this function itself makes true.
    pythonhashseed_env_value: str | None


def apply_framework_seed(framework_seed: int, attention_backend: str, cuda_available: bool) -> DeterminismPolicy:
    """Set Python's and PyTorch's RNG state from `framework_seed`. CUDA
    seeding and `cudnn.deterministic` are only requested when
    `cuda_available` is True (this repository's own CPU-only build never
    reaches that branch, matching every other GPU-only code path's
    deferred-import discipline). Returns the exact policy applied, never a
    bare boolean -- `bitwise_determinism_guaranteed` is always `False` for
    `attention_backend="flash_attention_2"` (FlashAttention's own kernels
    are not guaranteed bitwise-deterministic across runs, independent of
    seeding) and is documented as such in `tolerance_note`."""
    import os
    import random

    random.seed(framework_seed)

    torch_cpu_seeded = False
    torch_cuda_seeded = False
    cudnn_deterministic_requested = False
    try:
        import torch

        torch.manual_seed(framework_seed)
        torch_cpu_seeded = True
        if cuda_available:
            torch.cuda.manual_seed_all(framework_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch_cuda_seeded = True
            cudnn_deterministic_requested = True
    except ImportError:
        pass

    bitwise_guaranteed = attention_backend not in ("flash_attention_2",)
    tolerance_note = (
        "Greedy/argmax decoding removes sampling nondeterminism; RNG seeding covers residual "
        "framework-level nondeterminism (dict/set iteration order, etc.) only. "
        + (
            "FlashAttention 2 kernels are not guaranteed bitwise-deterministic across runs/hardware "
            "independent of seeding -- token-identical replay parity is verified by direct comparison "
            "(token_identical_replay evidence), never assumed from seeding alone."
            if attention_backend == "flash_attention_2"
            else "No known non-deterministic kernel is in use for this attention_backend."
        )
    )

    return DeterminismPolicy(
        framework_seed=framework_seed,
        python_random_seeded=True,
        torch_cpu_seeded=torch_cpu_seeded,
        torch_cuda_seeded=torch_cuda_seeded,
        cudnn_deterministic_requested=cudnn_deterministic_requested,
        attention_backend=attention_backend,
        bitwise_determinism_guaranteed=bitwise_guaranteed,
        tolerance_note=tolerance_note,
        pythonhashseed_env_value=os.environ.get("PYTHONHASHSEED"),
    )
