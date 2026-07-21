"""Discovery-only single-RTX-3090 preflight and strict local loaders."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable


class StrictDeviceError(RuntimeError):
    pass


# Independent-audit Gate H4: a real RTX 3090 reports total VRAM somewhat
# below its nominal 24 GiB (driver/firmware reservation) -- this range is
# deliberately generous (20-26 GiB) so it rejects a clearly-wrong card
# (e.g. a 16 GiB or 48 GiB device) without being sensitive to the exact
# reporting quirks of a specific driver version.
RTX3090_VRAM_MIN_BYTES = 20 * 1024**3
RTX3090_VRAM_MAX_BYTES = 26 * 1024**3


def verify_device_gate_from_raw_evidence(
    fullkv_device_evidence: dict[str, Any],
    rkv_device_evidence: dict[str, Any],
    cli_device_preflight: dict[str, Any] | None = None,
) -> bool:
    """Independent-audit Gate H4.1/H4.2/H4.3 repair: the final coordinator
    gate must not derive `single_rtx3090_verified` from a worker-reported
    `verified=True` boolean alone -- it must recompute the policy from RAW
    fields, and require every independently-launched observation to AGREE
    on what hardware was actually used.

    Requires, independently for EACH raw evidence dict (FullKV worker, R-KV
    worker, and -- when provided -- the CLI's own pre-launch preflight
    observation): `verified is True`; `visible_gpu_count == 1`;
    `device_index == 0` (the explicit `cuda:0` this repository always
    requests); `gpu_name` contains "RTX 3090"; `total_vram_bytes` falls
    inside the frozen plausibility range; and `driver_version`/
    `cuda_runtime`/`cudnn_version` are all non-empty.

    Then requires every provided observation to report the IDENTICAL
    `gpu_name`, `device_index`, `total_vram_bytes`, `compute_capability`,
    `driver_version`, `cuda_runtime`, and `cudnn_version` -- two (or three)
    independent observations disagreeing about the hardware is itself a
    failure, never silently accepted because each individually claimed
    `verified=True`. `cli_device_preflight` is optional (backward
    compatible with two-way FullKV/R-KV-only callers, e.g. CPU tests that
    never launch the real CLI) -- when `None`, only FullKV/R-KV are
    compared, exactly as before this three-way extension."""

    def _raw_ok(evidence: dict[str, Any]) -> bool:
        if evidence.get("verified") is not True:
            return False
        if evidence.get("visible_gpu_count") != 1:
            return False
        if evidence.get("device_index") != 0:
            return False
        gpu_name = str(evidence.get("gpu_name") or "")
        if "RTX 3090" not in gpu_name.upper():
            return False
        vram = evidence.get("total_vram_bytes")
        if not isinstance(vram, (int, float)) or isinstance(vram, bool):
            return False
        if not (RTX3090_VRAM_MIN_BYTES <= vram <= RTX3090_VRAM_MAX_BYTES):
            return False
        if not evidence.get("driver_version") or not evidence.get("cuda_runtime") or not evidence.get("cudnn_version"):
            return False
        return True

    observations = [fullkv_device_evidence, rkv_device_evidence]
    if cli_device_preflight is not None:
        observations.append(cli_device_preflight)

    if not all(_raw_ok(observation) for observation in observations):
        return False

    agreement_fields = (
        "gpu_name", "device_index", "total_vram_bytes", "compute_capability",
        "driver_version", "cuda_runtime", "cudnn_version",
    )
    reference = observations[0]
    return all(
        observation.get(field) == reference.get(field)
        for observation in observations[1:]
        for field in agreement_fields
    )


@dataclass(frozen=True)
class StrictDeviceEvidence:
    visible_gpu_count: int
    gpu_name: str
    device_index: int
    total_vram_bytes: int
    compute_capability: tuple[int, int]
    driver_version: str
    cuda_runtime: str
    cudnn_version: str
    policy_satisfied: bool


def _driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(values) != 1:
        raise StrictDeviceError(f"expected one driver version, observed {values}")
    return values[0]


def verify_single_rtx3090(
    cuda: Any,
    *,
    torch_module: Any,
    driver_version_fn: Callable[[], str] = _driver_version,
) -> StrictDeviceEvidence:
    count = int(cuda.device_count())
    if count != 1:
        raise StrictDeviceError(f"B2A requires exactly one visible CUDA device, observed {count}")
    index = int(cuda.current_device())
    props = cuda.get_device_properties(index)
    name = str(props.name)
    if "RTX 3090" not in name.upper():
        raise StrictDeviceError(f"B2A requires an RTX 3090, observed {name!r}")
    capability = tuple(int(v) for v in cuda.get_device_capability(index))
    runtime = getattr(getattr(torch_module, "version", None), "cuda", None)
    cudnn = cuda.cudnn.version()
    if runtime is None or cudnn is None:
        raise StrictDeviceError("CUDA runtime and cuDNN versions must both be available")
    return StrictDeviceEvidence(
        visible_gpu_count=count,
        gpu_name=name,
        device_index=index,
        total_vram_bytes=int(props.total_memory),
        compute_capability=capability,
        driver_version=str(driver_version_fn()),
        cuda_runtime=str(runtime),
        cudnn_version=str(cudnn),
        policy_satisfied=True,
    )


def load_fullkv_discovery_model(config: Any, model_snapshot_path: str, device: str = "cuda:0") -> Any:
    from kvcot.generation.state import declare_process_mode
    from transformers import AutoModelForCausalLM
    import torch

    declare_process_mode("stock")
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot_path,
        local_files_only=True,
        torch_dtype=getattr(torch, config.model.dtype),
        low_cpu_mem_usage=True,
        device_map={"": device},
        use_cache=True,
        attn_implementation=config.generation.attention_backend,
    )
    model.eval()
    return model


def load_rkv_discovery_model(
    config: Any, model_snapshot_path: str, tokenizer_snapshot_path: str, device: str = "cuda:0"
) -> Any:
    from kvcot.generation.policies import RKVPolicy, _set_static_token_id_attrs
    from kvcot.generation.state import declare_process_mode
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from kvcot.discovery.dispatch import resolve_patcher
    import torch

    declare_process_mode("patched")
    policy = RKVPolicy(
        budget=config.rkv.budget,
        window_size=config.rkv.window_size,
        mix_lambda=config.rkv.mix_lambda,
        retain_ratio=config.rkv.retain_ratio,
        retain_direction=config.rkv.retain_direction,
        divide_method=config.rkv.divide_method,
        divide_length=config.rkv.divide_length,
        compression_content=config.rkv.compression_content,
        kernel_size=config.rkv.kernel_size,
    )
    auto_config = AutoConfig.from_pretrained(model_snapshot_path, local_files_only=True)
    resolve_patcher(auto_config.model_type, policy._compression_config())
    model = AutoModelForCausalLM.from_pretrained(
        model_snapshot_path,
        local_files_only=True,
        torch_dtype=getattr(torch, config.model.dtype),
        low_cpu_mem_usage=True,
        device_map={"": device},
        use_cache=True,
        attn_implementation=config.generation.attention_backend,
    )
    model.eval()
    model.config.update({
        "divide_method": config.rkv.divide_method,
        "divide_length": config.rkv.divide_length,
        "compression_content": config.rkv.compression_content,
    })
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_snapshot_path, local_files_only=True, use_fast=True)
    _set_static_token_id_attrs(model, tokenizer)
    return model
