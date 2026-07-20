"""Discovery-only single-RTX-3090 preflight and strict local loaders."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable


class StrictDeviceError(RuntimeError):
    pass


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
