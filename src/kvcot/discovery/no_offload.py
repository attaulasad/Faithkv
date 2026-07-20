"""No-offload hard assertion for real GPU model construction (Part V.12,
repaired B1A Blocker 1: `docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md`).

Duck-typed against any object exposing `.named_parameters()` yielding
`(name, tensor)` pairs where `tensor.device.type` is a string — this is
`torch.nn.Module`'s real interface, but the function itself imports nothing
from torch, so it is testable with plain fake objects on a CPU-only,
no-GPU machine (this repository's build environment) without requiring a
GPU or even a torch install.

MUST be called unconditionally after every real model construction — never
guarded behind `if model.device.type == "cuda":` first. `model.device` is a
single reported property (often just the device of the first parameter, or
even a stale/misleading cached value) and is NOT a substitute for actually
walking every parameter: a `device_map="auto"` load can place some
parameters on `cuda`, some on `cpu`, and some on `disk`/`meta`, while
`model.device` still reports `cuda` for the first one. Checking
`model.device` first would let exactly that partially-offloaded model skip
the real per-parameter check entirely.
"""
from __future__ import annotations

_REJECTED_DEVICE_MAP_TARGETS = frozenset({"cpu", "disk", "meta"})


class ModelOffloadError(RuntimeError):
    pass


def assert_no_offloaded_parameters(model) -> None:
    """Every named parameter of `model` must have `device.type == "cuda"`,
    and (when present) every `model.hf_device_map` entry must not be
    assigned to `cpu`/`disk`/`meta`. Raises `ModelOffloadError`, naming
    every offending parameter/device and every offending device-map entry,
    on the first violation set found — never silently continues with a
    partially-offloaded model. Never reads `model.device` — that single
    reported property cannot detect a partially-offloaded model and is not
    an acceptable substitute for this per-parameter walk.

    A model with zero named parameters is also a failure (vacuous pass is
    not acceptable here — an empty parameter iterator gives no evidence
    anything is actually on `cuda`).
    """
    offenders: list[tuple[str, str]] = []
    n_params = 0
    for name, param in model.named_parameters():
        n_params += 1
        device_type = param.device.type
        if device_type != "cuda":
            offenders.append((name, device_type))

    device_map_offenders: list[tuple[str, str]] = []
    hf_device_map = getattr(model, "hf_device_map", None)
    if hf_device_map:
        for entry_name, target in hf_device_map.items():
            target_str = str(target).lower()
            if target_str in _REJECTED_DEVICE_MAP_TARGETS:
                device_map_offenders.append((entry_name, str(target)))

    errors: list[str] = []
    if n_params == 0:
        errors.append("model has zero named parameters -- cannot vacuously pass the no-offload assertion")
    if offenders:
        formatted = ", ".join(f"{name!r} (device.type={device_type!r})" for name, device_type in offenders)
        errors.append(f"{len(offenders)} model parameter(s) are not on cuda: {formatted}")
    if device_map_offenders:
        formatted_map = ", ".join(f"{name!r} -> {target!r}" for name, target in device_map_offenders)
        errors.append(f"{len(device_map_offenders)} hf_device_map entr(y/ies) assigned to cpu/disk/meta: {formatted_map}")

    if errors:
        raise ModelOffloadError(
            "offloading is not authorized for this repository's GPU path -- " + "; ".join(errors)
        )
