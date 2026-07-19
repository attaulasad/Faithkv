"""No-offload hard assertion for real GPU model construction (Part V.12).

Duck-typed against any object exposing `.named_parameters()` yielding
`(name, tensor)` pairs where `tensor.device.type` is a string — this is
`torch.nn.Module`'s real interface, but the function itself imports nothing
from torch, so it is testable with plain fake objects on a CPU-only,
no-GPU machine (this repository's build environment) without requiring a
GPU or even a torch install.
"""
from __future__ import annotations


class ModelOffloadError(RuntimeError):
    pass


def assert_no_offloaded_parameters(model) -> None:
    """Every parameter of `model` must have `device.type == "cuda"`. Raises
    `ModelOffloadError`, naming every offending parameter and its actual
    device, on the first violation set found — never silently continues
    with a partially-offloaded model."""
    offenders = []
    for name, param in model.named_parameters():
        device_type = param.device.type
        if device_type != "cuda":
            offenders.append((name, device_type))
    if offenders:
        formatted = ", ".join(f"{name!r} (device.type={device_type!r})" for name, device_type in offenders)
        raise ModelOffloadError(
            f"{len(offenders)} model parameter(s) are not on cuda -- offloading is not "
            f"authorized for this repository's GPU path: {formatted}"
        )
