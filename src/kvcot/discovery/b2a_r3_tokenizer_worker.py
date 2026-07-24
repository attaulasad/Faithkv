"""Subprocess-only B2A-R3 tokenizer rendering worker.

This module is launched with ``python -m`` by
``b2a_r3_production_tokenizer``. It exists so production prompt rendering
can prove a clean boundary: no Torch import, no CUDA module, and no model
weight file opened while using the exact local tokenizer snapshot.
"""
from __future__ import annotations

import builtins
import fnmatch
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

_FORBIDDEN_WEIGHT_NAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
}
_FORBIDDEN_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.pth")


def _torch_modules() -> tuple[str, ...]:
    return tuple(sorted(name for name in sys.modules if name == "torch" or name.startswith("torch.")))


def _install_torch_import_guard() -> None:
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
        root = name.split(".", 1)[0]
        if root == "torch":
            raise ImportError("B2A-R3 tokenizer worker forbids torch imports")
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def _is_forbidden_model_weight_path(path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> bool:
    try:
        raw = os.fsdecode(path)
    except TypeError:
        return False
    name = Path(raw).name
    return name in _FORBIDDEN_WEIGHT_NAMES or any(fnmatch.fnmatch(name, pattern) for pattern in _FORBIDDEN_WEIGHT_PATTERNS)


def _install_model_weight_open_guard(opened_forbidden: list[str]) -> None:
    real_builtin_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open

    def _record_if_forbidden(path: Any) -> None:
        if _is_forbidden_model_weight_path(path):
            rendered = os.fsdecode(path)
            opened_forbidden.append(rendered)
            raise RuntimeError(f"B2A-R3 tokenizer worker refused to open model-weight file: {rendered}")

    def guarded_builtin_open(file, *args, **kwargs):  # noqa: A002
        _record_if_forbidden(file)
        return real_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file, *args, **kwargs):  # noqa: A002
        _record_if_forbidden(file)
        return real_io_open(file, *args, **kwargs)

    def guarded_os_open(path, flags, mode=0o777, *, dir_fd=None):
        _record_if_forbidden(path)
        return real_os_open(path, flags, mode, dir_fd=dir_fd)

    builtins.open = guarded_builtin_open
    io.open = guarded_io_open
    os.open = guarded_os_open


def _render(payload: dict[str, Any]) -> dict[str, Any]:
    from kvcot.discovery.b2a_r3_contract import (
        PROMPT_ADD_GENERATION,
        PROMPT_MESSAGE_ROLES,
        PROMPT_SPECIAL_TOKENS_NOTE,
        PROMPT_TOKENIZE,
    )
    from kvcot.discovery.manifest import ChatTemplateRenderingConfig
    from kvcot.discovery.manifest_prepare import render_with_loaded_tokenizer
    from kvcot.utils.hashing import sha256_int_ids, sha256_json, sha256_text
    from transformers import AutoTokenizer

    local_path = payload["snapshot"]["local_path"]
    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True, use_fast=True)
    if not tokenizer.chat_template:
        raise RuntimeError("production tokenizer has no chat_template")

    user_message, messages, token_ids = render_with_loaded_tokenizer(tokenizer, payload["row"])
    if not token_ids:
        raise RuntimeError("production tokenizer rendered an empty prompt")

    config = ChatTemplateRenderingConfig(
        message_roles=PROMPT_MESSAGE_ROLES,
        add_generation_prompt=PROMPT_ADD_GENERATION,
        tokenize=PROMPT_TOKENIZE,
        add_special_tokens_note=PROMPT_SPECIAL_TOKENS_NOTE,
    )
    return {
        "rendered_user_message_sha256": sha256_text(user_message),
        "chat_template_source_sha256": sha256_text(tokenizer.chat_template),
        "chat_message_payload_sha256": sha256_json(messages),
        "prompt_token_ids": list(token_ids),
        "prompt_token_ids_sha256": sha256_int_ids(token_ids),
        "prompt_token_count": len(token_ids),
        "tokenizer_revision_used_for_prompt_hash": payload["snapshot"]["resolved_revision"],
        "prompt_rendering_config": config.model_dump(mode="json"),
    }


def main() -> int:
    payload = json.loads(sys.stdin.read())
    if _torch_modules():
        raise RuntimeError(f"torch modules were already loaded at worker start: {_torch_modules()}")
    _install_torch_import_guard()
    opened_forbidden: list[str] = []
    _install_model_weight_open_guard(opened_forbidden)

    result = _render(payload)
    torch_modules_after = _torch_modules()
    if torch_modules_after:
        raise RuntimeError(f"torch modules entered tokenizer worker: {torch_modules_after}")
    if opened_forbidden:
        raise RuntimeError(f"model-weight file open attempts occurred: {opened_forbidden}")

    json.dump(
        {
            "result": result,
            "audit": {
                "torch_modules_at_start": [],
                "torch_modules_at_exit": [],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                "model_weight_open_attempts": opened_forbidden,
            },
        },
        sys.stdout,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
