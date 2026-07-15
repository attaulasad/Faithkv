"""SHA-256 hashing helpers used for question identity, config identity, and
record/state fingerprints. Every hash in this repository goes through this
module so the digest format (lowercase hex, explicit UTF-8 encoding) is
consistent everywhere records are compared or deduplicated.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    """Hash a JSON-serializable object via its canonical (sorted-key, no
    whitespace) JSON encoding, so equal objects always hash identically
    regardless of dict insertion order.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(canonical)


def sha256_int_ids(ids: list[int]) -> str:
    """Hash a sequence of token/position ids without going through JSON, for
    hot paths (per-snapshot provenance hashing) where JSON overhead matters.
    """
    packed = ",".join(str(i) for i in ids)
    return sha256_text(packed)


def question_hash(question_text: str) -> str:
    """Canonical hash of a dataset question's exact text (§5): used to verify
    a manifest row still matches the live dataset row it was frozen from.
    Whitespace is NOT normalized here — the brief requires the *exact*
    question text to be hashed and verified, so accidental upstream
    whitespace changes must be caught, not silently absorbed.
    """
    return sha256_text(question_text)
