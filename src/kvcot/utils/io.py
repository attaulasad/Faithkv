"""Append-safe, resumable JSONL I/O.

Every writer in this repository goes through `JsonlWriter` so resume
semantics (§13: "Resume skips only schema-valid completed records with
matching config/model/upstream hashes") are implemented in exactly one place.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return
    with open(p, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no}: malformed JSONL: {e}") from e


def read_jsonl_auto(path: str | Path) -> Iterator[dict[str, Any]]:
    """Like `read_jsonl`, but transparently reads gzip-compressed JSONL when
    `path` ends in `.gz` (the committed convention for large per-record
    artifacts under `results/gate_artifacts/` — see README.md's "Expected
    artifacts"). Plain `.jsonl` still goes through the exact same line-by-
    line parsing as `read_jsonl` for non-gz paths, so behavior for existing
    callers is unchanged.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}:{line_no}: malformed JSONL: {e}") from e


def read_existing_record_ids(path: str | Path) -> set[str]:
    """Read only `record_id` values from an existing JSONL file, tolerating
    files whose later records are well-formed even if resume was interrupted
    mid-write on a previous run (each line is independently parsed).
    """
    ids: set[str] = set()
    for row in read_jsonl(path):
        rid = row.get("record_id")
        if rid is not None:
            ids.add(rid)
    return ids


class JsonlWriter:
    """Append-only JSONL writer with per-record validation and flush-per-line
    durability (so a crash mid-run loses at most the in-flight record, never
    corrupts prior ones).
    """

    def __init__(
        self,
        path: str | Path,
        validator: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._validator = validator
        self._written_ids: set[str] = read_existing_record_ids(self.path)

    def already_written(self, record_id: str) -> bool:
        return record_id in self._written_ids

    def append(self, record: dict[str, Any]) -> None:
        record_id = record.get("record_id")
        if record_id is not None and record_id in self._written_ids:
            raise ValueError(
                f"record_id {record_id!r} already written to {self.path}; "
                "resume logic should have skipped generating it"
            )
        if self._validator is not None:
            record = self._validator(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if record_id is not None:
            self._written_ids.add(record_id)


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
