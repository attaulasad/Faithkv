"""CPU-only tests for `kvcot failure-atlas` (kvcot.cli.cmd_failure_atlas).
Never imports torch/transformers -- this command reads only committed JSONL
gate artifacts and never loads a model.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kvcot.cli import cmd_failure_atlas
from kvcot.failure_atlas import EXPECTED_PAIR_COUNT
from kvcot.utils.io import read_json


def _record(src_idx: int, *, condition: str) -> dict:
    prompt = list(range(100 + src_idx, 100 + src_idx + 10))
    gen = [1, 2, 3, 4]
    return {
        "record_id": f"base-{condition}-ds-{src_idx}-seed42",
        "condition": condition,
        "global_seed": 42,
        "config_sha256": "cfg1",
        "model_revision": "rev1",
        "tokenizer_revision": "tok1",
        "provenance": {"upstream_rkv_commit": "up1"},
        "dataset": {"source_row_index": src_idx, "question_hash": f"qh{src_idx}"},
        "prompt_token_ids": prompt,
        "generated_token_ids": gen,
        "decoded_output": "reasoning </think> answer",
        "think_span": {"think_parse_status": "ok", "think_end_index": 1},
        "extracted_answer": "1",
        "extraction_method": "boxed",
        "is_correct": True,
        "cap_hit": False,
        "compaction_count": (0 if condition == "full" else 2),
        "compaction_event_steps": ([] if condition == "full" else [5, 6]),
        "retention": (
            None if condition == "full"
            else {
                "fullkv_equivalent_slots": 100,
                "physical_cache_slots_per_layer": [40],
                "instantaneous_retention_ratio": 0.4,
                "post_compaction_budget_tokens": 128,
                "tokens_since_last_compaction": 2,
            }
        ),
    }


def _write_gzip_jsonl(path: Path, records: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_sidecar(gz_path: Path, sha_path: Path) -> None:
    sha_path.write_text(f"{_sha256_file(gz_path)}  {gz_path.name}\n", encoding="utf-8")


def _args(tmp_path, full_path, rkv_path):
    return SimpleNamespace(
        full_artifact=str(full_path),
        rkv_artifact=str(rkv_path),
        output_csv=str(tmp_path / "atlas.csv"),
        output_markdown=str(tmp_path / "atlas.md"),
        output_summary=str(tmp_path / "atlas_summary.json"),
        dry_run=False,
    )


def _write_valid_artifacts(tmp_path):
    full_path = tmp_path / "full.jsonl.gz"
    rkv_path = tmp_path / "rkv_b128.jsonl.gz"
    _write_gzip_jsonl(full_path, [_record(i, condition="full") for i in range(EXPECTED_PAIR_COUNT)])
    _write_gzip_jsonl(rkv_path, [_record(i, condition="rkv_b128") for i in range(EXPECTED_PAIR_COUNT)])
    _write_sidecar(full_path, tmp_path / "full.sha256")
    _write_sidecar(rkv_path, tmp_path / "rkv_b128.sha256")
    return full_path, rkv_path


def test_cmd_failure_atlas_happy_path(tmp_path):
    full_path, rkv_path = _write_valid_artifacts(tmp_path)
    rc = cmd_failure_atlas(_args(tmp_path, full_path, rkv_path))
    assert rc == 0
    summary = read_json(tmp_path / "atlas_summary.json")
    assert summary["n_pairs"] == EXPECTED_PAIR_COUNT
    assert summary["input_artifacts"]["full"]["sha256"] == _sha256_file(full_path)
    assert (tmp_path / "atlas.csv").exists()
    assert (tmp_path / "atlas.md").exists()


def test_cmd_failure_atlas_dry_run_skips_checksum_verification(tmp_path, capsys):
    full_path = tmp_path / "full.jsonl.gz"
    rkv_path = tmp_path / "rkv_b128.jsonl.gz"
    # No .sha256 sidecars written at all -- dry-run must not require them.
    _write_gzip_jsonl(full_path, [])
    _write_gzip_jsonl(rkv_path, [])
    args = _args(tmp_path, full_path, rkv_path)
    args.dry_run = True
    rc = cmd_failure_atlas(args)
    assert rc == 0
    assert "failure-atlas plan" in capsys.readouterr().out


def test_cmd_failure_atlas_missing_checksum_sidecar_raises(tmp_path):
    full_path = tmp_path / "full.jsonl.gz"
    rkv_path = tmp_path / "rkv_b128.jsonl.gz"
    _write_gzip_jsonl(full_path, [_record(i, condition="full") for i in range(EXPECTED_PAIR_COUNT)])
    _write_gzip_jsonl(rkv_path, [_record(i, condition="rkv_b128") for i in range(EXPECTED_PAIR_COUNT)])
    _write_sidecar(rkv_path, tmp_path / "rkv_b128.sha256")
    # full.sha256 deliberately not written.
    with pytest.raises(SystemExit, match="missing checksum sidecar"):
        cmd_failure_atlas(_args(tmp_path, full_path, rkv_path))


def test_cmd_failure_atlas_tampered_artifact_raises(tmp_path):
    full_path, rkv_path = _write_valid_artifacts(tmp_path)
    # Tamper with the artifact bytes after the sidecar was computed.
    with gzip.open(full_path, "at", encoding="utf-8") as f:
        f.write(json.dumps({"tampered": True}) + "\n")
    with pytest.raises(SystemExit, match="checksum mismatch"):
        cmd_failure_atlas(_args(tmp_path, full_path, rkv_path))


def test_cmd_failure_atlas_sidecar_missing_entry_for_artifact_raises(tmp_path):
    full_path, rkv_path = _write_valid_artifacts(tmp_path)
    (tmp_path / "full.sha256").write_text("deadbeef  some_other_file.jsonl.gz\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not contain an entry"):
        cmd_failure_atlas(_args(tmp_path, full_path, rkv_path))
