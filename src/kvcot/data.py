"""Dataset manifest freezing and verification (§5).

Manifests are frozen once (`kvcot freeze-manifests`) and committed as JSONL.
Every manifest row carries only question text, its hash, and the normalized
gold answer — nothing else. This is deliberate: §5's "never copy stale
response/generation fields from a dataset row into evaluation" is grounded
in a real upstream incident (docs/UPSTREAM_AUDIT.md H8) where a leftover
`generation` field in a shipped dataset file silently got scored instead of
fresh model output. The manifest schema here makes that class of bug
structurally impossible — there is no field to copy stale content into.

This module requires the `datasets` and `huggingface_hub` packages
(`pip install -e ".[cpu-tools]"`) but never torch.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterator

from kvcot.utils.hashing import question_hash
from kvcot.utils.io import JsonlWriter, read_jsonl


@dataclass(frozen=True)
class ManifestRow:
    source_row_index: int
    question: str
    question_hash: str
    normalized_gold: str
    dataset_name: str
    dataset_config: str
    dataset_revision: str | None
    dataset_fingerprint: str | None

    def to_dict(self) -> dict:
        return {
            "source_row_index": self.source_row_index,
            "question": self.question,
            "question_hash": self.question_hash,
            "normalized_gold": self.normalized_gold,
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "dataset_revision": self.dataset_revision,
            "dataset_fingerprint": self.dataset_fingerprint,
        }


def _extract_gsm8k_gold(answer_field: str) -> str:
    """GSM8K's `answer` field is full worked solution ending in
    `#### <number>`. Only the part after `####` is the normalized gold —
    the worked solution itself is exactly the kind of stale/leftover field
    §5 warns against carrying into evaluation, so it is discarded here and
    never written to the manifest."""
    marker = "####"
    idx = answer_field.rfind(marker)
    if idx == -1:
        raise ValueError(f"GSM8K answer field has no '####' marker: {answer_field!r}")
    gold = answer_field[idx + len(marker) :].strip()
    gold = gold.replace(",", "")
    return gold


def freeze_gsm8k_manifest(
    n_rows: int,
    seed: int,
    exclude_indices: set[int] | None = None,
) -> list[ManifestRow]:
    """Sample `n_rows` rows from the GSM8K test split, seeded, disjoint from
    `exclude_indices` (used to keep smoke/calibration/main manifests
    disjoint per §5). Requires network + the `datasets` package.
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    fingerprint = getattr(ds, "_fingerprint", None)

    all_indices = list(range(len(ds)))
    if exclude_indices:
        all_indices = [i for i in all_indices if i not in exclude_indices]

    rng = random.Random(seed)
    chosen = sorted(rng.sample(all_indices, n_rows))

    rows: list[ManifestRow] = []
    for idx in chosen:
        row = ds[idx]
        question = row["question"]
        gold = _extract_gsm8k_gold(row["answer"])
        rows.append(
            ManifestRow(
                source_row_index=idx,
                question=question,
                question_hash=question_hash(question),
                normalized_gold=gold,
                dataset_name="gsm8k",
                dataset_config="main",
                dataset_revision=None,
                dataset_fingerprint=fingerprint,
            )
        )
    return rows


def freeze_math500_manifest(n_rows: int, seed: int, levels: tuple[int, ...] = (3, 4, 5)) -> list[ManifestRow]:
    """Backup manifest (§5), frozen but not run unless Stage 1A recommends
    switching off GSM8K. Filters to levels 3-5 before sampling."""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    fingerprint = getattr(ds, "_fingerprint", None)

    eligible_indices = [i for i, row in enumerate(ds) if int(row["level"]) in levels]
    rng = random.Random(seed)
    chosen = sorted(rng.sample(eligible_indices, min(n_rows, len(eligible_indices))))

    rows: list[ManifestRow] = []
    for idx in chosen:
        row = ds[idx]
        question = row["problem"]
        gold = str(row["answer"]).strip()
        rows.append(
            ManifestRow(
                source_row_index=idx,
                question=question,
                question_hash=question_hash(question),
                normalized_gold=gold,
                dataset_name="math500",
                dataset_config="default",
                dataset_revision=None,
                dataset_fingerprint=fingerprint,
            )
        )
    return rows


def write_manifest(rows: list[ManifestRow], path: str) -> None:
    writer = JsonlWriter(path, validator=None)
    for row in rows:
        writer.append({"record_id": f"manifest-{row.dataset_name}-{row.source_row_index}", **row.to_dict()})


def read_manifest(path: str) -> Iterator[dict]:
    yield from read_jsonl(path)


class QuestionHashMismatch(Exception):
    pass


def verify_manifest_row_against_live_question(row: dict, live_question_text: str) -> None:
    """§5: "Verify question hashes on every load." Raises if the live
    dataset's question text at this row no longer matches what was frozen —
    this is the mechanism that would have caught the exact class of bug
    described in docs/UPSTREAM_AUDIT.md H8, one layer earlier (before stale
    text ever reaches evaluation, not after)."""
    live_hash = question_hash(live_question_text)
    if live_hash != row["question_hash"]:
        raise QuestionHashMismatch(
            f"question_hash mismatch for source_row_index={row['source_row_index']}: "
            f"manifest has {row['question_hash']}, live dataset has {live_hash}"
        )
