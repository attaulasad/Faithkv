"""B2A-R2 candidate manifest construction (2026-07-22).

B2A-R1 (the single attempt authorized by CLAUDE.md §1c) executed against
`example_index=0` of the frozen MATH-500 manifest and produced ZERO
compaction events (prompt=105 tokens, generated=449 tokens, well under
R-KV budget=1024) -- an ineligible calibration that tested no eviction at
all. `docs/B2A_R1_FAILURE_AND_B2A_R2_PROTOCOL_2026-07-22.md` pre-registers
a deterministic, outcome-blind replacement-row selection procedure BEFORE
any qualification inference is run:

1. Build a candidate population from the SAME pinned MATH-500 revision
   (level-5 rows only -- the longest, hardest problems, most likely to
   produce a long enough generated trace to actually exercise R-KV's
   budget=1024 eviction trigger).
2. Order candidates by a fixed, content-derived hash (never by observed
   generation length or any other outcome) -- this module.
3. FullKV-only qualification (`kvcot.discovery.b2a_qualification`) attempts
   candidates in this committed order, stopping at the first one that
   satisfies every frozen qualification condition.

This module never loads a model, never imports torch, and performs the
ONE network fetch this procedure requires (the pinned dataset file itself)
directly via `urllib.request`, reusing `kvcot.discovery.manifest_prepare`'s
existing revision-pinned fetch conventions -- never the `datasets` library,
never the mutable `datasets-server` API (see that module's docstring for
why).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from kvcot.discovery.manifest_prepare import EXPECTED_MATH500_COLUMNS, _HF_DATASET_RESOLVE_URL, ManifestPreparationError
from kvcot.utils.hashing import sha256_json, sha256_text

CANDIDATE_MANIFEST_PROTOCOL_VERSION = "faithkv-b2a-r2-row-order-v1"
CANDIDATE_LEVEL = 5
CANDIDATE_COUNT = 12


def _ordering_hash(
    *,
    dataset_revision: str,
    model_revision: str,
    budget: int,
    unique_id: str,
    protocol_version: str = CANDIDATE_MANIFEST_PROTOCOL_VERSION,
) -> str:
    """The frozen, pre-registered ordering key -- a fixed function of
    identity fields ONLY (dataset revision, model revision, budget,
    unique_id). Never a function of anything observed about generation
    (length, answer, compaction count): the whole point of pre-registration
    is that this order is fixed BEFORE any qualification inference runs.

    `protocol_version` defaults to this module's own
    `CANDIDATE_MANIFEST_PROTOCOL_VERSION`, preserving B2A-R2's historical
    byte-for-byte hash construction exactly for every existing caller
    (`build_candidate_manifest` below never passes this argument).
    B2A-R3 (`kvcot.discovery.b2a_r3_candidates`) reuses this SAME function,
    parameterized with its own distinct
    `"faithkv-b2a-r3-row-order-v1"` protocol-version string
    (protocol §9), rather than reimplementing the payload construction
    independently."""
    payload = f"{protocol_version}|{dataset_revision}|{model_revision}|budget={budget}|{unique_id}"
    return sha256_text(payload)


def fetch_all_pinned_dataset_rows(dataset_repo: str, dataset_revision: str) -> list[dict[str, Any]]:
    """Every row of the revision-pinned `test.jsonl`, in file order. Same
    URL convention and same refusal-on-network-failure behavior as
    `kvcot.discovery.manifest_prepare._fetch_pinned_dataset_row`, generalized
    to the whole file since candidate-population construction needs every
    row, not one by index."""
    url = _HF_DATASET_RESOLVE_URL.format(repo=dataset_repo, revision=dataset_revision)
    rows: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 -- fixed https:// HF URL, not user input
            for raw_line in resp:
                text = raw_line.decode("utf-8").rstrip("\r\n")
                if not text:
                    continue
                rows.append(json.loads(text))
    except urllib.error.URLError as exc:
        raise ManifestPreparationError(f"failed to fetch {url}: {exc}") from exc
    return rows


@dataclass(frozen=True)
class CandidateRow:
    candidate_ordinal: int
    source_example_index: int
    unique_id: str
    subject: str
    level: int
    row: dict[str, Any]
    raw_row_sha256: str
    problem_sha256: str
    gold_answer_sha256: str
    ordering_hash: str
    dataset_revision: str
    model_revision: str
    tokenizer_revision: str
    budget: int
    protocol_version: str

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_ordinal": self.candidate_ordinal,
            "source_example_index": self.source_example_index,
            "unique_id": self.unique_id,
            "subject": self.subject,
            "level": self.level,
            "row": self.row,
            "raw_row_sha256": self.raw_row_sha256,
            "problem_sha256": self.problem_sha256,
            "gold_answer_sha256": self.gold_answer_sha256,
            "ordering_hash": self.ordering_hash,
            "dataset_revision": self.dataset_revision,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "budget": self.budget,
            "protocol_version": self.protocol_version,
        }


def build_candidate_manifest(
    all_rows_in_file_order: list[dict[str, Any]],
    *,
    dataset_repo: str,
    dataset_revision: str,
    model_revision: str,
    tokenizer_revision: str,
    budget: int,
    level: int = CANDIDATE_LEVEL,
    candidate_count: int = CANDIDATE_COUNT,
) -> dict[str, Any]:
    """Deterministic candidate-population construction:

    1. Filter to `level` (as a string, matching the raw dataset's own
       `"level"` column convention: HuggingFaceH4/MATH-500 stores level as
       a bare digit string, e.g. `"5"`) rows only, verifying every row's
       schema first.
    2. Reject any duplicate `unique_id` (fail loudly -- never silently keep
       the first/last).
    3. Compute the frozen ordering hash for every eligible row and sort
       ascending by it.
    4. Take exactly the first `candidate_count` rows in that order.

    Never selects by observed generation length or any other outcome --
    the entire population is fixed by dataset content and identity fields
    alone, before any model is ever loaded.
    """
    eligible: list[tuple[int, dict[str, Any]]] = []
    seen_unique_ids: set[str] = set()
    for index, row in enumerate(all_rows_in_file_order):
        got = tuple(row.keys())
        if got != EXPECTED_MATH500_COLUMNS:
            raise ManifestPreparationError(
                f"row {index} has unexpected columns: expected {EXPECTED_MATH500_COLUMNS}, got {got}"
            )
        if str(row["level"]) != str(level):
            continue
        unique_id = row["unique_id"]
        if unique_id in seen_unique_ids:
            raise ManifestPreparationError(f"duplicate unique_id {unique_id!r} in pinned dataset -- refusing to proceed")
        seen_unique_ids.add(unique_id)
        eligible.append((index, row))

    scored = [
        (
            _ordering_hash(
                dataset_revision=dataset_revision, model_revision=model_revision, budget=budget,
                unique_id=row["unique_id"],
            ),
            index,
            row,
        )
        for index, row in eligible
    ]
    scored.sort(key=lambda item: item[0])
    selected = scored[:candidate_count]

    candidates = [
        CandidateRow(
            candidate_ordinal=ordinal,
            source_example_index=index,
            unique_id=row["unique_id"],
            subject=row["subject"],
            level=int(row["level"]),
            row=row,
            raw_row_sha256=sha256_json(row),
            problem_sha256=sha256_text(row["problem"]),
            gold_answer_sha256=sha256_text(row["answer"]),
            ordering_hash=ordering_hash,
            dataset_revision=dataset_revision,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            budget=budget,
            protocol_version=CANDIDATE_MANIFEST_PROTOCOL_VERSION,
        )
        for ordinal, (ordering_hash, index, row) in enumerate(selected)
    ]

    manifest = {
        "protocol_version": CANDIDATE_MANIFEST_PROTOCOL_VERSION,
        "dataset_repo": dataset_repo,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "budget": budget,
        "level": level,
        "candidate_count": len(candidates),
        "eligible_population_size": len(eligible),
        "candidates": [c.to_json() for c in candidates],
    }
    manifest["canonical_sha256"] = sha256_json(manifest)
    return manifest
