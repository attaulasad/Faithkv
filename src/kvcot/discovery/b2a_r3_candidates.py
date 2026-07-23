"""B2A-R3 deterministic candidate-manifest construction (Step 3 Stage-A,
protocol §8, §8.1, §8.2, §9, §12.3, §12.4).

CPU-only: fetches the pinned MATH-500 revision's raw file directly (the
same revision-pinned `resolve/<revision>/test.jsonl` convention
`kvcot.discovery.b2a_r2_candidates`/`kvcot.discovery.manifest_prepare`
already use — never the mutable `datasets-server` API), computes every
hash with the repository's existing `kvcot.utils.hashing` helpers, and
never loads a model, a tokenizer, or touches CUDA.

Construction order (protocol §8.2), exactly:

1. Load every row from the pinned revision.
2. Verify every row's columns against the frozen embedded-row schema.
3. Reject any duplicate `unique_id` across the COMPLETE loaded population.
4. Remove every row whose `unique_id` is in the frozen 13-row exclusion
   set (protocol §8.1).
5. Keep only `level == "4"` or `level == "5"` rows.
6. Compute the frozen B2A-R3 ordering hash for every eligible row (reusing
   `kvcot.discovery.b2a_r2_candidates._ordering_hash`, parameterized by
   this round's own `candidate_order_protocol_version`, never a second,
   independently-written payload construction).
7/8. Sort the level-4 and level-5 subsets INDEPENDENTLY by
   `(ordering_hash, unique_id)` ascending.
9/10. Take the first 8 ranked rows of each level.
11. Interleave level-4-rank-i / level-5-rank-i, level-4 first.
12. Assign `candidate_ordinal` only after interleaving.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from kvcot.discovery.b2a_r2_candidates import _ordering_hash, fetch_all_pinned_dataset_rows
from kvcot.discovery.b2a_r3_contract import (
    CANDIDATE_MANIFEST_ARTIFACT_SCHEMA_VERSION,
    CANDIDATE_ORDER_PROTOCOL_VERSION,
    CANDIDATE_TOTAL_COUNT,
    CANDIDATES_PER_LEVEL,
    EMBEDDED_ROW_COLUMNS,
    EXCLUSION_SET,
    EXCLUSION_SET_SHA256,
    GENERATION_CONFIG_SHA256,
    QUALIFICATION_CANDIDATE_LIMIT,
    require_lowercase_hex64,
    verify_canonical_sha256,
)
from kvcot.discovery.manifest_prepare import ManifestPreparationError
from kvcot.utils.hashing import sha256_json, sha256_text

__all__ = [
    "CandidateRowR3",
    "CandidateManifestR3",
    "build_candidate_manifest",
    "verify_candidate_manifest_structure",
    "verify_candidate_manifest_against_dataset",
    "atomic_write_json",
]


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """The frozen 8-step atomic-JSON-write procedure (task §8): a temp file
    in the target directory, deterministic UTF-8 JSON (fixed field
    insertion order, never alphabetically re-sorted -- an embedded MATH-500
    `row`'s column ORDER is itself part of what gets verified on reload, so
    this must never use `sort_keys=True` the way `sha256_json`'s hashing
    path deliberately does), flush, fsync the temp file, atomic
    `os.replace`, best-effort fsync of the parent directory (skipped where
    the platform does not support directory fsync, e.g. Windows), then
    read back, parse, and require an exact round-trip."""
    import json
    import os
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=True) + "\n"

    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".b2a-r3-", suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, target)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise

    try:
        dir_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except (OSError, AttributeError):
        pass  # directory fsync unsupported on this platform -- best-effort only

    with open(target, "r", encoding="utf-8") as f:
        round_tripped = json.load(f)
    if round_tripped != payload:
        raise ManifestPreparationError(f"atomic write of {target} did not round-trip byte-identically")


class CandidateRowR3(BaseModel):
    """Protocol §12.4. Strict, immutable, extra fields forbidden."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

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

    @field_validator("candidate_ordinal")
    @classmethod
    def _ordinal_in_range(cls, v: int) -> int:
        if not (0 <= v <= 15):
            raise ValueError(f"candidate_ordinal must be in [0, 15], got {v}")
        return v

    @field_validator("source_example_index")
    @classmethod
    def _index_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"source_example_index must be >= 0, got {v}")
        return v

    @field_validator("level")
    @classmethod
    def _level_is_4_or_5(cls, v: int) -> int:
        if v not in (4, 5):
            raise ValueError(f"level must be 4 or 5, got {v}")
        return v

    @field_validator("raw_row_sha256", "problem_sha256", "gold_answer_sha256", "ordering_hash")
    @classmethod
    def _hex64(cls, v: str, info: Any) -> str:
        return require_lowercase_hex64(v, info.field_name)

    @model_validator(mode="after")
    def _row_matches_declared_identity_and_hashes(self) -> "CandidateRowR3":
        if tuple(self.row.keys()) != EMBEDDED_ROW_COLUMNS:
            raise ValueError(
                f"embedded row has columns {tuple(self.row.keys())}, expected {EMBEDDED_ROW_COLUMNS}"
            )
        if sha256_json(self.row) != self.raw_row_sha256:
            raise ValueError("raw_row_sha256 does not reproduce sha256_json(row)")
        if sha256_text(self.row["problem"]) != self.problem_sha256:
            raise ValueError("problem_sha256 does not reproduce sha256_text(row['problem'])")
        if sha256_text(self.row["answer"]) != self.gold_answer_sha256:
            raise ValueError("gold_answer_sha256 does not reproduce sha256_text(row['answer'])")
        if self.row["unique_id"] != self.unique_id:
            raise ValueError("row['unique_id'] does not match the candidate's own unique_id")
        if self.row["subject"] != self.subject:
            raise ValueError("row['subject'] does not match the candidate's own subject")
        if int(self.row["level"]) != self.level:
            raise ValueError("int(row['level']) does not match the candidate's own level")
        if self.unique_id in EXCLUSION_SET:
            raise ValueError(f"unique_id {self.unique_id!r} is in the frozen exclusion set")
        return self

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
        }


class CandidateManifestR3(BaseModel):
    """Protocol §12.3. Strict, immutable, extra fields forbidden."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    artifact_schema_version: str
    candidate_order_protocol_version: str
    dataset_repo: str
    dataset_config: str
    dataset_split: str
    dataset_revision: str
    model_name: str
    model_revision: str
    tokenizer_name: str
    tokenizer_revision: str
    budget: int
    config_path: str
    config_sha256: str
    generation_config_sha256: str
    exclusion_set_sha256: str
    candidate_count: int
    qualification_limit: int
    level_mixture: dict[str, int]
    candidates: list[CandidateRowR3]
    canonical_sha256: str

    @field_validator("canonical_sha256")
    @classmethod
    def _hex64(cls, v: str) -> str:
        return require_lowercase_hex64(v, "canonical_sha256")

    @model_validator(mode="after")
    def _cross_field_invariants(self) -> "CandidateManifestR3":
        if self.artifact_schema_version != CANDIDATE_MANIFEST_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("artifact_schema_version does not match the frozen value")
        if self.candidate_order_protocol_version != CANDIDATE_ORDER_PROTOCOL_VERSION:
            raise ValueError("candidate_order_protocol_version does not match the frozen value")
        if self.generation_config_sha256 != GENERATION_CONFIG_SHA256:
            raise ValueError("generation_config_sha256 does not match the frozen value")
        if self.exclusion_set_sha256 != EXCLUSION_SET_SHA256:
            raise ValueError("exclusion_set_sha256 does not match the frozen value")
        if self.qualification_limit != QUALIFICATION_CANDIDATE_LIMIT:
            raise ValueError(f"qualification_limit must be {QUALIFICATION_CANDIDATE_LIMIT}")
        if self.candidate_count != len(self.candidates):
            raise ValueError(
                f"candidate_count={self.candidate_count} disagrees with len(candidates)={len(self.candidates)}"
            )
        if self.candidate_count != CANDIDATE_TOTAL_COUNT:
            raise ValueError(f"candidate_count must be {CANDIDATE_TOTAL_COUNT}, got {self.candidate_count}")
        if self.level_mixture != {"level_4": CANDIDATES_PER_LEVEL, "level_5": CANDIDATES_PER_LEVEL}:
            raise ValueError(f"level_mixture must be exactly the frozen 8/8 split, got {self.level_mixture}")

        ordinals = [c.candidate_ordinal for c in self.candidates]
        if ordinals != list(range(CANDIDATE_TOTAL_COUNT)):
            raise ValueError("candidate_ordinal values are not exactly 0..15 in order")

        unique_ids = [c.unique_id for c in self.candidates]
        if len(set(unique_ids)) != len(unique_ids):
            raise ValueError("candidates contain a duplicate unique_id")

        level4 = [c for c in self.candidates if c.candidate_ordinal % 2 == 0]
        level5 = [c for c in self.candidates if c.candidate_ordinal % 2 == 1]
        if any(c.level != 4 for c in level4) or any(c.level != 5 for c in level5):
            raise ValueError("interleaving is broken: even ordinals must be level 4, odd ordinals level 5")

        for group in (level4, level5):
            keys = [(c.ordering_hash, c.unique_id) for c in group]
            if keys != sorted(keys):
                raise ValueError("a level's candidates are not ascending by (ordering_hash, unique_id)")

        for candidate in self.candidates:
            expected_hash = _ordering_hash(
                dataset_revision=self.dataset_revision,
                model_revision=self.model_revision,
                budget=self.budget,
                unique_id=candidate.unique_id,
                protocol_version=self.candidate_order_protocol_version,
            )
            if expected_hash != candidate.ordering_hash:
                raise ValueError(f"candidate {candidate.unique_id!r} ordering_hash does not reproduce")
        return self

    def to_json(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": self.artifact_schema_version,
            "candidate_order_protocol_version": self.candidate_order_protocol_version,
            "dataset_repo": self.dataset_repo,
            "dataset_config": self.dataset_config,
            "dataset_split": self.dataset_split,
            "dataset_revision": self.dataset_revision,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "budget": self.budget,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "generation_config_sha256": self.generation_config_sha256,
            "exclusion_set_sha256": self.exclusion_set_sha256,
            "candidate_count": self.candidate_count,
            "qualification_limit": self.qualification_limit,
            "level_mixture": dict(self.level_mixture),
            "candidates": [c.to_json() for c in self.candidates],
            "canonical_sha256": self.canonical_sha256,
        }


def build_candidate_manifest(
    all_rows_in_file_order: list[dict[str, Any]],
    *,
    dataset_repo: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: str,
    model_name: str,
    model_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    budget: int,
    config_path: str,
    config_sha256: str,
) -> dict[str, Any]:
    """Pure, deterministic, outcome-blind construction (protocol §8.2).
    Raises `ManifestPreparationError` if the eligible population cannot
    supply 8 level-4 AND 8 level-5 rows (never silently pads/shrinks the
    frozen 8/8 mixture)."""
    seen_unique_ids: set[str] = set()
    valid_rows: list[tuple[int, dict[str, Any]]] = []
    for index, row in enumerate(all_rows_in_file_order):
        got = tuple(row.keys())
        if got != EMBEDDED_ROW_COLUMNS:
            raise ManifestPreparationError(
                f"row {index} has unexpected columns: expected {EMBEDDED_ROW_COLUMNS}, got {got}"
            )
        unique_id = row["unique_id"]
        if unique_id in seen_unique_ids:
            raise ManifestPreparationError(
                f"duplicate unique_id {unique_id!r} in pinned dataset -- refusing to proceed"
            )
        seen_unique_ids.add(unique_id)
        valid_rows.append((index, row))

    exclusion = set(EXCLUSION_SET)
    remaining = [(i, r) for i, r in valid_rows if r["unique_id"] not in exclusion]

    def _eligible_for_level(level: str) -> list[tuple[str, int, dict[str, Any]]]:
        eligible = [(i, r) for i, r in remaining if str(r["level"]) == level]
        scored = [
            (
                _ordering_hash(
                    dataset_revision=dataset_revision, model_revision=model_revision, budget=budget,
                    unique_id=r["unique_id"], protocol_version=CANDIDATE_ORDER_PROTOCOL_VERSION,
                ),
                i,
                r,
            )
            for i, r in eligible
        ]
        scored.sort(key=lambda item: (item[0], item[2]["unique_id"]))
        return scored

    level4_ranked = _eligible_for_level("4")
    level5_ranked = _eligible_for_level("5")
    if len(level4_ranked) < CANDIDATES_PER_LEVEL or len(level5_ranked) < CANDIDATES_PER_LEVEL:
        raise ManifestPreparationError(
            f"insufficient eligible rows after exclusion: level-4={len(level4_ranked)}, "
            f"level-5={len(level5_ranked)}, need >= {CANDIDATES_PER_LEVEL} each"
        )
    level4_top = level4_ranked[:CANDIDATES_PER_LEVEL]
    level5_top = level5_ranked[:CANDIDATES_PER_LEVEL]

    interleaved: list[tuple[str, int, dict[str, Any]]] = []
    for rank in range(CANDIDATES_PER_LEVEL):
        interleaved.append(level4_top[rank])
        interleaved.append(level5_top[rank])

    candidates = [
        CandidateRowR3(
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
        )
        for ordinal, (ordering_hash, index, row) in enumerate(interleaved)
    ]

    manifest: dict[str, Any] = {
        "artifact_schema_version": CANDIDATE_MANIFEST_ARTIFACT_SCHEMA_VERSION,
        "candidate_order_protocol_version": CANDIDATE_ORDER_PROTOCOL_VERSION,
        "dataset_repo": dataset_repo,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "dataset_revision": dataset_revision,
        "model_name": model_name,
        "model_revision": model_revision,
        "tokenizer_name": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "budget": budget,
        "config_path": config_path,
        "config_sha256": config_sha256,
        "generation_config_sha256": GENERATION_CONFIG_SHA256,
        "exclusion_set_sha256": EXCLUSION_SET_SHA256,
        "candidate_count": len(candidates),
        "qualification_limit": QUALIFICATION_CANDIDATE_LIMIT,
        "level_mixture": {"level_4": CANDIDATES_PER_LEVEL, "level_5": CANDIDATES_PER_LEVEL},
        "candidates": [c.to_json() for c in candidates],
    }
    manifest["canonical_sha256"] = sha256_json(manifest)

    # Construct-and-validate against the strict schema before returning --
    # any internal inconsistency this function itself introduced is a hard
    # failure, never silently returned.
    CandidateManifestR3.model_validate(manifest)
    return manifest


def verify_candidate_manifest_structure(manifest: dict[str, Any]) -> CandidateManifestR3:
    """Schema-valid parse plus every internal-consistency check
    expressible without re-fetching the dataset: canonical self-hash,
    exact field set, level mixture, interleaving, per-candidate row-hash
    formulas, and every candidate's own `ordering_hash` recomputed from its
    declared identity fields."""
    verify_canonical_sha256(manifest)
    return CandidateManifestR3.model_validate(manifest)


def verify_candidate_manifest_against_dataset(
    manifest: dict[str, Any], all_rows_in_file_order: list[dict[str, Any]]
) -> None:
    """Full semantic re-derivation: rebuilds the candidate manifest fresh
    from the same raw dataset rows and requires byte-identical
    `canonical_sha256` agreement -- the one check that needs the complete
    dataset population (not just the 16 already-selected rows), so it can
    catch a hand-edited candidate list that happens to be internally
    self-consistent but does not reproduce the real deterministic
    selection."""
    typed = verify_candidate_manifest_structure(manifest)
    rebuilt = build_candidate_manifest(
        all_rows_in_file_order,
        dataset_repo=typed.dataset_repo,
        dataset_config=typed.dataset_config,
        dataset_split=typed.dataset_split,
        dataset_revision=typed.dataset_revision,
        model_name=typed.model_name,
        model_revision=typed.model_revision,
        tokenizer_name=typed.tokenizer_name,
        tokenizer_revision=typed.tokenizer_revision,
        budget=typed.budget,
        config_path=typed.config_path,
        config_sha256=typed.config_sha256,
    )
    if rebuilt["canonical_sha256"] != manifest["canonical_sha256"]:
        raise ManifestPreparationError(
            "candidate manifest does not reproduce from the pinned dataset: "
            f"rebuilt canonical_sha256={rebuilt['canonical_sha256']!r} != "
            f"stored canonical_sha256={manifest['canonical_sha256']!r}"
        )


def fetch_and_build_candidate_manifest(
    *,
    dataset_repo: str,
    dataset_config: str,
    dataset_split: str,
    dataset_revision: str,
    model_name: str,
    model_revision: str,
    tokenizer_name: str,
    tokenizer_revision: str,
    budget: int,
    config_path: str,
    config_sha256: str,
) -> dict[str, Any]:
    """The one network fetch this procedure requires (protocol §8): the
    pinned MATH-500 revision's raw file, then pure CPU construction."""
    rows = fetch_all_pinned_dataset_rows(dataset_repo, dataset_revision)
    return build_candidate_manifest(
        rows,
        dataset_repo=dataset_repo, dataset_config=dataset_config, dataset_split=dataset_split,
        dataset_revision=dataset_revision, model_name=model_name, model_revision=model_revision,
        tokenizer_name=tokenizer_name, tokenizer_revision=tokenizer_revision, budget=budget,
        config_path=config_path, config_sha256=config_sha256,
    )
