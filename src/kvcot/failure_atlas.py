"""Phase A2 — deterministic GSM8K protocol-v3 failure atlas.

CPU-only, post-hoc diagnostic analysis over the 50 committed FullKV/R-KV
result pairs at the RETIRED `gsm8k_calibration_50` / `RKV-B128` operating
point (`results/gate_artifacts/early_gap_v3_b128_{full,rkv_b128}.jsonl.gz`).
Never loads a model or tokenizer, never imports torch/transformers (mirrors
the discipline `tests/unit/test_no_analysis_torch_import.py` enforces for
`kvcot.analysis`/`kvcot.utils`), and never regenerates or mutates any
committed artifact.

CLAIM BOUNDARY — repeated here per CLAUDE.md/build brief §1, because every
summary string this module produces must carry it:

    This atlas analyzes a retired operating point where natural R-KV
    accuracy fell from 33/50 to 13/50. It is a post-hoc diagnostic analysis
    and cannot establish that any observed failure pattern occurs at an
    accuracy-preserving operating point. It generates hypotheses for later
    held-out testing only.

Coordinate conventions (load-bearing — see the mandatory regression test in
tests/unit/test_failure_atlas.py):

  * `generated_token_ids` indices are 0-based, matching
    `kvcot.schemas.ThinkSpanInfo`'s own documented convention
    ("index into generated_token_ids, inclusive/exclusive").
  * `kvcot.generation.decode.generate_base` records `compaction_event_steps`
    as ABSOLUTE positions in the (prompt + generated) token stream — the
    `self.length`/`absolute_position` value at the forward call that fired
    the event (see that module's docstring). Generated token at 0-based
    index `i` therefore occupies absolute position `prompt_token_count + i`.
    Divergence and compaction positions are only ever compared after both
    are converted into this ONE shared coordinate system:

        first_divergence_absolute_position
            = prompt_token_count + first_divergence_generated_index

    Do not compare a raw generated-index against an absolute compaction
    position directly — that is exactly the coordinate-system bug this
    module's regression test exists to catch.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from kvcot.config import FixedTraceSettings
from kvcot.probes.early_answering import CLAIM_BOUNDARY_NOTICE, find_subsequence
from kvcot.utils.io import read_jsonl_auto, write_json

ATLAS_SCHEMA_VERSION = "1.0.0"
DIAGNOSTIC_LABEL = "post_hoc_diagnostic"
EXPECTED_PAIR_COUNT = 50

THINK_PARSE_OK_STATUSES = ("ok", "generation_prompt_preopened_ok")

# Reused, not reinvented: the same "meaningful compression" retention
# threshold the fixed-trace screen uses (kvcot.config.FixedTraceSettings.
# meaningful_retention_ceiling), instantiated at its default. This is a
# descriptive threshold for the atlas's own `meaningful_compression_occurred`
# column, not a frozen §4 setting and not a gate.
MEANINGFUL_COMPRESSION_CEILING = FixedTraceSettings().meaningful_retention_ceiling

DivergenceRelation = Literal[
    "before_first_compaction", "at_first_compaction", "after_first_compaction",
    "no_divergence", "no_compaction",
]
ReasoningRegion = Literal[
    "reasoning", "reasoning_to_answer_transition", "post_think_answer",
    "malformed_or_missing_think_boundary", "no_divergence",
]
CorrectnessPair = Literal["correct_correct", "correct_wrong", "wrong_correct", "wrong_wrong"]
FlipDirection = Literal["correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"]
TerminationReason = Literal["eos", "max_new_tokens_cap"]


class FailureAtlasIntegrityError(ValueError):
    """Raised for any pairing/provenance/schema defect that must abort the
    atlas build rather than silently producing a partial or mis-paired
    table (§ Step 1/6 of the A2 brief: "Fail loudly")."""


# --------------------------------------------------------------- pairing


def _pair_key(record: dict) -> tuple[int, int]:
    return (record["dataset"]["source_row_index"], record["global_seed"])


_REQUIRED_TOP_LEVEL_FIELDS = (
    "record_id", "condition", "global_seed", "config_sha256", "model_revision",
    "tokenizer_revision", "provenance", "dataset", "prompt_token_ids",
    "generated_token_ids", "decoded_output", "think_span", "extracted_answer",
    "extraction_method", "is_correct", "cap_hit", "compaction_count",
    "compaction_event_steps", "retention",
)


def _check_required_fields(record: dict, label: str) -> None:
    missing = [f for f in _REQUIRED_TOP_LEVEL_FIELDS if f not in record]
    if missing:
        raise FailureAtlasIntegrityError(
            f"{label} record {record.get('record_id', '<unknown>')!r} is missing required field(s): {missing}"
        )
    if "source_row_index" not in record["dataset"] or "question_hash" not in record["dataset"]:
        raise FailureAtlasIntegrityError(
            f"{label} record {record['record_id']!r} has an incomplete 'dataset' block "
            "(missing source_row_index or question_hash)"
        )
    if "think_parse_status" not in record["think_span"] or "think_end_index" not in record["think_span"]:
        raise FailureAtlasIntegrityError(
            f"{label} record {record['record_id']!r} has an incomplete 'think_span' block"
        )


def _check_no_duplicate_keys(records: list[dict], label: str) -> dict[tuple[int, int], dict]:
    by_key: dict[tuple[int, int], dict] = {}
    dupes: list[tuple[int, int]] = []
    for r in records:
        key = _pair_key(r)
        if key in by_key:
            dupes.append(key)
        by_key[key] = r
    if dupes:
        raise FailureAtlasIntegrityError(f"{label} artifact contains duplicate (source_row_index, seed) keys: {sorted(set(dupes))}")
    return by_key


def pair_and_validate_records(full_records: list[dict], rkv_records: list[dict]) -> list[tuple[dict, dict]]:
    """Pair FullKV/R-KV records by the stable `(source_row_index, global_seed)`
    key (the same pairing convention `kvcot.analysis.fixed_trace.
    build_accuracy_screen`/`build_strict_accuracy_gate` use for this exact
    artifact pair) — never by file order/row position. Fails loudly (raises
    `FailureAtlasIntegrityError`) on any of: wrong record count, duplicate
    keys, an incomplete key intersection, mismatched question identity for a
    shared key, missing required fields, or inconsistent run provenance.
    """
    if len(full_records) != EXPECTED_PAIR_COUNT:
        raise FailureAtlasIntegrityError(
            f"expected exactly {EXPECTED_PAIR_COUNT} FullKV records, got {len(full_records)}"
        )
    if len(rkv_records) != EXPECTED_PAIR_COUNT:
        raise FailureAtlasIntegrityError(
            f"expected exactly {EXPECTED_PAIR_COUNT} R-KV records, got {len(rkv_records)}"
        )

    for r in full_records:
        _check_required_fields(r, "FullKV")
    for r in rkv_records:
        _check_required_fields(r, "R-KV")

    full_by_key = _check_no_duplicate_keys(full_records, "FullKV")
    rkv_by_key = _check_no_duplicate_keys(rkv_records, "R-KV")

    full_keys = set(full_by_key)
    rkv_keys = set(rkv_by_key)
    if full_keys != rkv_keys:
        raise FailureAtlasIntegrityError(
            "FullKV/R-KV key sets are not identical -- "
            f"missing from R-KV: {sorted(full_keys - rkv_keys)}; "
            f"missing from FullKV: {sorted(rkv_keys - full_keys)}"
        )

    def _identity(r: dict) -> tuple:
        return (r["config_sha256"], r["provenance"]["upstream_rkv_commit"], r["model_revision"], r["tokenizer_revision"])

    full_identities = {_identity(r) for r in full_records}
    rkv_identities = {_identity(r) for r in rkv_records}
    if len(full_identities) != 1:
        raise FailureAtlasIntegrityError(f"FullKV records do not share one run identity: {full_identities}")
    if len(rkv_identities) != 1:
        raise FailureAtlasIntegrityError(f"R-KV records do not share one run identity: {rkv_identities}")
    if full_identities != rkv_identities:
        raise FailureAtlasIntegrityError(
            f"FullKV and R-KV artifacts were produced under different provenance: {full_identities} vs {rkv_identities}"
        )

    pairs: list[tuple[dict, dict]] = []
    for key in sorted(full_keys):
        f, r = full_by_key[key], rkv_by_key[key]
        if f["dataset"]["question_hash"] != r["dataset"]["question_hash"]:
            raise FailureAtlasIntegrityError(
                f"source_row_index={key[0]} seed={key[1]}: FullKV/R-KV question_hash disagree -- "
                "the two artifacts do not describe the same problem"
            )
        if f["prompt_token_ids"] != r["prompt_token_ids"]:
            raise FailureAtlasIntegrityError(
                f"source_row_index={key[0]} seed={key[1]}: FullKV/R-KV prompt_token_ids differ"
            )
        pairs.append((f, r))
    return pairs


# ------------------------------------------------------------ pure logic


def common_prefix_and_divergence(a: list[int], b: list[int]) -> tuple[int, int | None]:
    """Returns (common_prefix_token_count, first_divergence_index).

    `first_divergence_index` is the 0-based index of the first token at
    which `a` and `b` disagree, or None if no mismatching token exists
    within the shorter sequence's length (the sequences are either fully
    identical, or one is an exact prefix of the other — see
    `fully_identical_sequences`/`identical_until_shorter_terminates` in
    `build_atlas_row`, which disambiguate those two cases).
    """
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, i
    return n, None


def divergence_relation_to_compaction(
    first_divergence_absolute_position: int | None,
    first_compaction_absolute_position: int | None,
) -> DivergenceRelation:
    """Both positions MUST already be in the same absolute (prompt +
    generated) coordinate system -- see this module's docstring. Never pass
    a raw generated-relative index here."""
    if first_divergence_absolute_position is None:
        return "no_divergence"
    if first_compaction_absolute_position is None:
        return "no_compaction"
    if first_divergence_absolute_position < first_compaction_absolute_position:
        return "before_first_compaction"
    if first_divergence_absolute_position == first_compaction_absolute_position:
        return "at_first_compaction"
    return "after_first_compaction"


def reasoning_region_category(
    first_divergence_generated_index: int | None,
    think_end_index_full: int | None,
    think_end_index_rkv: int | None,
    think_parse_ok_full: bool,
    think_parse_ok_rkv: bool,
) -> ReasoningRegion:
    """Classify where the first divergence begins relative to each side's
    own `</think>` close-marker start index (`ThinkSpanInfo.think_end_index`
    -- the index of the FIRST token of the close-marker subsequence, per
    `kvcot.probes.early_answering.find_think_span`'s documented contract).

    Malformed/missing boundaries on EITHER side are classified before any
    numeric comparison (never silently forced into "reasoning" or "post_think
    _answer" just because a comparison happened to be well-defined). When
    both sides parsed but disagree on WHERE the close marker starts (e.g.
    divergence pushed one side's reasoning to a different length), a
    divergence index that falls between the two markers -- at either marker
    exactly, or strictly between them -- is "reasoning_to_answer_transition":
    it is neither purely inside shared reasoning (both sides still open) nor
    purely inside a shared answer region (both sides already closed).
    """
    if first_divergence_generated_index is None:
        return "no_divergence"
    if not think_parse_ok_full or not think_parse_ok_rkv:
        return "malformed_or_missing_think_boundary"
    if think_end_index_full is None or think_end_index_rkv is None:
        return "malformed_or_missing_think_boundary"
    min_e = min(think_end_index_full, think_end_index_rkv)
    max_e = max(think_end_index_full, think_end_index_rkv)
    if first_divergence_generated_index < min_e:
        return "reasoning"
    if first_divergence_generated_index > max_e:
        return "post_think_answer"
    return "reasoning_to_answer_transition"


def _correctness_labels(full_correct: bool, rkv_correct: bool) -> tuple[CorrectnessPair, FlipDirection]:
    if full_correct and rkv_correct:
        return "correct_correct", "correct_to_correct"
    if full_correct and not rkv_correct:
        return "correct_wrong", "correct_to_wrong"
    if not full_correct and rkv_correct:
        return "wrong_correct", "wrong_to_correct"
    return "wrong_wrong", "wrong_to_wrong"


def _marker_repeats_after(generated_token_ids: list[int], think_end_index: int | None) -> bool:
    """True iff the exact token found at `think_end_index` (the close
    marker's own first token, read from the record itself -- never a
    hardcoded model-specific id) occurs again anywhere later in the
    sequence. A cheap, tokenizer-free proxy for "multiple close markers",
    reusing `find_subsequence` rather than a new scan."""
    if think_end_index is None or think_end_index >= len(generated_token_ids):
        return False
    marker_token = generated_token_ids[think_end_index]
    return find_subsequence(generated_token_ids, [marker_token], start=think_end_index + 1) is not None


def _decoded_suffix_after_think(decoded_output: str, max_len: int = 400) -> str | None:
    idx = decoded_output.find("</think>")
    if idx == -1:
        return None
    suffix = decoded_output[idx + len("</think>"):]
    return suffix[:max_len]


def _token_window(tokens: list[int], center: int | None, radius: int = 6) -> str | None:
    if center is None:
        return None
    lo = max(0, center - radius)
    hi = min(len(tokens), center + radius + 1)
    return ",".join(str(t) for t in tokens[lo:hi])


# ----------------------------------------------------------------- row


class AtlasRow(BaseModel):
    schema_version: str = ATLAS_SCHEMA_VERSION
    diagnostic_label: str = DIAGNOSTIC_LABEL
    analysis_code_commit: str

    source_row_index: int
    question_hash: str
    full_record_id: str
    rkv_record_id: str
    full_artifact_path: str
    rkv_artifact_path: str

    prompt_token_count: int
    full_generated_token_count: int
    rkv_generated_token_count: int
    length_delta: int

    rkv_compaction_count: int
    first_compaction_absolute_position: int | None
    final_retention_ratio: float | None
    meaningful_compression_ceiling: float
    meaningful_compression_occurred: bool

    common_prefix_token_count: int
    divergence_exists: bool
    first_divergence_generated_index: int | None
    first_divergence_absolute_position: int | None
    fully_identical_sequences: bool
    identical_until_shorter_terminates: bool
    divergence_relation_to_first_compaction: DivergenceRelation
    divergence_distance_from_first_compaction: int | None
    first_divergence_token_window_full: str | None
    first_divergence_token_window_rkv: str | None

    think_parse_status_full: str
    think_parse_status_rkv: str
    think_end_index_full: int | None
    think_end_index_rkv: int | None
    reasoning_region_category: ReasoningRegion
    identical_through_think: bool
    close_marker_token_id_full: int | None
    close_marker_token_id_rkv: int | None
    close_marker_repeats_after_first_full: bool
    close_marker_repeats_after_first_rkv: bool

    full_is_correct: bool | None
    rkv_is_correct: bool | None
    correctness_pair: CorrectnessPair
    flip_direction: FlipDirection

    extracted_answer_full: str | None
    extracted_answer_rkv: str | None
    extraction_method_full: str
    extraction_method_rkv: str
    malformed_answer_full: bool
    malformed_answer_rkv: bool
    cap_hit_full: bool
    cap_hit_rkv: bool
    termination_reason_full: TerminationReason
    termination_reason_rkv: TerminationReason
    post_think_decoded_snippet_full: str | None
    post_think_decoded_snippet_rkv: str | None


def build_atlas_row(
    full_record: dict, rkv_record: dict, *, analysis_code_commit: str,
    full_artifact_path: str, rkv_artifact_path: str,
) -> AtlasRow:
    prompt_token_count = len(full_record["prompt_token_ids"])

    full_tokens = full_record["generated_token_ids"]
    rkv_tokens = rkv_record["generated_token_ids"]
    common_prefix, div_idx = common_prefix_and_divergence(full_tokens, rkv_tokens)
    divergence_exists = div_idx is not None
    first_div_abs = (prompt_token_count + div_idx) if divergence_exists else None
    fully_identical = (not divergence_exists) and len(full_tokens) == len(rkv_tokens)
    identical_until_shorter = (not divergence_exists) and len(full_tokens) != len(rkv_tokens)

    compaction_steps = rkv_record["compaction_event_steps"]
    first_compaction_abs = min(compaction_steps) if compaction_steps else None
    divergence_relation = divergence_relation_to_compaction(first_div_abs, first_compaction_abs)
    divergence_distance = (
        first_div_abs - first_compaction_abs
        if (first_div_abs is not None and first_compaction_abs is not None) else None
    )

    retention = rkv_record.get("retention")
    final_retention_ratio = retention["instantaneous_retention_ratio"] if retention else None
    meaningful_compression = (
        final_retention_ratio is not None and final_retention_ratio <= MEANINGFUL_COMPRESSION_CEILING
    )

    think_full = full_record["think_span"]
    think_rkv = rkv_record["think_span"]
    ok_full = think_full["think_parse_status"] in THINK_PARSE_OK_STATUSES
    ok_rkv = think_rkv["think_parse_status"] in THINK_PARSE_OK_STATUSES
    region = reasoning_region_category(
        div_idx, think_full["think_end_index"], think_rkv["think_end_index"], ok_full, ok_rkv,
    )
    identical_through_think = (
        ok_full and ok_rkv
        and think_full["think_end_index"] is not None
        and think_full["think_end_index"] == think_rkv["think_end_index"]
        and common_prefix > think_full["think_end_index"]
    )
    close_marker_full = (
        full_tokens[think_full["think_end_index"]]
        if ok_full and think_full["think_end_index"] is not None and think_full["think_end_index"] < len(full_tokens)
        else None
    )
    close_marker_rkv = (
        rkv_tokens[think_rkv["think_end_index"]]
        if ok_rkv and think_rkv["think_end_index"] is not None and think_rkv["think_end_index"] < len(rkv_tokens)
        else None
    )

    full_correct = full_record["is_correct"] is True
    rkv_correct = rkv_record["is_correct"] is True
    correctness_pair, flip_direction = _correctness_labels(full_correct, rkv_correct)

    return AtlasRow(
        analysis_code_commit=analysis_code_commit,
        source_row_index=full_record["dataset"]["source_row_index"],
        question_hash=full_record["dataset"]["question_hash"],
        full_record_id=full_record["record_id"],
        rkv_record_id=rkv_record["record_id"],
        full_artifact_path=full_artifact_path,
        rkv_artifact_path=rkv_artifact_path,
        prompt_token_count=prompt_token_count,
        full_generated_token_count=len(full_tokens),
        rkv_generated_token_count=len(rkv_tokens),
        length_delta=len(rkv_tokens) - len(full_tokens),
        rkv_compaction_count=rkv_record["compaction_count"],
        first_compaction_absolute_position=first_compaction_abs,
        final_retention_ratio=final_retention_ratio,
        meaningful_compression_ceiling=MEANINGFUL_COMPRESSION_CEILING,
        meaningful_compression_occurred=meaningful_compression,
        common_prefix_token_count=common_prefix,
        divergence_exists=divergence_exists,
        first_divergence_generated_index=div_idx,
        first_divergence_absolute_position=first_div_abs,
        fully_identical_sequences=fully_identical,
        identical_until_shorter_terminates=identical_until_shorter,
        divergence_relation_to_first_compaction=divergence_relation,
        divergence_distance_from_first_compaction=divergence_distance,
        first_divergence_token_window_full=_token_window(full_tokens, div_idx),
        first_divergence_token_window_rkv=_token_window(rkv_tokens, div_idx),
        think_parse_status_full=think_full["think_parse_status"],
        think_parse_status_rkv=think_rkv["think_parse_status"],
        think_end_index_full=think_full["think_end_index"],
        think_end_index_rkv=think_rkv["think_end_index"],
        reasoning_region_category=region,
        identical_through_think=identical_through_think,
        close_marker_token_id_full=close_marker_full,
        close_marker_token_id_rkv=close_marker_rkv,
        close_marker_repeats_after_first_full=_marker_repeats_after(full_tokens, think_full["think_end_index"]),
        close_marker_repeats_after_first_rkv=_marker_repeats_after(rkv_tokens, think_rkv["think_end_index"]),
        full_is_correct=full_record["is_correct"],
        rkv_is_correct=rkv_record["is_correct"],
        correctness_pair=correctness_pair,
        flip_direction=flip_direction,
        extracted_answer_full=full_record["extracted_answer"],
        extracted_answer_rkv=rkv_record["extracted_answer"],
        extraction_method_full=full_record["extraction_method"],
        extraction_method_rkv=rkv_record["extraction_method"],
        malformed_answer_full=full_record["extracted_answer"] is None,
        malformed_answer_rkv=rkv_record["extracted_answer"] is None,
        cap_hit_full=full_record["cap_hit"],
        cap_hit_rkv=rkv_record["cap_hit"],
        termination_reason_full=("max_new_tokens_cap" if full_record["cap_hit"] else "eos"),
        termination_reason_rkv=("max_new_tokens_cap" if rkv_record["cap_hit"] else "eos"),
        post_think_decoded_snippet_full=_decoded_suffix_after_think(full_record["decoded_output"]),
        post_think_decoded_snippet_rkv=_decoded_suffix_after_think(rkv_record["decoded_output"]),
    )


def build_failure_atlas(
    full_records: list[dict], rkv_records: list[dict], *, analysis_code_commit: str,
    full_artifact_path: str, rkv_artifact_path: str,
) -> list[AtlasRow]:
    pairs = pair_and_validate_records(full_records, rkv_records)
    rows = [
        build_atlas_row(
            f, r, analysis_code_commit=analysis_code_commit,
            full_artifact_path=full_artifact_path, rkv_artifact_path=rkv_artifact_path,
        )
        for f, r in pairs
    ]
    rows.sort(key=lambda row: row.source_row_index)
    return rows


# ------------------------------------------------------------- summary


def _describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _counter(rows: list[AtlasRow], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, attr))
        counts[key] = counts.get(key, 0) + 1
    return counts


def build_atlas_summary(
    rows: list[AtlasRow], *, full_artifact_path: str, rkv_artifact_path: str,
    full_artifact_sha256: str, rkv_artifact_sha256: str, analysis_code_commit: str,
) -> dict[str, Any]:
    """Machine-readable recomputation of every headline aggregate over the
    atlas rows -- independently derived from the rows themselves (§ Step 6:
    "do not copy numbers from README/CHANGELOG"), never from any prior
    verdict."""
    n = len(rows)
    warnings: list[str] = []
    if n != EXPECTED_PAIR_COUNT:
        warnings.append(f"expected {EXPECTED_PAIR_COUNT} rows, got {n}")

    correctness_dist = _counter(rows, "correctness_pair")
    both_correct = correctness_dist.get("correct_correct", 0)
    full_only = correctness_dist.get("correct_wrong", 0)
    rkv_only = correctness_dist.get("wrong_correct", 0)
    both_wrong = correctness_dist.get("wrong_wrong", 0)
    full_correct_n = both_correct + full_only
    rkv_correct_n = both_correct + rkv_only

    retentions = [r.final_retention_ratio for r in rows if r.final_retention_ratio is not None]
    compaction_counts = [r.rkv_compaction_count for r in rows]
    length_deltas = [r.length_delta for r in rows]

    identical_through_think_rows = [r for r in rows if r.identical_through_think]
    identical_through_think_by_correctness = _counter(identical_through_think_rows, "correctness_pair")

    relation_dist = _counter(rows, "divergence_relation_to_first_compaction")
    region_dist = _counter(rows, "reasoning_region_category")

    invariants: dict[str, bool] = {
        "row_count_is_50": n == EXPECTED_PAIR_COUNT,
        "no_duplicate_source_row_index": len({r.source_row_index for r in rows}) == n,
        "four_way_correctness_sums_to_n": (both_correct + full_only + rkv_only + both_wrong) == n,
        "no_divergence_before_first_compaction": relation_dist.get("before_first_compaction", 0) == 0,
        "compaction_relation_counts_sum_to_n": sum(relation_dist.values()) == n,
        "reasoning_region_counts_sum_to_n": sum(region_dist.values()) == n,
    }

    return {
        "claim_boundary_notice": CLAIM_BOUNDARY_NOTICE,
        "atlas_claim_boundary": (
            "This atlas analyzes a retired operating point where natural R-KV accuracy fell from "
            "33/50 to 13/50. It is a post-hoc diagnostic analysis and cannot establish that any "
            "observed failure pattern occurs at an accuracy-preserving operating point. It generates "
            "hypotheses for later held-out testing only."
        ),
        "schema_version": ATLAS_SCHEMA_VERSION,
        "diagnostic_label": DIAGNOSTIC_LABEL,
        "operating_point_valid": False,
        "hypothesis_status": "not_tested",
        "analysis_code_commit": analysis_code_commit,
        "input_artifacts": {
            "full": {"path": full_artifact_path, "sha256": full_artifact_sha256},
            "rkv": {"path": rkv_artifact_path, "sha256": rkv_artifact_sha256},
        },
        "n_pairs": n,
        "correctness": {
            "both_correct": both_correct,
            "full_only_correct": full_only,
            "rkv_only_correct": rkv_only,
            "both_wrong": both_wrong,
            "full_correct_total": full_correct_n,
            "rkv_correct_total": rkv_correct_n,
            "distribution": correctness_dist,
        },
        "retention": {
            **_describe(retentions),
            "n_below_0.50": sum(1 for x in retentions if x < 0.50),
            "n_at_or_below_0.70": sum(1 for x in retentions if x <= 0.70),
        },
        "compaction_count": _describe(compaction_counts),
        "length_delta": _describe(length_deltas),
        "divergence": {
            "n_no_divergence": sum(1 for r in rows if not r.divergence_exists),
            "n_divergence": sum(1 for r in rows if r.divergence_exists),
            "relation_to_first_compaction": relation_dist,
        },
        "reasoning_region": region_dist,
        "identical_through_think": {
            "count": len(identical_through_think_rows),
            "source_rows": sorted(r.source_row_index for r in identical_through_think_rows),
            "by_correctness_pair": identical_through_think_by_correctness,
            "n_correct_to_wrong_flips": identical_through_think_by_correctness.get("correct_wrong", 0),
        },
        "cap_hit": {
            "n_full": sum(1 for r in rows if r.cap_hit_full),
            "n_rkv": sum(1 for r in rows if r.cap_hit_rkv),
        },
        "malformed_answer": {
            "n_full": sum(1 for r in rows if r.malformed_answer_full),
            "n_rkv": sum(1 for r in rows if r.malformed_answer_rkv),
        },
        "malformed_think_boundary": {
            "n_full": sum(1 for r in rows if r.think_parse_status_full not in THINK_PARSE_OK_STATUSES),
            "n_rkv": sum(1 for r in rows if r.think_parse_status_rkv not in THINK_PARSE_OK_STATUSES),
        },
        "invariant_checks": invariants,
        "warnings": warnings,
    }


# --------------------------------------------------------------- writers

_CSV_FIELDNAMES = list(AtlasRow.model_fields.keys())


def write_atlas_csv(rows: list[AtlasRow], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _fmt(v: Any) -> str:
    return "" if v is None else str(v)


def _fmt_float(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


def _markdown_identical_cot_table(rows: list[AtlasRow]) -> str:
    header = (
        "| source_row_index | correctness_pair | full_len | rkv_len | first_compaction_abs | "
        "compaction_count | final_retention | think_end_idx | first_divergence_gen_idx |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        lines.append(
            f"| {r.source_row_index} | {r.correctness_pair} | {r.full_generated_token_count} | "
            f"{r.rkv_generated_token_count} | {_fmt(r.first_compaction_absolute_position)} | "
            f"{r.rkv_compaction_count} | {_fmt_float(r.final_retention_ratio)} | {r.think_end_index_full} | "
            f"{_fmt(r.first_divergence_generated_index)} |"
        )
    return header + "\n".join(lines) + "\n"


def _markdown_case_detail(row: AtlasRow) -> str:
    return (
        f"### source row {row.source_row_index}\n\n"
        f"- prompt_token_count: {row.prompt_token_count}\n"
        f"- full generated length: {row.full_generated_token_count}; R-KV generated length: {row.rkv_generated_token_count}\n"
        f"- first_compaction_absolute_position: {_fmt(row.first_compaction_absolute_position)}; "
        f"compaction_count: {row.rkv_compaction_count}; final_retention_ratio: {_fmt_float(row.final_retention_ratio)}\n"
        f"- close marker (`</think>`) index -- full: {row.think_end_index_full}, R-KV: {row.think_end_index_rkv}\n"
        f"- first_divergence_generated_index: {_fmt(row.first_divergence_generated_index)}; "
        f"first_divergence_absolute_position: {_fmt(row.first_divergence_absolute_position)}\n"
        f"- divergence_relation_to_first_compaction: {row.divergence_relation_to_first_compaction}; "
        f"reasoning_region_category: {row.reasoning_region_category}\n"
        f"- extracted answer -- full: {row.extracted_answer_full!r} (correct={row.full_is_correct}), "
        f"R-KV: {row.extracted_answer_rkv!r} (correct={row.rkv_is_correct})\n"
        f"- decoded text after `</think>` -- full: {row.post_think_decoded_snippet_full!r}\n"
        f"- decoded text after `</think>` -- R-KV: {row.post_think_decoded_snippet_rkv!r}\n"
    )


def write_atlas_markdown(rows: list[AtlasRow], summary: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    identical_rows = [r for r in rows if r.identical_through_think]
    flip_rows = [r for r in identical_rows if r.correctness_pair == "correct_wrong"]

    parts: list[str] = []
    parts.append("# GSM8K protocol-v3 (RKV-B128) failure atlas\n")
    parts.append(f"\n> {summary['atlas_claim_boundary']}\n")
    parts.append(f"\n`hypothesis_status`: `{summary['hypothesis_status']}` -- `operating_point_valid`: `{summary['operating_point_valid']}`\n")

    parts.append("\n## Methodology and coordinate conventions\n")
    parts.append(
        "\nEach row pairs one FullKV and one R-KV base-generation record by "
        "`(source_row_index, global_seed)`. Token indices into `generated_token_ids` are "
        "0-based. `compaction_event_steps` are recorded by `kvcot.generation.decode.generate_base` "
        "as ABSOLUTE positions in the prompt+generated stream, so "
        "`first_divergence_absolute_position = prompt_token_count + first_divergence_generated_index` "
        "before any comparison against a compaction position. See `src/kvcot/failure_atlas.py`'s "
        "module docstring for the full derivation and the mandatory prompt-offset regression test.\n"
    )

    parts.append("\n## Aggregate statistics\n")
    parts.append(f"\n- n_pairs: {summary['n_pairs']}\n")
    parts.append(f"- retention: {summary['retention']}\n")
    parts.append(f"- compaction_count: {summary['compaction_count']}\n")
    parts.append(f"- length_delta (rkv - full): {summary['length_delta']}\n")
    parts.append(f"- cap_hit: {summary['cap_hit']}\n")
    parts.append(f"- malformed_answer: {summary['malformed_answer']}\n")
    parts.append(f"- malformed_think_boundary: {summary['malformed_think_boundary']}\n")

    parts.append("\n## Correctness-pair breakdown\n")
    c = summary["correctness"]
    parts.append(
        f"\n- both_correct: {c['both_correct']}\n"
        f"- full_only_correct: {c['full_only_correct']}\n"
        f"- rkv_only_correct: {c['rkv_only_correct']}\n"
        f"- both_wrong: {c['both_wrong']}\n"
        f"- FullKV accuracy: {c['full_correct_total']}/{summary['n_pairs']}\n"
        f"- R-KV accuracy: {c['rkv_correct_total']}/{summary['n_pairs']}\n"
    )

    parts.append("\n## Divergence-region breakdown (reasoning vs. answer)\n")
    parts.append(f"\n{summary['reasoning_region']}\n")

    parts.append("\n## Divergence relative to first compaction\n")
    parts.append(f"\n{summary['divergence']['relation_to_first_compaction']}\n")

    parts.append("\n## Identical reasoning with final-answer flips\n")
    parts.append(
        f"\n{len(identical_rows)} of {summary['n_pairs']} pairs are token-identical through the "
        f"close of the `</think>` marker (`identical_through_think`). "
        f"{len(flip_rows)} of those are correct-to-wrong flips.\n"
    )
    parts.append("\n" + _markdown_identical_cot_table(identical_rows))

    parts.append("\n### Detailed inspection: source rows 30, 271, 1115\n")
    by_idx = {r.source_row_index: r for r in rows}
    for idx in (30, 271, 1115):
        if idx in by_idx:
            parts.append("\n" + _markdown_case_detail(by_idx[idx]))
        else:
            parts.append(f"\n### source row {idx}\n\n(not present among the 50 paired examples)\n")

    parts.append("\n## Limitations and claim boundary\n")
    parts.append(
        "\n- This is a 50-example post-hoc diagnostic over a RETIRED operating point "
        "(accuracy dropped 0.40, exceeding the frozen 0.10 ceiling). No causal claims are "
        "supported by any correlation reported here.\n"
        "- Per-record retention is a single END-OF-GENERATION snapshot "
        "(`RetentionSummary.instantaneous_retention_ratio`); no per-step retention trajectory "
        "is committed, so `final_retention_ratio` is the only retention statistic directly "
        "available per pair -- not a minimum or a trajectory.\n"
        "- \"Decoded\" context fields are derived from each record's own already-decoded "
        "`decoded_output` string; no tokenizer was loaded or downloaded to produce them. "
        "The raw-token-id window fields are token ids, not decoded text -- decoding a token "
        "window precisely would require the pinned tokenizer, which this CPU-only, no-network "
        "analysis deliberately does not load.\n"
        f"\n{summary['claim_boundary_notice']}\n"
    )

    p.write_text("".join(parts), encoding="utf-8")
