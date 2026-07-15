"""End-to-end wiring from raw JSONL (base + probe records) to the frozen
primary result (§8). This is the glue that turns what `kvcot generate` and
`kvcot replay-probe` write on the GPU host into problem-level Delta_EAS, the
Wilcoxon/bootstrap tests, the paired accuracy headline, and the attrition
funnel — using ONLY the sanctioned building blocks in
`kvcot.analysis.metrics` / `.stats` / `.summaries`. In particular the single
per-problem seed averaging happens exclusively in
`kvcot.analysis.metrics.aggregate_problem_delta_eas` (§8.3); nothing here
re-derives it.

Must never import torch (tests/unit/test_no_analysis_torch_import.py) — every
input is a plain dict read from JSONL.

CLAIM BOUNDARY (§1): this pipeline measures counterfactual behavioral
dependence on the visible generated chain-of-thought under truncation.
Nothing here licenses any statement about internal faithfulness or cognition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from kvcot.analysis.metrics import (
    EligibilityCheck,
    aggregate_problem_delta_eas,
    compute_delta_eas,
    compute_eas,
    compute_match,
    think_parsed_ok,
)
from kvcot.config import PROBE_FRACTIONS_SCORED
from kvcot.utils.io import read_jsonl

F1_FRACTION = 1.0


@dataclass(frozen=True)
class ConditionRecords:
    condition: str
    # (source_row_index, seed) -> base record dict
    base_by_key: dict[tuple[int, int], dict]
    # base_record_id -> {fraction: probe record dict}
    probes_by_base: dict[str, dict[float, dict]]


def _base_key(rec: dict) -> tuple[int, int]:
    return (rec["dataset"]["source_row_index"], rec["global_seed"])


def load_condition_records(base_path: str | Path, probe_path: str | Path, condition: str) -> ConditionRecords:
    """Load one condition's base and probe JSONL into lookup tables. Missing
    files are tolerated as empty (a partially-completed run analyzes what
    exists; the attrition funnel then shows the loss explicitly)."""
    base_by_key: dict[tuple[int, int], dict] = {}
    for rec in read_jsonl(base_path):
        base_by_key[_base_key(rec)] = rec

    probes_by_base: dict[str, dict[float, dict]] = {}
    for rec in read_jsonl(probe_path):
        probes_by_base.setdefault(rec["base_record_id"], {})[float(rec["fraction"])] = rec

    return ConditionRecords(condition=condition, base_by_key=base_by_key, probes_by_base=probes_by_base)


def discover_conditions(output_dir: str | Path) -> tuple[str, str]:
    """Return (full_condition, rkv_condition) file stems present in
    `output_dir`. FullKV is always the literal 'full'; the R-KV condition is
    whichever `rkv_b{budget}.jsonl` base file is present (there is exactly one
    per Stage 2 run — the resolved operating point)."""
    d = Path(output_dir)
    rkv = sorted(
        p.stem for p in d.glob("rkv_b*.jsonl") if not p.stem.endswith("_probes")
    )
    if not rkv:
        raise FileNotFoundError(
            f"no rkv_b*.jsonl base file found in {output_dir} — has `kvcot generate` "
            "run for the R-KV condition yet?"
        )
    if len(rkv) > 1:
        raise ValueError(f"expected exactly one rkv_b* condition in {output_dir}, found {rkv}")
    return "full", rkv[0]


def _matches_by_fraction(
    cond: ConditionRecords, base_rec: dict, fractions: Iterable[float]
) -> dict[float, bool | None]:
    """match(f) for the given base record's own condition: probe answer at f
    vs this record's untruncated base answer, SAME condition, SAME seed
    (§8.1). Undefined (None) whenever the probe is missing or either side
    failed extraction — never silently coerced to a match or mismatch."""
    base_answer = base_rec["extracted_answer"]
    probes = cond.probes_by_base.get(base_rec["record_id"], {})
    out: dict[float, bool | None] = {}
    for f in fractions:
        probe = probes.get(f)
        if probe is None:
            continue  # leave absent -> compute_eas reports it as missing
        out[f] = compute_match(probe.get("normalized_probe_answer"), base_answer)
    return out


def _f1_stable(cond: ConditionRecords, base_rec: dict) -> bool:
    probe = cond.probes_by_base.get(base_rec["record_id"], {}).get(F1_FRACTION)
    if probe is None:
        return False
    return compute_match(probe.get("normalized_probe_answer"), base_rec["extracted_answer"]) is True


@dataclass(frozen=True)
class PairResult:
    source_row_index: int
    seed: int
    eligibility: EligibilityCheck
    delta_eas: float | None  # None unless eligible AND both EAS defined


def _all_scored_present(matches: dict[float, bool | None], scored: tuple[float, ...]) -> bool:
    return all(f in matches for f in scored)


def build_pair_results(
    full: ConditionRecords,
    rkv: ConditionRecords,
    scored_fractions: tuple[float, ...] = PROBE_FRACTIONS_SCORED,
) -> list[PairResult]:
    """One PairResult per (problem, seed) for which BOTH conditions have a base
    record. Computes each condition's EAS, the per-pair eligibility (§8.3), and
    Delta_EAS = EAS_full - EAS_rkv (only when eligible and both EAS defined)."""
    results: list[PairResult] = []
    shared_keys = sorted(set(full.base_by_key) & set(rkv.base_by_key))
    for key in shared_keys:
        src_idx, seed = key
        base_full = full.base_by_key[key]
        base_rkv = rkv.base_by_key[key]

        matches_full = _matches_by_fraction(full, base_full, scored_fractions)
        matches_rkv = _matches_by_fraction(rkv, base_rkv, scored_fractions)

        eas_full = compute_eas(matches_full, scored_fractions)
        eas_rkv = compute_eas(matches_rkv, scored_fractions)

        elig = EligibilityCheck(
            full_base_correct=base_full.get("is_correct") is True,
            rkv_base_correct=base_rkv.get("is_correct") is True,
            full_think_parsed=think_parsed_ok(base_full["think_span"]["think_parse_status"]),
            rkv_think_parsed=think_parsed_ok(base_rkv["think_span"]["think_parse_status"]),
            full_f1_stable=_f1_stable(full, base_full),
            rkv_f1_stable=_f1_stable(rkv, base_rkv),
            rkv_had_compaction=base_rkv.get("compaction_count", 0) >= 1,
            all_scored_probes_present=(
                _all_scored_present(matches_full, scored_fractions)
                and _all_scored_present(matches_rkv, scored_fractions)
            ),
        )

        delta = None
        if elig.eligible:
            delta = compute_delta_eas(eas_full.value, eas_rkv.value)
        results.append(
            PairResult(source_row_index=src_idx, seed=seed, eligibility=elig, delta_eas=delta)
        )
    return results


def problem_level_delta_eas(pairs: list[PairResult], min_eligible_seeds: int = 2):
    """Average Delta_EAS across a problem's eligible seeds — the ONLY place
    that averaging is allowed (delegates to metrics.aggregate_problem_delta_eas)
    — and return one aggregate per problem plus the flat list of the defined
    problem-level values that enter the primary tests (§8.3)."""
    per_problem: dict[int, dict[int, float | None]] = {}
    for p in pairs:
        per_problem.setdefault(p.source_row_index, {})[p.seed] = p.delta_eas

    aggregates = [
        aggregate_problem_delta_eas(str(src_idx), seed_map, min_eligible_seeds=min_eligible_seeds)
        for src_idx, seed_map in sorted(per_problem.items())
    ]
    primary_values = [a.delta_eas for a in aggregates if a.delta_eas is not None]
    return aggregates, primary_values


def paired_accuracy_inputs(
    full: ConditionRecords, rkv: ConditionRecords
) -> tuple[list[float], list[float]]:
    """Per-problem paired base accuracy over the full split (§8.5 accuracy
    headline — a distributional check over ALL shared problems, not the
    both-correct subset). Each problem contributes ONE value per condition:
    the mean over its seeds of `is_correct is True` (extraction failure counts
    as incorrect, never as a match). Problems must be present in both
    conditions to be paired; seeds are intersected per problem."""
    full_problems: dict[int, list[bool]] = {}
    rkv_problems: dict[int, list[bool]] = {}
    for (src_idx, _seed), rec in full.base_by_key.items():
        full_problems.setdefault(src_idx, []).append(rec.get("is_correct") is True)
    for (src_idx, _seed), rec in rkv.base_by_key.items():
        rkv_problems.setdefault(src_idx, []).append(rec.get("is_correct") is True)

    full_acc: list[float] = []
    rkv_acc: list[float] = []
    for src_idx in sorted(set(full_problems) & set(rkv_problems)):
        f = full_problems[src_idx]
        r = rkv_problems[src_idx]
        full_acc.append(sum(f) / len(f))
        rkv_acc.append(sum(r) / len(r))
    return full_acc, rkv_acc


def _funnel_record(cond: ConditionRecords, base_rec: dict) -> dict:
    """One attrition-funnel row-input per (problem, seed), with the boolean
    flags the funnel predicates in summaries.py read (§8.4)."""
    return {
        "base_generated": True,
        "cap_hit": bool(base_rec.get("cap_hit", False)),
        "think_parsed": think_parsed_ok(base_rec["think_span"]["think_parse_status"]),
        "extracted": base_rec.get("extracted_answer") is not None,
        "correct": base_rec.get("is_correct") is True,
        "f1_stable": _f1_stable(cond, base_rec),
        "compaction_occurred": base_rec.get("compaction_count", 0) >= 1,
    }


def funnel_records(cond: ConditionRecords) -> list[dict]:
    return [_funnel_record(cond, rec) for _, rec in sorted(cond.base_by_key.items())]


def count_answer_changed_at_any_scored_fraction(
    cond: ConditionRecords, scored_fractions: tuple[float, ...] = PROBE_FRACTIONS_SCORED
) -> tuple[int, int]:
    """Stage 1A measurability (§10): among base-correct records with a parsed
    think span and all scored probes present, how many change their answer at
    at least one scored fraction (a defined mismatch). Returns
    (n_eligible_base, n_changed)."""
    n_eligible = 0
    n_changed = 0
    for _, base_rec in cond.base_by_key.items():
        if base_rec.get("is_correct") is not True:
            continue
        if not think_parsed_ok(base_rec["think_span"]["think_parse_status"]):
            continue
        matches = _matches_by_fraction(cond, base_rec, scored_fractions)
        if not _all_scored_present(matches, scored_fractions):
            continue
        if any(m is None for m in matches.values()):
            continue
        n_eligible += 1
        if any(m is False for m in matches.values()):
            n_changed += 1
    return n_eligible, n_changed
