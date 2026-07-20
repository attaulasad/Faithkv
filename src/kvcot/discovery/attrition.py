"""Structured B1B attrition accounting
(`docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md` §9, authorized by CLAUDE.md
§1b/§4b). Every eligibility/validity filter in the B1B harness is
potentially correlated with whatever a future method would change, exactly
the same reasoning `kvcot.analysis.summaries.build_attrition_funnel_table`
already applies to the primary pipeline (CLAUDE.md §8.4) — this module is
the B1B-harness-shaped analogue, a separate set of stages, never merged
with the primary funnel's stage list.

Pure Python (no torch import) — usable from CPU-only planning/reporting
code exactly like `kvcot.discovery.sampling`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Every stage a single example can be dropped at, IN THE ORDER an example
# passes through them. `AttritionCounters.stage_order` is this exact tuple
# — never reordered independently anywhere else, so a funnel table's rows
# always read top-to-bottom in the order attrition actually happens.
STAGE_NATURAL_RUN_INVALID = "natural_run_invalid"
STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE = "answer_incorrect_or_unverifiable"
STAGE_CAP_HIT = "cap_hit"
STAGE_FEWER_THAN_THREE_ELIGIBLE_EVENTS = "fewer_than_three_eligible_events"
STAGE_INVALID_CANDIDATE_DONOR_POOL = "invalid_candidate_donor_pool"
STAGE_PASS2_TOKEN_MISMATCH = "pass2_token_mismatch"
STAGE_PREFILL_CONTRACT_VIOLATION = "prefill_contract_violation"
STAGE_COMPACTION_EVENT_MISMATCH = "compaction_event_mismatch"
STAGE_MISSING_TARGET_SNAPSHOT = "missing_target_snapshot"
STAGE_OBSERVED_SURVIVOR_MISMATCH = "observed_survivor_mismatch"
STAGE_CAPTURE_GATHER_PARITY_FAILURE = "capture_gather_parity_failure"
STAGE_UNCERTAINTY_MISSING = "uncertainty_missing"
STAGE_BRANCH_EVALUATION_FAILURE = "branch_evaluation_failure"
STAGE_SCHEMA_VALIDATION_FAILURE = "schema_validation_failure"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_NATURAL_RUN_INVALID,
    STAGE_ANSWER_INCORRECT_OR_UNVERIFIABLE,
    STAGE_CAP_HIT,
    STAGE_FEWER_THAN_THREE_ELIGIBLE_EVENTS,
    STAGE_INVALID_CANDIDATE_DONOR_POOL,
    STAGE_PASS2_TOKEN_MISMATCH,
    STAGE_PREFILL_CONTRACT_VIOLATION,
    STAGE_COMPACTION_EVENT_MISMATCH,
    STAGE_MISSING_TARGET_SNAPSHOT,
    STAGE_OBSERVED_SURVIVOR_MISMATCH,
    STAGE_CAPTURE_GATHER_PARITY_FAILURE,
    STAGE_UNCERTAINTY_MISSING,
    STAGE_BRANCH_EVALUATION_FAILURE,
    STAGE_SCHEMA_VALIDATION_FAILURE,
)


@dataclass
class AttritionCounters:
    """`total_entered` is the starting population (examples attempted).
    `dropped_at[stage]` counts examples that failed AT exactly that stage
    (never double-counted at a later stage once dropped). `passed_all`
    counts examples that cleared every stage. `total_entered ==
    passed_all + sum(dropped_at.values())` is an invariant this class
    enforces itself (`assert_consistent`) — a denominator can never
    silently shrink without being accounted for at some named stage."""

    total_entered: int = 0
    dropped_at: dict[str, int] = field(default_factory=lambda: {stage: 0 for stage in STAGE_ORDER})
    passed_all: int = 0

    def record_entered(self) -> None:
        self.total_entered += 1

    def record_dropped(self, stage: str) -> None:
        if stage not in self.dropped_at:
            raise ValueError(f"unknown attrition stage {stage!r}; must be one of {STAGE_ORDER}")
        self.dropped_at[stage] += 1

    def record_passed(self) -> None:
        self.passed_all += 1

    def assert_consistent(self) -> None:
        total_dropped = sum(self.dropped_at.values())
        if self.total_entered != self.passed_all + total_dropped:
            raise ValueError(
                f"attrition accounting is inconsistent: total_entered={self.total_entered}, "
                f"passed_all={self.passed_all}, sum(dropped_at)={total_dropped} -- "
                f"{self.total_entered} != {self.passed_all} + {total_dropped}. A denominator shrank "
                "silently somewhere -- every dropped example must be recorded at exactly one named stage."
            )

    def funnel_table(self) -> list[dict[str, object]]:
        """One row per stage, in `STAGE_ORDER`: `remaining_before` (how many
        examples were still alive entering this stage) and `dropped_here`.
        The final implicit row (`passed_all`) is reported separately, never
        folded into a stage row, since "passed everything" is not itself a
        drop reason."""
        self.assert_consistent()
        remaining = self.total_entered
        rows = []
        for stage in STAGE_ORDER:
            dropped_here = self.dropped_at[stage]
            rows.append({"stage": stage, "remaining_before": remaining, "dropped_here": dropped_here})
            remaining -= dropped_here
        assert remaining == self.passed_all
        return rows
