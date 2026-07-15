"""Figure generation. Matplotlib is imported lazily inside each function
(never at module scope) so `import kvcot.analysis.plots` succeeds even
without the optional `plots` extra installed, and so this module still
passes the "no torch" import check trivially (matplotlib is not torch, but
keeping every heavy/optional dependency lazy here is the same discipline).

Every figure embeds `CLAIM_BOUNDARY_NOTICE` as a caption so a reader who
only sees a saved PNG, not this source file, still gets the claim boundary
(§1).
"""
from __future__ import annotations

from pathlib import Path

from kvcot.probes.early_answering import CLAIM_BOUNDARY_NOTICE


def plot_agreement_curve(
    fractions: list[float],
    match_rate_full: list[float],
    match_rate_rkv: list[float],
    output_path: str | Path,
    title: str = "Early-answering agreement with own untruncated base answer",
) -> None:
    """Full agreement curve across all probe fractions (Stage 1A/1B/2), one
    line per condition. f=0 and f=1 are included on this descriptive plot
    even though they are excluded from EAS (§8.1) — the curve is meant to
    show the whole picture; EAS itself is computed elsewhere
    (kvcot.analysis.metrics) strictly over the 7 scored fractions.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fractions, match_rate_full, marker="o", label="FullKV")
    ax.plot(fractions, match_rate_rkv, marker="s", label="R-KV")
    ax.set_xlabel("Probe fraction of think span (f)")
    ax.set_ylabel("Match rate vs. own untruncated base answer")
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.text(0.01, 0.01, CLAIM_BOUNDARY_NOTICE, fontsize=6, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_delta_eas_distribution(
    problem_level_delta_eas: list[float],
    output_path: str | Path,
    title: str = "Problem-level Delta_EAS distribution (EAS_full - EAS_rkv)",
) -> None:
    """Histogram of the one-number-per-problem Delta_EAS values that feed
    the primary Wilcoxon test (kvcot.analysis.stats.wilcoxon_delta_eas).
    Positive values are the hypothesized direction (R-KV less sensitive to
    truncation) — a vertical reference line at 0 makes the split visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(problem_level_delta_eas, bins=20, edgecolor="black")
    ax.axvline(0.0, color="red", linestyle="--", linewidth=1, label="Delta_EAS = 0")
    ax.set_xlabel("Delta_EAS (positive = R-KV less sensitive to truncation)")
    ax.set_ylabel("Number of problems")
    ax.set_title(title)
    ax.legend()
    fig.text(0.01, 0.01, CLAIM_BOUNDARY_NOTICE, fontsize=6, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_realized_retention(
    fullkv_equivalent_slots: list[int],
    instantaneous_retention_ratio: list[float],
    output_path: str | Path,
    condition_label: str,
) -> None:
    """§9: realized retention over the course of generation for one
    example/condition — never label the resulting curve or its plateau
    value as a fixed percentage in isolation; it is per-example and
    per-snapshot, plotted here exactly to show that it moves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fullkv_equivalent_slots, instantaneous_retention_ratio, marker=".")
    ax.set_xlabel("FullKV-equivalent slots processed so far")
    ax.set_ylabel("Instantaneous retention ratio (physical / fullkv-equivalent)")
    ax.set_title(f"Realized retention over generation — {condition_label}")
    ax.set_ylim(0, 1.05)
    fig.text(0.01, 0.01, CLAIM_BOUNDARY_NOTICE, fontsize=6, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
