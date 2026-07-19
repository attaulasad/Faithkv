"""Part X.23 structural guard tests: fail if active files regress into any
of the defects this B1A pass repairs. Historical documents may contain the
superseded text only when clearly marked as historical/superseded -- this
file's exemption lists are exactly those documents, never a live schema,
source file, or the current-status section of a status document.
"""
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Documents allowed to still contain historical/superseded text verbatim
# (each one is explicitly marked superseded/historical inline, per
# docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md Part I.2's instruction to
# preserve prior protocol text rather than delete it).
HISTORICAL_DOCS = {
    REPO_ROOT / "docs" / "B0_5_R2_1_FINAL_PROTOCOL.md",
    REPO_ROOT / "docs" / "B0_5_R2_DENSE_CACHE_REPAIR.md",
    REPO_ROOT / "docs" / "B0_5_PROTOCOL_REPAIR.md",
    REPO_ROOT / "docs" / "B0_5_DISCOVERY_PROTOCOL.md",
    REPO_ROOT / "docs" / "b0_5_decision.json",
    REPO_ROOT / "CHANGELOG.md",
}

THIS_FILE = Path(__file__).resolve()


def _iter_active_python_files():
    for pkg in ("src", "tests"):
        for path in (REPO_ROOT / pkg).rglob("*.py"):
            if path == THIS_FILE:
                continue
            yield path


def test_no_unconditional_replace_qwen2_call_in_generation_code():
    # A real CALL (open paren) to replace_qwen2 outside the verified
    # dispatch table is exactly the repaired defect (Part V.9): it must
    # never appear anywhere in src/ again.
    pattern = re.compile(r"replace_qwen2\s*\(")
    offenders = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path)
    assert not offenders, f"unconditional replace_qwen2(...) call found outside the dispatch table: {offenders}"


def test_dispatch_table_is_the_only_place_naming_all_three_patchers():
    dispatch_path = REPO_ROOT / "src" / "kvcot" / "discovery" / "dispatch.py"
    text = dispatch_path.read_text(encoding="utf-8")
    for name in ("replace_qwen2", "replace_llama", "replace_qwen3"):
        assert name in text, f"{name} missing from the verified dispatch table"


def test_no_active_use_of_ambiguous_position_field_name():
    banned = "first_affected_logit_absolute_position"
    offenders = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path in HISTORICAL_DOCS or path == THIS_FILE:
            continue
        if any(part in {".git", ".venv", "third_party", "results", ".hf_cache_tmp"} for part in path.parts):
            continue
        if path.suffix not in {".py", ".yaml", ".yml", ".json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if banned in text:
            # A single explanatory docstring mention of the OLD name (to
            # document the fix) is fine; a field/attribute DEFINITION is not.
            if path.name == "schemas.py" and "Corrects the ambiguous" in text:
                continue
            offenders.append(path)
    assert not offenders, f"active use of the superseded ambiguous field name found in: {offenders}"


def test_active_schema_uses_corrected_split_timing_fields():
    from kvcot.discovery.schemas import SwapPairRecord

    fields = SwapPairRecord.model_fields
    assert "first_affected_forward_input_absolute_position" in fields
    assert "first_affected_logit_target_absolute_position" in fields
    assert "first_affected_logit_absolute_position" not in fields


def test_no_direct_chronology_to_depth_mapping_in_sampling_module():
    sampling_path = REPO_ROOT / "src" / "kvcot" / "discovery" / "sampling.py"
    text = sampling_path.read_text(encoding="utf-8")
    # The banned pattern this repairs: depth assigned directly FROM the
    # chronological ordinal (e.g. `depth_stratum_by_event = {event_id: k
    # for k, event_id in enumerate(...)}`), never independently permuted.
    banned_pattern = re.compile(r"depth_stratum(_by_event)?\s*[:=]\s*\{?\s*event_id\s*:\s*k\b")
    assert not banned_pattern.search(text), "direct chronology-to-depth mapping found in sampling.py"
    # Positive check: the actual (correct) mechanism -- an independent
    # permutation of the three depth strata -- must be present.
    assert "sample([0, 1, 2], 3)" in text


def test_entropy_and_logit_margin_are_pair_specific_not_event_shared():
    from kvcot.discovery.schemas import SwapPairRecord

    fields = SwapPairRecord.model_fields
    # Each swap PAIR carries its own candidate/donor source values -- never
    # one shared value for every candidate at an eviction event.
    for name in ("entropy_e", "entropy_r", "entropy_diff", "logit_margin_e", "logit_margin_r", "logit_margin_diff"):
        assert name in fields, f"{name} missing -- entropy/logit-margin must be recorded per pair, not per event"


def test_claude_md_states_no_gpu_or_method_authorization_in_discovery_exception():
    claude_md = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    normalized = " ".join(claude_md.lower().split())
    assert "no model inference or gpu use is authorized by this amendment" in normalized
    assert "no final faithfulness-aware compression method" in normalized


def test_correction_document_states_b1b_b2a_remain_unauthorized():
    doc_path = REPO_ROOT / "docs" / "B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md"
    assert doc_path.exists(), "docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md must exist"
    text = doc_path.read_text(encoding="utf-8")
    assert "B1B" in text and "B2A" in text
    assert "not authorize" in text.lower() or "does not authorize" in text.lower()


def test_no_document_claims_b0_5_discovery_has_passed():
    banned_claims = [
        "DISCOVERY-SUPPORTING" ,
    ]
    active_status_docs = [
        REPO_ROOT / "PLAN.md",
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md",
    ]
    for path in active_status_docs:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for claim in banned_claims:
            # DISCOVERY-SUPPORTING is only allowed to appear as part of the
            # enumerated OUTCOME SET description, never asserted as this
            # session's actual result.
            for line in text.splitlines():
                if claim in line:
                    assert (
                        "outcome" in line.lower()
                        or "possible" in line.lower()
                        or "not " in line.lower()
                        or "no inference result exists" in text.lower()
                    ), f"{path} appears to claim a discovery result: {line!r}"
