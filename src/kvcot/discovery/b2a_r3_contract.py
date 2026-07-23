"""B2A-R3 frozen CPU-only contract constants (Step 3 Stage-A,
`CLAUDE.md` §1h, `docs/B2A_R3_STEP2B_INDEPENDENT_REAUDIT_2026-07-23.md`).

Every literal in this module is copied verbatim from
`docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md` (the Step
2B-repaired, independently re-audited protocol) and cross-checked against
a real repository source at the point of use — never invented here. This
module is pure Python: no `torch`, no `transformers`, no
`kvcot.discovery.b2a_workers`/`kvcot.discovery.schemas` import, so it stays
safe to import from every CPU-only dry-run/verification path.

## Corrected note on `SAFETY_MULTIPLIER` (independent Step 2B re-audit,
## 2026-07-23)

The protocol's own §10.5.4 prose states that `1.20` "is exactly
representable in IEEE-754 binary64 as literal `1.2`." That sentence is
imprecise: `1.2` is a repeating binary fraction and is *not* exactly
representable in binary64. What *is* exact and reproducible is comparing
the literal Python float `1.2` against itself — both sides of `==` round
to the identical nearest representable double, so the gate below
(`safety_multiplier_exact`, `kvcot.discovery.b2a_r3_qualification`) is a
genuine, deterministic equality check, never a tolerance comparison. This
module's `SAFETY_MULTIPLIER` constant is the one and only frozen literal
every comparison must be made against — no `math.isclose`, no epsilon.
"""
from __future__ import annotations

import math
import re
from typing import Any, Final

from kvcot.utils.hashing import sha256_json, sha256_text

# --------------------------------------------------------------------------
# Repository identity (protocol §21)
# --------------------------------------------------------------------------
REQUIRED_REPOSITORY: Final[str] = "asad073-ui/Faithkv"

# --------------------------------------------------------------------------
# Per-artifact protocol-identity fields (protocol §12.9) -- four separate
# concepts, never one bare `protocol_version`.
# --------------------------------------------------------------------------
CANDIDATE_MANIFEST_ARTIFACT_SCHEMA_VERSION: Final[str] = "faithkv-b2a-r3-candidate-manifest-v1"
CANDIDATE_ORDER_PROTOCOL_VERSION: Final[str] = "faithkv-b2a-r3-row-order-v1"

QUALIFICATION_ARTIFACT_SCHEMA_VERSION: Final[str] = "faithkv-b2a-r3-qualification-artifact-v1"
QUALIFICATION_PROTOCOL_VERSION: Final[str] = "faithkv-b2a-r3-qualification-v1"
RUNTIME_PREDICTOR_VERSION: Final[str] = "faithkv-b2a-r3-runtime-predictor-v1"

SELECTION_PROVENANCE_ARTIFACT_SCHEMA_VERSION: Final[str] = "faithkv-b2a-r3-selection-provenance-v1"
SELECTION_PROTOCOL_VERSION: Final[str] = "faithkv-b2a-r3-selection-v1"

AUTHORIZATION_CLAIM_ARTIFACT_SCHEMA_VERSION: Final[str] = "faithkv-b2a-r3-authorization-claim-v1"

# A fifth, distinct identity concept (protocol §14.5): versions the
# IN-MEMORY `AttemptProvenancePolicy` object, never a persisted artifact.
PROVENANCE_POLICY_VERSION: Final[str] = "faithkv-b2a-r3-provenance-policy-v1"

# Names the algorithm `selected_manifest_sha256` used (protocol §12.1,
# §12.7, R3-AUDIT-21) -- an EXTERNAL hash over `B2AOneExampleManifest`,
# never a `canonical_sha256` field added to that historical schema.
SELECTED_MANIFEST_HASH_ALGORITHM: Final[str] = "B2AOneExampleManifest.manifest_hash-v1"

# --------------------------------------------------------------------------
# Dataset / model / tokenizer / config identity (protocol §4, cross-checked
# against configs/discovery/llama8b_math500_b1024.yaml)
# --------------------------------------------------------------------------
DATASET_REPO: Final[str] = "HuggingFaceH4/MATH-500"
DATASET_CONFIG: Final[str] = "default"
DATASET_SPLIT: Final[str] = "test"
DATASET_REVISION: Final[str] = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"

MODEL_NAME: Final[str] = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
MODEL_REVISION: Final[str] = "6a6f4aa4197940add57724a7707d069478df56b1"
TOKENIZER_NAME: Final[str] = MODEL_NAME
TOKENIZER_REVISION: Final[str] = MODEL_REVISION

BUDGET: Final[int] = 1024
CONFIG_PATH: Final[str] = "configs/discovery/llama8b_math500_b1024.yaml"

# --------------------------------------------------------------------------
# Frozen exact artifact paths (protocol §12.2) -- no "recommended"
# alternative; each is the one path Step 3 must use.
# --------------------------------------------------------------------------
CANDIDATE_MANIFEST_PATH: Final[str] = "configs/discovery/b2a_r3_candidate_manifest.json"
QUALIFICATION_ARTIFACT_PATH: Final[str] = "results/decisions/b2a_r3_qualification.json"
SELECTED_MANIFEST_PATH: Final[str] = "configs/discovery/b2a_one_example_manifest.json"
SELECTION_PROVENANCE_PATH: Final[str] = "results/decisions/b2a_r3_selection_provenance.json"
AUTHORIZATION_CLAIMS_DIR: Final[str] = "results/decisions/b2a_r3_authorization_claims"
RUNTIME_SOURCE_ARTIFACT_PATH: Final[str] = "docs/evidence/B2A_R2_ATTEMPT_INDEX_2026-07-22.json"

# sha256_file(RUNTIME_SOURCE_ARTIFACT_PATH), verified twice during the
# independent Step 2B re-audit against the real committed file.
RUNTIME_SOURCE_ARTIFACT_SHA256: Final[str] = (
    "f9eff6c1785df69406309157a6921a6fa7729dfb4fb5bfe43fa3401e0691443e"
)


def attempt_directory_name(utc_compact_timestamp: str, attempt_id: str) -> str:
    """`results/decisions/b2a_r3_attempt_<UTC-timestamp>_<attempt_id>/`
    (protocol §12.2) -- B2A-R2's exact naming convention with only the
    `b2a_` prefix becoming `b2a_r3_`."""
    return f"b2a_r3_attempt_{utc_compact_timestamp}_{attempt_id}"


# --------------------------------------------------------------------------
# Authorization-ID validation (protocol §14.4.1)
# --------------------------------------------------------------------------
_AUTHORIZATION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_authorization_id(value: Any) -> str:
    """`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`, PLUS an explicit `..`-sequence
    rejection the bare regex does not itself express (`a..b` matches the
    character class alone) -- rejects any `/`, `\\`, whitespace, control
    character, empty string, ID over 128 characters, or any `..`
    sequence. A hard failure, never sanitized or coerced."""
    if not isinstance(value, str) or not _AUTHORIZATION_ID_RE.match(value):
        raise ValueError(
            f"authorization_id must match {_AUTHORIZATION_ID_RE.pattern!r} (1-128 chars, "
            f"alphanumeric/._- only, starting alphanumeric), got {value!r}"
        )
    if ".." in value:
        raise ValueError(f"authorization_id must not contain '..', got {value!r}")
    return value


def global_claim_path(authorization_id: str) -> str:
    """`results/decisions/b2a_r3_authorization_claims/<authorization_id>.json`
    (protocol §12.2, §14.4.1) -- one deterministic path per
    `authorization_id`, a pure function, never a filesystem scan."""
    validate_authorization_id(authorization_id)
    return f"{AUTHORIZATION_CLAIMS_DIR}/{authorization_id}.json"


# --------------------------------------------------------------------------
# Canonical hashing rule (protocol §12.1)
# --------------------------------------------------------------------------
CANONICAL_HASH_FIELD: Final[str] = "canonical_sha256"
_FORBIDDEN_HASH_ALIASES: Final[frozenset[str]] = frozenset({"manifest_hash", "artifact_hash"})


def compute_canonical_sha256(payload: dict[str, Any]) -> str:
    """`canonical_sha256 = sha256_json(payload with canonical_sha256
    omitted)`. Raises if `payload` already carries the field -- callers
    must compute it from the payload BEFORE inserting the field, never
    hash a payload that already contains its own hash."""
    if CANONICAL_HASH_FIELD in payload:
        raise ValueError(
            "compute_canonical_sha256 must be called on a payload that does not yet contain "
            f"{CANONICAL_HASH_FIELD!r} -- insert the computed value afterward"
        )
    reject_hash_aliases(payload)
    return sha256_json(payload)


def verify_canonical_sha256(payload: dict[str, Any]) -> None:
    """The frozen five-step verification procedure (protocol §12.1):
    require exactly one lowercase-64-hex `canonical_sha256` field, remove
    it, recompute `sha256_json` over the remainder, require exact
    equality. Raises `ValueError` on any failure -- fails closed."""
    if CANONICAL_HASH_FIELD not in payload:
        raise ValueError(f"payload has no {CANONICAL_HASH_FIELD!r} field")
    stored = payload[CANONICAL_HASH_FIELD]
    require_lowercase_hex64(stored, CANONICAL_HASH_FIELD)
    without_hash = {k: v for k, v in payload.items() if k != CANONICAL_HASH_FIELD}
    reject_hash_aliases(without_hash)
    recomputed = sha256_json(without_hash)
    if recomputed != stored:
        raise ValueError(
            f"canonical_sha256 mismatch: stored={stored!r} recomputed_from_payload={recomputed!r}"
        )


def reject_hash_aliases(payload: dict[str, Any]) -> None:
    """`manifest_hash`/`artifact_hash` must never appear on a NEW B2A-R3
    schema (protocol §12.1) -- `canonical_sha256` is the only self-hash
    field name. Does not apply to historical B2A-R1/R2 artifacts, which
    keep their historical field names unchanged."""
    present = _FORBIDDEN_HASH_ALIASES & set(payload)
    if present:
        raise ValueError(
            f"payload contains forbidden hash-alias field name(s) {sorted(present)} -- "
            f"new B2A-R3 artifacts must use {CANONICAL_HASH_FIELD!r} only"
        )


# --------------------------------------------------------------------------
# Strict scalar validators -- reused by every new B2A-R3 pydantic schema and
# by the pure qualification evaluator so "reject bool-as-int",
# "reject numeric strings", "reject NaN/Infinity" are enforced identically
# everywhere, never reimplemented ad hoc per module.
# --------------------------------------------------------------------------


def require_strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a strict int (not bool, not a numeric string), got {value!r}")
    return value


def require_strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a strict bool (not 0/1/None/a string), got {value!r}")
    return value


def require_strict_finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a strict float (not bool, not a numeric string), got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise ValueError(f"{name} must be finite (NaN/Infinity are rejected), got {value!r}")
    return as_float


def require_lowercase_hex64(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase 64-hex-character string, got {value!r}")
    return value


def require_exact_string(value: Any, expected: str, name: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise ValueError(f"{name} must be exactly {expected!r}, got {value!r}")
    return value


# --------------------------------------------------------------------------
# Frozen prior-row exclusion set (protocol §8.1, repairs R3-AUDIT-05)
# --------------------------------------------------------------------------
EXCLUSION_SET: Final[tuple[str, ...]] = (
    "test/algebra/2277.json",
    "test/counting_and_probability/181.json",
    "test/counting_and_probability/51.json",
    "test/counting_and_probability/894.json",
    "test/intermediate_algebra/1354.json",
    "test/intermediate_algebra/2022.json",
    "test/intermediate_algebra/966.json",
    "test/number_theory/427.json",
    "test/number_theory/820.json",
    "test/prealgebra/1961.json",
    "test/precalculus/323.json",
    "test/precalculus/675.json",
    "test/precalculus/807.json",
)
EXCLUSION_SET_SHA256: Final[str] = "0c46510c79a22d08e8fd610104a527e867f821a540063a5059b51a660d25bc69"


def exclusion_set_canonical_payload() -> str:
    """UTF-8, ordered lexicographically ascending, one `unique_id` per
    line, trailing newline required (protocol §8.1)."""
    return "".join(f"{unique_id}\n" for unique_id in EXCLUSION_SET)


def verify_exclusion_set() -> None:
    """Fails closed if `EXCLUSION_SET` above was ever hand-edited: exactly
    13 unique, lexicographically sorted rows, hashing to
    `EXCLUSION_SET_SHA256` exactly."""
    if len(EXCLUSION_SET) != 13:
        raise ValueError(f"EXCLUSION_SET must have exactly 13 entries, has {len(EXCLUSION_SET)}")
    if len(set(EXCLUSION_SET)) != 13:
        raise ValueError("EXCLUSION_SET contains a duplicate unique_id")
    if list(EXCLUSION_SET) != sorted(EXCLUSION_SET):
        raise ValueError("EXCLUSION_SET is not lexicographically ascending")
    computed = sha256_text(exclusion_set_canonical_payload())
    if computed != EXCLUSION_SET_SHA256:
        raise ValueError(
            f"EXCLUSION_SET does not reproduce EXCLUSION_SET_SHA256: computed={computed!r} "
            f"expected={EXCLUSION_SET_SHA256!r}"
        )


# --------------------------------------------------------------------------
# Deterministic candidate-pool construction (protocol §8.2)
# --------------------------------------------------------------------------
CANDIDATE_LEVELS: Final[tuple[int, int]] = (4, 5)
CANDIDATES_PER_LEVEL: Final[int] = 8
CANDIDATE_TOTAL_COUNT: Final[int] = 16
QUALIFICATION_CANDIDATE_LIMIT: Final[int] = 8

EMBEDDED_ROW_COLUMNS: Final[tuple[str, ...]] = ("problem", "solution", "answer", "subject", "level", "unique_id")

# --------------------------------------------------------------------------
# Thinking-span validity / trace completeness (protocol §10.1)
# --------------------------------------------------------------------------
THINK_PARSE_SUCCESS_STATUSES: Final[frozenset[str]] = frozenset({"ok", "generation_prompt_preopened_ok"})

# --------------------------------------------------------------------------
# Frozen generation-configuration payload (protocol §10.5) -- byte-for-byte
# the `generation:` block of configs/discovery/llama8b_math500_b1024.yaml.
# --------------------------------------------------------------------------
FROZEN_GENERATION_CONFIG: Final[dict[str, Any]] = {
    "generation_mode": "greedy",
    "do_sample": False,
    "temperature": None,
    "top_p": None,
    "batch_size": 1,
    "max_new_tokens": 6144,
    "framework_seed": 13,
    "attention_backend": "flash_attention_2",
    "cache_implementation": "DynamicCache",
    "no_offload_required": True,
}
GENERATION_CONFIG_SHA256: Final[str] = "b67ed818bb94e0a674e01c1400caf2e005c41c12203d6a29b0f0f5c69eef01a3"


def verify_generation_config_hash() -> None:
    computed = sha256_json(FROZEN_GENERATION_CONFIG)
    if computed != GENERATION_CONFIG_SHA256:
        raise ValueError(
            f"FROZEN_GENERATION_CONFIG does not reproduce GENERATION_CONFIG_SHA256: "
            f"computed={computed!r} expected={GENERATION_CONFIG_SHA256!r}"
        )


# --------------------------------------------------------------------------
# Frozen exact runtime-predictor constants (protocol §7.3)
# --------------------------------------------------------------------------
REFERENCE_TOTAL_TOKENS: Final[int] = 4931
REFERENCE_EXAMPLE_SECONDS: Final[float] = 1378.3004406290129
REFERENCE_PAIR_SECONDS: Final[float] = 10.247917714063078
REFERENCE_PROJECTED_TOTAL_SECONDS: Final[float] = 18034.603590369457
# Algebraically derived (protocol §7.3): REFERENCE_PROJECTED_TOTAL_SECONDS
# - B2B_EXAMPLE_COUNT*REFERENCE_EXAMPLE_SECONDS -
# B2B_REAL_PAIR_COUNT*REFERENCE_PAIR_SECONDS. Recorded as the frozen
# literal (not recomputed at import time) so the 2775/2776 integer test
# boundary stays bit-for-bit reproducible regardless of floating-point
# operation ordering.
REFERENCE_SETUP_SECONDS: Final[float] = 19.298151996218968

SAFETY_MULTIPLIER: Final[float] = 1.20
QUALIFICATION_TARGET_HOURS: Final[float] = 3.60
FINAL_RUNTIME_LIMIT_HOURS: Final[float] = 4.00

B2B_EXAMPLE_COUNT: Final[int] = 12
B2B_REAL_PAIR_COUNT: Final[int] = 144

PER_CANDIDATE_WORKER_TIMEOUT_SECONDS: Final[int] = 7200
ABSOLUTE_TIMEOUT_ENVELOPE_SECONDS: Final[int] = QUALIFICATION_CANDIDATE_LIMIT * PER_CANDIDATE_WORKER_TIMEOUT_SECONDS

# --------------------------------------------------------------------------
# Frozen qualification-condition tuple (protocol §10.5.1, repairs
# R3-AUDIT-19) -- exact order and membership; no addition, removal,
# renaming, or reordering without a dated protocol amendment.
# --------------------------------------------------------------------------
B2A_R3_QUALIFICATION_CONDITIONS: Final[tuple[str, ...]] = (
    "no_cap_hit",
    "answer_verifiable",
    "fullkv_answer_correct",
    "thinking_span_valid",
    "trace_complete",
    "prompt_token_count_present",
    "generated_token_count_present",
    "fullkv_timing_complete",
    "candidate_manifest_hash_match",
    "config_hash_match",
    "dataset_identity_match",
    "model_identity_match",
    "tokenizer_identity_match",
    "generation_config_hash_match",
    "prompt_identity_match",
    "batch_size_is_one",
    "all_parameters_on_requested_cuda",
    "no_offload_verified",
    "peak_memory_within_limit",
    "sequence_exceeds_budget",
    "predicted_compaction_present",
    "predicted_event_count_at_least_six",
    "at_least_three_events_have_49_future_tokens",
    "runtime_inputs_complete",
    "runtime_predictor_version_match",
    "safety_multiplier_exact",
    "projected_runtime_within_qualification_target",
)


def verify_qualification_conditions_tuple() -> None:
    if len(B2A_R3_QUALIFICATION_CONDITIONS) != 27:
        raise ValueError(
            f"B2A_R3_QUALIFICATION_CONDITIONS must have exactly 27 entries, "
            f"has {len(B2A_R3_QUALIFICATION_CONDITIONS)}"
        )
    if len(set(B2A_R3_QUALIFICATION_CONDITIONS)) != 27:
        raise ValueError("B2A_R3_QUALIFICATION_CONDITIONS contains a duplicate name")


# Memory/device gates (protocol §10.5.4, reused from CLAUDE.md §4c).
QUALIFICATION_MEMORY_LIMIT_BYTES: Final[int] = 22 * 1024**3
QUALIFICATION_MINIMUM_PREDICTED_EVENTS: Final[int] = 6
QUALIFICATION_MINIMUM_ELIGIBLE_EVENTS: Final[int] = 3

# Run the frozen-constant self-checks once at import time -- a byte-level
# drift in any literal above is a hard import-time failure, never a latent
# bug discovered only when a test happens to exercise it.
verify_exclusion_set()
verify_generation_config_hash()
verify_qualification_conditions_tuple()
