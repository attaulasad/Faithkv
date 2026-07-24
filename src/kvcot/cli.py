"""kvcot CLI (§13). Every expensive command supports --limit,
--problem-index, --seed, --resume, --dry-run. `--dry-run` works without a
GPU or torch installed (§14: "Exercise --dry-run for every stage in this
build; that is your only end-to-end check available here") — this is
achieved by deferring every torch/transformers/kvcot.generation import to
inside the non-dry-run branch of `generate`/`replay-probe`, never at module
scope here or in kvcot.runtime.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kvcot.config import config_identity, load_stage_config
from kvcot.data import QuestionHashMismatch, read_manifest
from kvcot.runtime import (
    capture_version_info,
    git_commit,
    git_is_dirty,
    require_operating_point,
    resolve_conditions,
    upstream_submodule_commit,
)
from kvcot.schemas import RunManifest, utc_now_iso
from kvcot.utils.hashing import sha256_bytes, sha256_file, sha256_int_ids, sha256_json
from kvcot.utils.io import read_existing_record_ids, write_json
from kvcot.utils.logging import get_logger

logger = get_logger("kvcot.cli")


def _write_run_manifest(stage_name: str, label: str, manifest) -> None:
    """Protocol v3 fix (CHANGELOG.md 2026-07-17): every prior version of this
    function wrote to a FIXED filename inside `stage.output_dir` — which is
    itself under `results/raw/...`, gitignored per `README.md`'s own
    documented layout ("raw/ is gitignored; run_manifests/, decisions/,
    tables/, figures/ are committed"). Two independent defects followed from
    that: (1) `results/run_manifests/` (the location README actually
    promises "one JSON per invocation, committed") never received anything
    but its `.gitkeep`, and (2) even setting the location aside, a fixed
    filename meant a `--resume`d or re-run invocation silently overwrote the
    previous invocation's manifest — e.g. a one-example GPU gate's manifest
    getting clobbered by the subsequent full-run invocation, losing the
    provenance record of the gate ever having run at all.

    Fixed by writing to the documented `results/run_manifests/` directory
    with a filename that embeds the exact invocation timestamp, so every
    invocation gets its own immutable file and none are ever overwritten.

    The timestamp alone is not sufficient for uniqueness: `datetime.now()`'s
    resolution can be coarser than the time between two calls in the same
    process (observed on Windows — two manifests written back-to-back can
    land on the identical microsecond tick), which would silently overwrite
    one manifest with the other, exactly the bug this function exists to
    fix. A short random suffix guarantees uniqueness regardless of clock
    resolution.
    """
    import secrets

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_path = Path("results/run_manifests") / f"{stage_name}_{label}_{timestamp}_{secrets.token_hex(4)}.json"
    write_json(out_path, manifest.model_dump(mode="json") if hasattr(manifest, "model_dump") else manifest)


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--problem-index", type=int, default=None, help="restrict to a single source_row_index")
    p.add_argument("--seed", type=int, default=None, help="restrict to a single seed (must be one of the frozen seeds)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def _load_manifest_filtered(stage, args) -> list[dict]:
    """§ external review 2026-07-16: `--limit` on the CLI must NOT be the
    only way to cap a run — `StageConfig.limit` (e.g. `early_gap_v2_b128
    .yaml`'s `limit: 10`) is a config-declared cap that previously had no
    effect at all here, so every fixed-trace stage config's documented
    "ten-example screen" silently ran against the full manifest (50 rows)
    whenever the CLI invocation omitted `--limit`. `--limit` on the command
    line still wins when given (an explicit override of the config
    default), but the config's own `limit` is the floor otherwise."""
    rows = list(read_manifest(stage.dataset_manifest))
    if args.problem_index is not None:
        rows = [r for r in rows if r["source_row_index"] == args.problem_index]
    effective_limit = args.limit if args.limit is not None else stage.limit
    if effective_limit is not None:
        rows = rows[:effective_limit]
    return rows


def _filter_base_records_for_run(stage, args, base_records: list[dict]) -> list[dict]:
    """Restrict already-generated base records (read from a `{condition}.jsonl`
    file) to the ones `--limit`/`--problem-index`/`--seed` actually asked for.

    `replay-fixed-trace` reads its canonical trace from a base file that may
    contain more rows than this particular invocation was asked to probe
    (e.g. a shared `full.jsonl` reused across several `early_gap_b*.yaml`
    screens) — without this filter, `--limit 10` on the CLI would still probe
    every record in the file, which is exactly the "accidentally probing 50
    records when you intended 10" failure this function exists to prevent.
    Reuses `_load_manifest_filtered` for the problem-index/limit half (so
    that filtering logic isn't duplicated), then additionally filters by
    `--seed` directly against each base record's own `global_seed` — manifest
    rows have no seed dimension, so that part can't be expressed as a
    manifest filter.
    """
    manifest_rows = _load_manifest_filtered(stage, args)
    allowed_indices = {row["source_row_index"] for row in manifest_rows}

    filtered = [record for record in base_records if record["dataset"]["source_row_index"] in allowed_indices]

    if args.seed is not None:
        filtered = [record for record in filtered if record["global_seed"] == args.seed]

    return filtered


def _build_method_config(condition: str, policy):
    """Build the record's MethodConfig (§9/§12: *configured*, not measured,
    parameters). For R-KV/patched-noop, copy the concrete budget/window/etc.
    off the policy so a base record fully records the operating point it ran
    under, not just the method name. Realized retention stays a separate,
    measured concept (RetentionSummary) — never derived from these."""
    from kvcot.schemas import MethodConfig

    if condition == "full":
        return MethodConfig(method="fullkv")
    mc = getattr(policy, "method_config", None)
    method = "patched_noop" if condition == "patched_noop" else "rkv"
    if mc is None:
        return MethodConfig(method=method)
    return MethodConfig(
        method=method,
        budget=mc.budget,
        window_size=mc.window_size,
        mix_lambda=mc.mix_lambda,
        retain_ratio=mc.retain_ratio,
        retain_direction=mc.retain_direction,
        divide_method=mc.divide_method,
        divide_length=mc.divide_length,
        compression_content=mc.compression_content,
    )


def _resolve_condition_name(stage, condition: str) -> str:
    """Validate a condition string against this stage's declared conditions
    and resolve the `rkv_selected` placeholder (used only by
    stage2_main.yaml) into a concrete `rkv_b{budget}` condition name, by
    reading `configs/selected_operating_point.yaml`
    (kvcot.runtime.resolve_conditions).

    Every command that accepts a condition-like argument MUST use this same
    resolution — `generate`/`replay-probe` write/read the same
    `{condition}.jsonl` / `{condition}_probes.jsonl` file pair, and
    `replay-fixed-trace`/`analyze-fixed-trace` resolve two independent
    condition-like arguments (`--trace-condition`, `--replay-condition`)
    through it. `build_policy(condition, lock)` only understands concrete
    condition names ("full", "patched_noop", "rkv_b{budget}"), never the
    "rkv_selected" placeholder itself. Do not resolve this inline at each
    call site — that duplication is exactly what let `replay-probe` drift out
    of sync with `generate` before (`replay-probe --condition rkv_selected`
    looked for a nonexistent `rkv_selected.jsonl` and passed the literal
    placeholder string to `build_policy`, which raises `ValueError`).
    """
    resolved_conditions = resolve_conditions(stage)
    if condition not in stage.conditions and condition not in resolved_conditions:
        raise SystemExit(f"condition {condition!r} is not one of this stage's conditions {stage.conditions}")
    if condition == "rkv_selected":
        return resolved_conditions[stage.conditions.index("rkv_selected")]
    return condition


def _resolve_condition(stage, args) -> str:
    return _resolve_condition_name(stage, args.condition)


class ResumeIdentityMismatchError(RuntimeError):
    pass


def _get_dotted(row: dict, dotted_path: str):
    value = row
    for part in dotted_path.split("."):
        value = value[part]
    return value


def _verify_resumable_record_ids(path: Path, model_cls, expected_identity: dict[str, str]) -> set[str]:
    """§13: "Resume skips only schema-valid completed records with matching
    config/model/upstream hashes" — not just "any record_id we've seen
    before" (`kvcot.utils.io.read_existing_record_ids`'s literal behavior,
    which this function wraps with the two checks the docstring promises and
    the old `--resume` path never actually performed).

    Every existing row in `path` must (1) validate against `model_cls`'s
    schema and (2) match every `expected_identity` field (dotted paths, e.g.
    `"provenance.upstream_rkv_commit"`) exactly. A single mismatching or
    malformed row raises `ResumeIdentityMismatchError` rather than silently
    skipping just that row — a file is one run's output; if any record in it
    was produced under different settings, the file is not safely resumable
    at all (mixing identities in one condition's JSONL would be worse than
    refusing to start). The fix is to resume into a fresh output path, not to
    reconcile row-by-row.
    """
    from kvcot.utils.io import read_jsonl

    record_ids: set[str] = set()
    for row in read_jsonl(path):
        record_id = row.get("record_id")
        try:
            model_cls.model_validate(row)
        except Exception as e:
            raise ResumeIdentityMismatchError(
                f"{path}: existing record {record_id!r} fails schema validation for "
                f"{model_cls.__name__} ({e}) — cannot safely resume into this file."
            ) from e
        for dotted_path, expected_value in expected_identity.items():
            try:
                actual_value = _get_dotted(row, dotted_path)
            except KeyError:
                actual_value = None
            if actual_value != expected_value:
                raise ResumeIdentityMismatchError(
                    f"{path}: existing record {record_id!r} has {dotted_path}={actual_value!r}, "
                    f"but this run's identity is {dotted_path}={expected_value!r} — the output file "
                    "was produced under different settings and cannot be safely resumed into. "
                    "Use a fresh output path (or a different stage/condition output_dir) instead."
                )
        if record_id is not None:
            record_ids.add(record_id)
    return record_ids


def _resolve_seeds_for_run(stage, lock, args) -> list[int]:
    seeds = stage.resolve_seeds(lock)
    if args.seed is not None:
        if args.seed not in seeds:
            raise SystemExit(f"--seed {args.seed} is not among this stage's seeds {seeds}")
        return [args.seed]
    return seeds


def _expected_stage_record_count(stage, lock) -> int:
    """The PRE-REGISTERED natural record count for this stage — manifest rows
    (after the stage config's own `limit`, the config-declared cap) × the
    stage's resolved seeds. Deliberately takes no `args` (2026-07-18 external
    review): `check-fixed-trace-accuracy` and `analyze-fixed-trace` both
    derive their expected natural-population size from THIS function and
    nothing else, so no CLI flag (`--limit`/`--problem-index`/`--seed`) can
    shrink the expectation and let a partial pair of natural files pass the
    strict accuracy gate as if it were the complete pre-registered
    experiment. For the current protocol-v3 stage (50 manifest rows × 1
    seed) this is always 50."""
    rows = list(read_manifest(stage.dataset_manifest))
    if stage.limit is not None:
        rows = rows[: stage.limit]
    return len(rows) * len(stage.resolve_seeds(lock))


def _resolve_lock_path(config_path: str, stage) -> Path:
    """The same lock-file resolution `kvcot.config.load_stage_config` uses —
    sibling of the stage config first, then the literal configured path."""
    lock_path = Path(config_path).parent / Path(stage.lock_config_path).name
    if not lock_path.exists():
        lock_path = Path(stage.lock_config_path)
    return lock_path


# ---------------------------------------------------------------- freeze-manifests

def cmd_freeze_manifests(args: argparse.Namespace) -> int:
    from kvcot.data import freeze_gsm8k_manifest, freeze_math500_manifest, write_manifest

    if args.dry_run:
        print("freeze-manifests plan:")
        print("  gsm8k_smoke_20.jsonl      <- 20 rows,  seed=13")
        print("  gsm8k_calibration_50.jsonl <- 50 rows,  seed=13, disjoint from smoke")
        print("  gsm8k_main_200.jsonl      <- 200 rows, seed=13, disjoint from smoke+calibration")
        print("  math500_backup_100.jsonl  <- 100 rows (levels 3-5), seed=13 (frozen but not run)")
        return 0

    seed = 13
    smoke = freeze_gsm8k_manifest(n_rows=20, seed=seed)
    smoke_idx = {r.source_row_index for r in smoke}
    write_manifest(smoke, "data/manifests/gsm8k_smoke_20.jsonl")

    calibration = freeze_gsm8k_manifest(n_rows=50, seed=seed, exclude_indices=smoke_idx)
    calibration_idx = {r.source_row_index for r in calibration}
    write_manifest(calibration, "data/manifests/gsm8k_calibration_50.jsonl")

    main_split = freeze_gsm8k_manifest(n_rows=200, seed=seed, exclude_indices=smoke_idx | calibration_idx)
    write_manifest(main_split, "data/manifests/gsm8k_main_200.jsonl")

    backup = freeze_math500_manifest(n_rows=100, seed=seed)
    write_manifest(backup, "data/manifests/math500_backup_100.jsonl")

    logger.info(
        "froze manifests: smoke=%d calibration=%d main=%d math500_backup=%d",
        len(smoke), len(calibration), len(main_split), len(backup),
    )
    return 0


# ---------------------------------------------------------------------- generate

def cmd_generate(args: argparse.Namespace) -> int:
    stage, lock = load_stage_config(args.config)
    condition = _resolve_condition(stage, args)

    if stage.stage_name == "stage2_main":
        require_operating_point()  # §10: refuse to start Stage 2 without a reviewed decision

    rows = _load_manifest_filtered(stage, args)
    seeds = _resolve_seeds_for_run(stage, lock, args)
    output_path = Path(stage.output_dir) / f"{condition}.jsonl"

    dataset_name = Path(stage.dataset_manifest).stem
    planned_record_ids = [
        f"base-{condition}-{dataset_name}-{row['source_row_index']}-seed{seed}"
        for row in rows
        for seed in seeds
    ]

    # Identity this run's records must carry (§13/§12) — computed up front
    # (no torch needed for any of these) so both --dry-run and the real path
    # can refuse a --resume into a file produced under different settings,
    # per kvcot.utils.io's documented resume contract.
    config_sha256 = config_identity(args.config)
    upstream_commit = upstream_submodule_commit(lock)
    expected_identity = {
        "config_sha256": config_sha256,
        "model_revision": lock.model.revision,
        "tokenizer_revision": lock.model.tokenizer_revision,
        "provenance.upstream_rkv_commit": upstream_commit,
    }

    if args.dry_run:
        from kvcot.schemas import BaseRunRecord

        already_written = set()
        if args.resume and output_path.exists():
            already_written = _verify_resumable_record_ids(output_path, BaseRunRecord, expected_identity)
        to_do = [rid for rid in planned_record_ids if rid not in already_written]
        print(f"generate plan: stage={stage.stage_name} condition={condition}")
        print(f"  model: {lock.model.name}@{lock.model.revision}")
        print(f"  rows: {len(rows)}  seeds: {seeds}  total planned records: {len(planned_record_ids)}")
        if args.resume:
            print(f"  already written (resume): {len(already_written)}  remaining: {len(to_do)}")
        print(f"  output: {output_path}")
        return 0

    # ---- real path: every GPU-dependent import deferred to here ----
    import torch

    from kvcot.generation.decode import generate_base
    from kvcot.generation.policies import build_policy
    from kvcot.generation.sampling import make_generator
    from kvcot.generation.state import reset_patched_state
    from kvcot.probes.early_answering import find_think_span
    from kvcot.probes.templates import render_base_user_message
    from kvcot.utils.answers import extract_answer
    from kvcot.runtime import gpu_model_name
    from kvcot.utils.hashing import question_hash as qhash
    from kvcot.utils.io import JsonlWriter
    from kvcot.schemas import (
        BaseRunRecord, DatasetProvenance, ProvenanceState, RetentionSummary, ThinkSpanInfo,
    )
    from transformers.cache_utils import DynamicCache
    from transformers import AutoTokenizer

    policy = build_policy(condition, lock)
    dtype = getattr(torch, lock.model.dtype)
    model = policy.load(lock.model.name, lock.model.revision, dtype, lock.attention.primary)
    tokenizer = AutoTokenizer.from_pretrained(lock.model.tokenizer_name, revision=lock.model.tokenizer_revision, use_fast=True)
    open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    device = "cuda"

    already_written = set()
    if args.resume and output_path.exists():
        already_written = _verify_resumable_record_ids(output_path, BaseRunRecord, expected_identity)
    writer = JsonlWriter(output_path, validator=lambda r: BaseRunRecord.model_validate(r).model_dump(mode="json"))

    versions = capture_version_info()
    run_start_utc = utc_now_iso()
    run_start = time.monotonic()
    n_attempted = 0
    n_completed = 0
    n_skipped_resumed = 0
    n_cap_hits = 0
    n_think_parse_failures = 0
    n_extraction_failures = 0
    total_generated_tokens = 0
    total_compaction_events = 0

    for row in rows:
        # §5: verify the manifest row wasn't corrupted/hand-edited since
        # freezing before generating anything against it — the mechanism
        # that would catch an H8-class stale/mismatched-question bug one
        # layer earlier (docs/UPSTREAM_AUDIT.md H8, kvcot.data's own
        # verify_manifest_row_against_live_question docstring).
        live_hash = qhash(row["question"])
        if live_hash != row["question_hash"]:
            raise QuestionHashMismatch(
                f"question_hash mismatch for source_row_index={row['source_row_index']} in "
                f"{stage.dataset_manifest}: manifest row's question_hash is {row['question_hash']!r} "
                f"but hashing its own question field gives {live_hash!r} — the manifest file appears "
                "corrupted or was hand-edited since freezing."
            )
        for seed in seeds:
            n_attempted += 1
            record_id = f"base-{condition}-{Path(stage.dataset_manifest).stem}-{row['source_row_index']}-seed{seed}"
            if record_id in already_written:
                n_skipped_resumed += 1
                continue

            user_message = render_base_user_message(row["question"])
            prompt_token_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message}], tokenize=True, add_generation_prompt=True
            )
            generator, derived_seed = make_generator(seed, stage.dataset_manifest, row["source_row_index"], device)

            # B1 execution-boundary closure: `reset_patched_state` no longer
            # resets CUDA peak-memory stats itself -- this call preserves
            # this loop's pre-existing per-iteration reset exactly (the
            # run-wide `peak_vram_bytes` reported below therefore still
            # reflects only the most recent iteration, unchanged).
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            cache = reset_patched_state(model, lambda: DynamicCache())
            result = generate_base(
                model, cache, prompt_token_ids, lock.generation.base_max_new_tokens,
                lock.generation.base_temperature, lock.generation.base_top_p, generator,
                tokenizer.eos_token_id, device,
            )
            decoded = tokenizer.decode(result.generated_token_ids, skip_special_tokens=True)
            span = find_think_span(prompt_token_ids, result.generated_token_ids, open_ids, close_ids)
            answer = extract_answer(decoded)
            is_correct = (
                answer.normalized_value == row["normalized_gold"] if answer.normalized_value is not None else None
            )

            # Compaction counts come from generate_base, which tracks EVENTS
            # (one count, at absolute positions) during the decode loop —
            # never events * n_layers, and never a per-layer index enumeration.
            compaction_event_steps = result.compaction_event_steps
            compaction_count = result.compaction_count
            cache_lengths = [cache.key_cache[i].shape[-2] for i in range(len(model.model.layers))]

            # §9: realized retention is MEASURED at end of this base
            # generation, never derived from the configured budget. None for
            # FullKV (which never evicts, so "retention" is vacuously 1.0 and
            # not a meaningful measurement). fullkv_equivalent_slots is the
            # total tokens processed (prompt + generated); tokens_since_last_
            # compaction mirrors CompactionTracker.tokens_since_last's own
            # convention (0 baseline when no event has ever fired).
            retention = None
            if condition != "full" and cache_lengths:
                fullkv_equivalent_slots = result.final_absolute_position
                mean_physical_slots = sum(cache_lengths) / len(cache_lengths)
                last_event_position = compaction_event_steps[-1] if compaction_event_steps else 0
                mc = getattr(policy, "method_config", None)
                retention = RetentionSummary(
                    fullkv_equivalent_slots=fullkv_equivalent_slots,
                    physical_cache_slots_per_layer=cache_lengths,
                    instantaneous_retention_ratio=(
                        mean_physical_slots / fullkv_equivalent_slots if fullkv_equivalent_slots > 0 else 0.0
                    ),
                    post_compaction_budget_tokens=getattr(mc, "budget", None),
                    tokens_since_last_compaction=result.final_absolute_position - last_event_position,
                )

            record = BaseRunRecord(
                record_id=record_id,
                config_path=args.config,
                config_sha256=config_identity(args.config),
                provenance=ProvenanceState(upstream_rkv_commit=upstream_commit, git_commit=git_commit(), git_dirty=git_is_dirty()),
                versions=versions,
                gpu_model=gpu_model_name(),
                model_name=lock.model.name,
                model_revision=lock.model.revision,
                tokenizer_name=lock.model.tokenizer_name,
                tokenizer_revision=lock.model.tokenizer_revision,
                dataset=DatasetProvenance(
                    dataset_name=Path(stage.dataset_manifest).stem,
                    dataset_config=row.get("dataset_config"),
                    dataset_revision=row.get("dataset_revision"),
                    dataset_fingerprint=row.get("dataset_fingerprint"),
                    source_row_index=row["source_row_index"],
                    question_hash=row["question_hash"],
                    normalized_gold=row["normalized_gold"],
                ),
                condition=condition,
                method_config=_build_method_config(condition, policy),
                global_seed=seed,
                derived_seed=derived_seed,
                prompt_text=user_message,
                prompt_token_ids=list(prompt_token_ids),
                generated_token_ids=result.generated_token_ids,
                decoded_output=decoded,
                think_span=ThinkSpanInfo(
                    think_start_index=span.think_start_index,
                    think_end_index=span.think_end_index,
                    think_parse_status=span.think_parse_status,
                    generation_prompt_preopened_think=span.generation_prompt_preopened_think,
                ),
                extracted_answer=answer.normalized_value,
                extraction_method=answer.method,
                extraction_failure_reason=answer.failure_reason,
                is_correct=is_correct,
                cap_hit=result.cap_hit,
                wall_time_seconds=result.wall_time_seconds,
                generated_token_count=len(result.generated_token_ids),
                compaction_count=compaction_count,
                compaction_event_steps=compaction_event_steps,
                cache_length_final_per_layer=cache_lengths,
                retention=retention,
            )
            writer.append(record.model_dump(mode="json"))

            n_completed += 1
            total_generated_tokens += len(result.generated_token_ids)
            total_compaction_events += compaction_count
            if result.cap_hit:
                n_cap_hits += 1
            if span.think_parse_status not in ("ok", "generation_prompt_preopened_ok"):
                n_think_parse_failures += 1
            if answer.normalized_value is None:
                n_extraction_failures += 1

    manifest = RunManifest(
        command="generate",
        config_path=args.config,
        config_sha256=config_sha256,
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
        versions=versions,
        start_time_utc=run_start_utc,
        end_time_utc=utc_now_iso(),
        n_attempted=n_attempted,
        n_completed=n_completed,
        n_skipped_resumed=n_skipped_resumed,
        n_failed=0,  # a per-record failure raises and aborts the run rather than being counted and continuing
        total_generated_tokens=total_generated_tokens,
        n_cap_hits=n_cap_hits,
        n_think_parse_failures=n_think_parse_failures,
        n_extraction_failures=n_extraction_failures,
        total_compaction_events=total_compaction_events,
        wall_time_seconds=time.monotonic() - run_start,
        peak_vram_bytes=(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None),
    )
    _write_run_manifest(stage.stage_name, f"generate_{condition}", manifest)

    return 0


# ------------------------------------------------------------------ replay-probe

def _hash_snapshot_cache_content(snap) -> str:
    """Real content hash of the snapshot's K/V cache — NOT just shapes
    (the old `sha256_int_ids([t.shape[-2] for t in snap.key_cache])` would
    hash equal for two snapshots with the same lengths but different
    values, which defeats the point of a divergence-detecting hash).
    Upcast to float32 before extracting bytes: two bf16 tensors holding
    the same values upcast to identical float32 bytes, so this only
    changes when the actual content differs, not the storage dtype.

    Module-level (not a closure) so both `cmd_replay_probe` and
    `cmd_replay_fixed_trace` share exactly one implementation."""
    parts = [
        t.detach().float().cpu().contiguous().numpy().tobytes()
        for t in list(snap.key_cache) + list(snap.value_cache)
    ]
    return sha256_bytes(b"".join(parts))


def _hash_snapshot_provenance_content(snap) -> str:
    """Real content hash of per-layer, per-KV-head absolute source
    positions (kvcot.generation.provenance.LayerProvenance) plus the
    prompt/think boundaries — NOT just the compaction event-step list
    (the old hash), which says nothing about which positions survived."""
    prov = snap.provenance
    ints: list[int] = [
        prov.prompt_length,
        prov.think_start_absolute if prov.think_start_absolute is not None else -1,
        prov.think_end_absolute if prov.think_end_absolute is not None else -1,
    ]
    for layer_idx in sorted(prov.layers):
        ints.extend(int(x) for x in prov.layers[layer_idx].positions.flatten().tolist())
    return sha256_int_ids(ints)


def _hash_snapshot_state(snap) -> str:
    """Real content hash of the remaining scheduling/bookkeeping state —
    NOT just [model_length, absolute_position] (the old hash), which
    ignores compression flags, after_think, and R1KV's own eviction
    bookkeeping entirely."""
    return sha256_json(
        {
            "model_length": snap.model_length,
            "after_think": snap.after_think,
            "compression_flags_per_layer": snap.compression_flags_per_layer,
            "tokens_since_last_compaction": snap.tokens_since_last_compaction,
            "absolute_position": snap.absolute_position,
            "evicted_token_num_per_layer": [
                bk.get("evicted_token_num") for bk in (snap.kv_cluster_bookkeeping_per_layer or [])
            ],
        }
    )


def cmd_replay_probe(args: argparse.Namespace) -> int:
    stage, lock = load_stage_config(args.config)
    # Resolve the `rkv_selected` placeholder the SAME way `generate` does —
    # otherwise this looks for a nonexistent `rkv_selected.jsonl` and passes
    # the literal placeholder to build_policy(), which raises ValueError.
    condition = _resolve_condition(stage, args)
    base_path = Path(stage.output_dir) / f"{condition}.jsonl"
    probe_output_path = Path(stage.output_dir) / f"{condition}_probes.jsonl"

    config_sha256 = config_identity(args.config)
    upstream_commit = upstream_submodule_commit(lock)
    expected_identity = {
        "config_sha256": config_sha256,
        "provenance.upstream_rkv_commit": upstream_commit,
    }

    if args.dry_run:
        from kvcot.schemas import ProbeRunRecord

        base_records = list(read_manifest(base_path)) if base_path.exists() else []
        n_planned = len(base_records) * len(lock.probes.fractions_all)
        already_written = set()
        if probe_output_path.exists():
            already_written = _verify_resumable_record_ids(probe_output_path, ProbeRunRecord, expected_identity)
            print(f"  already written: {len(already_written)}")
        print(f"replay-probe plan: stage={stage.stage_name} condition={condition}")
        print(f"  base records available: {len(base_records)}")
        print(f"  probe fractions: {lock.probes.fractions_all}")
        print(f"  planned probe records: {n_planned}")
        print(f"  output: {probe_output_path}")
        return 0

    # ---- real path: deferred imports ----
    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.policies import build_policy
    from kvcot.generation.replay import branch_and_probe, replay_and_snapshot
    from kvcot.generation.state import reset_patched_state
    from kvcot.probes.early_answering import ThinkSpanResult, absolute_cut_position
    from kvcot.probes.templates import render_control_suffix
    from kvcot.schemas import ProbeRunRecord, ProvenanceState
    from kvcot.utils.answers import answers_match_or_none, extract_answer
    from kvcot.utils.io import JsonlWriter, read_jsonl

    policy = build_policy(condition, lock)
    dtype = getattr(torch, lock.model.dtype)
    model = policy.load(lock.model.name, lock.model.revision, dtype, lock.attention.primary)
    tokenizer = AutoTokenizer.from_pretrained(lock.model.tokenizer_name, revision=lock.model.tokenizer_revision, use_fast=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    suffix_ids = tokenizer.encode(render_control_suffix(), add_special_tokens=False)
    device = "cuda"

    # Unlike `generate`, this loop's skip decision (`writer.already_written`,
    # below) is NOT gated on `--resume` — probe records are idempotent per
    # (base_record, fraction), so re-running always dedupes against whatever
    # `probe_output_path` already contains. That means an identity mismatch
    # is reachable independent of `--resume` too: verify whenever the file
    # already has content, not just under `--resume`.
    if probe_output_path.exists():
        _verify_resumable_record_ids(probe_output_path, ProbeRunRecord, expected_identity)
    writer = JsonlWriter(probe_output_path, validator=lambda r: ProbeRunRecord.model_validate(r).model_dump(mode="json"))
    versions = capture_version_info()
    run_start_utc = utc_now_iso()
    run_start = time.monotonic()
    n_attempted = 0
    n_completed = 0
    n_skipped_resumed = 0
    n_extraction_failures = 0

    for base in read_jsonl(base_path):
        if base["think_span"]["think_parse_status"] not in ("ok", "generation_prompt_preopened_ok"):
            continue
        span = ThinkSpanResult(
            think_start_index=base["think_span"]["think_start_index"],
            think_end_index=base["think_span"]["think_end_index"],
            think_parse_status=base["think_span"]["think_parse_status"],
            generation_prompt_preopened_think=base["think_span"]["generation_prompt_preopened_think"],
        )
        # Absolute index into the (prompt + generated) token stream at which
        # to snapshot for each fraction, matching replay_and_snapshot's
        # documented contract EXACTLY: len(prompt) + absolute_cut_position(f),
        # where absolute_cut_position = think_start_index + floor(f * L). The
        # earlier `- span.think_start_index` term made every snapshot land
        # think_start_index tokens too early for any non-preopened ("ok")
        # trace; it was masked only because this model's chat template
        # pre-opens <think> (think_start_index == 0). cmd_replay_probe accepts
        # both "ok" and "generation_prompt_preopened_ok" traces, so the bug was
        # reachable — fixed here rather than relying on the template invariant.
        cut_positions = {
            f: len(base["prompt_token_ids"]) + absolute_cut_position(span, f)
            for f in lock.probes.fractions_all
        }
        snapshots = replay_and_snapshot(
            model=model, fresh_cache_factory=lambda: DynamicCache(),
            prompt_token_ids=base["prompt_token_ids"], generated_token_ids=base["generated_token_ids"],
            think_span=span, snapshot_absolute_positions=cut_positions, device=device,
        )
        base_answer = base["extracted_answer"]
        for fraction, snap in snapshots.items():
            n_attempted += 1
            record_id = f"probe-{base['record_id']}-f{fraction}"
            if writer.already_written(record_id):
                n_skipped_resumed += 1
                continue
            result = branch_and_probe(
                model, DynamicCache(), snap, close_ids, suffix_ids,
                lock.probes.max_new_tokens, tokenizer.eos_token_id, device,
            )
            probe_text = tokenizer.decode(result.probe_output_token_ids, skip_special_tokens=True)
            probe_answer = extract_answer(probe_text)
            record = ProbeRunRecord(
                record_id=record_id,
                parent_record_id=base["record_id"],
                config_path=args.config,
                config_sha256=config_identity(args.config),
                provenance=ProvenanceState(upstream_rkv_commit=upstream_commit, git_commit=git_commit(), git_dirty=git_is_dirty()),
                versions=versions,
                base_record_id=base["record_id"],
                condition=condition,
                fraction=fraction,
                think_span_length=span.think_token_count,
                # Relative cut index the schema documents as floor(fraction*L),
                # i.e. absolute cut position minus the think-span start.
                cut_index=cut_positions[fraction] - len(base["prompt_token_ids"]) - span.think_start_index,
                control_suffix_token_ids=suffix_ids,
                probe_decoding_max_new_tokens=lock.probes.max_new_tokens,
                probe_output_token_ids=result.probe_output_token_ids,
                probe_output_text=probe_text,
                normalized_probe_answer=probe_answer.normalized_value,
                probe_extraction_status=probe_answer.method,
                matches_own_condition_base_answer=answers_match_or_none(probe_answer.normalized_value, base_answer),
                is_f1_stability_probe=(fraction == 1.0),
                snapshot_cache_hash=_hash_snapshot_cache_content(snap),
                snapshot_provenance_hash=_hash_snapshot_provenance_content(snap),
                snapshot_state_hash=_hash_snapshot_state(snap),
            )
            writer.append(record.model_dump(mode="json"))
            n_completed += 1
            if probe_answer.normalized_value is None:
                n_extraction_failures += 1

    manifest = RunManifest(
        command="replay-probe",
        config_path=args.config,
        config_sha256=config_sha256,
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
        versions=versions,
        start_time_utc=run_start_utc,
        end_time_utc=utc_now_iso(),
        n_attempted=n_attempted,
        n_completed=n_completed,
        n_skipped_resumed=n_skipped_resumed,
        n_failed=0,
        n_extraction_failures=n_extraction_failures,
        wall_time_seconds=time.monotonic() - run_start,
        peak_vram_bytes=(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None),
    )
    _write_run_manifest(stage.stage_name, f"replay_probe_{condition}", manifest)

    return 0


# ------------------------------------------------------------------ replay-fixed-trace
#
# Secondary, additive diagnostic — NOT a replacement for `replay-probe`/EAS
# above. `replay-probe` matches each condition's probe answer against that
# SAME condition's own untruncated base answer (§8 sign convention,
# kvcot.analysis.metrics), which conflates two different things whenever
# FullKV and R-KV generate different natural traces for the same problem:
# how much the *cache policy* affects truncation sensitivity, and how much
# the *trace itself differs* between conditions. `replay-fixed-trace`
# isolates the first question alone: it takes ONE canonical trace (FullKV's
# own generated tokens) and replays those exact tokens under both FullKV and
# R-KV cache policies, so both conditions see identical prompt and reasoning
# tokens — only the cache policy varies. See kvcot.analysis.fixed_trace's
# module docstring for the resulting metric (PSS) and why it is scored
# against each policy's own f=1 answer, never the trace source's sampled
# natural answer.


def _add_fixed_trace_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace-condition", default="full")
    parser.add_argument("--replay-condition", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--problem-index", type=int, default=None, help="restrict to a single source_row_index")
    parser.add_argument("--seed", type=int, default=None, help="restrict to a single seed (must be one of the frozen seeds)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--selection-file", default=None,
        help="protocol v3: results/selections/{stage}.json from `inspect-fixed-trace --write-selection` -- "
        "restricts replay/analysis to exactly the selected base_record_ids, after verifying the "
        "selection was computed against this exact config/base-file/budget/divide_length",
    )


class SelectionFileMismatchError(ValueError):
    pass


def _read_jsonl_for_selection(path: Path):
    from kvcot.utils.io import read_jsonl

    return read_jsonl(path)


def _load_fixed_trace_selection(selection_path: str, args, stage, lock, base_path: Path) -> dict:
    """Load and verify a `results/selections/{stage}.json` file (protocol v3,
    CHANGELOG.md 2026-07-17/2026-07-18) before trusting it to restrict any
    replay or analysis run. A selection computed against a different
    config, base file, budget, or divide_length must never be silently
    applied — that would replay/analyze the wrong set of examples, or
    examples whose predicted retention no longer means what the selection
    file claims it means.
    """
    from kvcot.utils.io import read_json

    selection = read_json(selection_path)
    mismatches = []
    expected_config_sha256 = config_identity(args.config)
    if selection.get("config_sha256") != expected_config_sha256:
        mismatches.append(
            f"config_sha256: selection has {selection.get('config_sha256')!r}, current config is "
            f"{expected_config_sha256!r}"
        )
    if base_path.exists():
        expected_base_sha256 = sha256_file(base_path)
        if selection.get("base_file_sha256") != expected_base_sha256:
            mismatches.append(
                f"base_file_sha256: selection has {selection.get('base_file_sha256')!r}, {base_path} "
                f"currently hashes to {expected_base_sha256!r} -- the base file changed since the "
                "selection was written"
            )
    expected_budget = stage.rkv_budgets[0] if stage.rkv_budgets else None
    if selection.get("budget") != expected_budget:
        mismatches.append(f"budget: selection has {selection.get('budget')!r}, stage config has {expected_budget!r}")
    if selection.get("divide_length") != lock.rkv.divide_length:
        mismatches.append(
            f"divide_length: selection has {selection.get('divide_length')!r}, "
            f"lock.yaml has {lock.rkv.divide_length!r}"
        )
    if selection.get("stage_name") != stage.stage_name:
        mismatches.append(f"stage_name: selection has {selection.get('stage_name')!r}, current stage is {stage.stage_name!r}")

    # 2026-07-19 review: a selection whose top-level (config/base/budget/
    # divide_length/stage_name) identity matches can STILL be internally
    # inconsistent or stale relative to the CURRENT stage config's own
    # pre-registered cap -- none of the checks above would catch that.
    expected_max_selected = stage.fixed_trace.max_selected_examples if stage.fixed_trace is not None else None
    if expected_max_selected is not None and selection.get("max_selected") != expected_max_selected:
        mismatches.append(
            f"max_selected: selection has {selection.get('max_selected')!r}, current stage config's "
            f"pre-registered fixed_trace.max_selected_examples is {expected_max_selected!r} -- refusing "
            "to replay/analyze a selection whose cap does not match today's pre-registered value "
            "(regenerate the selection, or update the config to match, rather than silently trusting "
            "a selection written under a different --max-selected override)."
        )

    selected_ids = selection.get("selected_base_record_ids", [])
    selected_rows = selection.get("selected_source_row_indices", [])
    n_selected = selection.get("n_selected")
    candidates = selection.get("candidates", [])

    # 2026-07-18 external review: duplicate CANDIDATE entries were silently
    # collapsed by the candidates_by_id dict below (last row won), so a
    # corrupted/hand-edited selection with two conflicting entries for the
    # same base_record_id passed every check against whichever happened to
    # come last. Duplicates are structural corruption -- reject them, and
    # cross-check every summary count field against the actual entries.
    candidate_ids = [c.get("base_record_id") for c in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        dupes = sorted({cid for cid in candidate_ids if candidate_ids.count(cid) > 1})
        mismatches.append(f"candidates contain duplicate base_record_id entries: {dupes}")
    candidate_row_indices = [c.get("source_row_index") for c in candidates]
    if len(candidate_row_indices) != len(set(candidate_row_indices)):
        dupes = sorted({r for r in candidate_row_indices if candidate_row_indices.count(r) > 1})
        mismatches.append(f"candidates contain duplicate source_row_index entries: {dupes}")
    n_ranked = selection.get("n_ranked")
    if n_ranked is not None and n_ranked != len(candidates):
        mismatches.append(f"n_ranked ({n_ranked!r}) does not equal the number of candidate entries ({len(candidates)})")
    n_predicted_eligible = selection.get("n_predicted_eligible")
    actual_predicted_eligible = sum(1 for c in candidates if c.get("predicted_eligible") is True)
    if n_predicted_eligible is not None and n_predicted_eligible != actual_predicted_eligible:
        mismatches.append(
            f"n_predicted_eligible ({n_predicted_eligible!r}) does not equal the number of "
            f"predicted_eligible=true candidate entries ({actual_predicted_eligible})"
        )
    selected_flagged_ids = {c.get("base_record_id") for c in candidates if c.get("selected") is True}
    if selected_flagged_ids != set(selected_ids):
        mismatches.append(
            f"candidates' own selected=true flags ({sorted(selected_flagged_ids)}) do not agree with "
            f"selected_base_record_ids ({sorted(set(selected_ids))})"
        )
    candidates_by_id = {c["base_record_id"]: c for c in candidates}

    if len(selected_ids) != len(set(selected_ids)):
        mismatches.append(f"selected_base_record_ids contains duplicates: {selected_ids}")
    if len(selected_rows) != len(set(selected_rows)):
        mismatches.append(f"selected_source_row_indices contains duplicates: {selected_rows}")
    if n_selected != len(selected_ids):
        mismatches.append(
            f"n_selected ({n_selected!r}) does not equal len(selected_base_record_ids) ({len(selected_ids)})"
        )
    # Every selected id must name a record in the canonical base file itself
    # (only checkable when the base file exists on this machine -- the
    # base_file_sha256 check above already pins its exact bytes when it does).
    if base_path.exists():
        base_record_ids = {r.get("record_id") for r in _read_jsonl_for_selection(base_path)}
        missing_from_base = sorted(set(selected_ids) - base_record_ids)
        if missing_from_base:
            mismatches.append(
                f"selected_base_record_ids not present in the canonical base file {base_path}: "
                f"{missing_from_base}"
            )
    if len(selected_ids) != len(selected_rows):
        mismatches.append(
            f"selected_base_record_ids has {len(selected_ids)} entries but "
            f"selected_source_row_indices has {len(selected_rows)} -- these must be the same length "
            "(positionally paired, both built from the same ranked-and-capped selection)"
        )
    else:
        for base_id, row_idx in zip(selected_ids, selected_rows):
            candidate = candidates_by_id.get(base_id)
            if candidate is None:
                mismatches.append(f"selected base_record_id {base_id!r} does not appear in candidates at all")
                continue
            if candidate.get("source_row_index") != row_idx:
                mismatches.append(
                    f"selected_base_record_ids/selected_source_row_indices disagree with candidates for "
                    f"{base_id!r}: selection claims source_row_index={row_idx!r}, but the candidate's own "
                    f"recorded source_row_index is {candidate.get('source_row_index')!r}"
                )
            if candidate.get("predicted_eligible") is not True:
                mismatches.append(
                    f"selected base_record_id {base_id!r} has predicted_eligible="
                    f"{candidate.get('predicted_eligible')!r} in its own candidate entry -- every "
                    "selected example must be predicted_eligible=true"
                )

    if mismatches:
        raise SelectionFileMismatchError(
            f"{selection_path} does not match the current config/base file/lock -- refusing to use a "
            "stale or mismatched selection:\n" + "\n".join(f"  - {m}" for m in mismatches)
        )
    return selection


def cmd_replay_fixed_trace(args: argparse.Namespace) -> int:
    stage, lock = load_stage_config(args.config)
    if stage.fixed_trace is None:
        raise SystemExit(f"{stage.stage_name}: missing required fixed_trace settings")
    trace_condition = _resolve_condition_name(stage, args.trace_condition)
    replay_condition = _resolve_condition_name(stage, args.replay_condition)

    # Critical rule: the model is LOADED under replay_condition's policy, but
    # the token sequence being replayed is always read from trace_condition's
    # OWN base file — never from replay_condition's base file (which, for an
    # R-KV replay_condition, would be a different, independently-sampled
    # trace, defeating the entire point of holding the trace fixed).
    base_path = Path(stage.output_dir) / f"{trace_condition}.jsonl"
    output_path = Path(stage.output_dir) / f"{replay_condition}_on_{trace_condition}_fixed_trace_probes.jsonl"

    config_sha256 = config_identity(args.config)
    upstream_commit = upstream_submodule_commit(lock)
    expected_identity = {
        "config_sha256": config_sha256,
        "model_revision": lock.model.revision,
        "tokenizer_revision": lock.model.tokenizer_revision,
        "provenance.upstream_rkv_commit": upstream_commit,
    }

    # Protocol v3 (2026-07-18 review): --selection-file restricts this run to
    # exactly the base_record_ids `inspect-fixed-trace --write-selection`
    # predicted eligible, after verifying the selection matches this exact
    # config/base-file/budget/divide_length. Without this, the ONLY way to
    # run a selected subset was one --problem-index invocation per example
    # (reloading the model every time) -- selection was written but never
    # actually consumed.
    selected_base_record_ids: set[str] | None = None
    selection_path_str: str | None = None
    selection_file_sha256: str | None = None
    selection_file_arg = getattr(args, "selection_file", None)
    if selection_file_arg:
        selection = _load_fixed_trace_selection(selection_file_arg, args, stage, lock, base_path)
        selected_base_record_ids = set(selection["selected_base_record_ids"])
        selection_path_str = str(selection_file_arg)
        selection_file_sha256 = sha256_file(selection_file_arg)
        if len(selected_base_record_ids) < stage.fixed_trace.min_eligible_examples:
            raise SystemExit(
                f"{selection_file_arg}: n_selected ({len(selected_base_record_ids)}) is below "
                f"min_eligible_examples ({stage.fixed_trace.min_eligible_examples}) -- refusing to "
                "spend GPU time replaying a selection that cannot pass the screen's own validity gate."
            )

    if args.dry_run:
        from kvcot.schemas import FixedTraceProbeRecord
        from kvcot.utils.io import read_jsonl as _read_jsonl_dry_run

        manifest_rows = _load_manifest_filtered(stage, args)
        seeds = _resolve_seeds_for_run(stage, lock, args)
        n_examples = len(manifest_rows) * len(seeds)
        if selected_base_record_ids is not None:
            # Exact count from the selection intersected with --limit/
            # --problem-index/--seed, computed the same way the real path
            # filters base_records below -- never the raw manifest count
            # (2026-07-18 review: dry-run must reflect what a selection-file
            # run will actually replay, not the full manifest).
            base_records_for_count = _filter_base_records_for_run(
                stage, args, list(_read_jsonl_dry_run(base_path))
            ) if base_path.exists() else []
            n_examples = sum(1 for r in base_records_for_count if r["record_id"] in selected_base_record_ids)
        n_fractions = len(lock.probes.fractions_all)
        already_written = set()
        if output_path.exists():
            already_written = _verify_resumable_record_ids(output_path, FixedTraceProbeRecord, expected_identity)
            print(f"  already written: {len(already_written)}")
        print(f"replay-fixed-trace plan: stage={stage.stage_name}")
        print(f"  trace_condition={trace_condition}  replay_condition={replay_condition}")
        if selected_base_record_ids is not None:
            print(f"  selection_file={args.selection_file}  n_selected={len(selected_base_record_ids)}")
        print(f"  planned examples: {n_examples}")
        print(f"  fixed-trace fractions per example: {n_fractions}")
        print(f"  planned probe records: {n_examples * n_fractions}")
        print(f"  reads canonical trace: {base_path}")
        print(f"  output: {output_path}")
        return 0

    # ---- real path: every GPU-dependent import deferred to here ----
    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    from kvcot.generation.policies import build_policy
    from kvcot.generation.replay import branch_and_probe, replay_and_snapshot
    from kvcot.probes.early_answering import ThinkSpanResult, absolute_cut_position
    from kvcot.probes.templates import render_fixed_trace_suffix
    from kvcot.schemas import FixedTraceProbeRecord, ProvenanceState, RetentionSummary
    from kvcot.utils.answers import answers_match_or_none, extract_answer, has_complete_boxed_answer
    from kvcot.utils.io import JsonlWriter, read_jsonl

    policy = build_policy(replay_condition, lock)
    dtype = getattr(torch, lock.model.dtype)
    model = policy.load(lock.model.name, lock.model.revision, dtype, lock.attention.primary)
    tokenizer = AutoTokenizer.from_pretrained(lock.model.tokenizer_name, revision=lock.model.tokenizer_revision, use_fast=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    fixed_suffix_ids = tokenizer.encode(render_fixed_trace_suffix(), add_special_tokens=False)
    device = "cuda"
    probe_max_new_tokens = stage.fixed_trace.probe_max_new_tokens

    def boxed_answer_complete(generated_ids: list[int]) -> bool:
        """Stop as soon as the box closes (§ Step 5): reconstructs the same
        prefix+generated text `extract_answer` will later see and checks for
        a complete `\\boxed{...}`, so decoding never continues into a second
        solution attempt once the answer is already unambiguous."""
        text = tokenizer.decode(fixed_suffix_ids + generated_ids, skip_special_tokens=True)
        return has_complete_boxed_answer(text)

    if output_path.exists():
        _verify_resumable_record_ids(output_path, FixedTraceProbeRecord, expected_identity)
    writer = JsonlWriter(output_path, validator=lambda r: FixedTraceProbeRecord.model_validate(r).model_dump(mode="json"))
    versions = capture_version_info()
    run_start_utc = utc_now_iso()
    run_start = time.monotonic()
    n_attempted = 0
    n_completed = 0
    n_skipped_resumed = 0
    n_extraction_failures = 0

    base_records = list(read_jsonl(base_path))
    base_records = _filter_base_records_for_run(stage, args, base_records)
    if selected_base_record_ids is not None:
        base_records = [r for r in base_records if r["record_id"] in selected_base_record_ids]

    # f=1 first, always — every other fraction's record needs its answer as
    # the anchor before it can be written (§ kvcot.analysis.fixed_trace).
    ordered_fractions = [1.0] + [f for f in lock.probes.fractions_all if f != 1.0]

    for base in base_records:
        if base["think_span"]["think_parse_status"] not in ("ok", "generation_prompt_preopened_ok"):
            continue
        span = ThinkSpanResult(
            think_start_index=base["think_span"]["think_start_index"],
            think_end_index=base["think_span"]["think_end_index"],
            think_parse_status=base["think_span"]["think_parse_status"],
            generation_prompt_preopened_think=base["think_span"]["generation_prompt_preopened_think"],
        )
        cut_positions = {
            f: len(base["prompt_token_ids"]) + absolute_cut_position(span, f) for f in lock.probes.fractions_all
        }

        planned_ids = {
            f: f"fixed-probe-{replay_condition}-on-{trace_condition}-{base['record_id']}-f{f}"
            for f in lock.probes.fractions_all
        }
        # Whole-base-record resumability: the teacher-forced replay pass that
        # produces every fraction's snapshot is one sequential walk through
        # the trace regardless of how many of the resulting probes end up
        # written (snapshotting more points along an already-required pass is
        # not meaningfully more expensive), so there is no efficiency reason
        # to reconstruct a partially-written anchor from disk. Skip the whole
        # base record only when EVERY one of its fixed-trace records is
        # already present; otherwise recompute all 9 and let the per-record
        # `already_written` check below dedupe the append.
        if all(writer.already_written(rid) for rid in planned_ids.values()):
            n_attempted += len(planned_ids)
            n_skipped_resumed += len(planned_ids)
            continue

        # replay_and_snapshot resets state itself via fresh_cache_factory —
        # no explicit reset_patched_state call needed here (mirrors
        # cmd_replay_probe, which does not call it either).
        snapshots = replay_and_snapshot(
            model=model,
            fresh_cache_factory=lambda: DynamicCache(),
            prompt_token_ids=base["prompt_token_ids"],
            generated_token_ids=base["generated_token_ids"],
            think_span=span,
            snapshot_absolute_positions=cut_positions,
            device=device,
        )

        temp_results: dict[float, dict] = {}
        for fraction in ordered_fractions:
            snap = snapshots[fraction]
            result = branch_and_probe(
                model, DynamicCache(), snap, close_ids, fixed_suffix_ids,
                probe_max_new_tokens, tokenizer.eos_token_id, device,
                stop_predicate=boxed_answer_complete,
                probe_cache_mode=stage.fixed_trace.probe_cache_mode,
            )
            # §7: extraction must run over the RECONSTRUCTED text (teacher-
            # forced boxed-answer prefix + generated tokens), never the
            # generated tokens alone — the opening `\boxed{` lives in the
            # prefix, not in anything the model itself generated.
            probe_output_text = tokenizer.decode(result.probe_output_token_ids, skip_special_tokens=True)
            probe_extraction_text = tokenizer.decode(
                fixed_suffix_ids + result.probe_output_token_ids, skip_special_tokens=True
            )
            probe_answer = extract_answer(probe_extraction_text)

            # §7 realized retention/compression AT THE CUT (the snapshot this
            # probe branched from) — measured, never derived from the
            # configured budget (§9). A recorded compaction event count > 0
            # is not sufficient evidence of actual compression (the exact-
            # budget-boundary zero-eviction case, kvcot.generation.replay
            # module docstring), so this checks physical cache length
            # directly against what FullKV would have at this position.
            snapshot_cache_lengths = [int(t.shape[-2]) for t in snap.key_cache]
            fullkv_equivalent_slots = snap.absolute_position
            mean_physical_slots = sum(snapshot_cache_lengths) / len(snapshot_cache_lengths)
            retention_ratio = (
                mean_physical_slots / fullkv_equivalent_slots if fullkv_equivalent_slots > 0 else 0.0
            )
            actual_compression_at_cut = any(
                cache_length < fullkv_equivalent_slots for cache_length in snapshot_cache_lengths
            )
            # Protocol v3 (CHANGELOG.md 2026-07-17): a SUBSTANTIAL measured
            # retention drop, never just "not exactly 1.0" — see
            # kvcot.analysis.fixed_trace.FixedTraceEligibility.
            # rkv_meaningful_compression_at_f1's docstring for why
            # actual_compression_at_cut alone overstated real compression in
            # protocol v2 (a 0.9959-retention example counted as "active").
            meaningful_compression_at_cut = retention_ratio <= stage.fixed_trace.meaningful_retention_ceiling
            replay_retention_at_cut = RetentionSummary(
                fullkv_equivalent_slots=fullkv_equivalent_slots,
                physical_cache_slots_per_layer=snapshot_cache_lengths,
                instantaneous_retention_ratio=retention_ratio,
                post_compaction_budget_tokens=(
                    policy.method_config.budget if hasattr(policy, "method_config") else None
                ),
                tokens_since_last_compaction=snap.tokens_since_last_compaction,
            )

            # §8: detect compression that happened WHILE writing the answer
            # (after the cut), which the cut snapshot above cannot show —
            # the answer should reflect the cache state at the reasoning
            # cut, not a further eviction during its own decoding.
            tokens_fed_during_probe = result.final_absolute_position - snap.absolute_position
            expected_lengths_without_eviction = [
                start_length + tokens_fed_during_probe for start_length in snapshot_cache_lengths
            ]
            probe_actual_eviction_during_answer = any(
                final_length < expected_length
                for final_length, expected_length in zip(
                    result.final_cache_lengths_per_layer, expected_lengths_without_eviction
                )
            )

            temp_results[fraction] = {
                "snapshot": snap,
                "probe_output_token_ids": result.probe_output_token_ids,
                "probe_output_text": probe_output_text,
                "probe_extraction_text": probe_extraction_text,
                "probe_answer": probe_answer,
                "probe_stop_reason": result.stop_reason,
                "replay_retention_at_cut": replay_retention_at_cut,
                "actual_compression_at_cut": actual_compression_at_cut,
                "meaningful_compression_at_cut": meaningful_compression_at_cut,
                "probe_cache_length_final_per_layer": result.final_cache_lengths_per_layer,
                "probe_actual_eviction_during_answer": probe_actual_eviction_during_answer,
            }

        # Anchor: this replay policy's OWN greedy f=1 answer — never the
        # trace source's sampled natural base answer (kvcot.schemas.
        # FixedTraceProbeRecord docstring; kvcot.analysis.fixed_trace).
        f1_answer = temp_results[1.0]["probe_answer"].normalized_value
        source_base_answer = base["extracted_answer"]
        f1_matches_source_base = answers_match_or_none(f1_answer, source_base_answer)
        f1_is_correct = answers_match_or_none(f1_answer, base["dataset"]["normalized_gold"])

        for fraction in lock.probes.fractions_all:
            n_attempted += 1
            record_id = planned_ids[fraction]
            if writer.already_written(record_id):
                n_skipped_resumed += 1
                continue
            item = temp_results[fraction]
            probe_answer = item["probe_answer"]
            snap = item["snapshot"]
            record = FixedTraceProbeRecord(
                record_id=record_id,
                parent_record_id=base["record_id"],
                config_path=args.config,
                config_sha256=config_sha256,
                provenance=ProvenanceState(upstream_rkv_commit=upstream_commit, git_commit=git_commit(), git_dirty=git_is_dirty()),
                versions=versions,
                model_revision=lock.model.revision,
                tokenizer_revision=lock.model.tokenizer_revision,
                base_record_id=base["record_id"],
                trace_source_condition=trace_condition,
                replay_policy_condition=replay_condition,
                source_row_index=base["dataset"]["source_row_index"],
                global_seed=base["global_seed"],
                normalized_gold=base["dataset"]["normalized_gold"],
                source_base_answer=source_base_answer,
                source_base_is_correct=base.get("is_correct"),
                fraction=fraction,
                think_span_length=span.think_token_count,
                cut_index=cut_positions[fraction] - len(base["prompt_token_ids"]) - span.think_start_index,
                close_marker_token_ids=close_ids,
                control_suffix_token_ids=fixed_suffix_ids,
                probe_decoding_max_new_tokens=probe_max_new_tokens,
                probe_output_token_ids=item["probe_output_token_ids"],
                probe_output_text=item["probe_output_text"],
                probe_extraction_text=item["probe_extraction_text"],
                normalized_probe_answer=probe_answer.normalized_value,
                probe_extraction_status=probe_answer.method,
                probe_stop_reason=item["probe_stop_reason"],
                probe_cap_hit=(item["probe_stop_reason"] == "max_new_tokens"),
                replay_retention_at_cut=item["replay_retention_at_cut"],
                actual_compression_at_cut=item["actual_compression_at_cut"],
                protocol_version=("v3" if stage.fixed_trace.probe_cache_mode == "frozen_at_cut" else "v2"),
                probe_cache_mode=stage.fixed_trace.probe_cache_mode,
                meaningful_compression_at_cut=item["meaningful_compression_at_cut"],
                compressed_scored_fraction=(
                    fraction in lock.probes.fractions_scored and item["meaningful_compression_at_cut"]
                ),
                probe_cache_length_final_per_layer=item["probe_cache_length_final_per_layer"],
                probe_actual_eviction_during_answer=item["probe_actual_eviction_during_answer"],
                normalized_f1_anchor_answer=f1_answer,
                matches_f1_anchor_answer=answers_match_or_none(probe_answer.normalized_value, f1_answer),
                f1_anchor_matches_source_base_answer=f1_matches_source_base,
                f1_anchor_is_correct=f1_is_correct,
                replay_compaction_count_at_cut=len(snap.compaction_event_steps),
                replay_compaction_event_steps_at_cut=list(snap.compaction_event_steps),
                snapshot_cache_hash=_hash_snapshot_cache_content(snap),
                snapshot_provenance_hash=_hash_snapshot_provenance_content(snap),
                snapshot_state_hash=_hash_snapshot_state(snap),
            )
            writer.append(record.model_dump(mode="json"))
            n_completed += 1
            if probe_answer.normalized_value is None:
                n_extraction_failures += 1

    manifest = RunManifest(
        command="replay-fixed-trace",
        config_path=args.config,
        config_sha256=config_sha256,
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
        versions=versions,
        start_time_utc=run_start_utc,
        end_time_utc=utc_now_iso(),
        n_attempted=n_attempted,
        n_completed=n_completed,
        n_skipped_resumed=n_skipped_resumed,
        n_failed=0,
        n_extraction_failures=n_extraction_failures,
        wall_time_seconds=time.monotonic() - run_start,
        peak_vram_bytes=(torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None),
        selection_path=selection_path_str,
        selection_file_sha256=selection_file_sha256,
    )
    _write_run_manifest(stage.stage_name, f"replay_fixed_trace_{replay_condition}_on_{trace_condition}", manifest)
    return 0


# ------------------------------------------------------------------------- analyze

# §8: "≥1 compaction in ≥50% of valid calibration traces" (docs/EXPERIMENT.md
# §8). Not a per-stage-config value and not one of the frozen §4 settings —
# a fixed analysis constant, kept here (not buried inline) so it is a single
# grep target if it is ever revisited.
STAGE1B_COMPACTION_ACTIVATION_THRESHOLD = 0.5


def _stage1b_output_valid(record: dict) -> bool:
    """"Valid calibration trace" (docs/EXPERIMENT.md §8) = did not hit the
    generation cap — consistent with this repo's own "did_not_hit_cap" funnel
    stage vocabulary (kvcot.analysis.summaries.FUNNEL_STAGES)."""
    return not record.get("cap_hit", True)


def _cmd_analyze_stage1a(output_dir: Path) -> int:
    from kvcot.analysis.pipeline import count_answer_changed_at_any_scored_fraction, load_condition_records
    from kvcot.analysis.summaries import build_stage1a_measurability_decision

    full = load_condition_records(output_dir / "full.jsonl", output_dir / "full_probes.jsonl", "full")
    n_total, n_changed = count_answer_changed_at_any_scored_fraction(full)
    decision = build_stage1a_measurability_decision(
        n_total=max(n_total, 1), n_answer_changed_at_any_scored_fraction=n_changed
    )
    write_json("results/decisions/stage1a_baseline_measurability.json", decision)
    print(
        f"wrote results/decisions/stage1a_baseline_measurability.json: "
        f"n_eligible={n_total} n_changed={n_changed} recommendation={decision['recommendation']}"
    )
    return 0


def _cmd_analyze_stage1b(stage, output_dir: Path) -> int:
    """§8/§10: Stage 1B's calibration decision needs only `generate` output
    (compaction_count, is_correct) — it never uses probes/EAS at all (the
    both-correct-and-compression-active EAS pipeline is Stage 2's job). Do
    NOT route this stage through the generic probe-dependent pipeline below
    — Stage 1B's run scripts only replay-probe the R-KV condition (never
    FullKV, since nothing here needs it), so requiring FullKV probes here
    would make this stage's analysis silently produce nothing forever.
    """
    from kvcot.analysis.stats import bootstrap_ci_mean
    from kvcot.analysis.summaries import build_stage1b_budget_decision
    from kvcot.utils.io import read_jsonl

    if not stage.rkv_budgets:
        raise SystemExit(f"{stage.stage_name}: stage config has no rkv_budgets set")
    budget = stage.rkv_budgets[0]
    full_records = list(read_jsonl(output_dir / "full.jsonl"))
    rkv_records = list(read_jsonl(output_dir / f"rkv_b{budget}.jsonl"))
    if not full_records or not rkv_records:
        raise SystemExit(
            f"{stage.stage_name}: missing generate output in {output_dir} — run "
            f"`kvcot generate --condition full` and `kvcot generate --condition rkv_b{budget}` first."
        )

    valid_rkv = [r for r in rkv_records if _stage1b_output_valid(r)]
    n_with_compaction = sum(1 for r in valid_rkv if r.get("compaction_count", 0) >= 1)

    full_accuracy_point_estimate = sum(1 for r in full_records if r.get("is_correct") is True) / len(full_records)
    rkv_correct = [1.0 if r.get("is_correct") is True else 0.0 for r in rkv_records]
    ci = bootstrap_ci_mean(rkv_correct, seed=20260715)

    decision = build_stage1b_budget_decision(
        budget=budget,
        n_calibration=len(valid_rkv),
        n_with_compaction=n_with_compaction,
        compaction_activation_threshold=STAGE1B_COMPACTION_ACTIVATION_THRESHOLD,
        full_accuracy_point_estimate=full_accuracy_point_estimate,
        candidate_accuracy_ci=(ci.ci_low, ci.ci_high),
    )
    out_path = f"results/decisions/stage1b_budget_{budget}.json"
    write_json(out_path, decision)
    print(f"wrote {out_path}: overall_passed={decision['overall_passed']}")
    return 0


def _cmd_analyze_stage0(stage, output_dir: Path) -> int:
    """§7/stage0_smoke.yaml notes: "throughput measurement + Stage 2
    wall-clock extrapolation" — a rough order-of-magnitude estimate from
    observed Stage 0 generate timings, not a precise forecast. Stage 0's
    actual pass/fail criteria (coherence, parity, replay identity, f=1
    stability, ≥2 real compactions) are enforced independently by the GPU
    pytest suite (docs/EXPERIMENT.md §7), not by this command — this only
    covers the one §7 criterion pytest can't check (wall-clock).
    """
    from kvcot.probes.early_answering import CLAIM_BOUNDARY_NOTICE
    from kvcot.utils.io import read_jsonl

    all_records: list[dict] = []
    for condition in stage.conditions:
        p = output_dir / f"{condition}.jsonl"
        if p.exists():
            all_records.extend(read_jsonl(p))
    if not all_records:
        print(f"analyze: no generate output found in {output_dir} yet")
        return 0

    total_tokens = sum(r.get("generated_token_count", 0) for r in all_records)
    total_wall = sum(r.get("wall_time_seconds", 0.0) for r in all_records)
    tokens_per_second = total_tokens / total_wall if total_wall > 0 else 0.0
    mean_wall_per_generate_call = total_wall / len(all_records)

    # Stage 2's shape: gsm8k_main_200, 3 seeds, 2 conditions (full + the
    # selected R-KV budget) — each (problem, seed, condition) triple needs
    # one `generate` call and, on top of it, one `replay-probe` call that
    # replays a comparable number of forward passes up to its probed cut
    # points. Approximating replay-probe's cost as roughly equal to the
    # generate call it replays is a rough estimate, not a measurement.
    n_stage2_generate_calls = 200 * 3 * 2
    estimated_generate_seconds = n_stage2_generate_calls * mean_wall_per_generate_call
    estimated_total_seconds = estimated_generate_seconds * 2

    summary = {
        "claim_boundary_notice": CLAIM_BOUNDARY_NOTICE,
        "n_stage0_records": len(all_records),
        "observed_tokens_per_second": tokens_per_second,
        "observed_mean_wall_seconds_per_generate_call": mean_wall_per_generate_call,
        "stage2_generate_call_count_estimate": n_stage2_generate_calls,
        "stage2_estimated_generate_wall_seconds": estimated_generate_seconds,
        "stage2_estimated_total_wall_seconds_including_replay_probe": estimated_total_seconds,
        "note": (
            "rough order-of-magnitude extrapolation from Stage 0 smoke timings; "
            "replay-probe cost is approximated as comparable to the generate call "
            "it replays, not separately measured. Stage 0's own pass/fail criteria "
            "are enforced by the GPU pytest suite, not by this command."
        ),
    }
    write_json("results/decisions/stage0_throughput_extrapolation.json", summary)
    print(
        f"wrote results/decisions/stage0_throughput_extrapolation.json: "
        f"{tokens_per_second:.1f} tok/s observed, Stage 2 estimated "
        f"~{estimated_total_seconds / 3600:.1f}h wall-clock (generate + replay-probe)"
    )
    return 0


def _cmd_analyze_full_rkv_pipeline(output_dir: Path) -> int:
    """The frozen primary analysis (§8.5) — reachable only for stages that
    actually probe BOTH conditions (in practice, stage2_main; see
    _cmd_analyze_stage1b's docstring for why Stage 1B does not route here)."""
    from kvcot.analysis.pipeline import (
        agreement_curve_by_fraction,
        build_pair_results,
        discover_conditions,
        funnel_records,
        load_condition_records,
        paired_accuracy_inputs,
        problem_level_delta_eas,
    )
    from kvcot.analysis.summaries import (
        build_attrition_funnel_table,
        build_primary_analysis_summary,
        write_attrition_funnel_csv,
        write_primary_analysis_json,
    )
    from kvcot.config import PROBE_FRACTIONS_ALL

    full_cond, rkv_cond = discover_conditions(output_dir)
    full = load_condition_records(
        output_dir / f"{full_cond}.jsonl", output_dir / f"{full_cond}_probes.jsonl", full_cond
    )
    rkv = load_condition_records(
        output_dir / f"{rkv_cond}.jsonl", output_dir / f"{rkv_cond}_probes.jsonl", rkv_cond
    )

    # Attrition funnel first — it must always be emitted (§8.4), even if too few
    # problems survive to run the primary tests.
    funnel_rows = build_attrition_funnel_table(funnel_records(full), funnel_records(rkv))
    write_attrition_funnel_csv(funnel_rows, "results/tables/attrition_funnel.csv")
    print("wrote results/tables/attrition_funnel.csv")

    pairs = build_pair_results(full, rkv)
    _aggregates, primary_values = problem_level_delta_eas(pairs)

    # §6 descriptive agreement curve — plotted whenever there is anything to
    # plot, independent of whether the primary tests below have enough
    # problems to run (it doesn't require the >=2-eligible-seeds bar).
    try:
        from kvcot.analysis.plots import plot_agreement_curve

        full_curve = agreement_curve_by_fraction(full, PROBE_FRACTIONS_ALL)
        rkv_curve = agreement_curve_by_fraction(rkv, PROBE_FRACTIONS_ALL)
        plot_agreement_curve(
            list(PROBE_FRACTIONS_ALL),
            [full_curve[f] for f in PROBE_FRACTIONS_ALL],
            [rkv_curve[f] for f in PROBE_FRACTIONS_ALL],
            "results/figures/agreement_curve.png",
        )
        print("wrote results/figures/agreement_curve.png")
    except ImportError:
        print("analyze: matplotlib not installed (`pip install -e '.[plots]'`) — skipping figures")

    if not primary_values:
        print(
            "analyze: no problem reached the >=2-eligible-seeds bar, so the primary "
            "tests have no problem-level Delta_EAS to run on — see the attrition funnel "
            "for where problems were lost. (This is a real outcome, not an error.)"
        )
        return 0

    full_acc, rkv_acc = paired_accuracy_inputs(full, rkv)
    summary = build_primary_analysis_summary(primary_values, full_acc, rkv_acc)
    write_primary_analysis_json(summary, "results/decisions/stage2_primary_analysis.json")
    print(
        f"wrote results/decisions/stage2_primary_analysis.json: "
        f"n_problems_primary={summary.n_problems_primary} "
        f"delta_eas_mean={summary.delta_eas_bootstrap_ci.point_estimate:.4f} "
        f"CI=[{summary.delta_eas_bootstrap_ci.ci_low:.4f}, {summary.delta_eas_bootstrap_ci.ci_high:.4f}] "
        f"wilcoxon_pratt_p={summary.wilcoxon_pratt.p_value:.4g}"
    )

    try:
        from kvcot.analysis.plots import plot_delta_eas_distribution

        plot_delta_eas_distribution(primary_values, "results/figures/delta_eas_distribution.png")
        print("wrote results/figures/delta_eas_distribution.png")
    except ImportError:
        pass  # already reported above

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    stage, lock = load_stage_config(args.config)

    if args.dry_run:
        print(f"analyze plan: stage={stage.stage_name}")
        print(f"  reads: {stage.output_dir}/*.jsonl and {stage.output_dir}/*_probes.jsonl")
        print(f"  writes: results/tables/, results/decisions/, results/figures/")
        return 0

    output_dir = Path(stage.output_dir)

    if stage.stage_name == "stage1a_measurability":
        return _cmd_analyze_stage1a(output_dir)
    if stage.stage_name.startswith("stage1b_budget_"):
        return _cmd_analyze_stage1b(stage, output_dir)
    if stage.stage_name == "stage0_smoke":
        return _cmd_analyze_stage0(stage, output_dir)
    return _cmd_analyze_full_rkv_pipeline(output_dir)


# ------------------------------------------------------------------ analyze-fixed-trace

def cmd_analyze_fixed_trace(args: argparse.Namespace) -> int:
    stage, lock = load_stage_config(args.config)
    if stage.fixed_trace is None:
        raise SystemExit(f"{stage.stage_name}: missing required fixed_trace settings")
    trace_condition = _resolve_condition_name(stage, args.trace_condition)
    replay_condition = _resolve_condition_name(stage, args.replay_condition)

    if args.dry_run:
        print(f"analyze-fixed-trace plan: stage={stage.stage_name}")
        print(f"  trace_condition={trace_condition}  replay_condition={replay_condition}")
        print(
            f"  reads: {stage.output_dir}/{trace_condition}.jsonl, "
            f"{stage.output_dir}/{trace_condition}_on_{trace_condition}_fixed_trace_probes.jsonl, "
            f"{stage.output_dir}/{replay_condition}_on_{trace_condition}_fixed_trace_probes.jsonl"
        )
        # Protocol v3 (2026-07-18 external review): the dry run must report
        # the FULL input set of a v3 analysis -- the natural R-KV file, the
        # selection file, the pre-registered natural record count, and the
        # strict accuracy-gate requirement -- not just the three fixed-trace
        # files a v2 analysis reads.
        if stage.fixed_trace.require_meaningful_compression:
            print(f"  reads natural R-KV base file: {stage.output_dir}/{replay_condition}.jsonl")
            print(
                "  strict accuracy gate: REQUIRED -- expected natural record count "
                f"(both conditions, all-population, never selection-filtered): "
                f"{_expected_stage_record_count(stage, lock)}"
            )
        selection_arg_dry = getattr(args, "selection_file", None)
        if selection_arg_dry:
            print(f"  selection file: {selection_arg_dry}")
        print(f"  writes: results/decisions/{stage.stage_name}_fixed_trace.json")
        return 0

    from kvcot.analysis.fixed_trace import run_fixed_trace_analysis

    # § external review 2026-07-16: verify the three input files were
    # produced under THIS invocation's own config/lock, not just that they
    # agree with each other — `lock` was previously loaded and discarded.
    expected_identity = (
        config_identity(args.config),
        upstream_submodule_commit(lock),
        lock.model.revision,
        lock.model.tokenizer_revision,
    )

    selected_base_record_ids = None
    selection_arg = getattr(args, "selection_file", None)
    if selection_arg:
        base_path = Path(stage.output_dir) / f"{trace_condition}.jsonl"
        selection = _load_fixed_trace_selection(selection_arg, args, stage, lock, base_path)
        selected_base_record_ids = set(selection["selected_base_record_ids"])

    return run_fixed_trace_analysis(
        output_dir=Path(stage.output_dir),
        trace_condition=trace_condition,
        replay_condition=replay_condition,
        stage_name=stage.stage_name,
        settings=stage.fixed_trace,
        expected_identity=expected_identity,
        selected_base_record_ids=selected_base_record_ids,
        # 2026-07-18 external review: the natural accuracy population is the
        # PRE-REGISTERED stage count (manifest x seeds, shared helper with
        # check-fixed-trace-accuracy so the two commands can never derive
        # different expectations) -- never the selected subset's size.
        expected_accuracy_n=_expected_stage_record_count(stage, lock),
        selection_file_path=selection_arg or None,
    )


# ------------------------------------------------------------ check-fixed-trace-accuracy

def cmd_check_fixed_trace_accuracy(args: argparse.Namespace) -> int:
    """CPU-only pilot-accuracy gate (protocol v3, 2026-07-18 review). Reads
    ONLY the natural (non-fixed-trace) `full.jsonl`/`{replay_condition}.jsonl`
    base files — never the fixed-trace probe files. This is what makes the
    documented run order in `docs/GPU_VALIDATION_PLAN.md` actually work:
    `analyze-fixed-trace` requires the fixed-trace probe files to already
    exist (it reads them unconditionally), so running it right after natural
    generation — before any `replay-fixed-trace` has ever run — would crash.
    This command has no such dependency and can run immediately after
    `kvcot generate --condition {replay_condition}`.

    Wraps `kvcot.analysis.fixed_trace.build_strict_accuracy_gate`, which
    requires an EXACT expected record count and an IDENTICAL
    `(source_row_index, global_seed)` key set between the two files —
    `build_accuracy_screen`'s own intersection-only pairing could otherwise
    report `pilot_accuracy_plausible: True` from a single matching pair
    while the other 49 R-KV records were never generated at all.

    No `--limit`/`--problem-index`/`--seed` (removed 2026-07-18, external
    review): those flags fed `expected_n`, so a partial pair of natural
    files could be blessed as "the expected experiment" just by passing a
    matching restriction. The expected count comes only from the stage
    config + lock (`_expected_stage_record_count`, shared with
    `analyze-fixed-trace` so the two commands can never derive different
    expectations) — for the current stage, always 50.
    """
    stage, lock = load_stage_config(args.config)
    if stage.fixed_trace is None:
        raise SystemExit(f"{stage.stage_name}: missing required fixed_trace settings")
    replay_condition = _resolve_condition_name(stage, args.replay_condition)
    full_path = Path(stage.output_dir) / "full.jsonl"
    rkv_path = Path(stage.output_dir) / f"{replay_condition}.jsonl"
    expected_n = _expected_stage_record_count(stage, lock)

    if args.dry_run:
        print(f"check-fixed-trace-accuracy plan: stage={stage.stage_name}  replay_condition={replay_condition}")
        print(f"  reads: {full_path}, {rkv_path}")
        print(f"  expected_n={expected_n} (manifest rows x seeds; partial overrides disabled -- ")
        print("    this command accepts no --limit/--problem-index/--seed by design)")
        print(f"  writes: results/decisions/{stage.stage_name}_accuracy_gate.json")
        return 0

    from kvcot.analysis.fixed_trace import build_strict_accuracy_gate
    from kvcot.utils.io import read_jsonl

    full_records = list(read_jsonl(full_path))
    rkv_records = list(read_jsonl(rkv_path))

    expected_identity = (
        config_identity(args.config),
        upstream_submodule_commit(lock),
        lock.model.revision,
        lock.model.tokenizer_revision,
    )

    gate = build_strict_accuracy_gate(
        full_records, rkv_records, expected_n=expected_n, settings=stage.fixed_trace,
        expected_identity=expected_identity, expected_rkv_condition=replay_condition,
    )
    # 2026-07-18 external review: the gate decision must record which
    # analyzer code produced it and the exact input bytes it judged --
    # data-production and analysis code identities are separate facts.
    lock_path = _resolve_lock_path(args.config, stage)
    gate["analysis_provenance"] = {"git_commit": git_commit(), "git_dirty": git_is_dirty()}
    gate["config_sha256"] = config_identity(args.config)
    gate["lock_sha256"] = sha256_file(lock_path) if lock_path.exists() else None
    gate["input_sha256"] = {
        "full_base": sha256_file(full_path) if full_path.exists() else None,
        "natural_rkv": sha256_file(rkv_path) if rkv_path.exists() else None,
    }
    out_path = Path("results/decisions") / f"{stage.stage_name}_accuracy_gate.json"
    write_json(out_path, gate)
    print(
        f"wrote {out_path}: gate_passed={gate['gate_passed']} n_full={gate['n_full']} "
        f"n_rkv={gate['n_rkv']} expected_n={gate['expected_n']}"
    )
    if not gate["gate_passed"]:
        for reason in gate["reasons"]:
            print(f"  - {reason}")
    return 0 if gate["gate_passed"] else 1


# ------------------------------------------------------------------ inspect-fixed-trace

_THINK_PARSE_OK_STATUSES = ("ok", "generation_prompt_preopened_ok")


def cmd_inspect_fixed_trace(args: argparse.Namespace) -> int:
    """CPU-only trace-length preflight (§ Step 16; strengthened 2026-07-16
    per external review) for the fixed-trace screen — run BEFORE spending
    any GPU time on `replay-fixed-trace`. Reads one condition's already-
    generated base file (normally `--trace-condition full`, the canonical
    trace source) and reports think-span/prompt+think-span length
    statistics against the stage's configured R-KV budget.

    Three independent, purely-arithmetic reasons to stop before any GPU
    spend (none of these can *prove* compression will happen — a
    longer-than-budget trace can still fail to compact if e.g. its think
    span is short relative to divide_length — they only rule out cases
    where the configured screen definitely cannot pass):

    1. No trace is even longer than the budget — R-KV cannot compress a
       sequence that never exceeds its own budget, full stop.
    2. `fraction_of_traces_longer_than_budget` is an UPPER BOUND on the
       achievable `actual_compression_rate` (only a longer-than-budget
       trace can show `actual_compression_at_cut=True`) — if that upper
       bound is already below `FixedTraceSettings.min_actual_compression_
       rate`, the eligibility gate is mathematically unreachable at this
       budget, regardless of how the real GPU run behaves.
    3. `mean_optimistic_retention` is a LOWER BOUND on the achievable mean
       realized retention (`budget / length` per trace if `length >
       budget` else `1.0` — the most aggressive possible compaction; real
       retention sawtooths above this between compaction events, per
       docs/UPSTREAM_AUDIT.md H4). If even this best case already exceeds
       `FixedTraceSettings.max_mean_f1_retention_ratio`, no real run at
       this budget can clear the retention ceiling either.

    Checks 2 and 3 only run when the stage config declares `fixed_trace:`
    settings (they need its thresholds); check 1 always runs.
    """
    import statistics

    from kvcot.utils.io import read_jsonl

    stage, lock = load_stage_config(args.config)
    trace_condition = _resolve_condition_name(stage, args.trace_condition)
    base_path = Path(stage.output_dir) / f"{trace_condition}.jsonl"

    if args.dry_run:
        print(f"inspect-fixed-trace plan: stage={stage.stage_name} trace_condition={trace_condition}")
        print(f"  reads: {base_path}")
        return 0

    records = list(read_jsonl(base_path))
    if not records:
        print(f"inspect-fixed-trace: no records found in {base_path} — run `kvcot generate` first.")
        return 1

    n_total = len(records)
    n_correct = sum(1 for r in records if r.get("is_correct") is True)
    n_cap_hits = sum(1 for r in records if r.get("cap_hit"))
    n_think_parse_failures = sum(
        1 for r in records if r["think_span"]["think_parse_status"] not in _THINK_PARSE_OK_STATUSES
    )

    think_lengths: list[int] = []
    prompt_plus_think_lengths: list[int] = []
    for r in records:
        span = r["think_span"]
        if span["think_parse_status"] not in _THINK_PARSE_OK_STATUSES:
            continue
        start, end = span["think_start_index"], span["think_end_index"]
        if start is None or end is None:
            continue
        length = max(0, end - start)
        think_lengths.append(length)
        prompt_plus_think_lengths.append(len(r["prompt_token_ids"]) + length)

    budget = stage.rkv_budgets[0] if stage.rkv_budgets else None
    n_longer_than_budget = (
        sum(1 for L in prompt_plus_think_lengths if L > budget) if budget is not None else None
    )
    fraction_longer = (
        n_longer_than_budget / len(prompt_plus_think_lengths)
        if (budget is not None and prompt_plus_think_lengths)
        else None
    )
    # Lower bound on achievable mean retention: best case is the cache
    # compacted all the way down to the budget by the end of the think span;
    # a trace that never exceeds the budget cannot be compressed below 1.0
    # no matter what.
    mean_optimistic_retention = (
        sum((budget / L) if L > budget else 1.0 for L in prompt_plus_think_lengths) / len(prompt_plus_think_lengths)
        if (budget is not None and prompt_plus_think_lengths)
        else None
    )

    def _fmt(fn, values):
        return fn(values) if values else None

    print(f"inspect-fixed-trace: {base_path}")
    print(f"  number of records: {n_total}")
    print(f"  number correct: {n_correct}")
    print(f"  cap hits: {n_cap_hits}")
    print(f"  think parse failures: {n_think_parse_failures}")
    print(
        f"  think length: min={_fmt(min, think_lengths)} "
        f"median={_fmt(statistics.median, think_lengths)} max={_fmt(max, think_lengths)}"
    )
    print(
        f"  prompt+think length: min={_fmt(min, prompt_plus_think_lengths)} "
        f"median={_fmt(statistics.median, prompt_plus_think_lengths)} "
        f"max={_fmt(max, prompt_plus_think_lengths)}"
    )
    print(f"  configured R-KV budget: {budget}")
    print(f"  fraction of traces longer than budget: {fraction_longer}")
    print(f"  best-case (optimistic) mean retention achievable at this budget: {mean_optimistic_retention}")

    if budget is not None and prompt_plus_think_lengths and n_longer_than_budget == 0:
        print(
            "inspect-fixed-trace: STOP — no trace in this file is longer than the configured "
            f"budget ({budget}); R-KV compression can never fire during replay at this budget. "
            "Choose a smaller budget or a longer-trace manifest before running replay-fixed-trace."
        )
        return 1

    if stage.fixed_trace is not None and fraction_longer is not None:
        if fraction_longer < stage.fixed_trace.min_actual_compression_rate:
            print(
                "inspect-fixed-trace: STOP — fraction of traces longer than the budget "
                f"({fraction_longer:.3f}) is an upper bound on the achievable actual_compression_rate, "
                f"and it is already below min_actual_compression_rate "
                f"({stage.fixed_trace.min_actual_compression_rate}). The eligibility gate is "
                "mathematically unreachable at this budget on this manifest — choose a smaller "
                "budget or a longer-trace manifest, do not weaken the threshold."
            )
            return 1
        if (
            mean_optimistic_retention is not None
            and mean_optimistic_retention > stage.fixed_trace.max_mean_f1_retention_ratio
        ):
            print(
                "inspect-fixed-trace: STOP — even the best-case (most aggressive) achievable mean "
                f"retention ({mean_optimistic_retention:.3f}) already exceeds "
                f"max_mean_f1_retention_ratio ({stage.fixed_trace.max_mean_f1_retention_ratio}); real "
                "retention only sawtooths higher than this optimistic floor between compaction "
                "events. No real run at this budget can clear the retention ceiling either — choose "
                "a smaller budget or a longer-trace manifest, do not weaken the threshold."
            )
            return 1

    if getattr(args, "write_selection", False):
        if stage.fixed_trace is None:
            raise SystemExit(f"{stage.stage_name}: --write-selection requires stage fixed_trace settings")
        if budget is None:
            raise SystemExit(f"{stage.stage_name}: --write-selection requires stage.rkv_budgets[0]")
        n_selected = _write_fixed_trace_selection(
            stage=stage, lock=lock, records=records, base_path=base_path,
            budget=budget, args=args,
        )
        if n_selected < stage.fixed_trace.min_eligible_examples:
            return 1
    return 0


def _write_fixed_trace_selection(stage, lock, records: list[dict], base_path: Path, budget: int, args) -> int:
    """Protocol v3 (CHANGELOG.md 2026-07-17): deterministic, outcome-blind
    trace selection using ONLY this FullKV base file's own correctness/cap/
    think-parse validity plus `kvcot.analysis.rkv_schedule`'s PREDICTED
    retention — never any fixed-trace probe answer, PSS, or CPSS value (a
    selection built from outcomes would not be a pre-registered screen; it
    would let the data that determines which examples get analyzed also
    influence what the analysis finds).

    Ranking/capping order (fixed 2026-07-18 review — a real bug): ALL valid
    candidates are ranked (sorted by `source_row_index`) and their
    `predicted_eligible` flag computed FIRST, uncapped. Only THEN is
    `predicted_eligible` used to build the eligible subset, which is capped
    at `max_selected` (if given) — capping the full ranked list BEFORE
    filtering to eligible candidates (an earlier version of this function)
    could select zero examples even when plenty of eligible ones existed
    later in `source_row_index` order, whenever the first `max_selected`
    candidates by row index happened to be ineligible. The cap itself is
    still independent of any retention VALUE (only of the boolean
    eligibility outcome and the deterministic row-index sort), so this
    remains a pre-registered, outcome-blind selection.
    """
    from kvcot.analysis.rkv_schedule import meaningfully_compressed_fractions, predict_retention_by_fraction
    from kvcot.probes.early_answering import ThinkSpanResult, absolute_cut_position

    ft = stage.fixed_trace
    max_selected_arg = getattr(args, "max_selected", None)
    effective_max_selected = max_selected_arg if max_selected_arg is not None else ft.max_selected_examples
    if (
        max_selected_arg is not None
        and ft.max_selected_examples is not None
        and max_selected_arg != ft.max_selected_examples
    ):
        # 2026-07-19 review: a CLI --max-selected that silently overrides the
        # stage config's pre-registered fixed_trace.max_selected_examples
        # defeats the entire point of pre-registering it. Not refused
        # outright (a quick manual/debugging override still has legitimate
        # uses), but the resulting file's own "max_selected" will then
        # disagree with the CURRENT config -- _load_fixed_trace_selection
        # refuses to replay/analyze against it until the config is updated
        # to match or a matching selection is regenerated.
        print(
            f"inspect-fixed-trace --write-selection: WARNING -- --max-selected {max_selected_arg} "
            f"overrides this stage's pre-registered fixed_trace.max_selected_examples "
            f"({ft.max_selected_examples}). The written selection file will record max_selected="
            f"{max_selected_arg}, which will NOT match the current config -- replay-fixed-trace/"
            "analyze-fixed-trace will refuse this selection file until the config's "
            "max_selected_examples is updated to match, or a matching selection is regenerated."
        )
    scored_fractions = set(lock.probes.fractions_scored)
    candidates = []
    n_rejected_invalid = 0
    for r in records:
        span_info = r["think_span"]
        if r.get("is_correct") is not True or r.get("cap_hit"):
            n_rejected_invalid += 1
            continue
        if span_info["think_parse_status"] not in _THINK_PARSE_OK_STATUSES:
            n_rejected_invalid += 1
            continue
        span = ThinkSpanResult(
            think_start_index=span_info["think_start_index"],
            think_end_index=span_info["think_end_index"],
            think_parse_status=span_info["think_parse_status"],
            generation_prompt_preopened_think=span_info["generation_prompt_preopened_think"],
        )
        prompt_length = len(r["prompt_token_ids"])
        cut_positions = {f: prompt_length + absolute_cut_position(span, f) for f in lock.probes.fractions_all}
        predicted_retention = predict_retention_by_fraction(
            prompt_length=prompt_length, target_absolute_positions=cut_positions,
            budget=budget, divide_length=lock.rkv.divide_length,
        )
        active = meaningfully_compressed_fractions(predicted_retention, ft.meaningful_retention_ceiling)
        active_scored = active & scored_fractions
        f1_retention = predicted_retention.get(1.0)
        predicted_eligible = (
            f1_retention is not None
            and f1_retention <= ft.meaningful_retention_ceiling
            and len(active_scored) >= ft.min_meaningfully_compressed_scored_fractions
        )
        candidates.append(
            {
                "source_row_index": r["dataset"]["source_row_index"],
                "global_seed": r["global_seed"],
                "base_record_id": r["record_id"],
                "predicted_retention_by_fraction": {str(k): v for k, v in predicted_retention.items()},
                "predicted_active_scored_fraction_count": len(active_scored),
                "predicted_eligible": predicted_eligible,
            }
        )

    # Rank ALL candidates by source_row_index FIRST, uncapped -- this is the
    # complete, deterministic ranking `n_ranked` describes.
    candidates.sort(key=lambda c: c["source_row_index"])

    # THEN filter to predicted-eligible candidates (still in source_row_index
    # order) and cap THAT list -- never the other way around.
    eligible_candidates = [c for c in candidates if c["predicted_eligible"]]
    selected = eligible_candidates[:effective_max_selected] if effective_max_selected is not None else eligible_candidates
    selected_ids = {c["base_record_id"] for c in selected}
    for c in candidates:
        c["selected"] = c["base_record_id"] in selected_ids

    selection = {
        "stage_name": stage.stage_name,
        "config_path": args.config,
        "config_sha256": config_identity(args.config),
        "base_path": str(base_path),
        "base_file_sha256": sha256_file(base_path),
        "budget": budget,
        "divide_length": lock.rkv.divide_length,
        "meaningful_retention_ceiling": ft.meaningful_retention_ceiling,
        "min_meaningfully_compressed_scored_fractions": ft.min_meaningfully_compressed_scored_fractions,
        "max_selected": effective_max_selected,
        "n_candidates_considered": len(records),
        "n_rejected_invalid_base": n_rejected_invalid,
        "n_ranked": len(candidates),
        "n_predicted_eligible": len(eligible_candidates),
        "n_selected": len(selected),
        "selected_source_row_indices": [c["source_row_index"] for c in selected],
        "selected_base_record_ids": [c["base_record_id"] for c in selected],
        "candidates": candidates,
    }
    out_path = Path("results/selections") / f"{stage.stage_name}.json"
    write_json(out_path, selection)
    print(
        f"wrote {out_path}: n_ranked={len(candidates)} n_predicted_eligible={len(eligible_candidates)} "
        f"n_selected={len(selected)} (of {len(records)} base records, {n_rejected_invalid} rejected on "
        "correctness/cap/think-parse)"
    )
    if len(selected) < ft.min_eligible_examples:
        print(
            f"inspect-fixed-trace --write-selection: STOP -- n_selected ({len(selected)}) is below "
            f"min_eligible_examples ({ft.min_eligible_examples}); a full replay under this selection "
            "cannot pass the screen's own validity gate. Consider a smaller budget, a longer-trace "
            "manifest, or generating more natural traces before spending GPU time on replay."
        )
    return len(selected)


# ------------------------------------------------------------------ calibrate-budget

def _stage1b_candidate_budgets(config_dir: Path) -> list[int]:
    import re

    budgets = []
    for p in sorted(config_dir.glob("stage1b_budget_*.yaml")):
        m = re.match(r"stage1b_budget_(\d+)\.yaml$", p.name)
        if m:
            budgets.append(int(m.group(1)))
    return sorted(budgets)


def cmd_calibrate_budget(args: argparse.Namespace) -> int:
    """§10: reads each candidate budget's `results/decisions/stage1b_budget_
    <N>.json` (written by `kvcot analyze --config configs/stage1b_budget_
    <N>.yaml`, once per budget, by `scripts/run_stage1b.sh`'s loop) and
    reports the smallest budget whose `overall_passed` is true — never
    silently the one numerically closest to any target retention fraction.
    Does NOT write `configs/selected_operating_point.yaml` itself: that file
    also records `selected_by`/`selected_on_utc` and a human review of the
    decision basis, which `scripts/run_stage1b.sh` already documents as a
    manual step (§10: "recommends... but does not silently apply"). This
    command's job stops at producing the recommendation that step copies from.
    """
    config_dir = Path(args.config_dir)
    candidates = _stage1b_candidate_budgets(config_dir)

    if args.dry_run:
        print("calibrate-budget plan:")
        if not candidates:
            print(f"  (no configs/stage1b_budget_*.yaml files found under {config_dir})")
        for b in candidates:
            print(f"  {config_dir}/stage1b_budget_{b}.yaml -> results/decisions/stage1b_budget_{b}.json")
        print(
            "  reports the smallest budget passing both gates -> "
            "results/decisions/stage1b_recommendation.json (does not write "
            "configs/selected_operating_point.yaml itself — that stays a manual, "
            "reviewed step per §10)"
        )
        return 0

    if not candidates:
        raise SystemExit(f"no configs/stage1b_budget_*.yaml files found under {config_dir}")

    from kvcot.probes.early_answering import CLAIM_BOUNDARY_NOTICE
    from kvcot.utils.io import read_json

    per_budget: dict[int, dict] = {}
    missing: list[int] = []
    for b in candidates:
        decision_path = Path("results/decisions") / f"stage1b_budget_{b}.json"
        if not decision_path.exists():
            missing.append(b)
            continue
        per_budget[b] = read_json(decision_path)

    if missing:
        print(
            f"calibrate-budget: missing decision file(s) for budget(s) {missing} — "
            f"run `kvcot analyze --config configs/stage1b_budget_<N>.yaml` for each first."
        )

    passing = sorted(b for b, d in per_budget.items() if d.get("overall_passed"))
    recommended = passing[0] if passing else None

    recommendation_text = (
        f"smallest budget passing both gates: {recommended}"
        if recommended is not None
        else (
            "no candidate budget passed both gates — GSM8K provides no accuracy-plausible, "
            "compression-active operating point at these candidates (§10); do not pick one anyway."
        )
    )
    summary = {
        "claim_boundary_notice": CLAIM_BOUNDARY_NOTICE,
        "candidates_checked": candidates,
        "missing_decisions": missing,
        "per_budget_overall_passed": {str(b): d.get("overall_passed") for b, d in per_budget.items()},
        "recommended_budget": recommended,
        "recommendation": recommendation_text,
    }
    write_json("results/decisions/stage1b_recommendation.json", summary)
    print(f"wrote results/decisions/stage1b_recommendation.json: {recommendation_text}")
    if recommended is not None:
        print(
            f"Next: copy configs/selected_operating_point.yaml.example to "
            f"configs/selected_operating_point.yaml and fill it in from "
            f"results/decisions/stage1b_budget_{recommended}.json (§10: never silently apply)."
        )
        return 0
    return 1


# ------------------------------------------------------------------ failure-atlas

def cmd_failure_atlas(args: argparse.Namespace) -> int:
    """Phase A2: deterministic, CPU-only failure atlas over the 50 committed
    FullKV/R-KV protocol-v3 GSM8K pairs (`kvcot.failure_atlas`). Reads only
    the committed `.jsonl.gz` gate artifacts, verifies their `.sha256`
    sidecars first (never trusts unverified bytes), never loads a model or
    tokenizer, and never mutates any historical artifact.
    """
    from kvcot.failure_atlas import build_atlas_summary, build_failure_atlas, write_atlas_csv, write_atlas_markdown
    from kvcot.utils.io import read_jsonl_auto

    full_path = Path(args.full_artifact)
    rkv_path = Path(args.rkv_artifact)

    if args.dry_run:
        print(f"failure-atlas plan: full={full_path} rkv={rkv_path}")
        print(f"  output_csv={args.output_csv}")
        print(f"  output_markdown={args.output_markdown}")
        print(f"  output_summary={args.output_summary}")
        return 0

    def _verify_checksum(artifact_path: Path) -> str:
        sha_path = artifact_path.with_name(artifact_path.name.replace(".jsonl.gz", ".sha256"))
        if not sha_path.exists():
            raise SystemExit(f"missing checksum sidecar for {artifact_path}: expected {sha_path}")
        actual = sha256_file(artifact_path)
        found = False
        for line in sha_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            recorded_hash, recorded_name = parts
            if Path(recorded_name.strip()).name == artifact_path.name:
                found = True
                if recorded_hash.strip() != actual:
                    raise SystemExit(
                        f"checksum mismatch for {artifact_path}: {sha_path} records {recorded_hash.strip()}, "
                        f"actual sha256 is {actual}"
                    )
        if not found:
            raise SystemExit(f"{sha_path} does not contain an entry for {artifact_path.name}")
        return actual

    full_sha256 = _verify_checksum(full_path)
    rkv_sha256 = _verify_checksum(rkv_path)

    full_records = list(read_jsonl_auto(full_path))
    rkv_records = list(read_jsonl_auto(rkv_path))

    analysis_code_commit = git_commit()

    rows = build_failure_atlas(
        full_records, rkv_records, analysis_code_commit=analysis_code_commit,
        full_artifact_path=str(full_path), rkv_artifact_path=str(rkv_path),
    )
    summary = build_atlas_summary(
        rows, full_artifact_path=str(full_path), rkv_artifact_path=str(rkv_path),
        full_artifact_sha256=full_sha256, rkv_artifact_sha256=rkv_sha256,
        analysis_code_commit=analysis_code_commit,
    )

    write_atlas_csv(rows, args.output_csv)
    write_atlas_markdown(rows, summary, args.output_markdown)
    write_json(args.output_summary, summary)

    print(f"failure-atlas: wrote {len(rows)} rows to {args.output_csv}, {args.output_markdown}, {args.output_summary}")
    print(f"  correctness: {summary['correctness']}")
    if summary["warnings"]:
        for w in summary["warnings"]:
            print(f"  warning: {w}")
    return 0


# --------------------------------------------------------------------- validate-run

def _schema_for_record(row: dict):
    """Dispatch on the row's own `record_type` field, not the filename it
    came from. Filename-based dispatch (`path.name.endswith("_probes.jsonl")`)
    silently misclassified fixed-trace probe files once they existed: a
    `{replay_condition}_on_{trace_condition}_fixed_trace_probes.jsonl` file
    still ends in `_probes.jsonl`, so it would have validated against
    `ProbeRunRecord` (missing `trace_source_condition`/`replay_policy_condition`/
    the f=1-anchor fields) instead of `FixedTraceProbeRecord`."""
    from kvcot.schemas import BaseRunRecord, FixedTraceProbeRecord, ProbeRunRecord

    record_type = row.get("record_type")
    mapping = {
        "base_generation": BaseRunRecord,
        "probe": ProbeRunRecord,
        "fixed_trace_probe": FixedTraceProbeRecord,
    }
    if record_type not in mapping:
        raise ValueError(f"unknown record_type: {record_type!r}")
    return mapping[record_type]


def cmd_validate_run(args: argparse.Namespace) -> int:
    from kvcot.utils.io import read_jsonl

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"{run_dir} does not exist")

    n_valid = 0
    n_invalid = 0
    for path in sorted(run_dir.glob("*.jsonl")):
        for row in read_jsonl(path):
            try:
                model_cls = _schema_for_record(row)
                model_cls.model_validate(row)
                n_valid += 1
            except Exception as e:
                n_invalid += 1
                logger.error("invalid record in %s: %s", path, e)

    print(f"validate-run: {run_dir}: {n_valid} valid, {n_invalid} invalid")
    return 0 if n_invalid == 0 else 1


# ------------------------------------------------------------------- plan-discovery
#
# B1B CPU harness dry-run planner (docs/B1A_REPAIR_AND_B1B_CPU_INTEGRATION.md
# §10, authorized by CLAUDE.md §1b/§4b). ALWAYS behaves as a planning-only
# command -- there is no non-dry-run mode, on purpose: B2A/B2B/GPU/Vast.ai
# execution is not authorized by this pass. Every import in this function
# body is CPU-only, pure-Python (kvcot.discovery.discovery_config,
# kvcot.discovery.constants -- deliberately NOT kvcot.discovery.branch_eval
# or kvcot.discovery.pass1, which import torch at module scope); none of
# them import torch, transformers, datasets, or huggingface_hub, so this
# command cannot load model weights, download a dataset, or initialize CUDA
# even if asked to -- matching every other `--dry-run` command in this CLI.


def cmd_plan_discovery(args: argparse.Namespace) -> int:
    from kvcot.discovery.constants import (
        B2B_PILOT_EXAMPLE_COUNT,
        B2B_PILOT_TOTAL_REAL_BRANCHES,
        CANDIDATES_PER_EVENT,
        DONORS_PER_EVENT,
        EVENTS_SELECTED_PER_EXAMPLE,
        MINIMUM_FUTURE_TOKENS_AFTER_EVENT,
        PAIR_BRANCHES_PER_EVENT,
        SCORED_HORIZON,
    )
    from kvcot.discovery.discovery_config import load_discovery_config

    config = load_discovery_config(args.config)

    print("plan-discovery plan:")
    print(f"  model: {config.model.name}@{config.model.revision}")
    print(f"  tokenizer: {config.model.tokenizer_name}@{config.model.tokenizer_revision}")
    print(f"  dataset: {config.dataset.name} (revision_frozen={config.dataset.revision_is_frozen})")
    print(f"  rkv: budget={config.rkv.budget} upstream_revision={config.rkv.upstream_revision}")
    print("  bridge_tokens=1")
    print(f"  scored_horizon={SCORED_HORIZON}")
    print(f"  minimum_future_tokens_after_event={MINIMUM_FUTURE_TOKENS_AFTER_EVENT}")
    print(
        f"  events={EVENTS_SELECTED_PER_EXAMPLE} candidates={CANDIDATES_PER_EVENT} donors={DONORS_PER_EVENT} "
        f"pair_branches_per_event={PAIR_BRANCHES_PER_EVENT}"
    )
    print(
        f"  B2B pilot cost model (planning information only, NOT authorized by this command): "
        f"{B2B_PILOT_EXAMPLE_COUNT} examples x {EVENTS_SELECTED_PER_EXAMPLE} events x "
        f"{PAIR_BRANCHES_PER_EVENT} real swaps = {B2B_PILOT_TOTAL_REAL_BRANCHES} real branches "
        "(the mandatory no-op control is excluded from this total -- it is a separate, CPU-mandatory "
        "parity mechanism, never one of the 144 real branches)."
    )
    print(
        "  B2A (one-example GPU calibration), B2B (the bounded discovery pilot), and any Vast.ai "
        "(or other GPU host) activity of any kind are BLOCKED -- not authorized by this command, by "
        "configs/discovery/*.yaml, or by CLAUDE.md §1a/§1b."
    )
    if not config.dataset.revision_is_frozen:
        print(
            "  BLOCKED: dataset.revision is not frozen -- actual B2A execution remains blocked until "
            "the one-example manifest identity and dataset revision are independently frozen from an "
            "authoritative source (never guessed)."
        )
    print("  no result files created")
    return 0


# ----------------------------------------------------------------- prepare-b2a-manifest
#
# CPU-only real resolution of the B2A one-example manifest's prompt
# identity (B1B-R3 §5/§20). `--dry-run` prints the exact plan (dataset
# repo/revision/index, tokenizer name/revision, existing manifest state)
# without downloading or writing anything. `--execute` fetches the pinned
# MATH-500 row and loads the pinned tokenizer ONLY (never model weights),
# then atomically writes the resolved manifest -- refuses to overwrite an
# already-populated manifest without `--force`.


def cmd_prepare_b2a_manifest(args: argparse.Namespace) -> int:
    from kvcot.discovery.discovery_config import load_discovery_config
    from kvcot.discovery.manifest_prepare import (
        ManifestPreparationError,
        ManifestPreparationRefused,
        build_plan,
        prepare_manifest,
    )

    config = load_discovery_config(args.config)
    plan = build_plan(config, force=args.force)

    if args.dry_run:
        print("prepare-b2a-manifest dry-run plan:")
        print(f"  dataset: {plan.dataset_repo}@{plan.dataset_revision} config={plan.dataset_config} split={plan.dataset_split}")
        print(f"  example_index: {plan.example_index}")
        print(f"  tokenizer: {plan.tokenizer_name}@{plan.tokenizer_revision}")
        print(f"  existing_manifest_path: {plan.existing_manifest_path}")
        print(f"  existing_manifest_is_prompt_resolved: {plan.existing_manifest_is_prompt_resolved}")
        print(f"  force: {plan.force}")
        print("  no download, no write performed by --dry-run")
        return 0

    if not args.execute:
        raise SystemExit("prepare-b2a-manifest requires exactly one of --dry-run or --execute.")

    try:
        manifest = prepare_manifest(config, force=args.force)
    except ManifestPreparationRefused as e:
        raise SystemExit(str(e)) from e
    except ManifestPreparationError as e:
        raise SystemExit(str(e)) from e

    print(f"prepare-b2a-manifest: wrote {plan.existing_manifest_path}")
    print(f"  unique_id={manifest.unique_id}")
    print(f"  raw_content_hash={manifest.raw_content_hash}")
    print(f"  prompt_token_ids_sha256={manifest.prompt_token_ids_sha256}")
    print(f"  prompt_token_count={manifest.prompt_token_count}")
    print(f"  manifest_hash={manifest.manifest_hash()}")
    return 0


# ------------------------------------------------------------- qualify-b2a-row
#
# B2A-R2 FullKV-only row qualification (2026-07-22). `--dry-run` performs no
# CUDA access, model load, or inference -- structural/identity validation of
# the committed candidate manifest against the config only. `--execute`
# loads FullKV ONLY (never R-KV) and attempts candidates, in their committed
# deterministic order, stopping at the first one satisfying every frozen
# qualification condition. Writes an immutable qualification artifact.


def cmd_qualify_b2a_row(args: argparse.Namespace) -> int:
    import json

    from kvcot.discovery.b2a_qualification import (
        run_qualification_dry_run,
        run_qualification_execute,
    )
    from kvcot.discovery.discovery_config import canonical_config_hash, load_discovery_config
    from kvcot.discovery.manifest_prepare import ManifestPreparationError

    config = load_discovery_config(args.config)
    with open(args.candidates, "r", encoding="utf-8") as f:
        candidate_manifest = json.load(f)

    if args.dry_run:
        try:
            plan = run_qualification_dry_run(config, candidate_manifest)
        except ManifestPreparationError as e:
            raise SystemExit(f"qualify-b2a-row --dry-run refused: {e}") from e
        print("qualify-b2a-row dry-run plan:")
        print(f"  candidate_manifest_hash: {plan['candidate_manifest_hash']}")
        print(f"  candidates_to_attempt_in_order: {plan['candidates_to_attempt_in_order']}")
        print(f"  qualification_conditions: {plan['qualification_conditions']}")
        print("  no CUDA access, no model load, no inference performed by --dry-run")
        return 0

    if not args.execute:
        raise SystemExit("qualify-b2a-row requires exactly one of --dry-run or --execute.")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("qualify-b2a-row --execute requires CUDA; none is available on this machine.")

    config_hash = canonical_config_hash(config)
    try:
        artifact = run_qualification_execute(
            config, candidate_manifest, args.candidates, config_hash=config_hash,
        )
    except ManifestPreparationError as e:
        raise SystemExit(f"qualify-b2a-row --execute refused: {e}") from e

    from kvcot.discovery.attempt_artifacts import atomic_write_json
    from pathlib import Path

    output_path = Path(args.output) if args.output else Path("results/decisions/b2a_r2_qualification.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, artifact.to_json())

    print(f"qualify-b2a-row: attempted {len(artifact.attempted)} candidate(s)")
    for outcome in artifact.attempted:
        print(
            f"  ordinal={outcome.candidate_ordinal} unique_id={outcome.unique_id} "
            f"qualified={outcome.qualified} failed_conditions={outcome.failed_conditions}"
        )
    if artifact.selected_ordinal is None:
        print("qualify-b2a-row: NO QUALIFIED ROW")
        print(f"  artifact written: {output_path}")
        return 1
    print(f"qualify-b2a-row: SELECTED ordinal={artifact.selected_ordinal} unique_id={artifact.selected_unique_id}")
    print(f"  artifact written: {output_path}")
    return 0


# --------------------------------------------------------------------- b2a-calibrate
#
# One-example-only B2A calibration command (B1B-R2 §11). "B2A is a one-
# example engineering calibration. It does not authorize the 12-example
# pilot." `--dry-run` never touches CUDA, never downloads a model, never
# runs inference. `--execute` is a separate, explicit flag; it requires
# CUDA and refuses to proceed on ANY unresolved/inconsistent identity
# field -- in this build that is unconditional, since the frozen
# manifest's prompt-token identity is not yet resolved.


def cmd_b2a_calibrate(args: argparse.Namespace) -> int:
    from kvcot.discovery.constants import (
        B2B_PILOT_EXAMPLE_COUNT,
        B2B_PILOT_TOTAL_REAL_BRANCHES,
        BRIDGE_TOKEN_COUNT,
        EVENTS_SELECTED_PER_EXAMPLE,
        PAIR_BRANCHES_PER_EVENT,
        SCORED_HORIZON,
    )
    from kvcot.discovery.discovery_config import (
        canonical_config_hash,
        generation_config_hash,
        load_discovery_config,
        prompt_template_hash,
        rkv_config_hash,
    )
    from kvcot.discovery.manifest import load_b2a_one_example_manifest

    attempt = None
    if args.execute:
        import os
        import sys
        from datetime import datetime, timezone

        from kvcot.discovery.attempt_artifacts import atomic_write_json, create_attempt_directory
        from kvcot.discovery.manifest import DEFAULT_MANIFEST_PATH as _DEFAULT_MANIFEST_PATH

        attempt = create_attempt_directory()
        atomic_write_json(
            attempt.path / "invocation.json",
            {
                "attempt_id": attempt.attempt_id,
                # F6: a real UTC start timestamp -- completion.json's
                # finished_at is validated against this.
                "started_at": datetime.now(timezone.utc).isoformat(),
                "argv": [arg for arg in sys.argv if "token" not in arg.lower() and "secret" not in arg.lower()],
                "working_directory": str(Path.cwd()),
                "pid": os.getpid(),
                "config_path": str(args.config),
                "manifest_path": str(_DEFAULT_MANIFEST_PATH),
                "environment": {
                    "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                    "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
                },
            },
        )

    try:
        config = load_discovery_config(args.config)
        manifest = load_b2a_one_example_manifest()
    except Exception as exc:
        if attempt is not None:
            from kvcot.discovery.attempt_artifacts import atomic_write_json

            atomic_write_json(
                attempt.path / "failure.json",
                {"stage": "config_or_manifest_validation", "error_type": type(exc).__name__, "error": str(exc)},
            )
        raise

    if attempt is not None:
        from kvcot.discovery.attempt_artifacts import collect_execution_provenance

        provenance = collect_execution_provenance(
            repository=Path.cwd(), expected_rkv_sha=config.rkv.upstream_revision, artifact_root=attempt.path,
        )
        atomic_write_json(attempt.path / "provenance.json", provenance)

    blockers: list[str] = []
    if not config.dataset.revision_is_frozen:
        blockers.append("config dataset.revision is not frozen")
    if config.dataset.revision != manifest.dataset_revision:
        blockers.append(
            f"config dataset.revision ({config.dataset.revision!r}) does not match manifest.dataset_revision "
            f"({manifest.dataset_revision!r})"
        )
    if not manifest.prompt_identity_is_resolved:
        blockers.append(
            "manifest prompt-token identity is unresolved (prompt_token_ids_sha256 / "
            "tokenizer_revision_used_for_prompt_hash require a live tokenizer)"
        )

    if args.dry_run:
        print("b2a-calibrate dry-run plan:")
        print("  B2A is a one-example engineering calibration. It does not authorize the 12-example pilot.")
        print(f"  model: {config.model.name}@{config.model.revision}")
        print(f"  tokenizer: {config.model.tokenizer_name}@{config.model.tokenizer_revision}")
        print(
            f"  dataset: {manifest.dataset_repo}@{manifest.dataset_revision} "
            f"config={manifest.dataset_config} split={manifest.dataset_split}"
        )
        print(f"  selected example: index={manifest.example_index} unique_id={manifest.unique_id}")
        print("  one_example_only=True (structural: exactly one manifest entry; no range/list is accepted)")
        print(
            f"  generation: mode={config.generation.generation_mode} do_sample={config.generation.do_sample} "
            f"batch_size={config.generation.batch_size} max_new_tokens={config.generation.max_new_tokens} "
            f"framework_seed={config.generation.framework_seed} "
            f"attention_backend={config.generation.attention_backend} "
            f"no_offload_required={config.generation.no_offload_required}"
        )
        print(
            f"  rkv: budget={config.rkv.budget} window_size={config.rkv.window_size} "
            f"mix_lambda={config.rkv.mix_lambda} divide_length={config.rkv.divide_length}"
        )
        print(
            f"  call plan: exactly 1 prefill call (complete prompt) + 1 decode_one call per continuation token "
            f"(never repeated one-token prefill calls); target-only capture at exactly "
            f"{EVENTS_SELECTED_PER_EXAMPLE} preselected (position, layer) pairs; {BRIDGE_TOKEN_COUNT} unscored "
            f"bridge token then {SCORED_HORIZON} scored teacher-forced tokens per branch"
        )
        print(
            f"  B2B pilot cost model (planning information only, NOT authorized here): "
            f"{B2B_PILOT_EXAMPLE_COUNT} examples x {EVENTS_SELECTED_PER_EXAMPLE} events x "
            f"{PAIR_BRANCHES_PER_EVENT} real pair evaluations = {B2B_PILOT_TOTAL_REAL_BRANCHES} real pair "
            "evaluations; 0 GPU no-op pair evaluations"
        )
        print(
            f"  B2A pair-evaluation accounting (this example only): {EVENTS_SELECTED_PER_EXAMPLE} selected events "
            f"x {PAIR_BRANCHES_PER_EVENT} real pair evaluations = 12 real pair evaluations, PLUS exactly 1 no-op "
            "pair evaluation -- reported separately, never added to the 144 B2B total"
        )
        print("  strict hardware boundary: exactly one visible NVIDIA RTX 3090 at explicit cuda:0")
        print("  immutable snapshots: exact 40-SHA local model/tokenizer paths; local_files_only=True; no offload")
        print("  compact capture: full Pass-2 records convert to selected-only CompactBranchTarget objects")
        print("  capture-release boundary: full captured K/V, returned K/V, and score tensors released before pair 1")
        print("  branch-memory guard: synchronized, shape/dtype-derived estimate before every real pair and no-op")
        print("  pair topology: 3 unique selected events; 4 unique real pairs/event; 12 total; 1 deterministic no-op")
        print("  branch scoring: one bridge token plus exactly 48 scored targets; NLL scalars retained, logits/cache released")
        print("  timing: synchronized startup/load/pass1/pass2/compact/per-pair phases; load-inclusive projection")
        print("  memory: before-load, model-load, post-load, inference/pass/capture/compact/pair/no-op/worker phases")
        print("  artifacts: immutable b2a_attempt_<UTC>_<UUID>/ with atomic results/envelopes/logs/journals/hashes")
        print("  mandatory gates: identity, expected generation, actual batch, complete traces, raw pair/no-op,")
        print("    semantic swap, timings, memory phases, runtime/VRAM, envelopes, and attempt artifacts all fail closed")
        from kvcot.discovery.final_contract import FINAL_MANDATORY_GATE_CONDITIONS

        for gate_name in FINAL_MANDATORY_GATE_CONDITIONS:
            print(f"    - {gate_name}")
        print(f"  canonical_config_hash={canonical_config_hash(config)}")
        print(f"  generation_config_hash={generation_config_hash(config.generation)}")
        print(f"  rkv_config_hash={rkv_config_hash(config.rkv)}")
        print(f"  prompt_template_hash={prompt_template_hash()}")
        print(f"  manifest_hash={manifest.manifest_hash()}")
        if blockers:
            print("  BLOCKED:")
            for blocker in blockers:
                print(f"    - {blocker}")
        else:
            print("  no unresolved/inconsistent identity fields detected")
        print("  dry-run only: no CUDA/model execution performed; no result files created; no model loaded; no CUDA required")
        return 0 if not blockers else 2

    # ---- explicit GPU execution mode ----
    if not args.execute:
        raise SystemExit(
            "b2a-calibrate requires exactly one of --dry-run or --execute; refusing to silently default "
            "to running GPU inference."
        )
    if args.problem_index is not None or args.limit is not None:
        raise SystemExit(
            "b2a-calibrate --execute accepts no --problem-index/--limit override -- exactly the one frozen "
            "manifest example runs, never a range, multiple ids, or unrestricted dataset iteration."
        )
    if blockers:
        if attempt is not None:
            from kvcot.discovery.attempt_artifacts import atomic_write_json
            atomic_write_json(attempt.path / "failure.json", {"stage": "preflight", "blockers": blockers})
        raise SystemExit("b2a-calibrate --execute refused:\n" + "\n".join(f"  - {b}" for b in blockers))

    import torch

    if not torch.cuda.is_available():
        if attempt is not None:
            from kvcot.discovery.attempt_artifacts import atomic_write_json
            atomic_write_json(
                attempt.path / "failure.json", {"stage": "cuda_preflight", "reason": "CUDA unavailable"}
            )
        raise SystemExit("b2a-calibrate --execute requires CUDA; none is available on this machine.")

    # Independent-audit Gate H4.3 repair: the CLI used to write a trivial
    # `preflight.json` (`{"passed": True, ...}` -- a literal, never a real
    # device observation) BEFORE even checking CUDA availability, and never
    # verified the single-RTX-3090 policy at all before launching workers.
    # `verify_single_rtx3090` is called here -- the SAME raw-evidence
    # producer the workers themselves use (`kvcot.discovery.strict_device`)
    # -- and its result is both written into `preflight.json` AND passed
    # into the coordinator so the final gate can cross-check THREE
    # independent observations (CLI, FullKV worker, R-KV worker), never
    # just two.
    from kvcot.discovery.strict_device import StrictDeviceError, verify_single_rtx3090

    try:
        device_preflight = verify_single_rtx3090(torch.cuda, torch_module=torch)
    except StrictDeviceError as exc:
        if attempt is not None:
            from kvcot.discovery.attempt_artifacts import atomic_write_json
            atomic_write_json(
                attempt.path / "failure.json", {"stage": "device_preflight", "reason": str(exc)}
            )
        raise SystemExit(f"b2a-calibrate --execute refused: device preflight failed: {exc}")

    if attempt is not None:
        from kvcot.discovery.attempt_artifacts import atomic_write_json

        atomic_write_json(
            attempt.path / "preflight.json",
            {
                "passed": True, "blockers": [], "config_hash": canonical_config_hash(config),
                "manifest_hash": manifest.manifest_hash(), "device": dict(device_preflight.__dict__),
            },
        )

    from kvcot.discovery.b2a_execute import run_b2a_calibration
    from kvcot.discovery.manifest import DEFAULT_MANIFEST_PATH

    # Independent-audit Gate H7.2 repair: `invocation.json` is immutable
    # (like every attempt artifact) and was never rewritten with an end
    # timestamp -- a SEPARATE `completion.json` is now written exactly
    # once, in a `finally` block, so it is guaranteed to exist whether this
    # command reaches a gate result, a refusal, or an uncaught exception.
    completion = {"outcome": "exception", "exit_code": None, "artifact_path": None, "gate_passed": None}
    try:
        # The coordinator never loads a model itself (B1B-R3 §11) -- it
        # launches the FullKV and R-KV workers as separate subprocesses,
        # each re-loading config/manifest from these same paths
        # independently.
        artifact = run_b2a_calibration(
            config, manifest, config_path=str(args.config), manifest_path=str(DEFAULT_MANIFEST_PATH),
            attempt_directory=attempt.path,
            # Same `{"verified": True, **device.__dict__}` convention the
            # workers themselves use (`kvcot.discovery.b2a_workers`) -- so
            # the coordinator's raw-evidence gate treats all three
            # observations (CLI, FullKV, R-KV) uniformly.
            cli_device_preflight={"verified": True, **device_preflight.__dict__},
        )
        # B2A-R2 forensic repair (audit round 3,
        # docs/B2A_R2_FORENSIC_PAIR_RECORD_PERSISTENCE_2026-07-22.md §11):
        # `artifact.overall_passed` is the ONE authoritative outcome the
        # coordinator itself already computed (legacy gate AND final gate
        # AND pair-artifact verification) -- this CLI must never recompute
        # success from a subset of gates, or it can print/return a
        # different verdict than the coordinator's own `completion.json`.
        overall_passed = artifact.overall_passed
        completion["artifact_path"] = str(artifact.artifact_path)
        completion["gate_passed"] = overall_passed
        print(f"b2a-calibrate result: passed={overall_passed}")
        print(f"  artifact written: {artifact.artifact_path}")
        if not overall_passed:
            print(f"  failed_conditions={artifact.gate_result.failed_conditions}")
            if artifact.final_gate_result is None:
                print("  final_failed_conditions=['final_gate_result_missing']")
            elif not artifact.final_gate_result.passed:
                print(f"  final_failed_conditions={artifact.final_gate_result.failed_conditions}")
            if not artifact.scientific_pair_artifacts_verified:
                print(f"  pair_artifact_verification_reasons={list(artifact.pair_record_verification_reasons)}")
            completion["outcome"] = "gate_failed"
            completion["exit_code"] = 2
            return 2

        print("  stop: B2B pilot NOT started by this command.")
        completion["outcome"] = "gate_passed"
        completion["exit_code"] = 0
        return 0
    finally:
        # F5: on the successful path the coordinator itself already wrote
        # `completion.json` BEFORE `final.json` (so the final reference
        # manifest includes it); this fallback covers only paths where the
        # coordinator never got that far, and NEVER overwrites it.
        if attempt is not None and not (attempt.path / "completion.json").exists():
            import datetime as datetime_module

            from kvcot.discovery.attempt_artifacts import atomic_write_json

            atomic_write_json(
                attempt.path / "completion.json",
                {
                    "attempt_id": attempt.attempt_id,
                    "finished_at": datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(),
                    **completion,
                },
            )


# --------------------------------------------------------------- B2A-R3 (Step 3 Stage-A)
#
# CPU-only planning/verification commands for the B2A-R3 runtime-qualified
# calibration protocol (docs/B2A_R3_RUNTIME_QUALIFIED_PROTOCOL_2026-07-22.md,
# CLAUDE.md §1h). Every command below is either a deterministic CPU-only
# builder/verifier or a --dry-run planning command -- none of them ever
# initializes CUDA, loads a model, loads a tokenizer for execution, or
# imports R-KV. Stage B (FullKV qualification), Stage C (B2A-R3 execution),
# and authorization-claim creation each require a separate, future, dated
# authorization and are deliberately NOT exposed as CLI commands here.


def cmd_prepare_b2a_r3_candidates(args: argparse.Namespace) -> int:
    from kvcot.config import config_identity
    from kvcot.discovery import b2a_r3_contract as c
    from kvcot.discovery.b2a_r2_candidates import fetch_all_pinned_dataset_rows
    from kvcot.discovery.b2a_r3_candidates import (
        atomic_write_json,
        fetch_and_build_candidate_manifest,
        verify_candidate_manifest_against_dataset,
    )

    output_path = args.output or c.CANDIDATE_MANIFEST_PATH

    if args.dry_run:
        print("prepare-b2a-r3-candidates dry-run plan:")
        print(f"  dataset: {c.DATASET_REPO}@{c.DATASET_REVISION}")
        print(f"  model: {c.MODEL_NAME}@{c.MODEL_REVISION}")
        print(f"  budget: {c.BUDGET}")
        print(f"  exclusion_set_sha256: {c.EXCLUSION_SET_SHA256}")
        print(
            f"  candidate_count: {c.CANDIDATE_TOTAL_COUNT} "
            f"(level_4={c.CANDIDATES_PER_LEVEL}, level_5={c.CANDIDATES_PER_LEVEL})"
        )
        print(f"  output_path: {output_path}")
        print("  no network fetch, no write performed by --dry-run")
        return 0

    if not args.execute:
        raise SystemExit("prepare-b2a-r3-candidates requires exactly one of --dry-run or --execute.")

    config_sha256 = config_identity(c.CONFIG_PATH)
    manifest = fetch_and_build_candidate_manifest(
        dataset_repo=c.DATASET_REPO, dataset_config=c.DATASET_CONFIG, dataset_split=c.DATASET_SPLIT,
        dataset_revision=c.DATASET_REVISION, model_name=c.MODEL_NAME, model_revision=c.MODEL_REVISION,
        tokenizer_name=c.TOKENIZER_NAME, tokenizer_revision=c.TOKENIZER_REVISION, budget=c.BUDGET,
        config_path=c.CONFIG_PATH, config_sha256=config_sha256,
    )
    rows = fetch_all_pinned_dataset_rows(c.DATASET_REPO, c.DATASET_REVISION)
    verify_candidate_manifest_against_dataset(manifest, rows)
    atomic_write_json(output_path, manifest)

    print(f"prepare-b2a-r3-candidates: wrote {output_path}")
    print(f"  canonical_sha256={manifest['canonical_sha256']}")
    print(f"  candidate_count={manifest['candidate_count']} level_mixture={manifest['level_mixture']}")
    return 0


def cmd_verify_b2a_r3_candidates(args: argparse.Namespace) -> int:
    import json

    from kvcot.config import config_identity
    from kvcot.discovery import b2a_r3_contract as c
    from kvcot.discovery.b2a_r3_candidates import (
        verify_candidate_manifest_against_dataset,
        verify_candidate_manifest_structure,
    )

    manifest_path = args.manifest or c.CANDIDATE_MANIFEST_PATH
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    try:
        typed = verify_candidate_manifest_structure(
            manifest, expected_config_sha256=config_identity(c.CONFIG_PATH)
        )
    except Exception as e:  # noqa: BLE001 -- report the reason, exit nonzero
        print(f"verify-b2a-r3-candidates: STRUCTURAL VERIFICATION FAILED: {e}")
        return 1

    print(f"verify-b2a-r3-candidates: structural verification PASSED for {manifest_path}")
    print(f"  candidate_count={typed.candidate_count} level_mixture={typed.level_mixture}")

    if args.verify_existing:
        from kvcot.discovery.b2a_r2_candidates import fetch_all_pinned_dataset_rows

        rows = fetch_all_pinned_dataset_rows(c.DATASET_REPO, c.DATASET_REVISION)
        try:
            verify_candidate_manifest_against_dataset(manifest, rows)
        except Exception as e:  # noqa: BLE001
            print(f"verify-b2a-r3-candidates: --verify-existing FAILED: {e}")
            return 1
        print("verify-b2a-r3-candidates: --verify-existing reproduced the manifest from the pinned dataset")
    return 0


def cmd_plan_b2a_r3_qualification(args: argparse.Namespace) -> int:
    import json

    from kvcot.config import config_identity
    from kvcot.discovery import b2a_r3_contract as c
    from kvcot.discovery.b2a_r3_candidates import verify_candidate_manifest_structure

    if not args.dry_run:
        raise SystemExit("plan-b2a-r3-qualification only supports --dry-run -- Stage B remains unauthorized.")

    candidates_path = args.candidates or c.CANDIDATE_MANIFEST_PATH
    with open(candidates_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    typed = verify_candidate_manifest_structure(
        manifest, expected_config_sha256=config_identity(c.CONFIG_PATH)
    )

    print("plan-b2a-r3-qualification dry-run plan:")
    print(f"  candidate_manifest_canonical_sha256: {typed.canonical_sha256}")
    print(
        "  candidates_to_attempt_in_order: "
        f"{[cand.unique_id for cand in typed.candidates[:c.QUALIFICATION_CANDIDATE_LIMIT]]}"
    )
    print(f"  qualification_candidate_limit = {c.QUALIFICATION_CANDIDATE_LIMIT}")
    print(f"  per_candidate_worker_timeout_seconds = {c.PER_CANDIDATE_WORKER_TIMEOUT_SECONDS}")
    print(f"  absolute_timeout_envelope_seconds = {c.ABSOLUTE_TIMEOUT_ENVELOPE_SECONDS}")
    print("  qualification_phase_wall_time_limit = null")
    print("  gpu_qualification_authorized = false")
    print("  wall_time_authorization_required = true")
    print("  authorization_claim_created = false")
    print("  authorization_consumed = false")
    print("  would_initialize_cuda = false")
    print("  would_load_model = false")
    print("  would_load_tokenizer_for_execution = false")
    print("  would_import_rkv = false")
    return 0


def cmd_run_b2a_r3_stage_b_qualification(args: argparse.Namespace) -> int:
    import json

    from kvcot.discovery.b2a_r3_stage_b import run_b2a_r3_stage_b_qualification

    with open(args.claim, "r", encoding="utf-8") as f:
        claim_payload = json.load(f)
    try:
        result = run_b2a_r3_stage_b_qualification(claim_payload=claim_payload)
    except Exception as e:  # noqa: BLE001
        print(f"run-b2a-r3-stage-b-qualification: EXECUTION FAILED: {e}")
        return 1
    print("run-b2a-r3-stage-b-qualification: execution PASSED")
    print(f"  authorization_claim_path={result.authorization_claim_path}")
    print(f"  qualification_artifact_path={result.qualification_artifact_path}")
    print(f"  selection_status={result.artifact['selection_status']}")
    print(f"  selected_unique_id={result.artifact['selected_unique_id']}")
    return 0


def cmd_verify_b2a_r3_qualification(args: argparse.Namespace) -> int:
    import json

    from kvcot.config import config_identity
    from kvcot.discovery import b2a_r3_contract as c
    from kvcot.discovery.b2a_r3_artifacts import verify_qualification_artifact
    from kvcot.discovery.b2a_r3_authorization import verify_persisted_stage_b_authorization_binding
    from kvcot.discovery.b2a_r3_provenance import SubprocessGitStateProvider

    artifact_path = args.artifact or c.QUALIFICATION_ARTIFACT_PATH
    candidates_path = args.candidates or c.CANDIDATE_MANIFEST_PATH
    config_path = args.config or c.CONFIG_PATH
    if config_path != c.CONFIG_PATH:
        print(f"verify-b2a-r3-qualification: VERIFICATION FAILED: config path must be {c.CONFIG_PATH}")
        return 1

    with open(artifact_path, "r", encoding="utf-8") as f:
        artifact = json.load(f)
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidate_manifest = json.load(f)
    expected_config_sha256 = config_identity(config_path)

    try:
        stage_b_binding = verify_persisted_stage_b_authorization_binding(
            authorization_id=artifact["stage_b_authorization_id"],
            repository_root=".",
            git_state=SubprocessGitStateProvider("."),
            candidate_manifest=candidate_manifest,
            expected_config_sha256=expected_config_sha256,
        )
        typed = verify_qualification_artifact(
            artifact, candidate_manifest=candidate_manifest, expected_config_sha256=expected_config_sha256,
            stage_b_authorization_context=stage_b_binding,
        )
    except Exception as e:  # noqa: BLE001
        print(f"verify-b2a-r3-qualification: VERIFICATION FAILED: {e}")
        return 1

    print(f"verify-b2a-r3-qualification: verification PASSED for {artifact_path}")
    print(f"  selection_status={typed.selection_status} selected_unique_id={typed.selected_unique_id}")
    return 0


def cmd_freeze_b2a_r3_selected_row(args: argparse.Namespace) -> int:
    import json

    from kvcot.config import config_identity
    from kvcot.discovery import b2a_r3_contract as c
    from kvcot.discovery.b2a_r3_authorization import verify_persisted_stage_b_authorization_binding
    from kvcot.discovery.b2a_r3_freeze import plan_freeze_dry_run
    from kvcot.discovery.b2a_r3_provenance import SubprocessGitStateProvider

    if args.dry_run and args.execute:
        raise SystemExit("freeze-b2a-r3-selected-row: pass exactly one of --dry-run or --execute, not both.")
    if not args.dry_run and not args.execute:
        raise SystemExit("freeze-b2a-r3-selected-row: pass exactly one of --dry-run or --execute.")

    if args.execute:
        # Execute mode uses fixed production paths only -- no caller-selected
        # candidate/artifact/config path is ever accepted, even if it would
        # happen to equal the fixed path.
        if args.candidates or args.artifact or args.config:
            raise SystemExit(
                "freeze-b2a-r3-selected-row --execute does not accept --candidates/--artifact/--config -- "
                "it always uses the fixed production paths."
            )
        return _cmd_freeze_b2a_r3_selected_row_execute()

    candidates_path = args.candidates or c.CANDIDATE_MANIFEST_PATH
    artifact_path = args.artifact or c.QUALIFICATION_ARTIFACT_PATH
    config_path = args.config or c.CONFIG_PATH
    if config_path != c.CONFIG_PATH:
        raise SystemExit(f"freeze-b2a-r3-selected-row config path must be {c.CONFIG_PATH}")

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidate_manifest = json.load(f)
    with open(artifact_path, "r", encoding="utf-8") as f:
        qualification_artifact = json.load(f)
    expected_config_sha256 = config_identity(config_path)
    stage_b_binding = verify_persisted_stage_b_authorization_binding(
        authorization_id=qualification_artifact["stage_b_authorization_id"],
        repository_root=".",
        git_state=SubprocessGitStateProvider("."),
        candidate_manifest=candidate_manifest,
        expected_config_sha256=expected_config_sha256,
    )

    plan = plan_freeze_dry_run(
        candidate_manifest=candidate_manifest, qualification_artifact=qualification_artifact,
        expected_config_sha256=expected_config_sha256,
        stage_b_authorization_context=stage_b_binding,
    )
    print("freeze-b2a-r3-selected-row dry-run plan:")
    for key, value in plan.items():
        print(f"  {key} = {value}")
    return 0 if plan["would_freeze"] else 1


def _cmd_freeze_b2a_r3_selected_row_execute() -> int:
    """The one production execute path -- fixed paths only, exact local
    tokenizer resolution, complete in-memory plan construction, guarded
    publication, immediate reload, and complete provenance verification."""
    from kvcot.discovery.b2a_r3_freeze import construct_production_freeze_plan, publish_production_freeze

    plan = construct_production_freeze_plan(
        repository_root=".",
    )
    result = publish_production_freeze(plan)

    print("freeze-b2a-r3-selected-row --execute result:")
    for key in (
        "selected_unique_id", "selected_ordinal", "selected_manifest_path", "selected_manifest_sha256",
        "selection_provenance_path", "selection_provenance_canonical_sha256", "tokenizer_snapshot_revision",
        "publication_state_before", "publication_state_after", "already_frozen", "verification_passed",
    ):
        print(f"  {key} = {result[key]}")
    return 0 if result["verification_passed"] else 1


def cmd_verify_b2a_r3_selection(args: argparse.Namespace) -> int:
    import json

    from kvcot.config import config_identity
    from kvcot.discovery import b2a_r3_contract as c
    from kvcot.discovery.b2a_r3_authorization import verify_persisted_stage_b_authorization_binding
    from kvcot.discovery.b2a_r3_freeze import verify_selection_provenance
    from kvcot.discovery.b2a_r3_provenance import SubprocessGitStateProvider
    from kvcot.discovery.manifest import B2AOneExampleManifest

    provenance_path = args.provenance or c.SELECTION_PROVENANCE_PATH
    manifest_path = args.manifest or c.SELECTED_MANIFEST_PATH
    candidates_path = args.candidates or c.CANDIDATE_MANIFEST_PATH
    artifact_path = args.artifact or c.QUALIFICATION_ARTIFACT_PATH

    with open(provenance_path, "r", encoding="utf-8") as f:
        provenance = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        selected_manifest = B2AOneExampleManifest.model_validate_json(f.read())
    with open(candidates_path, "r", encoding="utf-8") as f:
        candidate_manifest = json.load(f)
    with open(artifact_path, "r", encoding="utf-8") as f:
        qualification_artifact = json.load(f)

    try:
        expected_config_sha256 = config_identity(c.CONFIG_PATH)
        stage_b_binding = verify_persisted_stage_b_authorization_binding(
            authorization_id=qualification_artifact["stage_b_authorization_id"],
            repository_root=".",
            git_state=SubprocessGitStateProvider("."),
            candidate_manifest=candidate_manifest,
            expected_config_sha256=expected_config_sha256,
        )
        typed = verify_selection_provenance(
            provenance, selected_manifest=selected_manifest, candidate_manifest=candidate_manifest,
            qualification_artifact=qualification_artifact,
            expected_config_sha256=expected_config_sha256,
            stage_b_authorization_context=stage_b_binding,
        )
    except Exception as e:  # noqa: BLE001
        print(f"verify-b2a-r3-selection: VERIFICATION FAILED: {e}")
        return 1
    print(f"verify-b2a-r3-selection: verification PASSED for {provenance_path}")
    print(f"  selected_unique_id={typed.selected_unique_id}")
    return 0


def cmd_verify_b2a_r3_authorization(args: argparse.Namespace) -> int:
    import json

    from kvcot.config import config_identity
    from kvcot.discovery import b2a_r3_contract as c
    from kvcot.discovery.b2a_r3_authorization import (
        AUTHORIZATION_STAGE_B2A_R3_EXECUTION,
        AuthorizationClaimR3,
        verify_authorization_preconditions,
    )
    from kvcot.discovery.b2a_r3_contract import verify_canonical_sha256
    from kvcot.discovery.b2a_r3_provenance import SubprocessGitStateProvider
    from kvcot.discovery.manifest import B2AOneExampleManifest

    with open(args.claim, "r", encoding="utf-8") as f:
        claim = json.load(f)

    try:
        verify_canonical_sha256(claim)
        typed = AuthorizationClaimR3.model_validate(claim)
        candidates_path = args.candidates or c.CANDIDATE_MANIFEST_PATH
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidate_manifest = json.load(f)
        expected_config_sha256 = config_identity(c.CONFIG_PATH)
        # Step 3R4 Finding 3: the policy is no longer constructed here from
        # the claim's own fields -- verify_authorization_preconditions now
        # parses the authorization document itself and builds the policy
        # from it, then requires the claim to agree with the document.
        qualification_artifact = None
        selected_manifest = None
        selection_provenance = None
        if typed.authorization_stage == AUTHORIZATION_STAGE_B2A_R3_EXECUTION:
            qualification_path = args.qualification or c.QUALIFICATION_ARTIFACT_PATH
            selected_path = args.manifest or c.SELECTED_MANIFEST_PATH
            provenance_path = args.provenance or c.SELECTION_PROVENANCE_PATH
            with open(qualification_path, "r", encoding="utf-8") as f:
                qualification_artifact = json.load(f)
            with open(selected_path, "r", encoding="utf-8") as f:
                selected_manifest = B2AOneExampleManifest.model_validate_json(f.read())
            with open(provenance_path, "r", encoding="utf-8") as f:
                selection_provenance = json.load(f)
        verify_authorization_preconditions(
            claim,
            git_state=SubprocessGitStateProvider(args.repository_root),
            authorization_document_path=args.document or typed.authorization_document_path,
            candidate_manifest=candidate_manifest,
            expected_config_sha256=expected_config_sha256,
            qualification_artifact=qualification_artifact,
            selected_manifest=selected_manifest,
            selection_provenance=selection_provenance,
            repository_root=args.repository_root,
        )
    except Exception as e:  # noqa: BLE001
        print(f"verify-b2a-r3-authorization: VERIFICATION FAILED: {e}")
        return 1
    print(f"verify-b2a-r3-authorization: verification PASSED for {args.claim}")
    print(f"  authorization_id={typed.authorization_id} authorization_stage={typed.authorization_stage}")
    return 0


# ----------------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvcot")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("freeze-manifests")
    p.add_argument("--config", default="configs/lock.yaml")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_freeze_manifests)

    p = sub.add_parser("generate")
    _add_common_run_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("replay-probe")
    _add_common_run_args(p)
    p.set_defaults(func=cmd_replay_probe)

    p = sub.add_parser("replay-fixed-trace")
    _add_fixed_trace_run_args(p)
    p.set_defaults(func=cmd_replay_fixed_trace)

    p = sub.add_parser("analyze")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("analyze-fixed-trace")
    p.add_argument("--config", required=True)
    p.add_argument("--trace-condition", default="full")
    p.add_argument("--replay-condition", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--selection-file", default=None,
        help="protocol v3: restrict analysis to exactly the selected base_record_ids and require "
        "complete (all 9 fractions, both policies) coverage of every one of them -- abort otherwise",
    )
    p.set_defaults(func=cmd_analyze_fixed_trace)

    # Deliberately NO --limit/--problem-index/--seed here (2026-07-18
    # external review): they fed expected_n, letting a partial pair of
    # natural files pass the strict gate as "the expected experiment". The
    # expected count comes only from the stage config + lock
    # (_expected_stage_record_count).
    p = sub.add_parser("check-fixed-trace-accuracy")
    p.add_argument("--config", required=True)
    p.add_argument("--replay-condition", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_check_fixed_trace_accuracy)

    p = sub.add_parser("inspect-fixed-trace")
    p.add_argument("--config", required=True)
    p.add_argument("--trace-condition", default="full")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--write-selection", action="store_true",
        help="protocol v3: write results/selections/{stage_name}.json using kvcot.analysis.rkv_schedule's "
        "predicted retention (deterministic, outcome-blind -- never a fixed-trace probe answer/PSS/CPSS)",
    )
    p.add_argument(
        "--max-selected", type=int, default=None,
        help="truncate ranked candidates (sorted by source_row_index) to at most this many before "
        "reporting predicted_eligible; omit for no truncation",
    )
    p.set_defaults(func=cmd_inspect_fixed_trace)

    p = sub.add_parser("calibrate-budget")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_calibrate_budget)

    p = sub.add_parser("validate-run")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_validate_run)

    p = sub.add_parser("plan-discovery")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_plan_discovery)

    p = sub.add_parser(
        "prepare-b2a-manifest",
        help="CPU-only: resolve the B2A one-example manifest's prompt identity from the live tokenizer.",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_prepare_b2a_manifest)

    p = sub.add_parser(
        "qualify-b2a-row",
        help="B2A-R2: FullKV-only, R-KV-free row qualification against the committed candidate manifest.",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_qualify_b2a_row)

    p = sub.add_parser(
        "b2a-calibrate",
        help="One-example engineering calibration. Does not authorize the 12-example B2B pilot.",
    )
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--execute", action="store_true",
        help="Explicit GPU execution flag. Requires CUDA and a fully-resolved one-example manifest. "
        "B2A is a one-example engineering calibration. It does not authorize the 12-example pilot.",
    )
    p.add_argument("--problem-index", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(func=cmd_b2a_calibrate)

    p = sub.add_parser("failure-atlas")
    p.add_argument("--full-artifact", required=True)
    p.add_argument("--rkv-artifact", required=True)
    p.add_argument("--output-csv", default="results/tables/gsm8k_v3_b128_failure_atlas.csv")
    p.add_argument("--output-markdown", default="results/tables/gsm8k_v3_b128_failure_atlas.md")
    p.add_argument("--output-summary", default="results/decisions/gsm8k_v3_b128_failure_atlas_summary.json")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_failure_atlas)

    p = sub.add_parser(
        "prepare-b2a-r3-candidates",
        help="B2A-R3: CPU-only deterministic candidate-manifest construction from the pinned MATH-500 dataset.",
    )
    p.add_argument("--output", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=cmd_prepare_b2a_r3_candidates)

    p = sub.add_parser("verify-b2a-r3-candidates", help="B2A-R3: verify the committed candidate manifest.")
    p.add_argument("--manifest", default=None)
    p.add_argument("--verify-existing", action="store_true")
    p.set_defaults(func=cmd_verify_b2a_r3_candidates)

    p = sub.add_parser(
        "plan-b2a-r3-qualification",
        help="B2A-R3: CPU-only qualification planning. Stage B FullKV qualification remains unauthorized.",
    )
    p.add_argument("--candidates", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_plan_b2a_r3_qualification)

    p = sub.add_parser(
        "run-b2a-r3-stage-b-qualification",
        help="B2A-R3: consume a Stage-B authorization claim and run fixed-path FullKV qualification.",
    )
    p.add_argument("--claim", required=True)
    p.set_defaults(func=cmd_run_b2a_r3_stage_b_qualification)

    p = sub.add_parser("verify-b2a-r3-qualification", help="B2A-R3: verify a qualification artifact.")
    p.add_argument("--artifact", default=None)
    p.add_argument("--candidates", default=None)
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_verify_b2a_r3_qualification)

    p = sub.add_parser(
        "freeze-b2a-r3-selected-row",
        help=(
            "B2A-R3: freeze planning (--dry-run, CPU-only, read-only overrides accepted) or production "
            "publication (--execute, fixed production paths only, exact one required)."
        ),
    )
    p.add_argument("--candidates", default=None)
    p.add_argument("--artifact", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(func=cmd_freeze_b2a_r3_selected_row)

    p = sub.add_parser("verify-b2a-r3-selection", help="B2A-R3: verify a selection-provenance artifact.")
    p.add_argument("--provenance", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--candidates", default=None)
    p.add_argument("--artifact", default=None)
    p.set_defaults(func=cmd_verify_b2a_r3_selection)

    p = sub.add_parser("verify-b2a-r3-authorization", help="B2A-R3: fully verify an authorization claim and provenance chain.")
    p.add_argument("--claim", required=True)
    p.add_argument("--document", default=None)
    p.add_argument("--candidates", default=None)
    p.add_argument("--qualification", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--provenance", default=None)
    p.add_argument("--repository-root", default=".")
    p.set_defaults(func=cmd_verify_b2a_r3_authorization)

    return parser


def main(argv: list[str] | None = None) -> int:
    from kvcot.runtime import OperatingPointMissingError

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OperatingPointMissingError, ResumeIdentityMismatchError, QuestionHashMismatch) as e:
        # Prerequisite/identity/manifest-integrity failures (§5, §10, §13).
        # Report cleanly — including under --dry-run, whose whole purpose is
        # to surface exactly this before GPU time is spent — instead of
        # dumping a raw traceback.
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
