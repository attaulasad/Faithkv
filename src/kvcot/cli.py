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
from kvcot.utils.io import read_existing_record_ids, write_json
from kvcot.utils.logging import get_logger

logger = get_logger("kvcot.cli")


def _add_common_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--problem-index", type=int, default=None, help="restrict to a single source_row_index")
    p.add_argument("--seed", type=int, default=None, help="restrict to a single seed (must be one of the frozen seeds)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def _load_manifest_filtered(stage, args) -> list[dict]:
    rows = list(read_manifest(stage.dataset_manifest))
    if args.problem_index is not None:
        rows = [r for r in rows if r["source_row_index"] == args.problem_index]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


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


def _resolve_condition(stage, args) -> str:
    """Validate `--condition` against this stage's declared conditions and
    resolve the `rkv_selected` placeholder (used only by stage2_main.yaml)
    into a concrete `rkv_b{budget}` condition name, by reading
    `configs/selected_operating_point.yaml` (kvcot.runtime.resolve_conditions).

    Both `generate` and `replay-probe` MUST use this same resolution — they
    write/read the same `{condition}.jsonl` / `{condition}_probes.jsonl`
    file pair, and `build_policy(condition, lock)` only understands concrete
    condition names ("full", "patched_noop", "rkv_b{budget}"), never the
    "rkv_selected" placeholder itself. Do not resolve this inline at each
    call site — that duplication is exactly what let `replay-probe` drift out
    of sync with `generate` before (`replay-probe --condition rkv_selected`
    looked for a nonexistent `rkv_selected.jsonl` and passed the literal
    placeholder string to `build_policy`, which raises `ValueError`).
    """
    resolved_conditions = resolve_conditions(stage)
    if args.condition not in stage.conditions and args.condition not in resolved_conditions:
        raise SystemExit(f"condition {args.condition!r} is not one of this stage's conditions {stage.conditions}")
    if args.condition == "rkv_selected":
        return resolved_conditions[stage.conditions.index("rkv_selected")]
    return args.condition


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
    write_json(Path(stage.output_dir) / f"{condition}_generate_manifest.json", manifest.model_dump(mode="json"))

    return 0


# ------------------------------------------------------------------ replay-probe

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
    from kvcot.utils.answers import answers_match, extract_answer
    from kvcot.utils.hashing import sha256_bytes, sha256_int_ids, sha256_json
    from kvcot.utils.io import JsonlWriter, read_jsonl

    def _hash_snapshot_cache_content(snap) -> str:
        """Real content hash of the snapshot's K/V cache — NOT just shapes
        (the old `sha256_int_ids([t.shape[-2] for t in snap.key_cache])` would
        hash equal for two snapshots with the same lengths but different
        values, which defeats the point of a divergence-detecting hash).
        Upcast to float32 before extracting bytes: two bf16 tensors holding
        the same values upcast to identical float32 bytes, so this only
        changes when the actual content differs, not the storage dtype."""
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
                matches_own_condition_base_answer=answers_match(probe_answer.normalized_value, base_answer),
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
    write_json(Path(stage.output_dir) / f"{condition}_replay_probe_manifest.json", manifest.model_dump(mode="json"))

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


# --------------------------------------------------------------------- validate-run

def cmd_validate_run(args: argparse.Namespace) -> int:
    from kvcot.schemas import BaseRunRecord, ProbeRunRecord
    from kvcot.utils.io import read_jsonl

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"{run_dir} does not exist")

    n_valid = 0
    n_invalid = 0
    for path in sorted(run_dir.glob("*.jsonl")):
        model_cls = ProbeRunRecord if path.name.endswith("_probes.jsonl") else BaseRunRecord
        for row in read_jsonl(path):
            try:
                model_cls.model_validate(row)
                n_valid += 1
            except Exception as e:
                n_invalid += 1
                logger.error("invalid record in %s: %s", path, e)

    print(f"validate-run: {run_dir}: {n_valid} valid, {n_invalid} invalid")
    return 0 if n_invalid == 0 else 1


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

    p = sub.add_parser("analyze")
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("calibrate-budget")
    p.add_argument("--config-dir", default="configs")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_calibrate_budget)

    p = sub.add_parser("validate-run")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_validate_run)

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
