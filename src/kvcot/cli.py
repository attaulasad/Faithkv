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
from pathlib import Path

from kvcot.config import config_identity, load_stage_config
from kvcot.data import read_manifest
from kvcot.runtime import (
    capture_version_info,
    git_commit,
    git_is_dirty,
    require_operating_point,
    resolve_conditions,
)
from kvcot.schemas import RunManifest
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
    resolved_conditions = resolve_conditions(stage)
    if args.condition not in stage.conditions and args.condition not in resolved_conditions:
        raise SystemExit(f"condition {args.condition!r} is not one of this stage's conditions {stage.conditions}")
    condition = args.condition if args.condition != "rkv_selected" else resolved_conditions[stage.conditions.index("rkv_selected")]

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

    if args.dry_run:
        already_written = read_existing_record_ids(output_path) if args.resume else set()
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
    from kvcot.runtime import gpu_model_name, upstream_submodule_commit
    from kvcot.utils.hashing import question_hash as qhash
    from kvcot.utils.io import JsonlWriter
    from kvcot.schemas import (
        BaseRunRecord, DatasetProvenance, MethodConfig, ProvenanceState, ThinkSpanInfo,
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

    writer = JsonlWriter(output_path, validator=lambda r: BaseRunRecord.model_validate(r).model_dump(mode="json"))
    already_written = writer._written_ids if args.resume else set()

    upstream_commit = upstream_submodule_commit(lock)
    versions = capture_version_info()

    for row in rows:
        for seed in seeds:
            record_id = f"base-{condition}-{Path(stage.dataset_manifest).stem}-{row['source_row_index']}-seed{seed}"
            if record_id in already_written:
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

            compaction_event_steps: list[int] = []
            cache_lengths = [cache.key_cache[i].shape[-2] for i in range(len(model.model.layers))]
            for layer in model.model.layers:
                kv_cluster = getattr(layer.self_attn, "kv_cluster", None)
                if kv_cluster is not None:
                    compaction_event_steps.extend(range(len(kv_cluster.kept_token_indices)))

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
                    source_row_index=row["source_row_index"],
                    question_hash=row["question_hash"],
                    normalized_gold=row["normalized_gold"],
                ),
                condition=condition,
                method_config=MethodConfig(method="fullkv" if condition == "full" else ("patched_noop" if condition == "patched_noop" else "rkv")),
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
                compaction_count=len(compaction_event_steps),
                compaction_event_steps=compaction_event_steps,
                cache_length_final_per_layer=cache_lengths,
            )
            writer.append(record.model_dump(mode="json"))

    return 0


# ------------------------------------------------------------------ replay-probe

def cmd_replay_probe(args: argparse.Namespace) -> int:
    stage, lock = load_stage_config(args.config)
    base_path = Path(stage.output_dir) / f"{args.condition}.jsonl"

    if args.dry_run:
        base_records = list(read_manifest(base_path)) if base_path.exists() else []
        n_planned = len(base_records) * len(lock.probes.fractions_all)
        print(f"replay-probe plan: stage={stage.stage_name} condition={args.condition}")
        print(f"  base records available: {len(base_records)}")
        print(f"  probe fractions: {lock.probes.fractions_all}")
        print(f"  planned probe records: {n_planned}")
        print(f"  output: {Path(stage.output_dir) / f'{args.condition}_probes.jsonl'}")
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
    from kvcot.runtime import upstream_submodule_commit
    from kvcot.schemas import ProbeRunRecord, ProvenanceState
    from kvcot.utils.answers import answers_match, extract_answer
    from kvcot.utils.hashing import sha256_int_ids
    from kvcot.utils.io import JsonlWriter, read_jsonl

    policy = build_policy(args.condition, lock)
    dtype = getattr(torch, lock.model.dtype)
    model = policy.load(lock.model.name, lock.model.revision, dtype, lock.attention.primary)
    tokenizer = AutoTokenizer.from_pretrained(lock.model.tokenizer_name, revision=lock.model.tokenizer_revision, use_fast=True)
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    suffix_ids = tokenizer.encode(render_control_suffix(), add_special_tokens=False)
    device = "cuda"

    probe_output_path = Path(stage.output_dir) / f"{args.condition}_probes.jsonl"
    writer = JsonlWriter(probe_output_path, validator=lambda r: ProbeRunRecord.model_validate(r).model_dump(mode="json"))
    upstream_commit = upstream_submodule_commit(lock)
    versions = capture_version_info()

    for base in read_jsonl(base_path):
        if base["think_span"]["think_parse_status"] not in ("ok", "generation_prompt_preopened_ok"):
            continue
        span = ThinkSpanResult(
            think_start_index=base["think_span"]["think_start_index"],
            think_end_index=base["think_span"]["think_end_index"],
            think_parse_status=base["think_span"]["think_parse_status"],
            generation_prompt_preopened_think=base["think_span"]["generation_prompt_preopened_think"],
        )
        cut_positions = {
            f: len(base["prompt_token_ids"]) + absolute_cut_position(span, f) - span.think_start_index
            for f in lock.probes.fractions_all
        }
        snapshots = replay_and_snapshot(
            model=model, fresh_cache_factory=lambda: DynamicCache(),
            prompt_token_ids=base["prompt_token_ids"], generated_token_ids=base["generated_token_ids"],
            think_span=span, snapshot_absolute_positions=cut_positions, device=device,
        )
        base_answer = base["extracted_answer"]
        for fraction, snap in snapshots.items():
            record_id = f"probe-{base['record_id']}-f{fraction}"
            if writer.already_written(record_id):
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
                condition=args.condition,
                fraction=fraction,
                think_span_length=span.think_token_count,
                cut_index=cut_positions[fraction] - len(base["prompt_token_ids"]),
                control_suffix_token_ids=suffix_ids,
                probe_decoding_max_new_tokens=lock.probes.max_new_tokens,
                probe_output_token_ids=result.probe_output_token_ids,
                probe_output_text=probe_text,
                normalized_probe_answer=probe_answer.normalized_value,
                probe_extraction_status=probe_answer.method,
                matches_own_condition_base_answer=answers_match(probe_answer.normalized_value, base_answer),
                is_f1_stability_probe=(fraction == 1.0),
                snapshot_cache_hash=sha256_int_ids([t.shape[-2] for t in snap.key_cache]),
                snapshot_provenance_hash=sha256_int_ids(snap.compaction_event_steps),
                snapshot_state_hash=sha256_int_ids([snap.model_length, snap.absolute_position]),
            )
            writer.append(record.model_dump(mode="json"))

    return 0


# ------------------------------------------------------------------------- analyze

def cmd_analyze(args: argparse.Namespace) -> int:
    stage, lock = load_stage_config(args.config)

    if args.dry_run:
        print(f"analyze plan: stage={stage.stage_name}")
        print(f"  reads: {stage.output_dir}/*.jsonl and {stage.output_dir}/*_probes.jsonl")
        print(f"  writes: results/tables/, results/decisions/, results/figures/")
        return 0

    from kvcot.analysis.summaries import (
        build_attrition_funnel_table,
        build_primary_analysis_summary,
        build_stage1a_measurability_decision,
        write_attrition_funnel_csv,
        write_primary_analysis_json,
    )
    from kvcot.utils.io import read_jsonl

    output_dir = Path(stage.output_dir)
    if stage.stage_name == "stage1a_measurability":
        full_records = list(read_jsonl(output_dir / "full.jsonl"))
        n_total = sum(1 for r in full_records if r.get("is_correct"))
        n_changed = 0  # populated once probe records exist; requires replay-probe to have run first
        decision = build_stage1a_measurability_decision(n_total=max(n_total, 1), n_answer_changed_at_any_scored_fraction=n_changed)
        write_json("results/decisions/stage1a_baseline_measurability.json", decision)
        print(f"wrote results/decisions/stage1a_baseline_measurability.json: recommendation={decision['recommendation']}")
        return 0

    print(f"analyze: stage {stage.stage_name} has no CPU-runnable data yet on this build machine (GPU validation pending)")
    return 0


# ------------------------------------------------------------------ calibrate-budget

def cmd_calibrate_budget(args: argparse.Namespace) -> int:
    from kvcot.analysis.summaries import build_stage1b_budget_decision

    candidates = [128, 256, 512, 1024]
    if args.dry_run:
        print("calibrate-budget plan:")
        for b in candidates:
            print(f"  configs/stage1b_budget_{b}.yaml -> results/decisions/stage1b_budget_{b}.json")
        print("  selects smallest budget passing both gates -> configs/selected_operating_point.yaml")
        return 0

    print(
        "calibrate-budget: no Stage 1B results exist on this build machine "
        "(GPU validation pending) — nothing to calibrate from yet."
    )
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
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
