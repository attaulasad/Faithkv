"""B2A-R2 FullKV-only row qualification tests (2026-07-22). Every test uses
injected fakes -- no CUDA, no model, no R-KV, no network access.
"""
from __future__ import annotations

import builtins
import copy
from types import SimpleNamespace

import pytest

from kvcot.discovery.b2a_qualification import (
    MAX_CANDIDATES_ATTEMPTED,
    QUALIFICATION_CONDITIONS,
    _default_fullkv_runner,
    build_candidate_outcome,
    evaluate_candidate_qualification,
    run_qualification_dry_run,
    run_qualification_execute,
    select_first_qualified,
    validate_candidate_manifest_identity,
)
from kvcot.discovery.discovery_config import load_discovery_config
from kvcot.discovery.manifest_prepare import ManifestPreparationError

CONFIG_PATH = "configs/discovery/llama8b_math500_b1024.yaml"


@pytest.fixture
def config():
    return load_discovery_config(CONFIG_PATH)


def _candidate(ordinal, unique_id, *, dataset_revision, budget=1024, model_revision=None, tokenizer_revision=None):
    return {
        "candidate_ordinal": ordinal,
        "source_example_index": 1000 + ordinal,
        "unique_id": unique_id,
        "subject": "Algebra",
        "level": 5,
        "row": {
            "problem": f"problem {ordinal}", "solution": "sol", "answer": f"answer-{ordinal}",
            "subject": "Algebra", "level": "5", "unique_id": unique_id,
        },
        "raw_row_sha256": "a" * 64,
        "problem_sha256": "b" * 64,
        "gold_answer_sha256": "c" * 64,
        "ordering_hash": f"{ordinal:064x}",
        "dataset_revision": dataset_revision,
        "model_revision": model_revision or "modelrev",
        "tokenizer_revision": tokenizer_revision or "modelrev",
        "budget": budget,
        "protocol_version": "faithkv-b2a-r2-row-order-v1",
    }


def _manifest(config, *, n=3):
    candidates = [
        _candidate(i, f"id-{i}", dataset_revision=config.dataset.revision, model_revision=config.model.revision,
                    tokenizer_revision=config.model.tokenizer_revision, budget=config.rkv.budget)
        for i in range(n)
    ]
    m = {
        "protocol_version": "faithkv-b2a-r2-row-order-v1",
        "dataset_repo": "HuggingFaceH4/MATH-500",
        "dataset_revision": config.dataset.revision,
        "model_revision": config.model.revision,
        "tokenizer_revision": config.model.tokenizer_revision,
        "budget": config.rkv.budget,
        "level": 5,
        "candidate_count": n,
        "eligible_population_size": n,
        "candidates": candidates,
    }
    from kvcot.utils.hashing import sha256_json

    m["canonical_sha256"] = sha256_json(m)
    return m


# --- evaluate_candidate_qualification (pure conditions) ---

def _good_kwargs(**overrides):
    defaults = dict(
        cap_hit=False, answer_status="correct", predicted_event_count=5, eligible_event_count=3,
        identity_ok=True, batch_size=1, every_parameter_on_cuda=True, no_offload_verified=True,
        peak_cuda_allocated_bytes=1 * 1024**3, peak_cuda_reserved_bytes=2 * 1024**3,
    )
    defaults.update(overrides)
    return defaults


def test_all_conditions_true_when_everything_passes():
    result = evaluate_candidate_qualification(**_good_kwargs())
    assert all(result.values())
    assert set(result) == set(QUALIFICATION_CONDITIONS)


def test_cap_hit_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(cap_hit=True))
    assert result["no_cap_hit"] is False
    assert not all(result.values())


def test_unverifiable_answer_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(answer_status="unverifiable"))
    assert result["fullkv_answer_verifiable"] is False
    assert result["fullkv_answer_correct"] is False


def test_incorrect_answer_rejects_correctness_but_not_verifiability():
    result = evaluate_candidate_qualification(**_good_kwargs(answer_status="incorrect"))
    assert result["fullkv_answer_verifiable"] is True
    assert result["fullkv_answer_correct"] is False
    assert not all(result.values())


def test_insufficient_predicted_events_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(predicted_event_count=2))
    assert result["predicted_schedule_has_at_least_three_events"] is False


def test_insufficient_eligible_events_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(eligible_event_count=2))
    assert result["at_least_three_events_have_49_future_tokens"] is False


def test_identity_mismatch_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(identity_ok=False))
    assert result["identity_checks_pass"] is False


def test_batch_size_other_than_one_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(batch_size=2))
    assert result["batch_size_is_one"] is False


def test_offload_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(no_offload_verified=False))
    assert result["no_offload"] is False


def test_not_all_parameters_on_cuda_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(every_parameter_on_cuda=False))
    assert result["all_parameters_on_cuda"] is False


def test_memory_over_limit_rejects():
    result = evaluate_candidate_qualification(**_good_kwargs(peak_cuda_reserved_bytes=23 * 1024**3))
    assert result["peak_memory_within_limit"] is False


def test_memory_exactly_at_limit_passes():
    result = evaluate_candidate_qualification(**_good_kwargs(peak_cuda_reserved_bytes=22 * 1024**3, peak_cuda_allocated_bytes=0))
    assert result["peak_memory_within_limit"] is True


def test_invalid_answer_status_raises():
    with pytest.raises(ValueError):
        evaluate_candidate_qualification(**_good_kwargs(answer_status="maybe"))


# --- build_candidate_outcome (wires in real eligibility rule) ---

def test_build_candidate_outcome_computes_eligible_events_via_shared_rule():
    # prompt_length=10, events at [15, 100, 200, 300, 950] with total_len=1000:
    # first/last excluded -> [100, 200, 300] remain, all with future_tokens >= 49.
    outcome = build_candidate_outcome(
        candidate_ordinal=0, source_example_index=5, unique_id="u",
        prompt_token_count=10, prompt_token_ids_sha256="x" * 64,
        generated_token_ids=list(range(990)),  # total_len = 10 + 990 = 1000
        generated_token_ids_sha256="y" * 64,
        cap_hit=False, extracted_answer="42", answer_status="correct",
        fullkv_wall_seconds=12.0, peak_cuda_allocated_bytes=1024, peak_cuda_reserved_bytes=2048,
        predicted_event_positions=[15, 100, 200, 300, 950],
        identity_ok=True, batch_size=1, every_parameter_on_cuda=True, no_offload_verified=True,
    )
    assert outcome.predicted_event_count == 5
    assert outcome.eligible_event_count == 3
    assert outcome.qualified is True
    assert outcome.failed_conditions == []


def test_build_candidate_outcome_reports_failed_conditions_when_not_qualified():
    outcome = build_candidate_outcome(
        candidate_ordinal=0, source_example_index=5, unique_id="u",
        prompt_token_count=10, prompt_token_ids_sha256="x" * 64,
        generated_token_ids=[1, 2, 3], generated_token_ids_sha256="y" * 64,
        cap_hit=False, extracted_answer=None, answer_status="unverifiable",
        fullkv_wall_seconds=1.0, peak_cuda_allocated_bytes=0, peak_cuda_reserved_bytes=0,
        predicted_event_positions=[], identity_ok=True, batch_size=1,
        every_parameter_on_cuda=True, no_offload_verified=True,
    )
    assert outcome.qualified is False
    assert "fullkv_answer_verifiable" in outcome.failed_conditions
    assert "fullkv_answer_correct" in outcome.failed_conditions
    assert "predicted_schedule_has_at_least_three_events" in outcome.failed_conditions
    assert "at_least_three_events_have_49_future_tokens" in outcome.failed_conditions


# --- select_first_qualified ---

def test_select_first_qualified_returns_first_true_never_best():
    def outcome(ordinal, qualified):
        return build_candidate_outcome(
            candidate_ordinal=ordinal, source_example_index=ordinal, unique_id=f"u{ordinal}",
            prompt_token_count=10, prompt_token_ids_sha256="x" * 64,
            generated_token_ids=list(range(990)), generated_token_ids_sha256="y" * 64,
            cap_hit=False, extracted_answer="4", answer_status=("correct" if qualified else "incorrect"),
            fullkv_wall_seconds=1.0, peak_cuda_allocated_bytes=0, peak_cuda_reserved_bytes=0,
            predicted_event_positions=[10, 100, 200, 300, 950], identity_ok=True, batch_size=1,
            every_parameter_on_cuda=True, no_offload_verified=True,
        )

    outcomes = [outcome(0, False), outcome(1, True), outcome(2, True)]
    selected = select_first_qualified(outcomes)
    assert selected.candidate_ordinal == 1


def test_select_first_qualified_returns_none_when_nothing_qualifies():
    outcome = build_candidate_outcome(
        candidate_ordinal=0, source_example_index=0, unique_id="u0",
        prompt_token_count=10, prompt_token_ids_sha256="x" * 64,
        generated_token_ids=[1], generated_token_ids_sha256="y" * 64,
        cap_hit=True, extracted_answer=None, answer_status="unverifiable",
        fullkv_wall_seconds=1.0, peak_cuda_allocated_bytes=0, peak_cuda_reserved_bytes=0,
        predicted_event_positions=[], identity_ok=True, batch_size=1,
        every_parameter_on_cuda=True, no_offload_verified=True,
    )
    assert select_first_qualified([outcome]) is None


# --- validate_candidate_manifest_identity ---

def test_identity_validation_accepts_matching_manifest(config):
    validate_candidate_manifest_identity(_manifest(config), config=config)  # must not raise


def test_identity_validation_rejects_dataset_revision_mismatch(config):
    manifest = _manifest(config)
    manifest["dataset_revision"] = "wrong" * 12 + "aaaa"
    with pytest.raises(ManifestPreparationError, match="dataset_revision"):
        validate_candidate_manifest_identity(manifest, config=config)


def test_identity_validation_rejects_budget_mismatch(config):
    manifest = _manifest(config)
    manifest["budget"] = 999
    with pytest.raises(ManifestPreparationError, match="budget"):
        validate_candidate_manifest_identity(manifest, config=config)


def test_identity_validation_rejects_more_than_twelve_candidates(config):
    manifest = _manifest(config, n=3)
    extra = [_candidate(i, f"extra-{i}", dataset_revision=config.dataset.revision) for i in range(3, 13)]
    manifest["candidates"] = manifest["candidates"] + extra
    with pytest.raises(ManifestPreparationError, match="qualification limit"):
        validate_candidate_manifest_identity(manifest, config=config)


def test_identity_validation_rejects_duplicate_ordinals(config):
    manifest = _manifest(config, n=3)
    manifest["candidates"][2]["candidate_ordinal"] = manifest["candidates"][0]["candidate_ordinal"]
    with pytest.raises(ManifestPreparationError, match="duplicate candidate_ordinal"):
        validate_candidate_manifest_identity(manifest, config=config)


# --- run_qualification_dry_run: no CUDA/model/rkv/network ---

_DISALLOWED_MODULES = {"torch", "transformers", "datasets", "huggingface_hub", "rkv", "rkv.monkeypatch"}


@pytest.fixture
def block_disallowed_imports(monkeypatch):
    real_import = builtins.__import__

    def _guarded_import(name, *args, **kwargs):
        if name in _DISALLOWED_MODULES or name.split(".")[0] in _DISALLOWED_MODULES:
            raise AssertionError(f"qualify-b2a-row --dry-run must never import {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _guarded_import)


def test_dry_run_never_imports_cuda_model_or_rkv(config, block_disallowed_imports):
    manifest = _manifest(config)
    plan = run_qualification_dry_run(config, manifest)
    assert plan["would_touch_cuda"] is False
    assert plan["would_load_model"] is False
    assert plan["would_import_rkv"] is False


def test_dry_run_reports_candidates_in_committed_order(config):
    manifest = _manifest(config, n=3)
    plan = run_qualification_dry_run(config, manifest)
    assert plan["candidates_to_attempt_in_order"] == ["id-0", "id-1", "id-2"]


def test_dry_run_rejects_identity_mismatch(config):
    manifest = _manifest(config)
    manifest["model_revision"] = "wrong-model-rev"
    with pytest.raises(ManifestPreparationError):
        run_qualification_dry_run(config, manifest)


# --- run_qualification_execute: full orchestration with a fake FullKV runner ---

def _fake_runner(outcomes_by_unique_id):
    calls = []

    def runner(candidate):
        calls.append(candidate["unique_id"])
        return outcomes_by_unique_id[candidate["unique_id"]]

    return runner, calls


def _raw_result(*, prompt_token_count=10, generated_len=2000, cap_hit=False, answer="4", status="correct",
                 model_revision="modelrev", tokenizer_revision="modelrev", dataset_revision="datarev",
                 peak_allocated=1 * 1024**3, peak_reserved=2 * 1024**3, batch_size=1, every_parameter_on_cuda=True,
                 no_offload_verified=True):
    return {
        "prompt_token_count": prompt_token_count,
        "natural_generated_token_ids": list(range(generated_len)),
        "cap_hit": cap_hit,
        "natural_answer": answer,
        "natural_answer_status": status,
        "wall_seconds": 12.0,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "batch_size": batch_size,
        "every_parameter_on_cuda": every_parameter_on_cuda,
        "parameter_placement": {"no_offload_verified": no_offload_verified},
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "dataset_revision": dataset_revision,
        "prompt_token_ids_sha256": "z" * 64,
    }


def test_execute_selects_first_qualified_and_stops(config):
    manifest = _manifest(config, n=3)
    manifest["dataset_revision"] = "datarev"
    manifest["model_revision"] = "modelrev"
    manifest["tokenizer_revision"] = "modelrev"
    for c in manifest["candidates"]:
        c["dataset_revision"] = "datarev"
        c["model_revision"] = "modelrev"
        c["tokenizer_revision"] = "modelrev"
    from kvcot.utils.hashing import sha256_json
    manifest["canonical_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "canonical_sha256"})

    # id-0: unqualified (unverifiable). id-1: qualified. id-2: must never be attempted.
    runner, calls = _fake_runner({
        "id-0": _raw_result(status="unverifiable", answer=None, dataset_revision="datarev"),
        "id-1": _raw_result(status="correct", dataset_revision="datarev"),
        "id-2": _raw_result(status="correct", dataset_revision="datarev"),
    })

    class _FakeConfig:
        dataset = type("D", (), {"revision": "datarev"})()
        model = type("M", (), {"revision": "modelrev", "tokenizer_revision": "modelrev"})()
        rkv = type("R", (), {"budget": 1024, "divide_length": 128})()

    artifact = run_qualification_execute(
        _FakeConfig(), manifest, "configs/discovery/b2a_r2_candidate_manifest.json",
        config_hash="cfg" * 20 + "a", fullkv_runner=runner,
    )

    assert calls == ["id-0", "id-1"], "must stop immediately after the first qualified candidate"
    assert artifact.selected_ordinal == 1
    assert artifact.selected_unique_id == "id-1"
    assert len(artifact.attempted) == 2
    assert artifact.attempted[0].qualified is False
    assert artifact.attempted[1].qualified is True


def test_execute_returns_no_selection_when_nothing_qualifies(config):
    manifest = _manifest(config, n=2)
    for c in manifest["candidates"]:
        c["dataset_revision"] = "datarev"
        c["model_revision"] = "modelrev"
        c["tokenizer_revision"] = "modelrev"
    manifest["dataset_revision"] = "datarev"
    manifest["model_revision"] = "modelrev"
    manifest["tokenizer_revision"] = "modelrev"
    from kvcot.utils.hashing import sha256_json
    manifest["canonical_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "canonical_sha256"})

    runner, calls = _fake_runner({
        "id-0": _raw_result(status="incorrect", dataset_revision="datarev"),
        "id-1": _raw_result(cap_hit=True, dataset_revision="datarev"),
    })

    class _FakeConfig:
        dataset = type("D", (), {"revision": "datarev"})()
        model = type("M", (), {"revision": "modelrev", "tokenizer_revision": "modelrev"})()
        rkv = type("R", (), {"budget": 1024, "divide_length": 128})()

    artifact = run_qualification_execute(
        _FakeConfig(), manifest, "path.json", config_hash="h" * 64, fullkv_runner=runner,
    )
    assert calls == ["id-0", "id-1"], "every candidate must be attempted when none qualifies"
    assert artifact.selected_ordinal is None
    assert artifact.selected_unique_id is None
    assert len(artifact.attempted) == 2


def test_execute_never_mutates_the_input_candidate_manifest(config):
    manifest = _manifest(config, n=2)
    for c in manifest["candidates"]:
        c["dataset_revision"] = "datarev"
        c["model_revision"] = "modelrev"
        c["tokenizer_revision"] = "modelrev"
    manifest["dataset_revision"] = "datarev"
    manifest["model_revision"] = "modelrev"
    manifest["tokenizer_revision"] = "modelrev"
    from kvcot.utils.hashing import sha256_json
    manifest["canonical_sha256"] = sha256_json({k: v for k, v in manifest.items() if k != "canonical_sha256"})
    original = copy.deepcopy(manifest)

    runner, _ = _fake_runner({
        "id-0": _raw_result(status="correct", dataset_revision="datarev"),
        "id-1": _raw_result(status="correct", dataset_revision="datarev"),
    })

    class _FakeConfig:
        dataset = type("D", (), {"revision": "datarev"})()
        model = type("M", (), {"revision": "modelrev", "tokenizer_revision": "modelrev"})()
        rkv = type("R", (), {"budget": 1024, "divide_length": 128})()

    run_qualification_execute(_FakeConfig(), manifest, "path.json", config_hash="h" * 64, fullkv_runner=runner)
    assert manifest == original


def test_execute_respects_max_twelve_even_if_more_are_somehow_present(config):
    # validate_candidate_manifest_identity already rejects >12, so this
    # proves the defense is real, not merely documented.
    manifest = _manifest(config, n=3)
    extra = [_candidate(i, f"extra-{i}", dataset_revision=config.dataset.revision) for i in range(3, 13)]
    manifest["candidates"] = manifest["candidates"] + extra
    runner, calls = _fake_runner({})

    with pytest.raises(ManifestPreparationError, match="qualification limit"):
        run_qualification_execute(config, manifest, "path.json", config_hash="h" * 64, fullkv_runner=runner)
    assert calls == []


# ---------------------------------------------------------------------------
# Regression (2026-07-22): the FIRST real GPU qualification run crashed with
# `AttributeError: 'dict' object has no attribute 'model_dump'` -- production
# `run_fullkv_worker` already returns a JSON-serializable dict (its own
# internal `.model_dump(mode="json")` call), but `_default_fullkv_runner`'s
# inner closure called `.model_dump()` on that dict AGAIN. No candidate was
# ever evaluated before this crash (it happened at result serialization,
# after generation but before any qualification condition was computed), so
# rerunning qualification after this fix does not re-attempt anything that
# had already produced a verdict.
# ---------------------------------------------------------------------------


def test_default_fullkv_runner_returns_the_dict_run_fullkv_worker_produces_unmodified(monkeypatch, config):
    """`run_fullkv_worker` returns a plain dict (its own
    `.model_dump(mode='json')` already applied) -- `_default_fullkv_runner`'s
    closure must return that dict AS-IS, never call `.model_dump()` on it
    again (that call raises `AttributeError` on a plain dict, exactly the
    real crash this regression test reproduces)."""
    import torch

    from kvcot.discovery import b2a_qualification

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("kvcot.discovery.strict_device.verify_single_rtx3090", lambda *a, **k: None)

    fake_snapshot = SimpleNamespace(local_path="/fake/snapshot")
    monkeypatch.setattr(
        "kvcot.discovery.snapshot_boundary.resolve_local_snapshot", lambda *a, **k: fake_snapshot
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained", classmethod(lambda cls, *a, **k: SimpleNamespace())
    )
    monkeypatch.setattr(
        "kvcot.discovery.strict_device.load_fullkv_discovery_model", lambda *a, **k: SimpleNamespace()
    )

    real_worker_dict_result = {
        "role": "fullkv", "natural_generated_token_ids": [1, 2, 3], "natural_answer_status": "correct",
        "prompt_token_count": 5, "cap_hit": False, "wall_seconds": 1.0,
    }
    monkeypatch.setattr(
        "kvcot.discovery.b2a_workers.run_fullkv_worker", lambda *a, **k: real_worker_dict_result
    )
    monkeypatch.setattr(
        "kvcot.discovery.manifest_prepare._render_and_tokenize",
        lambda *a, **k: (SimpleNamespace(chat_template="t"), "msg", [{"role": "user", "content": "x"}], [1, 2, 3]),
    )

    runner = _default_fullkv_runner(config)
    candidate = {
        "row": {"problem": "p", "answer": "4", "unique_id": "u", "subject": "Algebra", "level": "5"},
        "dataset_revision": config.dataset.revision,
        "source_example_index": 0,
        "unique_id": "u",
        "raw_row_sha256": "a" * 64,
    }
    result = runner(candidate)
    assert result == real_worker_dict_result, "the runner must return run_fullkv_worker's dict unmodified"
