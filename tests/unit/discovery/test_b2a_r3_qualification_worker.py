"""Step 3R4-Repair-2 Finding 1: real R3 FullKV qualification worker tests.

Exercises `run_fullkv_r3_qualification_worker` end-to-end via the SAME
internal dependency-injection seams
`tests/unit/discovery/test_b2a_workers_real_bodies.py` already uses for
`run_fullkv_worker` directly -- a real, deterministic, torch-CPU-tensor
fake model/tokenizer/cuda backend, never a preconstructed
`FullKVWorkerResultR3`. `test_real_worker_output_adapts_into_qualification_evidence`
is the exact repair this module exists for: it feeds the REAL (injected)
worker's own output through the existing, already-tested
`adapt_fullkv_worker_result_to_r3_evidence` adapter, proving there is now a
genuine production path from an R3 candidate row all the way to
qualification evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from kvcot.discovery.b2a_r3_candidates import CandidateRowR3
from kvcot.discovery.b2a_r3_contract import (
    CANDIDATE_MANIFEST_PATH,
    DATASET_CONFIG,
    DATASET_REPO,
    DATASET_REVISION,
    DATASET_SPLIT,
    GENERATION_CONFIG_SHA256,
    MODEL_NAME,
    MODEL_REVISION,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
)
from kvcot.discovery.b2a_r3_qualification import B2AR3FullKVQualificationEvidence
from kvcot.discovery.b2a_r3_qualification_worker import run_fullkv_r3_qualification_worker
from kvcot.discovery.b2a_r3_worker_adapter import (
    FullKVWorkerResultR3,
    adapt_fullkv_worker_result_to_r3_evidence,
)
from tests.unit.discovery.test_b2a_workers_real_bodies import (
    NUM_LAYERS,
    _build_fake_discovery_config,
    _FakeCache,
    _FakeCudaFacade,
    _FakeModel,
)

CONFIG_SHA = "de8ac65a348c307c4f00089da07914666332935981bcaa7c98a150a9e7e778b3"


class _FakeQualificationTokenizer:
    """Duck-types just enough of `transformers`' tokenizer surface for the
    canonical renderer (`render_with_loaded_tokenizer`) and
    `find_think_span`'s marker encoding -- never a real network/model
    load."""

    chat_template = "{{ messages }}"
    eos_token_id = 999

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        return [1, 2, 3, 4, 5]

    def encode(self, text, add_special_tokens=False):
        return {"<think>": [100], "</think>": [101]}[text]

    def decode(self, ids, skip_special_tokens=True):
        return "no boxed answer here"


def _candidate_manifest() -> dict:
    return json.loads(Path(CANDIDATE_MANIFEST_PATH).read_text(encoding="utf-8"))


def _candidate() -> CandidateRowR3:
    return CandidateRowR3.model_validate(_candidate_manifest()["candidates"][0])


def _config():
    config = _build_fake_discovery_config()
    return config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "name": MODEL_NAME,
                    "revision": MODEL_REVISION,
                    "tokenizer_name": TOKENIZER_NAME,
                    "tokenizer_revision": TOKENIZER_REVISION,
                    "model_type": "llama",
                }
            ),
        }
    )


def _run_worker():
    config = _config()
    candidate = _candidate()
    model = _FakeModel()
    fake_cuda = _FakeCudaFacade()

    return run_fullkv_r3_qualification_worker(
        config,
        candidate,
        _load_model=lambda: model,
        _load_tokenizer=lambda: _FakeQualificationTokenizer(),
        _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS),
        _cuda=fake_cuda,
        _device="cpu",
    )


def test_real_worker_produces_schema_valid_r3_result():
    result = _run_worker()

    typed = FullKVWorkerResultR3.model_validate(result)
    assert typed.role == "fullkv"
    assert typed.model_name == MODEL_NAME
    assert typed.model_revision == MODEL_REVISION
    assert typed.tokenizer_name == TOKENIZER_NAME
    assert typed.dataset_repo == DATASET_REPO
    assert typed.dataset_config == DATASET_CONFIG
    assert typed.dataset_split == DATASET_SPLIT
    assert typed.dataset_revision == DATASET_REVISION
    assert typed.cap_hit is True  # deterministic fake never emits EOS
    assert typed.observed_prompt_token_ids_sha256 == typed.expected_prompt_token_ids_sha256


def test_worker_generation_config_matches_frozen_value_for_the_real_frozen_config():
    """`worker_generation_config_sha256` is reconstructed from
    `config.generation` (never the worker's own, structurally different
    `runtime_generation` dict) -- proving that a genuine run against the
    frozen generation config reproduces `GENERATION_CONFIG_SHA256` exactly,
    so `generation_config_hash_match` (one of the 27 qualification
    conditions) can legitimately pass for a real worker result."""
    config = _config().model_copy(
        update={"generation": _config().generation.model_copy(update={"max_new_tokens": 6144})}
    )
    candidate = _candidate()
    model = _FakeModel()
    fake_cuda = _FakeCudaFacade()

    result = run_fullkv_r3_qualification_worker(
        config,
        candidate,
        _load_model=lambda: model,
        _load_tokenizer=lambda: _FakeQualificationTokenizer(),
        _fresh_cache_factory=lambda: _FakeCache(NUM_LAYERS),
        _cuda=fake_cuda,
        _device="cpu",
    )
    typed = FullKVWorkerResultR3.model_validate(result)
    assert typed.worker_generation_config_sha256 == GENERATION_CONFIG_SHA256


def test_real_worker_output_adapts_into_qualification_evidence():
    """The exact Blocking Finding 1 requirement: feed the REAL (injected)
    worker's own output through the existing adapter -- never a hand-built
    FullKVWorkerResultR3."""
    candidate_manifest = _candidate_manifest()
    result = _run_worker()
    worker_result = FullKVWorkerResultR3.model_validate(result)

    evidence = adapt_fullkv_worker_result_to_r3_evidence(
        worker_result=worker_result,
        candidate_manifest=candidate_manifest,
        candidate_ordinal=0,
        expected_config_sha256=CONFIG_SHA,
    )
    assert isinstance(evidence, B2AR3FullKVQualificationEvidence)
    assert evidence.candidate_ordinal == 0
    assert evidence.fullkv_wall_seconds >= 0.0


def test_no_hand_built_worker_result_used():
    """Guards against silent regression back to a hand-built
    `FullKVWorkerResultR3`: the timing/memory evidence must actually come
    from the real `SynchronizedTimer`/`CudaMemoryMeasurer` inside
    `run_fullkv_worker`, not a fixture -- there must be more than one
    `fullkv_decode` timing record for a multi-token generation."""
    result = _run_worker()
    decode_records = [r for r in result["timing_evidence"] if r["phase"] == "fullkv_decode"]
    assert len(decode_records) == _config().generation.max_new_tokens
