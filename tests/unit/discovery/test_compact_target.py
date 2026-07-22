from __future__ import annotations

import dataclasses
import gc
import weakref

import torch

from _synthetic_harness import (
    EOS_TOKEN_ID,
    NUM_HEADS,
    NUM_LAYERS,
    HarnessState,
    branch_step_fn,
    build_snapshot_from_state,
    fresh_state_factory,
    install_fake_rkv_compression_module,
    make_step_fns,
)
from kvcot.discovery.attrition import AttritionCounters
from kvcot.discovery.compact_target import build_compact_branch_target
from kvcot.discovery.orchestrator import run_example
from kvcot.discovery.pass1 import NaturalRunProvenance, build_pass1_plan, run_natural_pass1
from kvcot.discovery.pass2 import run_pass2_capture
from kvcot.discovery.sampling import IdentitySeedParts

PROMPT = list(range(1, 11))
MAX_NEW = 295
STOP_AT = 300
IDENTITY = IdentitySeedParts(13, "synthetic", 0, "rev-a", "rkv-rev")
PROVENANCE = NaturalRunProvenance(
    "synthetic", "rev-a", "synthetic-tokenizer", "rev-a", "rkv-rev", "deadbeef", "synthetic", "ex-1"
)


def _correct(_ids):
    return "42", "correct"


def _capture(monkeypatch):
    install_fake_rkv_compression_module(monkeypatch)
    prefill, decode = make_step_fns(stop_at_predicted_position=STOP_AT)
    trace = run_natural_pass1(
        PROVENANCE, PROMPT, HarnessState(), prefill, decode, MAX_NEW, EOS_TOKEN_ID, _correct
    )
    plan, failure = build_pass1_plan(trace, NUM_LAYERS, NUM_HEADS, IDENTITY)
    assert failure is None
    result = run_pass2_capture(
        plan, trace.full_token_ids, HarnessState(), prefill, decode, build_snapshot_from_state
    )
    assert result.valid
    return result.target_captures[0]


def test_compact_non_snapshot_storage_is_independent_of_twenty_thousand_position_capture(monkeypatch):
    target = _capture(monkeypatch)
    short = build_compact_branch_target(target)
    record = target.capture_record
    cache_length = 20_000
    head_dim = record.pre_call_key_states.shape[-1]
    dtype = record.pre_call_key_states.dtype
    long_record = dataclasses.replace(
        record,
        pre_call_key_states=torch.zeros((1, NUM_HEADS, cache_length, head_dim), dtype=dtype),
        pre_call_value_states=torch.zeros((1, NUM_HEADS, cache_length, head_dim), dtype=dtype),
        pre_call_key_shape=(1, NUM_HEADS, cache_length, head_dim),
        pre_call_value_shape=(1, NUM_HEADS, cache_length, head_dim),
        recomputed_final_score=torch.zeros((1, NUM_HEADS, cache_length - record.window_size), dtype=dtype),
        recomputed_attention_component=torch.zeros(
            (1, NUM_HEADS, cache_length - record.window_size), dtype=dtype
        ),
        recomputed_similarity_component=torch.zeros(
            (1, NUM_HEADS, cache_length - record.window_size), dtype=dtype
        ),
        pre_event_absolute_position_map=torch.arange(cache_length).unsqueeze(0).expand(NUM_HEADS, -1),
    )
    long = build_compact_branch_target(dataclasses.replace(target, capture_record=long_record))
    assert long.persistent_tensor_numel == short.persistent_tensor_numel
    assert long.persistent_tensor_bytes == short.persistent_tensor_bytes
    assert long.persistent_tensor_bytes == long.derived_tensor_byte_bound
    assert len(long.positions) == 4


def test_full_capture_and_large_capture_tensors_are_unreachable_before_first_pair(monkeypatch):
    import kvcot.discovery.orchestrator as orchestrator

    install_fake_rkv_compression_module(monkeypatch)
    prefill, decode = make_step_fns(stop_at_predicted_position=STOP_AT)
    real_pass2 = orchestrator.run_pass2_capture
    refs: list[weakref.ReferenceType] = []

    def observed_pass2(*args, **kwargs):
        result = real_pass2(*args, **kwargs)
        for target in result.target_captures:
            refs.extend(
                weakref.ref(value)
                for value in (
                    target.capture_record,
                    target.capture_record.pre_call_key_states,
                    target.capture_record.pre_call_value_states,
                    target.capture_record.returned_key_states,
                    target.capture_record.returned_value_states,
                    target.capture_record.recomputed_final_score,
                )
            )
        return result

    real_build = orchestrator.build_swap_pair_record
    checked = False

    def observed_build(**kwargs):
        nonlocal checked
        if not checked:
            gc.collect()
            assert refs and all(ref() is None for ref in refs)
            target = kwargs["target_capture"]
            assert target.pristine_snapshot.key_cache
            assert target.positions
            checked = True
        return real_build(**kwargs)

    monkeypatch.setattr(orchestrator, "run_pass2_capture", observed_pass2)
    monkeypatch.setattr(orchestrator, "build_swap_pair_record", observed_build)
    result = run_example(
        example_id="ex-1",
        model_revision="rev-a",
        rkv_revision="rkv-rev",
        provenance=PROVENANCE,
        prompt_token_ids=PROMPT,
        pass1_initial_state=HarnessState(),
        pass2_initial_state_factory=fresh_state_factory(),
        prefill_fn=prefill,
        decode_one_fn=decode,
        snapshot_fn=build_snapshot_from_state,
        max_new_tokens=MAX_NEW,
        eos_token_id=EOS_TOKEN_ID,
        answer_fn=_correct,
        num_hidden_layers=NUM_LAYERS,
        num_key_value_heads=NUM_HEADS,
        identity=IDENTITY,
        branch_step_fn=branch_step_fn,
        example_attrition=AttritionCounters(),
        pair_attrition=AttritionCounters(),
    )
    assert checked and result.valid
