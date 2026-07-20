"""Call-boundary instrumentation shared by the FullKV and R-KV B2A workers
(B1B-R4 §5/§8). Wraps a real `PrefillFn`/`DecodeOneFn` pair and records
EXACTLY how many prefill/decode calls happened and with what token inputs,
in order -- never inferred after the fact from an indirect side effect
(e.g. counting generated tokens and assuming one decode call per token).

Used identically by FullKV's natural-generation loop and by R-KV's Pass 1
and Pass 2, so `prefill_decode_boundary_parity` (B1B-R4 §8) compares two
INDEPENDENTLY recorded traces against each other, never a trace asserted
equal to itself.

Pure Python -- no torch import (token ids are plain ints); safe to import
from any CPU test.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kvcot.discovery.harness_types import DecodeOneFn, PrefillFn
from kvcot.utils.hashing import sha256_json


@dataclass(frozen=True)
class CallBoundaryEvent:
    kind: str  # "prefill" or "decode"
    token_ids: tuple[int, ...]  # prompt tokens for prefill; exactly one token for decode


@dataclass
class CallTraceRecorder:
    """Instrumented `(prefill, decode_one)` pair. `prefill`/`decode_one` are
    the two methods a caller threads into Pass 1 / Pass 2 / the FullKV
    natural loop in place of the raw `PrefillFn`/`DecodeOneFn` -- every
    other behavior (return value, state threading) is passed through
    unchanged; this class only OBSERVES, it never alters what gets fed to
    the model."""

    prefill_fn: PrefillFn
    decode_one_fn: DecodeOneFn
    events: list[CallBoundaryEvent] = field(default_factory=list)

    @property
    def prefill_call_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "prefill")

    @property
    def decode_call_count(self) -> int:
        return sum(1 for e in self.events if e.kind == "decode")

    @property
    def prefill_token_count(self) -> int:
        return sum(len(e.token_ids) for e in self.events if e.kind == "prefill")

    def prefill(self, state: Any, prompt_token_ids):
        prompt_token_ids = list(prompt_token_ids)
        self.events.append(CallBoundaryEvent(kind="prefill", token_ids=tuple(prompt_token_ids)))
        return self.prefill_fn(state, prompt_token_ids)

    def decode_one(self, state: Any, token_id: int):
        self.events.append(CallBoundaryEvent(kind="decode", token_ids=(token_id,)))
        return self.decode_one_fn(state, token_id)

    def ordered_call_kinds_and_tokens_hash(self) -> str:
        """One stable hash over the COMPLETE ordered call sequence (kind +
        exact token inputs) -- the strongest single piece of evidence that
        two passes fed the model through an identical call boundary, not
        merely the same COUNT of calls with different contents."""
        return sha256_json([{"kind": e.kind, "token_ids": list(e.token_ids)} for e in self.events])


@dataclass(frozen=True)
class CallBoundaryComparison:
    prefill_call_count_match: bool
    prefill_token_count_match: bool
    decode_call_count_match: bool
    ordered_trace_hash_match: bool

    @property
    def all_match(self) -> bool:
        return (
            self.prefill_call_count_match
            and self.prefill_token_count_match
            and self.decode_call_count_match
            and self.ordered_trace_hash_match
        )


def compare_call_boundary_traces(a: CallTraceRecorder, b: CallTraceRecorder) -> CallBoundaryComparison:
    """Exact boundary comparison (B1B-R4 §8): prefill-call count, prefill
    token count, decode-call count, AND the ordered call-kinds-and-tokens
    hash must all agree -- every one of the four is reported independently
    so a caller can see exactly which check (if any) failed, never a single
    collapsed boolean."""
    return CallBoundaryComparison(
        prefill_call_count_match=a.prefill_call_count == b.prefill_call_count,
        prefill_token_count_match=a.prefill_token_count == b.prefill_token_count,
        decode_call_count_match=a.decode_call_count == b.decode_call_count,
        ordered_trace_hash_match=a.ordered_call_kinds_and_tokens_hash() == b.ordered_call_kinds_and_tokens_hash(),
    )
