"""Test-only fixtures for Parts VIII/IX/X of
docs/B0_5_R2_2_AUTHORITY_AND_IMPLEMENTATION.md.

`FakeR1KV.update_kv`, `fake_compute_attention_scores`, and
`fake_cal_similarity` are deliberately small, self-contained ports of the
REAL formulas in `third_party/R-KV/HuggingFace/rkv/compression/r1_kv.py`
and `third_party/R-KV/HuggingFace/rkv/utils.py` (verified by direct
inspection this session) — restricted to the multi-head-attention case
(`query_group_size == 1`, i.e. `num_attention_heads == num_key_value_heads`)
since this repository's frozen model does not use GQA in a way that matters
for exercising `kvcot.discovery.capture`'s recomputation logic. This lets
CPU tests exercise real eviction-formula behavior without requiring the
pinned R-KV submodule to be importable (`third_party/R-KV` is a GPU-host-
only, sometimes-uninitialized dependency; see `CLAUDE.md`'s session notes) —
never used by any file under `src/`.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_compute_attention_scores(query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
    """Ported from rkv/utils.py:compute_attention_scores, MHA branch only."""
    head_dim = query_states.shape[-1]
    return torch.matmul(query_states, key_states.transpose(2, 3)) / (head_dim**0.5)


def fake_cal_similarity(
    key_states: torch.Tensor,
    threshold: float = 0.5,
    retain_ratio: float = 0.2,
    retain_direction: str = "last",
) -> torch.Tensor:
    """Ported verbatim (formula-for-formula) from rkv/utils.py:cal_similarity."""
    _, _, seq_len, _ = key_states.shape
    k_norm = key_states / (key_states.norm(dim=-1, keepdim=True) + 1e-8)
    similarity_cos = torch.matmul(k_norm, k_norm.transpose(-1, -2))
    diag = torch.eye(seq_len, dtype=torch.bool, device=key_states.device)
    similarity_cos.masked_fill_(diag.view(1, 1, seq_len, seq_len), 0.0)
    similarity_mask = similarity_cos > threshold
    indices = torch.where(
        similarity_mask,
        torch.arange(seq_len, device=similarity_mask.device).view(1, 1, 1, seq_len),
        torch.zeros_like(similarity_mask, dtype=torch.long),
    )
    if retain_direction == "last":
        similarity_retain = torch.max(indices, dim=-1)[0]
    elif retain_direction == "first":
        similarity_retain = torch.min(indices, dim=-1)[0]
    else:
        raise ValueError(f"fake_cal_similarity only supports last/first, got {retain_direction!r}")
    similarity_cos.scatter_(-1, similarity_retain.unsqueeze(-1), 0)
    return similarity_cos.mean(dim=-2).softmax(dim=-1)


class FakeR1KV:
    """Ported from rkv/compression/r1_kv.py:R1KV -- same public interface
    (`budget`, `window_size`, `kernel_size`, `mix_lambda`, `retain_ratio`,
    `retain_direction`, `record_kept_token_indices`, `kept_token_indices`,
    `evicted_token_num`, `update_kv`) and the identical eviction formula,
    calling THIS module's fake helpers (never the real `rkv.compression`)."""

    def __init__(
        self,
        budget=12,
        window_size=4,
        kernel_size=3,
        mix_lambda=0.1,
        retain_ratio=0.2,
        retain_direction="last",
        record_kept_token_indices=True,
        **kwargs,
    ):
        assert budget - window_size > 0
        self.budget = budget
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.mix_lambda = mix_lambda
        self.retain_ratio = retain_ratio
        self.retain_direction = retain_direction
        self.record_kept_token_indices = record_kept_token_indices
        if self.record_kept_token_indices:
            self.evicted_token_num = 0
            self.kept_token_indices = []
            self.kept_attention_scores = []
            self.kept_similarity_scores = []
            self.kept_final_scores = []

    def update_kv(self, key_states, query_states, value_states):
        kv_cache_len = key_states.shape[-2]
        head_dim = query_states.shape[-1]

        if kv_cache_len < self.budget:
            return key_states, value_states

        attn_weights = fake_compute_attention_scores(query_states, key_states)
        attn_weights_sum = (
            F.softmax(
                attn_weights[:, :, -self.window_size :, : -self.window_size], dim=-1, dtype=torch.float32
            )
            .mean(dim=-2)
            .to(query_states.dtype)
        )
        attn_cache = F.max_pool1d(
            attn_weights_sum, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1
        )
        similarity_cos = fake_cal_similarity(
            key_states, retain_ratio=self.retain_ratio, retain_direction=self.retain_direction
        )[:, :, : -self.window_size]
        final_score = attn_cache * self.mix_lambda - similarity_cos * (1 - self.mix_lambda)
        indices = final_score.topk(self.budget - self.window_size, dim=-1).indices

        if self.record_kept_token_indices:
            indices_cl = indices.clone().squeeze(0).to("cpu")
            recent_window_indices = torch.arange(kv_cache_len - self.window_size, kv_cache_len).expand(
                indices_cl.shape[0], -1
            )
            cur_indices = torch.cat([indices_cl, recent_window_indices], dim=-1)
            if self.evicted_token_num > 0:
                prev_indices = self.kept_token_indices[-1]
                mask = cur_indices < self.budget
                for i in range(cur_indices.shape[0]):
                    positions = torch.where(mask[i])[0]
                    for pos in positions:
                        val = cur_indices[i, pos].item()
                        cur_indices[i, pos] = prev_indices[i, val]
                cur_indices[~mask] += self.evicted_token_num
            self.kept_token_indices.append(cur_indices)
            self.evicted_token_num += kv_cache_len - self.budget

        gather_indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        k_past_compress = key_states[:, :, : -self.window_size, :].gather(dim=2, index=gather_indices)
        v_past_compress = value_states[:, :, : -self.window_size, :].gather(dim=2, index=gather_indices)
        k_cur = key_states[:, :, -self.window_size :, :]
        v_cur = value_states[:, :, -self.window_size :, :]
        return torch.cat([k_past_compress, k_cur], dim=2), torch.cat([v_past_compress, v_cur], dim=2)


def install_fake_rkv_compression_module(monkeypatch) -> None:
    """Inject a fake `rkv`/`rkv.compression` module pair into `sys.modules`
    so `kvcot.discovery.capture`'s deferred `from rkv.compression import
    cal_similarity, compute_attention_scores` resolves to this file's fake
    (formula-identical) implementations instead of requiring the real,
    sometimes-uninitialized pinned submodule."""
    fake_rkv = types.ModuleType("rkv")
    fake_compression = types.ModuleType("rkv.compression")
    fake_compression.cal_similarity = fake_cal_similarity
    fake_compression.compute_attention_scores = fake_compute_attention_scores
    fake_rkv.compression = fake_compression
    monkeypatch.setitem(sys.modules, "rkv", fake_rkv)
    monkeypatch.setitem(sys.modules, "rkv.compression", fake_compression)
