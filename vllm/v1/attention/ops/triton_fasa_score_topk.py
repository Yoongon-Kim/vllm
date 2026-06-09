# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FASA-fc score + top-K — paper-faithful "fetch the dominant FCs each step".

Unlike LRoSA (which stores a projected ``proj_K`` in the combined slot) and
unlike the fasa_fc_cached optimization (which caches the selected channels),
the paper's fasa_fc reads the I_dom frequency-component channels straight out
of the FULL K cache on every decode step — no auxiliary cache. The combined
slot here is just ``[K | V]`` (2*head_size, no proj_K region).

Score for a (request r, kv-head h, cached token t):
    s[r,h,t] = Σ_{c in I_dom[h]} q[r,h,c] * K[t,h,c]
where I_dom[h] is the set of ``2*n_tip`` raw channels (the n_tip RoPE pairs
selected at calibration: channels {2*fc, 2*fc+1}). q is the GQA group-mean
query (reduced to per-kv-head) gathered at those channels (``q_sub``); K is
read from the K region (slot[:head_size]) at the scattered channel offsets.

Mirrors triton_lrosa_score_topk: same masking / window / radix-topk reuse,
only the score inner product changes from a contiguous proj_K dot to a
channel-gathered dot.
"""

import torch

from vllm.triton_utils import tl, triton

# Reuse LRoSA's radix top-K (DSA kernel) + the score-buffer top-K plumbing.
from vllm.v1.attention.ops.triton_lrosa_score_topk import (
    _radix_topk,
    _radix_topk_available,
)


@triton.jit
def _fasa_score_kernel(
    q_kv_ptr,  # [num_reqs, H_kv, head_size]  group-mean (per-kv-head) query
    kv_cache_ptr,  # [num_blocks, block_size, H_kv, slot_size]  ([K|V])
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens_ptr,  # [num_reqs]
    ch_ptr,  # [H_kv, n_ch] int32 — raw K channel offsets in [0, head_size)
    scores_ptr,  # [num_reqs, H_kv, max_kv_len]  fp32
    max_kv_len,
    n_ch,
    block_size,
    q_kv_stride_r,
    q_kv_stride_h,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    ch_stride_h,
    scores_stride_r,
    scores_stride_h,
    window,
    recent_w,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + pid_r)
    # Sliding-window restriction identical to LRoSA: caller passes window >=
    # max context for full attention (lower bound is then a no-op).
    in_seq = (t_offs < seq_len) & (t_offs >= seq_len - window)
    in_range = t_offs < max_kv_len

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < n_ch

    # I_dom channel offsets for this kv-head. Gather BOTH q and K at these
    # scattered channels (reading q directly from the group-mean query avoids a
    # separate pre-gather + int64 index buffer, which was CUDA-graph-unsafe).
    ch = tl.load(ch_ptr + pid_h * ch_stride_h + c_offs, mask=c_mask, other=0)
    qs = tl.load(
        q_kv_ptr + pid_r * q_kv_stride_r + pid_h * q_kv_stride_h + ch,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    block_idx = t_offs // block_size
    pos_in_block = t_offs % block_size
    bt_offset = pid_r * bt_stride_r + block_idx
    # int64: physical block ids overflow int32 cache-offset arithmetic once the
    # KV cache fills large memory -> intermittent OOB at long context.
    block_id = tl.load(block_table_ptr + bt_offset, mask=in_seq, other=0).to(tl.int64)

    # K region starts at offset 0 within the slot; gather the I_dom channels.
    K_base = (
        block_id * cache_stride_block
        + pos_in_block * cache_stride_pos
        + pid_h * cache_stride_head
    )
    K_vals = tl.load(
        kv_cache_ptr + K_base[:, None] + ch[None, :],
        mask=in_seq[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    score = tl.sum(K_vals * qs[None, :], axis=1)  # [BLOCK_T]
    # Recency-keep: bias last recent_w in-seq positions into the top-K (mirrors
    # LRoSA so FASA gets the same recency prior — fair comparison). recent_w=0 →
    # predicate contradicts in_seq → no-op.
    recent_keep = (t_offs >= seq_len - recent_w) & in_seq
    score = tl.where(recent_keep, score + 1e4, score)
    score = tl.where(in_seq, score, float("-inf"))

    out_off = pid_r * scores_stride_r + pid_h * scores_stride_h + t_offs
    tl.store(scores_ptr + out_off, score, mask=in_range)


def fasa_score(
    q_kv: torch.Tensor,  # (num_reqs, H_kv, head_size)  group-mean query
    kv_cache: torch.Tensor,  # (num_blocks, block_size, H_kv, slot_size)
    block_table: torch.Tensor,  # (num_reqs, max_blocks) int32
    seq_lens: torch.Tensor,  # (num_reqs,) int32
    ch: torch.Tensor,  # (H_kv, n_ch) int32 — raw channel offsets
    block_t: int = 64,
    scores_out: torch.Tensor | None = None,
    window: int = 0,
    recent_w: int = 0,
) -> torch.Tensor:
    """Score every cached position via the I_dom channel-gathered dot.
    ``window`` > 0 restricts to the last ``window`` positions; 0 = full."""
    num_reqs, H_kv, _ = q_kv.shape
    n_ch = ch.shape[1]
    block_size = kv_cache.shape[1]
    max_blocks = block_table.shape[1]
    max_kv_len = max_blocks * block_size

    if scores_out is None:
        scores = torch.empty(
            (num_reqs, H_kv, max_kv_len), dtype=torch.float32, device=q_kv.device
        )
    else:
        scores = scores_out[:num_reqs, :, :max_kv_len]

    BLOCK_C = triton.next_power_of_2(n_ch)
    BLOCK_T = block_t
    grid = (num_reqs, H_kv, triton.cdiv(max_kv_len, BLOCK_T))
    win_eff = window if window > 0 else (max_kv_len + 1)

    _fasa_score_kernel[grid](
        q_kv,
        kv_cache,
        block_table,
        seq_lens,
        ch,
        scores,
        max_kv_len,
        n_ch,
        block_size,
        q_kv.stride(0),
        q_kv.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        ch.stride(0),
        scores.stride(0),
        scores.stride(1),
        win_eff,
        recent_w,
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
    )
    return scores


def fasa_score_topk(
    q_kv: torch.Tensor,  # (num_reqs, H_kv, head_size)  group-mean query
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    ch: torch.Tensor,  # (H_kv, n_ch) int32
    n_fac: int,
    scores_out: torch.Tensor | None = None,
    top_idx_out: torch.Tensor | None = None,
    top_scores_out: torch.Tensor | None = None,
    use_radix: bool = True,
    window: int = 0,
    recent_w: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FASA-fc score -> top-K. Returns (top_idx, top_scores) shaped
    (num_reqs, H_kv, k), k = min(n_fac, max_kv_len). Mirrors lrosa_score_topk."""
    scores = fasa_score(
        q_kv, kv_cache, block_table, seq_lens, ch,
        scores_out=scores_out, window=window, recent_w=recent_w,
    )
    k = min(n_fac, scores.shape[-1])
    use_radix_eff = use_radix and _radix_topk_available()

    if top_idx_out is not None and top_scores_out is not None:
        num_reqs, H_kv, _ = scores.shape
        if use_radix_eff and top_idx_out.dtype == torch.int32:
            idx_view = _radix_topk(scores, seq_lens, k, idx_out=top_idx_out)
            score_view = top_scores_out[:num_reqs, :, :k]
            torch.gather(scores, -1, idx_view.to(torch.int64), out=score_view)
            return idx_view, score_view
        if top_idx_out.dtype == torch.int32:
            ts, ti = torch.topk(scores, k=k, dim=-1)
            idx_view = top_idx_out[:num_reqs, :, :k]
            score_view = top_scores_out[:num_reqs, :, :k]
            idx_view.copy_(ti.to(torch.int32))
            score_view.copy_(ts)
            return idx_view, score_view
        idx_view = top_idx_out[:num_reqs, :, :k]
        score_view = top_scores_out[:num_reqs, :, :k]
        torch.topk(scores, k=k, dim=-1, out=(score_view, idx_view))
        return idx_view, score_view

    if use_radix_eff:
        top_idx = _radix_topk(scores, seq_lens, k)
        top_scores = torch.gather(scores, -1, top_idx.to(torch.int64))
        return top_idx, top_scores
    top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
    return top_idx.to(torch.int32), top_scores
