# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRoSA score + top-K (Step 3b-2, CG-friendly variant since Step 4b).

Given per-(request, kv_head) projected queries and the paged combined-slot
KV cache, compute ``score[r,h,t] = Σ_c proj_q[r,h,c] · proj_K[r,h,t,c]``
(read from the proj_K region of each slot) for every cached position, mask
positions beyond ``seq_lens`` with -inf, then ``torch.topk`` to get the
sparse selection indices.

This is the two-pass implementation flagged in
``sprightly-gliding-kettle.md`` as the fallback for the eventual streaming
top-K kernel. It materializes a ``(num_reqs, H_kv, max_kv_len)`` fp32 score
tensor; the fused-streaming variant is left as a follow-up (Step 4a).

Step 4b: the helpers accept optional pre-allocated output buffers so the
decode hot-path can avoid ``torch.empty`` calls inside the captured CUDA
graph. The legacy "no-buffer" call signature is preserved for tests.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _lrosa_score_kernel(
    proj_q_ptr,  # [num_reqs, H_kv, cs_h]
    kv_cache_ptr,  # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens_ptr,  # [num_reqs]
    scores_ptr,  # [num_reqs, H_kv, max_kv_len]  fp32
    max_kv_len,
    head_size,
    cs_h,
    block_size,
    proj_q_stride_r,
    proj_q_stride_h,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    scores_stride_r,
    scores_stride_h,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + pid_r)
    in_seq = t_offs < seq_len
    in_range = t_offs < max_kv_len

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h

    # proj_q[r, h, :] — single vector per program.
    pq = tl.load(
        proj_q_ptr + pid_r * proj_q_stride_r + pid_h * proj_q_stride_h + c_offs,
        mask=c_mask,
        other=0.0,
    ).to(tl.float32)

    block_idx = t_offs // block_size
    pos_in_block = t_offs % block_size
    bt_offset = pid_r * bt_stride_r + block_idx
    block_id = tl.load(block_table_ptr + bt_offset, mask=in_seq, other=0)

    # Slot base for proj_K region: skip K (head_size) + V (head_size).
    proj_K_base = (
        block_id * cache_stride_block
        + pos_in_block * cache_stride_pos
        + pid_h * cache_stride_head
        + 2 * head_size
    )

    proj_K = tl.load(
        kv_cache_ptr + proj_K_base[:, None] + c_offs[None, :],
        mask=in_seq[:, None] & c_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    score = tl.sum(proj_K * pq[None, :], axis=1)  # [BLOCK_T]
    score = tl.where(in_seq, score, float("-inf"))

    out_off = pid_r * scores_stride_r + pid_h * scores_stride_h + t_offs
    tl.store(scores_ptr + out_off, score, mask=in_range)


def _launch_score_kernel(
    proj_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scores: torch.Tensor,
    head_size: int,
    cs_h: int,
    block_t: int,
) -> None:
    num_reqs, H_kv, _ = proj_q.shape
    block_size = kv_cache.shape[1]
    max_kv_len = scores.shape[-1]

    BLOCK_C = triton.next_power_of_2(cs_h)
    BLOCK_T = block_t
    grid = (num_reqs, H_kv, triton.cdiv(max_kv_len, BLOCK_T))

    _lrosa_score_kernel[grid](
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        scores,
        max_kv_len,
        head_size,
        cs_h,
        block_size,
        proj_q.stride(0),
        proj_q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        scores.stride(0),
        scores.stride(1),
        BLOCK_T=BLOCK_T,
        BLOCK_C=BLOCK_C,
    )


def lrosa_score(
    proj_q: torch.Tensor,  # (num_reqs, H_kv, cs_h)  bf16/fp16
    kv_cache: torch.Tensor,  # (num_blocks, block_size, H_kv, slot_size)
    block_table: torch.Tensor,  # (num_reqs, max_blocks)  int32
    seq_lens: torch.Tensor,  # (num_reqs,)  int32
    head_size: int,
    cs_h: int,
    block_t: int = 64,
    scores_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scores for every cached KV position.

    Returns an fp32 tensor of shape ``(num_reqs, H_kv, max_kv_len)`` where
    positions ≥ ``seq_lens[r]`` are -inf. When ``scores_out`` is passed,
    writes into that buffer and returns it (decode hot-path); otherwise
    allocates a fresh tensor (test/eager path).
    """
    num_reqs, H_kv, cs_h_q = proj_q.shape
    assert cs_h_q == cs_h, f"proj_q.shape[-1] {cs_h_q} != cs_h {cs_h}"
    block_size = kv_cache.shape[1]
    max_blocks = block_table.shape[1]
    max_kv_len = max_blocks * block_size

    if scores_out is None:
        scores = torch.empty(
            (num_reqs, H_kv, max_kv_len),
            dtype=torch.float32,
            device=proj_q.device,
        )
    else:
        assert scores_out.shape[0] >= num_reqs and scores_out.shape[1] == H_kv
        assert scores_out.shape[-1] >= max_kv_len
        scores = scores_out[:num_reqs, :, :max_kv_len]

    _launch_score_kernel(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        scores,
        head_size,
        cs_h,
        block_t,
    )
    return scores


def _radix_topk_available() -> bool:
    """True iff the DSA radix top-K C++ binding is compiled into this build.

    Borrowed from DeepSeek Sparse Attention (csrc/topk.cu): a CTA-persistent
    radix-select that is O(seq) vs torch.topk's O(seq log seq) full sort.
    MLA-independent — operates on a plain (rows, seq) fp32 logits tensor."""
    if not hasattr(torch.ops._C, "top_k_per_row_decode"):
        # The op may not be registered yet (lazy). Importing _custom_ops
        # triggers the torch.library registrations from the compiled _C ext.
        try:
            import vllm._custom_ops  # noqa: F401
        except Exception:
            return False
    return hasattr(torch.ops._C, "top_k_per_row_decode")


def _radix_topk(
    scores: torch.Tensor,       # (num_reqs, H_kv, max_kv_len) fp32
    seq_lens: torch.Tensor,     # (num_reqs,) int32 — valid kv_len per request
    k: int,
    idx_out: torch.Tensor | None = None,  # (num_reqs, H_kv, k) int32
) -> torch.Tensor:
    """Radix top-K over the last dim via DSA's top_k_per_row_decode.

    Flattens (num_reqs, H_kv, seq) → (num_reqs*H_kv, seq) rows, repeats
    seq_lens per kv-head, runs the radix kernel, reshapes back. Returns
    int32 indices (num_reqs, H_kv, k). Indices are NOT guaranteed sorted
    by score (radix select) — fine for gather since order is irrelevant.
    """
    num_reqs, H_kv, max_kv = scores.shape
    rows = num_reqs * H_kv
    logits2d = scores.reshape(rows, max_kv).contiguous()    # kernel needs row-major
    # seq_lens per row: each request's length repeated for its H_kv heads.
    seq_lens_rows = (
        seq_lens.to(torch.int32).view(num_reqs, 1)
        .expand(num_reqs, H_kv).reshape(rows).contiguous()
    )
    # The radix kernel writes a dense (rows, k) int32 output. Always use a
    # fresh contiguous buffer here — writing through a non-contiguous view
    # of a caller buffer (e.g. a [:num_decodes] slice whose last dim is
    # n_fac != k) silently lands in a temporary copy and never reaches the
    # caller's buffer, producing stale indices downstream. We copy into
    # idx_out (if given) after reshaping back to (num_reqs, H_kv, k).
    topk_idx = torch.empty(rows, k, dtype=torch.int32, device=scores.device)
    next_n = 1  # LRoSA decode: one query token per step (no spec-decode)
    torch.ops._C.top_k_per_row_decode(
        logits2d,
        next_n,
        seq_lens_rows,
        topk_idx,
        rows,
        logits2d.stride(0),
        logits2d.stride(1),
        k,
    )
    # When seq_len < k the radix kernel pads unfilled slots with -1. Those
    # slots correspond to non-existent tokens (the row had fewer than k
    # valid positions). torch.topk would instead return real-but-low-score
    # indices in [0, seq_len). Downstream the index feeds both a score
    # gather and the block-table gather; a -1 there is an OOB access. Clamp
    # -1 → 0: token 0 is always valid, already attended, and its duplicate
    # selection is harmless (its true score is unchanged in the SDPA).
    topk_idx.clamp_(min=0)
    topk_idx = topk_idx.view(num_reqs, H_kv, k)
    if idx_out is not None:
        idx_out[:num_reqs, :, :k].copy_(topk_idx)
        return idx_out[:num_reqs, :, :k]
    return topk_idx


def lrosa_score_topk(
    proj_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    n_fac: int,
    head_size: int,
    cs_h: int,
    scores_out: torch.Tensor | None = None,
    top_idx_out: torch.Tensor | None = None,
    top_scores_out: torch.Tensor | None = None,
    use_radix: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score → top-K.

    Returns ``(top_idx, top_scores)`` each shaped ``(num_reqs, H_kv, k)``
    where ``k = min(n_fac, max_kv_len)``. ``top_idx`` is int64 when written
    into a caller-supplied ``top_idx_out`` buffer (which must be int64 —
    ``torch.topk`` requires that for ``out=``); when no buffer is supplied
    the legacy int32 cast is preserved for back-compat with existing tests.
    """
    scores = lrosa_score(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        head_size,
        cs_h,
        scores_out=scores_out,
    )
    k = min(n_fac, scores.shape[-1])

    # Radix top-K path (borrowed from DSA csrc/topk.cu). O(seq) vs
    # torch.topk O(seq log seq). Only used when explicitly requested AND
    # the binding is compiled in; otherwise transparently falls back to
    # torch.topk. Radix returns int32 indices, unsorted by score — the
    # downstream gather only needs the index set, not the order.
    #
    # CRITICAL: radix pads under-filled rows (seq_len < k) with -1, which we
    # clamp to 0. Unlike DSA's MLA sparse kernel (which treats -1 as a skip
    # marker), our downstream flash_attn_varlen has no skip semantics, so a
    # clamped-0 duplicate is attended (n_fac - seq_len) times and corrupts
    # the softmax. torch.topk instead returns distinct in-range padding
    # indices. So: only take the radix path when EVERY row has at least k
    # valid positions (seq_lens.min() >= k); otherwise torch.topk. radix's
    # speed win is in the long-context regime anyway, exactly where this
    # condition holds.
    use_radix_eff = (
        use_radix
        and _radix_topk_available()
        and int(seq_lens.min()) >= k
    )

    if top_idx_out is not None and top_scores_out is not None:
        num_reqs, H_kv, _ = scores.shape
        assert top_scores_out.dtype == torch.float32
        assert (
            top_idx_out.shape[0] >= num_reqs
            and top_idx_out.shape[1] == H_kv
            and top_idx_out.shape[-1] >= k
        ), (
            f"top_idx_out shape {tuple(top_idx_out.shape)} too small for "
            f"({num_reqs}, {H_kv}, {k})"
        )
        if use_radix_eff and top_idx_out.dtype == torch.int32:
            idx_view = _radix_topk(scores, seq_lens, k, idx_out=top_idx_out)
            # Gather the corresponding scores for callers that consume them.
            score_view = top_scores_out[:num_reqs, :, :k]
            torch.gather(scores, -1, idx_view.to(torch.int64), out=score_view)
            return idx_view, score_view
        if top_idx_out.dtype == torch.int32:
            # i32 buffer (radix was requested) but the radix gate declined
            # (e.g. seq_len < k → would need -1 padding our flash_attn can't
            # skip). Fall back to torch.topk, which returns distinct in-range
            # padding indices, but torch.topk(out=) requires int64 — so call
            # it without out= and copy into the i32 buffer.
            ts, ti = torch.topk(scores, k=k, dim=-1)
            idx_view = top_idx_out[:num_reqs, :, :k]
            score_view = top_scores_out[:num_reqs, :, :k]
            idx_view.copy_(ti.to(torch.int32))
            score_view.copy_(ts)
            return idx_view, score_view
        assert top_idx_out.dtype == torch.int64, (
            "top_idx_out must be int64 (torch.topk constraint)"
        )
        idx_view = top_idx_out[:num_reqs, :, :k]
        score_view = top_scores_out[:num_reqs, :, :k]
        torch.topk(scores, k=k, dim=-1, out=(score_view, idx_view))
        return idx_view, score_view

    if use_radix_eff:
        top_idx = _radix_topk(scores, seq_lens, k)              # int32
        top_scores = torch.gather(scores, -1, top_idx.to(torch.int64))
        return top_idx, top_scores

    top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
    return top_idx.to(torch.int32), top_scores
