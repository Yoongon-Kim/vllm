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
    window,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + pid_r)
    # Sliding-window restriction: only positions in [seq_len - window, seq_len)
    # are eligible (older ones masked to -inf so they're never selected). The
    # caller passes a window >= max context for full-attention layers, making
    # ``seq_len - window`` <= 0 so the lower bound is a no-op.
    in_seq = (t_offs < seq_len) & (t_offs >= seq_len - window)
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
    # int64: physical block ids can exceed 2^31 / cache_stride_block (~58k for
    # an lrosa slot) once the KV cache fills B200-class memory, overflowing
    # int32 address arithmetic into an OOB read (data-dependent on which blocks
    # the allocator hands out → intermittent illegal-access at long context).
    block_id = tl.load(block_table_ptr + bt_offset, mask=in_seq, other=0).to(tl.int64)

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


@triton.jit
def _lrosa_score_contig_kernel(
    proj_q_ptr,   # [num_reqs, H_kv, cs_h]
    projk_ptr,    # [num_blocks, block_size, H_kv, cs_h]  CONTIGUOUS proj_K cache
    block_table_ptr,
    seq_lens_ptr,
    scores_ptr,
    max_kv_len,
    cs_h,
    block_size,
    proj_q_stride_r, proj_q_stride_h,
    pk_stride_block, pk_stride_pos, pk_stride_head,
    bt_stride_r,
    scores_stride_r, scores_stride_h,
    window,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Same score as _lrosa_score_kernel but proj_K lives in its own contiguous
    cache (cs_h-wide slot) so consecutive positions' proj_K are adjacent in
    memory → coalesced read, no strided cache-line waste from the [K|V|proj_K]
    interleave."""
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_t = tl.program_id(2)

    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + pid_r)
    in_seq = (t_offs < seq_len) & (t_offs >= seq_len - window)
    in_range = t_offs < max_kv_len

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h
    pq = tl.load(
        proj_q_ptr + pid_r * proj_q_stride_r + pid_h * proj_q_stride_h + c_offs,
        mask=c_mask, other=0.0,
    ).to(tl.float32)

    block_idx = t_offs // block_size
    pos_in_block = t_offs % block_size
    block_id = tl.load(
        block_table_ptr + pid_r * bt_stride_r + block_idx, mask=in_seq, other=0
    ).to(tl.int64)
    proj_K_base = (
        block_id * pk_stride_block
        + pos_in_block * pk_stride_pos
        + pid_h * pk_stride_head
    )
    proj_K = tl.load(
        projk_ptr + proj_K_base[:, None] + c_offs[None, :],
        mask=in_seq[:, None] & c_mask[None, :], other=0.0,
    ).to(tl.float32)

    score = tl.sum(proj_K * pq[None, :], axis=1)
    score = tl.where(in_seq, score, float("-inf"))
    out_off = pid_r * scores_stride_r + pid_h * scores_stride_h + t_offs
    tl.store(scores_ptr + out_off, score, mask=in_range)


@triton.jit
def _lrosa_score_layer_kernel(
    proj_q_ptr,      # [num_reqs, cs_h_layer]  single per-layer query proj
    kv_cache_ptr,    # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr, # [num_reqs, max_blocks]
    seq_lens_ptr,    # [num_reqs]
    scores_ptr,      # [num_reqs, max_kv_len]  fp32 (single head dim)
    max_kv_len,
    head_size,
    cs_h_slot,       # cs_h_layer // H_kv
    block_size,
    proj_q_stride_r,
    cache_stride_block, cache_stride_pos, cache_stride_head,
    bt_stride_r,
    scores_stride_r,
    H_KV: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,   # >= cs_h_slot
):
    pid_r = tl.program_id(0)
    pid_t = tl.program_id(1)
    t_offs = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    seq_len = tl.load(seq_lens_ptr + pid_r)
    in_seq = t_offs < seq_len
    in_range = t_offs < max_kv_len

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h_slot

    block_idx = t_offs // block_size
    pos_in_block = t_offs % block_size
    block_id = tl.load(block_table_ptr + pid_r * bt_stride_r + block_idx,
                       mask=in_seq, other=0).to(tl.int64)  # int64: avoid OOB overflow

    # score[t] = sum over heads of (proj_K_slot[h] · proj_q_layer[h-chunk]).
    # proj_q_layer is split the same way proj_K was scattered at store time:
    # head h owns proj_q_layer[h*cs_h_slot : (h+1)*cs_h_slot].
    score = tl.zeros([BLOCK_T], dtype=tl.float32)
    for h in range(H_KV):
        proj_K_base = (block_id * cache_stride_block
                       + pos_in_block * cache_stride_pos
                       + h * cache_stride_head + 2 * head_size)
        pK = tl.load(kv_cache_ptr + proj_K_base[:, None] + c_offs[None, :],
                     mask=in_seq[:, None] & c_mask[None, :], other=0.0).to(tl.float32)
        pq_h = tl.load(proj_q_ptr + pid_r * proj_q_stride_r + h * cs_h_slot + c_offs,
                       mask=c_mask, other=0.0).to(tl.float32)
        score += tl.sum(pK * pq_h[None, :], axis=1)
    score = tl.where(in_seq, score, float("-inf"))
    tl.store(scores_ptr + pid_r * scores_stride_r + t_offs, score, mask=in_range)


def lrosa_score_layer(
    proj_q_layer: torch.Tensor,  # (num_reqs, cs_h_layer)
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    head_size: int,
    cs_h_slot: int,
    H_kv: int,
    scores_out: torch.Tensor | None = None,
    partial_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-layer CONCAT score → single score row per request (head-shared).

    Reuses the per-kv-head score kernel for H_kv-parallel throughput: the
    per-layer query projection splits the same way proj_K was scattered at
    store time, so reshaping proj_q_layer to (num_reqs, H_kv, cs_h_slot)
    makes it a drop-in for the per-head kernel. That produces a per-head
    partial[r, h, t] = proj_q[r,h] · proj_K_slot[r,h,t]; summing over h
    recovers the full per-layer score Σ_c proj_q_layer·proj_K_layer.

    The dedicated _lrosa_score_layer_kernel (single program loops over
    H_kv) is left in place as a reference but no longer the default — it
    launched H_kv× fewer programs and was ~2.4% slower at 32K.
    """
    num_reqs = proj_q_layer.shape[0]
    max_blocks = block_table.shape[1]
    max_kv_len = max_blocks * kv_cache.shape[1]
    # (num_reqs, H_kv, cs_h_slot) view — head h owns the c-slice of proj_q.
    proj_q_heads = proj_q_layer.view(num_reqs, H_kv, cs_h_slot).contiguous()
    if partial_out is not None:
        partial = partial_out[:num_reqs]
    else:
        partial = torch.empty(num_reqs, H_kv, max_kv_len, dtype=torch.float32,
                              device=proj_q_layer.device)
    # Per-head partial scores via the existing H_kv-parallel kernel.
    _launch_score_kernel(
        proj_q_heads, kv_cache, block_table, seq_lens, partial,
        head_size, cs_h_slot, block_t=128,
    )
    # Sum over heads → per-layer score. Out-of-seq positions are -inf in
    # every head, so their sum stays -inf (no +inf, so no NaN).
    scores = partial.sum(dim=1)  # (num_reqs, max_kv_len)
    if scores_out is not None:
        scores_out[:num_reqs].copy_(scores)
        return scores_out[:num_reqs]
    return scores


def _launch_score_kernel(
    proj_q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scores: torch.Tensor,
    head_size: int,
    cs_h: int,
    block_t: int,
    window: int = 0,
) -> None:
    num_reqs, H_kv, _ = proj_q.shape
    block_size = kv_cache.shape[1]
    max_kv_len = scores.shape[-1]

    BLOCK_C = triton.next_power_of_2(cs_h)
    BLOCK_T = block_t
    grid = (num_reqs, H_kv, triton.cdiv(max_kv_len, BLOCK_T))
    # window<=0 means full attention: a sentinel > any seq_len makes the
    # ``t_offs >= seq_len - window`` lower bound always true (no restriction).
    win_eff = window if window > 0 else (max_kv_len + 1)

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
        win_eff,
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
    window: int = 0,
    projk_cache: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute scores for every cached KV position.

    Returns an fp32 tensor of shape ``(num_reqs, H_kv, max_kv_len)`` where
    positions ≥ ``seq_lens[r]`` are -inf. When ``scores_out`` is passed,
    writes into that buffer and returns it (decode hot-path); otherwise
    allocates a fresh tensor (test/eager path).

    When ``projk_cache`` is given, proj_K is read from that separate contiguous
    cache (cs_h-wide slot) instead of the interleaved [K|V|proj_K] kv slot —
    coalesced scan, ~1.4x faster at long context.
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

    if projk_cache is not None:
        BLOCK_C = triton.next_power_of_2(cs_h)
        BLOCK_T = block_t
        grid = (num_reqs, H_kv, triton.cdiv(max_kv_len, BLOCK_T))
        win_eff = window if window > 0 else (max_kv_len + 1)
        _lrosa_score_contig_kernel[grid](
            proj_q, projk_cache, block_table, seq_lens, scores,
            max_kv_len, cs_h, block_size,
            proj_q.stride(0), proj_q.stride(1),
            projk_cache.stride(0), projk_cache.stride(1), projk_cache.stride(2),
            block_table.stride(0), scores.stride(0), scores.stride(1),
            win_eff, BLOCK_T=BLOCK_T, BLOCK_C=BLOCK_C,
        )
        return scores

    _launch_score_kernel(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        scores,
        head_size,
        cs_h,
        block_t,
        window=window,
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


# Shape-keyed persistent scratch for the radix path. Fresh torch.empty /
# .contiguous() / arange / where ALLOCATIONS inside the CUDA-graph-captured
# decode path corrupt at bsz>1 (each captured batch size allocates from the
# graph memory pool; addresses get reused/overlap → illegal memory access,
# observed only at bsz>=2 long-context where the graph is replayed). Reusing a
# per-shape persistent buffer gives a stable address that stays valid across
# capture+replay. Keyed by shape so distinct capture batch sizes never alias.
_RADIX_SCRATCH: dict = {}


def _radix_scratch(name, shape, dtype, device):
    key = (name, tuple(shape), dtype, device)
    t = _RADIX_SCRATCH.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _RADIX_SCRATCH[key] = t
    return t


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

    All scratch comes from the per-shape persistent cache (no allocation in
    the captured decode path) so it is CUDA-graph safe at any batch size.
    """
    num_reqs, H_kv, max_kv = scores.shape
    rows = num_reqs * H_kv
    dev = scores.device
    # Row-major logits copy into a persistent buffer (scores may be a strided
    # slice of the static scores_buf; copy_ handles the strides).
    logits2d = _radix_scratch("logits", (rows, max_kv), torch.float32, dev)
    logits2d.view(num_reqs, H_kv, max_kv).copy_(scores)
    # seq_lens per row: each request's length repeated for its H_kv heads.
    seq_lens_rows = _radix_scratch("seqrows", (rows,), torch.int32, dev)
    seq_lens_rows.view(num_reqs, H_kv).copy_(
        seq_lens.to(torch.int32).view(num_reqs, 1).expand(num_reqs, H_kv)
    )
    topk_idx = _radix_scratch("topk", (rows, k), torch.int32, dev)
    next_n = 1  # LRoSA decode: one query token per step (no spec-decode)
    # top_k_per_row_decode picks its kernel by row count: num_rows > 32 (with
    # >=128KB smem, i.e. Blackwell) takes FilteredTopKRaggedTransform, which has
    # an intermittent illegal-memory-access bug at large row counts; num_rows<=32
    # takes the (correct, CUDA-graph-safe) cooperative persistent_topk path.
    # bsz=1 has rows = H_kv (<=32) so it always hit the good path; bsz>=4 with
    # H_kv=8 (rows>=32) fell into the buggy one. Chunk the rows to <=32 so every
    # call uses the persistent path. The chunk count is fixed per captured batch
    # size, so the Python loop is CUDA-graph capturable.
    _RADIX_ROW_CHUNK = 32
    if rows <= _RADIX_ROW_CHUNK:
        torch.ops._C.top_k_per_row_decode(
            logits2d, next_n, seq_lens_rows, topk_idx, rows,
            logits2d.stride(0), logits2d.stride(1), k,
        )
    else:
        for r0 in range(0, rows, _RADIX_ROW_CHUNK):
            r1 = min(r0 + _RADIX_ROW_CHUNK, rows)
            torch.ops._C.top_k_per_row_decode(
                logits2d[r0:r1], next_n, seq_lens_rows[r0:r1], topk_idx[r0:r1],
                r1 - r0, logits2d.stride(0), logits2d.stride(1), k,
            )
    # When seq_len < k the radix kernel pads unfilled slots with -1. A -1 is
    # an OOB access downstream (score gather + block-table gather). DSA's MLA
    # sparse kernel treats -1 as a skip marker, but our flash_attn has no skip
    # semantics, so we must hand it valid indices.
    #
    # The earlier scheme filled with ``seq_len + offset`` (distinct, in-range)
    # on the assumption those positions score ~-inf and contribute ~0 to the
    # softmax. That holds for the *selection* score but NOT for the gathered
    # attention: flash_attn recomputes q·K on the gathered K/V, and seq_len +
    # offset points past seq_len into UNWRITTEN cache slots — whatever stale
    # bytes live there become real (non-masked) attention logits, corrupting
    # the decode output whenever the slot isn't zero (observed as degenerate
    # short-prompt decode on Ministral; Llama/Qwen3 only survived because their
    # unwritten slots happened to be zero).
    #
    # Instead cycle the padding through the REAL written tokens [0, seq_len):
    # every cached token is duplicated ~uniformly, and equal multiplicity
    # cancels in the softmax normalizer (exact when k % seq_len == 0, near-exact
    # otherwise), so kv_len < n_fac decode reduces to honest dense attention
    # over all real tokens — no garbage rows. All tensor ops, no host sync, so
    # CUDA-Graph safe (the previous seq_lens.min() Python gate broke capture
    # with cudaErrorStreamCaptureInvalidated).
    # All persistent-scratch + out= ops (no allocation → CUDA-graph safe).
    slot_off = _radix_scratch("slotoff", (k,), torch.int32, dev)
    torch.arange(k, out=slot_off)
    sl = _radix_scratch("sl", (rows, 1), torch.int32, dev)  # avoid modulo-by-zero
    torch.clamp(seq_lens_rows.view(rows, 1), min=1, out=sl)
    fill = _radix_scratch("fill", (rows, k), torch.int32, dev)
    torch.remainder(slot_off.view(1, k).expand(rows, k), sl, out=fill)  # [rows,k]
    # Treat an index as invalid if it is negative (the radix -1 pad) OR >=
    # seq_len. The latter guards a heisenbug: the DSA radix kernel intermittently
    # emits a stale out-of-range *positive* index (only when kernels run async —
    # CUDA_LAUNCH_BLOCKING=1 serializes and hides it). The old code rewrote only
    # negatives, so a too-large index slipped through into the downstream K/V and
    # block-table gather as a real OOB read; whether it faulted depended on the
    # caching allocator's page layout (timing-dependent → looked intermittent,
    # worse under profiler/concurrent memory pressure). Cycling every invalid
    # slot through a real token [0, seq_len) makes the gather OOB-proof regardless
    # of what radix returns. All scratch + out= → CUDA-graph safe.
    neg = _radix_scratch("neg", (rows, k), torch.bool, dev)
    torch.lt(topk_idx, 0, out=neg)
    oob = _radix_scratch("oob", (rows, k), torch.bool, dev)
    torch.ge(topk_idx, seq_lens_rows.view(rows, 1), out=oob)
    torch.logical_or(neg, oob, out=neg)
    res = _radix_scratch("res", (rows, k), torch.int32, dev)
    torch.where(neg, fill, topk_idx, out=res)
    out3d = res.view(num_reqs, H_kv, k)
    if idx_out is not None:
        idx_out[:num_reqs, :, :k].copy_(out3d)
        return idx_out[:num_reqs, :, :k]
    return out3d


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
    window: int = 0,
    projk_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score → top-K. ``window`` > 0 restricts selection to the last
    ``window`` positions (sliding-window layers); 0 = full attention.

    Returns ``(top_idx, top_scores)`` each shaped ``(num_reqs, H_kv, k)``
    where ``k = min(n_fac, max_kv_len)``. ``top_idx`` is int64 when written
    into a caller-supplied ``top_idx_out`` buffer (which must be int64 —
    ``torch.topk`` requires that for ``out=``); when no buffer is supplied
    the legacy int32 cast is preserved for back-compat with existing tests.

    ``projk_cache`` (optional) routes the score read to a separate contiguous
    proj_K cache (coalesced scan) instead of the interleaved kv slot.
    """
    scores = lrosa_score(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        head_size,
        cs_h,
        scores_out=scores_out,
        window=window,
        projk_cache=projk_cache,
    )
    k = min(n_fac, scores.shape[-1])

    # Radix top-K path (borrowed from DSA csrc/topk.cu). O(seq) vs
    # torch.topk O(seq log seq). Only used when explicitly requested AND
    # the binding is compiled in; otherwise transparently falls back to
    # torch.topk. Radix returns int32 indices, unsorted by score — the
    # downstream gather only needs the index set, not the order.
    #
    # _radix_topk handles the seq_len < k under-fill internally with distinct
    # in-range padding indices (no host sync), so the radix path is always
    # safe — including for short sequences and under CUDA-Graph capture.
    use_radix_eff = use_radix and _radix_topk_available()

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
