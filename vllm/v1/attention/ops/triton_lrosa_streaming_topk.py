# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRoSA streaming top-K (Step 4a, V2 chunk-parallel two-stage).

Ported from pca's ``fasa/triton_kernels.py:fused_score_topk_gather_triton_v2``.
The V1 single-stage variant (one program per (req, kv-head)) was correct
but ~3× slower than the two-pass score+topk path on Blackwell because
only ``num_reqs * H_kv`` SMs are busy — for bsz=1, that's 8 SMs out of
144. V2 fixes that by splitting kv_len into chunks and computing per-
chunk local top-K in parallel, then merging across chunks.

Layout:
  Stage 1 — grid ``(num_reqs, H_kv, MAX_NUM_CHUNKS)``. Each program owns
            one chunk of ``CHUNK_SIZE`` positions, streams the local
            top-K, writes ``N_FAC`` packed (score, idx) uint64 entries
            into a candidates buffer of shape
            ``(num_reqs, H_kv, MAX_NUM_CHUNKS, N_FAC)`` uint64. Chunks
            past ``kv_len`` write NEG_INF_PACKED padding.
  Stage 2 — grid ``(num_reqs, H_kv)``. Each program loads the flat
            ``MAX_NUM_CHUNKS * N_FAC`` candidates and runs one
            ``tl.topk`` to take the global top-``N_FAC``, unpacks, and
            writes ``int32`` indices.

Never materializes the full ``(num_reqs, H_kv, max_kv_len)`` fp32 score
buffer. Memory cost is the candidates buffer ``(max_num_reqs, H_kv,
MAX_NUM_CHUNKS, N_FAC) × 8B`` — at our default CHUNK_SIZE=4096 and
context=40K this is ~10 MB total, vs ~80 MB for the 2-pass score buffer.
"""

import torch

from vllm.triton_utils import tl, triton

# fp32 -inf bits = 0xFF800000; after our flip (negative → ~bits) it
# becomes 0x007FFFFF — the smallest non-zero positive uint32 in our
# encoding, so it sorts to the bottom under tl.topk (descending).
#
# Low 32 bits hold the kv-position index. CRITICAL: for padded slots
# (chunk_start >= seq_len) we use idx=0 rather than 0xFFFFFFFF. Two
# reasons:
#   1. When tl.topk ties on the (NEG_INF, *) score-bits, the low-32-bit
#      tie-breaker decides which entry wins. With idx=0xFFFFFFFF the
#      padded entry beats real in-seq positions whose -inf-masked
#      packing carries idx ∈ [0, kv_len). The result was top_idx
#      containing -1, which the downstream gather then dereferenced as
#      a negative offset into block_table → cudaErrorIllegalAddress in
#      vLLM's full-CG dummy capture (seq_len=1).
#   2. Even when all candidates are padded (seq_len < n_fac × 0), the
#      fallback idx=0 stays in valid range for the gather kernel.
_NEG_INF_FLIPPED_HI = 0x007FFFFF
_NEG_INF_PACKED_PY = (_NEG_INF_FLIPPED_HI << 32) | 0x00000000


@triton.jit
def _pack_score_idx(score, idx):
    """Pack (fp32 score, int32 idx) → uint64 with order-preserving encoding."""
    bits = score.to(tl.uint32, bitcast=True)
    sign = bits & 0x80000000
    flipped = tl.where(sign != 0, bits ^ 0xFFFFFFFF, bits ^ 0x80000000)
    return (flipped.to(tl.uint64) << 32) | (idx.to(tl.uint32).to(tl.uint64))


@triton.jit
def _unpack_idx(packed):
    return (packed & tl.cast(0xFFFFFFFF, tl.uint64)).to(tl.int32)


@triton.jit
def _stage1_local_topk_kernel(
    proj_q_ptr,  # [num_reqs, H_kv, cs_h]
    kv_cache_ptr,  # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens_ptr,  # [num_reqs]
    candidates_ptr,  # [num_reqs, H_kv, MAX_NUM_CHUNKS, N_FAC] uint64
    head_size,
    block_size,
    proj_q_stride_r,
    proj_q_stride_h,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    cand_stride_r,
    cand_stride_h,
    cand_stride_chunk,
    CS_H: tl.constexpr,
    N_FAC: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NEG_INF_PACKED: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_chunk = tl.program_id(2)
    chunk_start = pid_chunk * CHUNK_SIZE

    out_offs = tl.arange(0, N_FAC)
    out_ptrs = (
        candidates_ptr
        + pid_r * cand_stride_r
        + pid_h * cand_stride_h
        + pid_chunk * cand_stride_chunk
        + out_offs
    )

    seq_len = tl.load(seq_lens_ptr + pid_r)
    # Chunks entirely past seq_len skip the streaming loop and emit
    # NEG_INF_PACKED padding (which has idx=0 — see the constant's
    # docstring). Without this short-circuit, the inner loop would
    # produce candidates carrying ``t_offs`` values that overshoot the
    # block_table column range when seq_len < max_kv_len (e.g. CG
    # dummy warmup with seq_len=1) — the gather kernel then dereferences
    # block_table[r, t_offs // block_size] OOB.
    if chunk_start >= seq_len:
        pad = tl.full([N_FAC], NEG_INF_PACKED, dtype=tl.uint64)
        tl.store(out_ptrs, pad)
        return

    cs_offs = tl.arange(0, CS_H)
    q = tl.load(
        proj_q_ptr + pid_r * proj_q_stride_r + pid_h * proj_q_stride_h + cs_offs
    ).to(tl.float32)

    running_packed = tl.full([N_FAC], NEG_INF_PACKED, dtype=tl.uint64)

    for tile_local in range(0, CHUNK_SIZE, BLOCK_T):
        tile_start = chunk_start + tile_local
        t_offs = tile_start + tl.arange(0, BLOCK_T)
        in_seq = t_offs < seq_len

        block_idx = t_offs // block_size
        pos_in_block = t_offs % block_size
        # int64: physical block ids overflow int32 cache-offset arithmetic once
        # the KV cache fills large memory -> intermittent OOB at long context.
        block_id = tl.load(
            block_table_ptr + pid_r * bt_stride_r + block_idx,
            mask=in_seq,
            other=0,
        ).to(tl.int64)

        proj_K_base = (
            block_id * cache_stride_block
            + pos_in_block * cache_stride_pos
            + pid_h * cache_stride_head
            + 2 * head_size
        )
        proj_K = tl.load(
            kv_cache_ptr + proj_K_base[:, None] + cs_offs[None, :],
            mask=in_seq[:, None],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(proj_K * q[None, :], axis=1)
        scores = tl.where(in_seq, scores, -float("inf"))

        tile_packed = _pack_score_idx(scores, t_offs.to(tl.int32))
        merged = tl.cat(running_packed, tile_packed, can_reorder=True)
        running_packed = tl.topk(merged, N_FAC, dim=0)

    tl.store(out_ptrs, running_packed)


@triton.jit
def _stage2_merge_kernel(
    candidates_ptr,  # [num_reqs, H_kv, MAX_NUM_CHUNKS, N_FAC] uint64
    top_idx_ptr,  # [num_reqs, H_kv, N_FAC] int32
    cand_stride_r,
    cand_stride_h,
    top_stride_r,
    top_stride_h,
    N_FAC: tl.constexpr,
    MERGE_SIZE: tl.constexpr,  # MAX_NUM_CHUNKS * N_FAC
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)

    cand_offs = tl.arange(0, MERGE_SIZE)
    cand_ptrs = (
        candidates_ptr + pid_r * cand_stride_r + pid_h * cand_stride_h + cand_offs
    )
    cands = tl.load(cand_ptrs)
    top_packed = tl.topk(cands, N_FAC, dim=0)
    final_idx = _unpack_idx(top_packed)

    idx_offs = tl.arange(0, N_FAC)
    tl.store(
        top_idx_ptr + pid_r * top_stride_r + pid_h * top_stride_h + idx_offs,
        final_idx,
    )


def lrosa_streaming_topk(
    proj_q: torch.Tensor,  # (num_reqs, H_kv, cs_h)
    kv_cache: torch.Tensor,  # (num_blocks, block_size, H_kv, slot_size)
    block_table: torch.Tensor,  # (num_reqs, max_blocks)
    seq_lens: torch.Tensor,  # (num_reqs,)
    n_fac: int,
    head_size: int,
    cs_h: int,
    candidates_buf: torch.Tensor,  # (>=num_reqs, H_kv, MAX_NUM_CHUNKS, N_FAC) uint64
    chunk_size: int,
    top_idx_out: torch.Tensor | None = None,
    block_t: int = 256,
) -> torch.Tensor:
    """Streaming top-K (V2: chunk-parallel two-stage). Returns
    ``(num_reqs, H_kv, n_fac)`` int32 indices, equivalent to a top-K on
    ``score = proj_q · proj_K^T`` masked to ``seq_lens[r]`` positions.

    Caller owns the candidates buffer (sized at engine init in the
    LRoSA backend's MetadataBuilder); this avoids per-step allocations
    and keeps the pointer stable across CUDA-Graph captures.
    """
    num_reqs, H_kv, cs_h_q = proj_q.shape
    assert cs_h_q == cs_h

    # MAX_NUM_CHUNKS is the candidates buffer's third dim; it must be
    # large enough for the worst-case kv_len in this engine. The wrapper
    # asserts it, the kernel sees it as a constexpr.
    max_num_chunks = candidates_buf.shape[2]
    assert (max_num_chunks * n_fac & (max_num_chunks * n_fac - 1)) == 0, (
        f"MAX_NUM_CHUNKS * N_FAC must be a power of 2 (got "
        f"{max_num_chunks} * {n_fac} = {max_num_chunks * n_fac})"
    )
    assert (n_fac & (n_fac - 1)) == 0 and n_fac > 0
    assert (chunk_size & (chunk_size - 1)) == 0 and chunk_size > 0
    assert (block_t & (block_t - 1)) == 0 and block_t > 0
    assert (cs_h & (cs_h - 1)) == 0 and cs_h > 0
    # Stage 1 merges the running top-N_FAC with each BLOCK_T tile via
    # ``tl.cat(running[N_FAC], tile[BLOCK_T])`` -> ``tl.topk(.., N_FAC)``; Triton
    # block shapes (and tl.topk inputs) must be powers of 2, so N_FAC + BLOCK_T
    # must be a power of 2. With a pow2 N_FAC that forces BLOCK_T == N_FAC. This
    # is fine for the LongBench regime (n_fac=256 -> block_t=256, merge 512), but
    # the reasoning regime (n_fac=2048) would need block_t=2048 whose proj_K tile
    # (2048 x cs_h fp32) overflows shared memory — streaming there needs a
    # sub-tiled-accumulate redesign (and reads in-slot bf16, so it does not beat
    # the fp8/contig score + radix path on bandwidth-rich GPUs anyway). Fail
    # loudly here instead of crashing inside the kernel with a cryptic
    # "Shape element 0 must be a power of 2".
    nfbt = n_fac + block_t
    assert (nfbt & (nfbt - 1)) == 0, (
        f"streaming top-K needs n_fac + block_t to be a power of 2 (the stage-1 "
        f"cat+topk width); got n_fac={n_fac} + block_t={block_t} = {nfbt}. "
        f"Set block_t={n_fac} (smem-permitting) or use the radix/contig score "
        f"path for this n_fac."
    )

    block_size = kv_cache.shape[1]

    if top_idx_out is None:
        top_idx = torch.empty(
            (num_reqs, H_kv, n_fac),
            dtype=torch.int32,
            device=proj_q.device,
        )
    else:
        assert top_idx_out.dtype == torch.int32
        top_idx = top_idx_out[:num_reqs, :, :n_fac]

    cand_view = candidates_buf[:num_reqs]

    # Stage 1: chunk-parallel local top-K.
    grid1 = (num_reqs, H_kv, max_num_chunks)
    _stage1_local_topk_kernel[grid1](
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        cand_view,
        head_size,
        block_size,
        proj_q.stride(0),
        proj_q.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        cand_view.stride(0),
        cand_view.stride(1),
        cand_view.stride(2),
        CS_H=cs_h,
        N_FAC=n_fac,
        CHUNK_SIZE=chunk_size,
        BLOCK_T=block_t,
        NEG_INF_PACKED=_NEG_INF_PACKED_PY,
    )

    # Stage 2: cross-chunk merge.
    grid2 = (num_reqs, H_kv)
    _stage2_merge_kernel[grid2](
        cand_view,
        top_idx,
        cand_view.stride(0),
        cand_view.stride(1),
        top_idx.stride(0),
        top_idx.stride(1),
        N_FAC=n_fac,
        MERGE_SIZE=max_num_chunks * n_fac,
    )
    return top_idx


def alloc_candidates_buf(
    max_num_reqs: int,
    num_kv_heads: int,
    max_num_chunks: int,
    n_fac: int,
    device: torch.device,
) -> torch.Tensor:
    """Pre-allocate the candidates buffer for the streaming V2 kernel.

    Shape: ``(max_num_reqs, num_kv_heads, max_num_chunks, n_fac)`` uint64.
    Caller is expected to ``torch._dynamo.mark_static_address`` this so
    inductor's CG path treats it as stable.
    """
    return torch.empty(
        (max_num_reqs, num_kv_heads, max_num_chunks, n_fac),
        dtype=torch.uint64,
        device=device,
    )
