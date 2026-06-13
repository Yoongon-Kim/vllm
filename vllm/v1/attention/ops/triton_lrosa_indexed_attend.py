"""Gather-free indexed attention for LRoSA decode.

The default LRoSA decode does ``lrosa_gather`` (materialize K_sel/V_sel into a
contiguous [num_decodes * n_fac, H_kv, head_size] buffer) followed by a dense
``flash_attn_varlen_func`` over that buffer. The gather buffer scales with
``num_decodes * n_fac``, so at the throughput operating point (large batch) the
buffer write+read dominates the attention side.

This kernel fuses selection-gather and attention: stream the n_fac selected
tokens directly out of the paged kv cache via ``top_idx`` (online-softmax flash
style, ``tl.dot`` for QK/PV), so the intermediate buffer never exists. K|V are
read from the combined LRoSA slot ([K | V | proj_K]); proj_K is untouched, so
this is independent of the fp8/contig proj_K score path.

Two grids:
  - v2 (``_indexed_attend_kv_kernel``): one program per (request, kv-head). Best
    at high batch where num_decodes*H_kv already saturates the SMs (microbench:
    2.3x over gather+attend at batch 32).
  - v3 split-N (``_split_attend_kernel`` + ``_split_reduce_kernel``): splits the
    n_fac scan across S programs (grid num_decodes*H_kv*S) + an online-softmax
    reduce, so the GPU stays occupied at SMALL batch where v2 under-fills the
    SMs and would regress vs gather+flash (microbench: v2 0.98x but v3 1.50x at
    batch 4). S is chosen adaptively from the SM count.

``lrosa_indexed_attend`` picks v2 vs v3 by batch, so it is a net win across all
batch sizes. Caller gates (see lrosa_attn.py): head_size<=256, set attention,
no sinks/softcap/alibi, steady-state non-partial; everything else falls back to
gather+flash. ``int64`` slot addressing handles num_blocks>~58k (long ctx).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _indexed_attend_kv_kernel(
    q_ptr, kv_cache_ptr, block_table_ptr, top_idx_ptr, o_ptr,
    scale, n_fac, hs, block_size,
    q_sr, q_sh, kv_sb, kv_sp, kv_sh, bt_sr, top_sr, top_sh, o_sr, o_sh,
    G: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # one program per (request, kv_head): the whole group of G q-heads sharing
    # this kv-head is processed together so K,V are read ONCE per kv-head.
    pid_r = tl.program_id(0)
    pid_hk = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dm = d < hs
    g = tl.arange(0, G)
    qh = pid_hk * G + g
    q = tl.load(q_ptr + pid_r * q_sr + qh[:, None] * q_sh + d[None, :],
                mask=dm[None, :], other=0.0).to(tl.float32)
    m_i = tl.full([G], -float("inf"), tl.float32)
    l_i = tl.zeros([G], tl.float32)
    acc = tl.zeros([G, BLOCK_D], tl.float32)
    for start in range(0, n_fac, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        nm = n < n_fac
        tok = tl.load(top_idx_ptr + pid_r * top_sr + pid_hk * top_sh + n,
                      mask=nm, other=0)
        blk = tl.load(block_table_ptr + pid_r * bt_sr + (tok // block_size),
                      mask=nm, other=0)
        base = (blk.to(tl.int64) * kv_sb.to(tl.int64)
                + (tok % block_size).to(tl.int64) * kv_sp.to(tl.int64)
                + pid_hk.to(tl.int64) * kv_sh.to(tl.int64))
        k = tl.load(kv_cache_ptr + base[:, None] + d[None, :],
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        v = tl.load(kv_cache_ptr + base[:, None] + (hs + d[None, :]),
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(nm[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + pid_r * o_sr + qh[:, None] * o_sh + d[None, :],
             o.to(o_ptr.dtype.element_ty), mask=dm[None, :])


@triton.jit
def _split_attend_kernel(
    q_ptr, kv_cache_ptr, block_table_ptr, top_idx_ptr,
    m_ptr, l_ptr, acc_ptr,
    scale, n_fac, hs, block_size, S,
    q_sr, q_sh, kv_sb, kv_sp, kv_sh, bt_sr, top_sr, top_sh,
    m_sr, m_sh, m_ss, a_sr, a_sh, a_ss,
    G: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # grid (num_decodes, H_kv, S): this program owns split pid_s of the n_fac
    # scan for (request pid_r, kv-head pid_hk), producing a partial (m,l,acc).
    pid_r = tl.program_id(0)
    pid_hk = tl.program_id(1)
    pid_s = tl.program_id(2)
    d = tl.arange(0, BLOCK_D)
    dm = d < hs
    g = tl.arange(0, G)
    qh = pid_hk * G + g
    q = tl.load(q_ptr + pid_r * q_sr + qh[:, None] * q_sh + d[None, :],
                mask=dm[None, :], other=0.0).to(tl.float32)
    m_i = tl.full([G], -float("inf"), tl.float32)
    l_i = tl.zeros([G], tl.float32)
    acc = tl.zeros([G, BLOCK_D], tl.float32)
    nblk = (n_fac + BLOCK_N - 1) // BLOCK_N
    per = (nblk + S - 1) // S
    for j in range(0, per):
        tile = pid_s * per + j
        start = tile * BLOCK_N
        n = start + tl.arange(0, BLOCK_N)
        nm = n < n_fac
        tok = tl.load(top_idx_ptr + pid_r * top_sr + pid_hk * top_sh + n,
                      mask=nm, other=0)
        blk = tl.load(block_table_ptr + pid_r * bt_sr + (tok // block_size),
                      mask=nm, other=0)
        base = (blk.to(tl.int64) * kv_sb.to(tl.int64)
                + (tok % block_size).to(tl.int64) * kv_sp.to(tl.int64)
                + pid_hk.to(tl.int64) * kv_sh.to(tl.int64))
        k = tl.load(kv_cache_ptr + base[:, None] + d[None, :],
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        v = tl.load(kv_cache_ptr + base[:, None] + (hs + d[None, :]),
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(nm[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # A split may own only empty tiles (tile >= nblk) -> m_new stays -inf ->
        # exp(-inf-(-inf))=nan. Use a finite m_safe; the empty contributions are
        # 0 (acc/l stay 0, m_i stays -inf so the reduce zeroes this split out).
        m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    tl.store(m_ptr + pid_r * m_sr + qh * m_sh + pid_s * m_ss, m_i)
    tl.store(l_ptr + pid_r * m_sr + qh * m_sh + pid_s * m_ss, l_i)
    tl.store(acc_ptr + pid_r * a_sr + qh[:, None] * a_sh + pid_s * a_ss + d[None, :],
             acc, mask=dm[None, :])


@triton.jit
def _split_reduce_kernel(
    m_ptr, l_ptr, acc_ptr, o_ptr,
    hs, S,
    m_sr, m_sh, m_ss, a_sr, a_sh, a_ss, o_sr, o_sh,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # grid (num_decodes, H_q): merge the S partials for this (request, q-head)
    # with the online-softmax (log-sum-exp) combine.
    pid_r = tl.program_id(0)
    pid_hq = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dm = d < hs
    sm = tl.arange(0, BLOCK_S)
    smask = sm < S
    base_m = pid_r * m_sr + pid_hq * m_sh
    m_s = tl.load(m_ptr + base_m + sm * m_ss, mask=smask, other=-float("inf"))
    l_s = tl.load(l_ptr + base_m + sm * m_ss, mask=smask, other=0.0)
    m_g = tl.max(m_s)
    alpha = tl.where(smask, tl.exp(m_s - m_g), 0.0)
    l_g = tl.sum(l_s * alpha)
    acc_s = tl.load(
        acc_ptr + pid_r * a_sr + pid_hq * a_sh + sm[:, None] * a_ss + d[None, :],
        mask=smask[:, None] & dm[None, :], other=0.0)
    acc_g = tl.sum(acc_s * alpha[:, None], axis=0)
    o = acc_g / l_g
    tl.store(o_ptr + pid_r * o_sr + pid_hq * o_sh + d,
             o.to(o_ptr.dtype.element_ty), mask=dm)


def indexed_split_count(num_decodes, n_kv_heads, n_fac, sm_count, block_n=128):
    """Adaptive split count S. v2 (S=1) already saturates the SMs once the grid
    num_decodes*H_kv reaches the SM count and wins there; below that, split the
    n_fac scan to ~2*SM blocks for occupancy. Capped so each split keeps >=1
    BLOCK_N tile."""
    max_splits = max(1, (n_fac + block_n - 1) // block_n)
    if num_decodes * n_kv_heads >= sm_count:
        return 1
    return min(max_splits, max(2, -(-(2 * sm_count) // (num_decodes * n_kv_heads))))


def lrosa_indexed_attend(q, kv_cache, block_table, top_idx, head_size, scale,
                         out, scratch=None, sm_count=None):
    """Fused gather-free attention over the top_idx-selected paged kv slots.

    q          : (num_decodes, H_q, head_size)
    kv_cache   : (num_blocks, block_size, H_kv, slot)  slot = [K | V | proj_K]
    block_table: (num_decodes, max_blocks) int32
    top_idx    : (num_decodes, H_kv, n_fac) int32/int64
    out        : (num_decodes, H_q, head_size)         preallocated (CG-safe)
    scratch    : optional (name, shape, dtype, device)->Tensor persistent-buffer
                 allocator (lrosa_attn._decode_scratch). Required CG-safety for
                 the split path's partials; falls back to torch.empty if None.
    sm_count   : SM count for the split heuristic (queried if None).

    Returns ``out``. No host sync; with ``scratch`` provided, no allocation —
    safe under CUDA-graph capture.
    """
    R, Hq, hs = q.shape
    Hk = top_idx.shape[1]
    n_fac = top_idx.shape[2]
    G = Hq // Hk
    BLOCK_N = 128
    BLOCK_D = triton.next_power_of_2(hs)
    if sm_count is None:
        sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count
    S = indexed_split_count(R, Hk, n_fac, sm_count, BLOCK_N)

    if S == 1:
        _indexed_attend_kv_kernel[(R, Hk)](
            q, kv_cache, block_table, top_idx, out,
            scale, n_fac, hs, kv_cache.shape[1],
            q.stride(0), q.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            block_table.stride(0), top_idx.stride(0), top_idx.stride(1),
            out.stride(0), out.stride(1),
            G=G, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
        )
        return out

    if scratch is None:
        m = torch.empty((R, Hq, S), dtype=torch.float32, device=q.device)
        l = torch.empty((R, Hq, S), dtype=torch.float32, device=q.device)
        acc = torch.empty((R, Hq, S, BLOCK_D), dtype=torch.float32, device=q.device)
    else:
        m = scratch("lrosa_idx_m", (R, Hq, S), torch.float32, q.device)
        l = scratch("lrosa_idx_l", (R, Hq, S), torch.float32, q.device)
        acc = scratch("lrosa_idx_acc", (R, Hq, S, BLOCK_D), torch.float32, q.device)
    _split_attend_kernel[(R, Hk, S)](
        q, kv_cache, block_table, top_idx, m, l, acc,
        scale, n_fac, hs, kv_cache.shape[1], S,
        q.stride(0), q.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), top_idx.stride(0), top_idx.stride(1),
        m.stride(0), m.stride(1), m.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2),
        G=G, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )
    _split_reduce_kernel[(R, Hq)](
        m, l, acc, out, hs, S,
        m.stride(0), m.stride(1), m.stride(2),
        acc.stride(0), acc.stride(1), acc.stride(2),
        out.stride(0), out.stride(1),
        BLOCK_S=triton.next_power_of_2(S), BLOCK_D=BLOCK_D,
    )
    return out
