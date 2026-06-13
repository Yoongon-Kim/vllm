"""Gather-free indexed attention for LRoSA decode.

The default LRoSA decode does ``lrosa_gather`` (materialize K_sel/V_sel into a
contiguous [num_decodes * n_fac, H_kv, head_size] buffer) followed by a dense
``flash_attn_varlen_func`` over that buffer. The gather buffer scales with
``num_decodes * n_fac``, so at the throughput operating point (large batch) the
buffer write+read dominates the attention side (microbench: at batch 32 /
n_fac 2048, gather+attend = 8.1 ms/step vs 3.5 ms for this fused kernel — a
2.3x reduction of the per-step attention-side fixed overhead).

This kernel fuses selection-gather and attention: each program handles one
(request, kv-head) and streams the n_fac selected tokens directly out of the
paged kv cache via ``top_idx`` (online-softmax flash style, ``tl.dot`` for
QK/PV), so the intermediate buffer never exists. The K|V are read from the
combined LRoSA slot ([K | V | proj_K]); proj_K is untouched, so this is
independent of the fp8/contig proj_K score path.

Constraints (gated by the caller — see lrosa_attn.py):
  - head_size <= 256 (BLOCK_D = next_pow2(head_size); Gemma-4 full layers use
    head_size 512 and keep the manual einsum path).
  - selected K is treated as an unordered set (causal=False), matching the
    flash path; no sinks / softcap / alibi support — those fall back to gather.
  - steady state only (every request's kv_len >= n_fac); the seq<k_eff partial
    tail compaction stays on the gather+flash path.
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
    qh = pid_hk * G + g  # the G query heads mapping to this kv-head
    q = tl.load(q_ptr + pid_r * q_sr + qh[:, None] * q_sh + d[None, :],
                mask=dm[None, :], other=0.0).to(tl.float32)  # [G, BLOCK_D]
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
        # int64 throughout: blk*kv_sb exceeds int32 once num_blocks > ~58k
        # (long ctx / big batch). Cast EVERY operand — i64*i32-scalar promotion
        # is unreliable, so force i64 on the strides too.
        base = (blk.to(tl.int64) * kv_sb.to(tl.int64)
                + (tok % block_size).to(tl.int64) * kv_sp.to(tl.int64)
                + pid_hk.to(tl.int64) * kv_sh.to(tl.int64))  # [BLOCK_N]
        k = tl.load(kv_cache_ptr + base[:, None] + d[None, :],
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        v = tl.load(kv_cache_ptr + base[:, None] + (hs + d[None, :]),
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        s = tl.dot(q, tl.trans(k)) * scale  # [G, BLOCK_N]
        s = tl.where(nm[None, :], s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])  # [G, BLOCK_N]
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    o = acc / l_i[:, None]
    tl.store(o_ptr + pid_r * o_sr + qh[:, None] * o_sh + d[None, :],
             o.to(o_ptr.dtype.element_ty), mask=dm[None, :])


def lrosa_indexed_attend(q, kv_cache, block_table, top_idx, head_size, scale,
                         out):
    """Fused gather-free attention over the top_idx-selected paged kv slots.

    q          : (num_decodes, H_q, head_size)        — decode queries
    kv_cache   : (num_blocks, block_size, H_kv, slot)  — slot = [K | V | proj_K]
    block_table: (num_decodes, max_blocks) int32
    top_idx    : (num_decodes, H_kv, n_fac) int32/int64 — per-kv-head selection
    out        : (num_decodes, H_q, head_size)         — preallocated (CG-safe)

    Returns ``out``. No host sync, no allocation — safe under CUDA-graph capture.
    """
    R, Hq, hs = q.shape
    Hk = top_idx.shape[1]
    n_fac = top_idx.shape[2]
    G = Hq // Hk
    BLOCK_D = triton.next_power_of_2(hs)
    _indexed_attend_kv_kernel[(R, Hk)](
        q, kv_cache, block_table, top_idx, out,
        scale, n_fac, hs, kv_cache.shape[1],
        q.stride(0), q.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), top_idx.stride(0), top_idx.stride(1),
        out.stride(0), out.stride(1),
        G=G, BLOCK_N=128, BLOCK_D=BLOCK_D,
    )
    return out
