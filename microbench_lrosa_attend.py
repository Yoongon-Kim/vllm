"""Kernel-level microbench for the LRoSA decode gather+attend path.

Isolates the decode-time cost the gather-free "indexed attention" optimization
targets — no model, no prefill, synthetic paged combined-slot KV cache + top-k
indices.

  baseline = lrosa_gather (scatter selected K,V -> K_sel/V_sel buffers)
           + flash_attn_varlen_func (dense attend over the gathered buffers)
  indexed  = fused flash-style attend that reads K,V from the paged slot
             directly via top_idx (NO gather buffer)

Reports gather_ms vs attend_ms vs indexed_ms + correctness (indexed vs ref).

Run (qwen3-8b decode dims):
  CUDA_VISIBLE_DEVICES=0 python microbench_lrosa_attend.py \
    --num_reqs 8 --ctx 32768 --n_fac 2048 --head_size 128 --n_kv 8 --n_q 32
"""
import argparse
import time

import torch
import triton
import triton.language as tl

from vllm.v1.attention.ops.triton_lrosa_gather import lrosa_gather
from vllm.v1.attention.ops.triton_lrosa_score_topk import lrosa_score_topk
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func


@triton.jit
def _indexed_attend_kernel(
    q_ptr, kv_cache_ptr, block_table_ptr, top_idx_ptr, o_ptr,
    scale, n_fac, hs, block_size, group,
    q_sr, q_sh, kv_sb, kv_sp, kv_sh, bt_sr, top_sr, top_sh, o_sr, o_sh,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # one program per (req, q_head). GQA: q_head -> kv_head = q_head // group.
    pid_r = tl.program_id(0)
    pid_hq = tl.program_id(1)
    pid_hk = pid_hq // group
    d = tl.arange(0, BLOCK_D)
    dm = d < hs
    q = tl.load(q_ptr + pid_r * q_sr + pid_hq * q_sh + d, mask=dm, other=0.0).to(tl.float32)
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for start in range(0, n_fac, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        nm = n < n_fac
        tok = tl.load(top_idx_ptr + pid_r * top_sr + pid_hk * top_sh + n, mask=nm, other=0)
        blk = tl.load(block_table_ptr + pid_r * bt_sr + (tok // block_size), mask=nm, other=0)
        pos = tok % block_size
        # int64 throughout: blk*kv_sb can exceed int32 (num_blocks>~58k at long
        # ctx / big batch). Cast EVERY operand — i64*i32-scalar promotion is
        # unreliable, so force i64 on the strides too.
        base = (blk.to(tl.int64) * kv_sb.to(tl.int64)
                + pos.to(tl.int64) * kv_sp.to(tl.int64)
                + pid_hk.to(tl.int64) * kv_sh.to(tl.int64))  # [BLOCK_N]
        k = tl.load(kv_cache_ptr + base[:, None] + d[None, :],
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        v = tl.load(kv_cache_ptr + base[:, None] + (hs + d[None, :]),
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        s = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_N]
        s = tl.where(nm, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new
    tl.store(o_ptr + pid_r * o_sr + pid_hq * o_sh + d, (acc / l_i).to(o_ptr.dtype.element_ty), mask=dm)


@triton.jit
def _indexed_attend_kv_kernel(
    q_ptr, kv_cache_ptr, block_table_ptr, top_idx_ptr, o_ptr,
    scale, n_fac, hs, block_size,
    q_sr, q_sh, kv_sb, kv_sp, kv_sh, bt_sr, top_sr, top_sh, o_sr, o_sh,
    G: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    # one program per (req, kv_head); processes the whole group of G q-heads
    # together (K,V read ONCE per kv-head). tl.dot for QK / PV.
    pid_r = tl.program_id(0)
    pid_hk = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dm = d < hs
    g = tl.arange(0, G)
    # q_grp [G, BLOCK_D] : the G query heads mapping to this kv-head
    qh = pid_hk * G + g
    q = tl.load(q_ptr + pid_r * q_sr + qh[:, None] * q_sh + d[None, :],
                mask=dm[None, :], other=0.0).to(tl.float32)
    m_i = tl.full([G], -float("inf"), tl.float32)
    l_i = tl.zeros([G], tl.float32)
    acc = tl.zeros([G, BLOCK_D], tl.float32)
    for start in range(0, n_fac, BLOCK_N):
        n = start + tl.arange(0, BLOCK_N)
        nm = n < n_fac
        tok = tl.load(top_idx_ptr + pid_r * top_sr + pid_hk * top_sh + n, mask=nm, other=0)
        blk = tl.load(block_table_ptr + pid_r * bt_sr + (tok // block_size), mask=nm, other=0)
        # int64 base: blk*kv_sb can exceed int32 (num_blocks>~58k at long ctx /
        # big batch) -> overflow -> illegal access. Cast EVERY operand to i64 —
        # i64*i32-scalar promotion is unreliable, so force the strides too.
        base = (blk.to(tl.int64) * kv_sb.to(tl.int64)
                + (tok % block_size).to(tl.int64) * kv_sp.to(tl.int64)
                + pid_hk.to(tl.int64) * kv_sh.to(tl.int64))
        k = tl.load(kv_cache_ptr + base[:, None] + d[None, :],
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)  # [BLOCK_N, BLOCK_D]
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


def indexed_attend_kv(q, kv_cache, block_table, top_idx, head_size, scale, out=None):
    R, Hq, hs = q.shape
    Hk = top_idx.shape[1]
    n_fac = top_idx.shape[2]
    G = Hq // Hk
    if out is None:
        out = torch.empty_like(q)
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


def indexed_attend(q, kv_cache, block_table, top_idx, head_size, scale, out=None):
    R, Hq, hs = q.shape
    Hk = top_idx.shape[1]
    n_fac = top_idx.shape[2]
    if out is None:
        out = torch.empty_like(q)
    BLOCK_D = triton.next_power_of_2(hs)
    _indexed_attend_kernel[(R, Hq)](
        q, kv_cache, block_table, top_idx, out,
        scale, n_fac, hs, kv_cache.shape[1], Hq // Hk,
        q.stride(0), q.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), top_idx.stride(0), top_idx.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_N=64, BLOCK_D=BLOCK_D,
    )
    return out


def _time(fn, reps=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_reqs", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--n_fac", type=int, default=2048)
    ap.add_argument("--head_size", type=int, default=128)
    ap.add_argument("--n_kv", type=int, default=8)
    ap.add_argument("--n_q", type=int, default=32)
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--reps", type=int, default=50)
    a = ap.parse_args()

    dev = "cuda"
    dt = torch.bfloat16
    R, L, n_fac, hs, Hk, Hq = a.num_reqs, a.ctx, a.n_fac, a.head_size, a.n_kv, a.n_q
    n_fac = min(n_fac, L)
    slot = 2 * hs + a.cs_h
    bpr = (L + a.block_size - 1) // a.block_size
    num_blocks = R * bpr + 1
    torch.manual_seed(0)

    kv_cache = torch.randn(num_blocks, a.block_size, Hk, slot, dtype=dt, device=dev)
    block_table = torch.arange(R * bpr, device=dev, dtype=torch.int32).reshape(R, bpr)
    top_idx = torch.stack([
        torch.stack([torch.randperm(L, device=dev)[:n_fac] for _ in range(Hk)])
        for _ in range(R)]).to(torch.int32)
    q = torch.randn(R, Hq, hs, dtype=dt, device=dev)
    K_sel = torch.empty(R * n_fac, Hk, hs, dtype=dt, device=dev)
    V_sel = torch.empty(R * n_fac, Hk, hs, dtype=dt, device=dev)
    cu_q = torch.arange(R + 1, device=dev, dtype=torch.int32)
    cu_k = (cu_q * n_fac).to(torch.int32)
    scale = 1.0 / (hs ** 0.5)

    def gather():
        lrosa_gather(kv_cache, block_table, top_idx, hs, K_sel_out=K_sel, V_sel_out=V_sel)

    def attend():
        flash_attn_varlen_func(q=q, k=K_sel, v=V_sel, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                               max_seqlen_q=1, max_seqlen_k=n_fac, softmax_scale=scale, causal=False)

    def both():
        gather(); attend()

    o_idx = torch.empty_like(q)
    o_idx2 = torch.empty_like(q)

    def indexed():
        indexed_attend(q, kv_cache, block_table, top_idx, hs, scale, out=o_idx)

    def indexed_kv():
        indexed_attend_kv(q, kv_cache, block_table, top_idx, hs, scale, out=o_idx2)

    # correctness vs (gather + varlen) reference
    gather()
    ref = flash_attn_varlen_func(q=q, k=K_sel, v=V_sel, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                                 max_seqlen_q=1, max_seqlen_k=n_fac, softmax_scale=scale, causal=False)
    indexed(); indexed_kv()
    torch.cuda.synchronize()
    refmax = ref.float().abs().max().item() + 1e-6
    rel1 = (o_idx.float() - ref.float()).abs().max().item() / refmax
    rel2 = (o_idx2.float() - ref.float()).abs().max().item() / refmax

    # SCORE+topk path (scales with context: reads proj_K over full ctx)
    proj_q = torch.randn(R, Hk, a.cs_h, dtype=dt, device=dev)
    seq_lens = torch.full((R,), L, dtype=torch.int32, device=dev)
    # separate contiguous proj_K caches: bf16 and fp8 (the score-scan optimization)
    pk_bf16 = torch.randn(num_blocks, a.block_size, Hk, a.cs_h, dtype=dt, device=dev)
    pk_fp8 = pk_bf16.to(torch.float8_e4m3fn)

    def score_inslot():
        lrosa_score_topk(proj_q, kv_cache, block_table, seq_lens, n_fac, hs, a.cs_h, use_radix=True)

    def score_contig():
        lrosa_score_topk(proj_q, kv_cache, block_table, seq_lens, n_fac, hs, a.cs_h,
                         use_radix=True, projk_cache=pk_bf16)

    def score_fp8():
        lrosa_score_topk(proj_q, kv_cache, block_table, seq_lens, n_fac, hs, a.cs_h,
                         use_radix=True, projk_cache=pk_fp8)

    s_ms = _time(score_inslot, a.reps)
    sc_ms = _time(score_contig, a.reps)
    try:
        sf_ms = _time(score_fp8, a.reps)
    except Exception as e:
        sf_ms = float("nan")
        print(f"  [score_fp8 failed: {type(e).__name__}]")
    g_ms = _time(gather, a.reps)
    a_ms = _time(attend, a.reps)
    t_ms = _time(both, a.reps)

    def _safe(fn):
        try:
            return _time(fn, a.reps)
        except Exception as e:
            print(f"  [{fn.__name__} failed: {type(e).__name__}: {str(e)[:80]}]")
            return float("nan")

    i_ms = _safe(indexed)
    i2_ms = _safe(indexed_kv)
    NL = 36  # qwen3-8b layers (per-step = NL * per-layer)
    print(f"[microbench] R={R} ctx={L} n_fac={n_fac} hs={hs} Hk={Hk} Hq={Hq}")
    print(f"  SCORE per-layer: in-slot={s_ms:.4f} contig-bf16={sc_ms:.4f} contig-fp8={sf_ms:.4f}"
          f"  (contig {s_ms/sc_ms:.2f}x, fp8 {s_ms/sf_ms:.2f}x)")
    print(f"  ATTEND-side per-layer: gather={g_ms:.4f} attend={a_ms:.4f} gather+attend={t_ms:.4f}"
          f"  indexed-v2={i2_ms:.4f} ({t_ms/i2_ms:.2f}x, rel={rel2:.4f})")
    print(f"  per-step (x{NL}): SCORE={s_ms*NL:.2f}ms GATHER={g_ms*NL:.2f} ATTEND={a_ms*NL:.2f}"
          f" | gather+attend={t_ms*NL:.2f} -> indexed-v2 {i2_ms*NL:.2f}  saves {(t_ms-i2_ms)*NL:.2f}ms/step")


if __name__ == "__main__":
    main()
