"""JSSA (LRoSA) GQA decode component-latency breakdown — Gemma-4-26B-A4B.

Kernel-level, synthetic paged KV (no model / no prefill), matching the PRODUCTION
default decode dispatch for a Gemma FULL (global) attention layer:
  - fp8 contiguous proj_K score cache (lrosa_fp8_projk default)
  - gather-free indexed attend (lrosa_indexed_attend default)

Gemma-4-26B-A4B full/global layers: head_dim=512, num_kv_heads=2 (global),
num_q_heads=16, cs_h=64; 5 full layers (JSSA runs here) + 25 sliding (dense
windowed FA, no selection — NOT part of this breakdown).

Four decode components timed separately (min-of-reps, perf_counter+synchronize):
  1. proj_k        : project the NEW token's K -> fp8 proj_K cache (O(batch), decode)
  2. score-scan    : q_proj . proj_K^T over the FULL ctx (O(batch*ctx*cs_h))
  3. top-k         : select n_fac from ctx scores (radix; = combined - scan)
  4. indexed-attend: gather-free attend over the n_fac selected slots (O(batch*n_fac))

Representative point (user-chosen): ctx=131072, batch=1, n_fac=2048 — at fixed ctx
the score/attend RATIO is ~batch-independent (both scale with batch), so batch 1
gives the cleanest measurement; 128k maximizes score-scan dominance.
"""
import argparse
import time

import torch

from vllm.v1.attention.ops.triton_lrosa_store import (
    lrosa_project_and_store_contig_fp8,
)
from vllm.v1.attention.ops.triton_lrosa_score_topk import (
    lrosa_score, lrosa_score_topk,
)
from vllm.v1.attention.ops.triton_lrosa_indexed_attend import lrosa_indexed_attend


def _time(fn, reps, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    # Gemma-4-26B-A4B FULL (global) layer dims by default
    ap.add_argument("--num_reqs", type=int, default=1)     # batch
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--n_fac", type=int, default=2048)
    ap.add_argument("--head_size", type=int, default=512)  # global_head_dim
    ap.add_argument("--n_kv", type=int, default=2)         # num_global_key_value_heads
    ap.add_argument("--n_q", type=int, default=16)         # num_attention_heads
    ap.add_argument("--cs_h", type=int, default=64)
    ap.add_argument("--block_size", type=int, default=16)
    ap.add_argument("--n_full_layers", type=int, default=5)
    ap.add_argument("--n_sliding_layers", type=int, default=25)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--with_full", action="store_true",
                    help="also time DENSE full-KV attend (indexed kernel, top_idx=all ctx)")
    ap.add_argument("--md", type=str, default="")          # optional md output path
    a = ap.parse_args()

    dev, dt = "cuda", torch.bfloat16
    R, L, hs, Hk, Hq, cs_h = a.num_reqs, a.ctx, a.head_size, a.n_kv, a.n_q, a.cs_h
    n_fac = min(a.n_fac, L)
    slot = 2 * hs + cs_h
    bpr = (L + a.block_size - 1) // a.block_size
    num_blocks = R * bpr + 1
    torch.manual_seed(0)
    scale = 1.0 / (hs ** 0.5)

    # --- shared paged fixtures ---
    kv_cache = torch.randn(num_blocks, a.block_size, Hk, slot, dtype=dt, device=dev)
    block_table = torch.arange(R * bpr, device=dev, dtype=torch.int32).reshape(R, bpr)
    seq_lens = torch.full((R,), L, dtype=torch.int32, device=dev)

    # --- (1) proj_k fixtures: new decode token K/V -> fp8 contig proj_K cache ---
    key = torch.randn(R, Hk, hs, dtype=dt, device=dev)
    value = torch.randn(R, Hk, hs, dtype=dt, device=dev)
    projk_fp8 = torch.zeros(num_blocks, a.block_size, Hk, cs_h,
                            dtype=torch.float8_e4m3fn, device=dev)
    projk_scale = torch.full((Hk, cs_h), 0.05, dtype=torch.float32, device=dev)
    M = torch.randn(Hk, cs_h, hs, dtype=dt, device=dev)
    slot_mapping = torch.arange(R, device=dev, dtype=torch.long)

    # --- (2/3) score-scan / top-k fixtures ---
    proj_q = torch.randn(R, Hk, cs_h, dtype=dt, device=dev)
    scores_buf = torch.empty(R, Hk, bpr * a.block_size, dtype=torch.float32, device=dev)

    # --- (4) indexed-attend fixtures ---
    q = torch.randn(R, Hq, hs, dtype=dt, device=dev)
    top_idx = torch.stack([
        torch.stack([torch.randperm(L, device=dev)[:n_fac] for _ in range(Hk)])
        for _ in range(R)]).to(torch.int32)
    out = torch.empty_like(q)

    # --- component closures (production default dispatch) ---
    def proj_k():
        lrosa_project_and_store_contig_fp8(
            key, value, kv_cache, projk_fp8, projk_scale, slot_mapping, M)

    def score_scan():
        lrosa_score(proj_q, kv_cache, block_table, seq_lens, hs, cs_h,
                    projk_cache=projk_fp8, scores_out=scores_buf)

    def combined():  # score-scan + radix top-k fused
        lrosa_score_topk(proj_q, kv_cache, block_table, seq_lens, n_fac, hs, cs_h,
                         use_radix=True, projk_cache=projk_fp8)

    def indexed_attend():
        lrosa_indexed_attend(q, kv_cache, block_table, top_idx, hs, scale, out)

    # warm _C op registration on the fused/topk path first
    combined(); torch.cuda.synchronize()

    def _safe(fn):
        try:
            return _time(fn, a.reps)
        except Exception as e:
            print(f"  [{fn.__name__} FAILED: {type(e).__name__}: {str(e)[:100]}]")
            return float("nan")

    pk = _safe(proj_k)
    sc = _safe(score_scan)
    cb = _safe(combined)
    tk = max(cb - sc, 0.0)          # radix top-k = combined - scan (op-reg safe)
    ia = _safe(indexed_attend)

    # optional: DENSE full-KV attend — same indexed kernel, top_idx = all ctx
    fa = float("nan")
    if a.with_full:
        top_full = (torch.arange(L, device=dev, dtype=torch.int32)
                    .view(1, 1, L).expand(R, Hk, L).contiguous())
        out_full = torch.empty_like(q)

        def full_attend():
            lrosa_indexed_attend(q, kv_cache, block_table, top_full, hs, scale, out_full)
        fa = _safe(full_attend)

    total = pk + sc + tk + ia       # JSSA per-full-layer decode cost
    NF, NS = a.n_full_layers, a.n_sliding_layers

    def pct(x):
        return f"{x/total*100:5.1f}%" if total > 0 else "   -"

    lines = []
    lines.append(f"# JSSA decode component breakdown — Gemma-4-26B-A4B full/global layer")
    lines.append("")
    lines.append(f"- ctx = {L} ({L//1024}k), batch = {R}, n_fac = {n_fac}")
    lines.append(f"- head_dim = {hs}, num_kv_heads = {Hk}, num_q_heads = {Hq}, "
                 f"cs_h = {cs_h}, block_size = {a.block_size}")
    lines.append(f"- proj_K = fp8 contig cache, attend = gather-free indexed "
                 f"(production default)")
    lines.append(f"- layers: {NF} full/global (JSSA) + {NS} sliding (dense windowed FA, "
                 f"not shown)")
    lines.append("")
    lines.append(f"| component | ms/layer | share |")
    lines.append(f"|---|---|---|")
    lines.append(f"| proj_k (new-token proj) | {pk:.4f} | {pct(pk)} |")
    lines.append(f"| **score-scan** (q·projK over ctx) | **{sc:.4f}** | **{pct(sc)}** |")
    lines.append(f"| top-k (radix select {n_fac}) | {tk:.4f} | {pct(tk)} |")
    lines.append(f"| indexed-attend ({n_fac} slots) | {ia:.4f} | {pct(ia)} |")
    lines.append(f"| **JSSA total / full layer** | **{total:.4f}** | 100% |")
    lines.append("")
    lines.append(f"- per-step JSSA (×{NF} full layers): "
                 f"proj_k={pk*NF:.3f}  score-scan={sc*NF:.3f}  top-k={tk*NF:.3f}  "
                 f"indexed-attend={ia*NF:.3f}  total={total*NF:.3f} ms")
    lines.append(f"- score-scan / indexed-attend ratio = "
                 f"{sc/ia:.1f}x" if ia > 0 else "-")
    if a.with_full and fa == fa:  # not nan
        lines.append(f"- **DENSE full-KV attend** (all {L} tokens, same kernel) = "
                     f"**{fa:.4f} ms/layer**  "
                     f"(indexed-sparse attend {ia:.4f} → dense/sparse = {fa/ia:.1f}x; "
                     f"dense-attend vs JSSA-total = {fa/total:.2f}x)")
    report = "\n".join(lines)
    print("\n" + report + "\n")

    if a.md:
        with open(a.md, "w") as f:
            f.write(report + "\n")
        print(f"[written] {a.md}")


if __name__ == "__main__":
    main()
