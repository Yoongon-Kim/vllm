"""JSSA (LRoSA) GQA decode component breakdown — Gemma-4-26B-A4B — CUDA-GRAPH.

Deployment-mode timing: every op captured in a CUDA graph and replayed
(per-replay launch floor amortized), unlike the eager _gemma_breakdown.py.
DENSE attention reference = real FlashInfer paged decode (BatchDecodeWithPagedKV),
NOT the indexed kernel with a full arange. Same Gemma full/global-layer dims:
head_dim 512, num_kv_heads 2, num_q_heads 16, cs_h 64, page 16, ctx 131072, n_fac 2048.

Components (cudagraph): proj_k / score-scan / top-k(=combined-scan) / indexed-attend,
plus dense-attend (FlashInfer, all pages). Batch sweep.
"""
import argparse
import sys
import time

import torch

sys.path.insert(0, "/NHNHOME/jiwonsong/vllm")
import flashinfer  # noqa: E402
from vllm.v1.attention.ops.triton_lrosa_store import (  # noqa: E402
    lrosa_project_and_store_contig_fp8,
)
from vllm.v1.attention.ops.triton_lrosa_score_topk import (  # noqa: E402
    lrosa_score, lrosa_score_topk,
)
from vllm.v1.attention.ops.triton_lrosa_indexed_attend import (  # noqa: E402
    lrosa_indexed_attend,
)

dev, dt = "cuda", torch.bfloat16
L, n_fac, hs, Hk, Hq, cs_h, page = 131072, 2048, 512, 2, 16, 64, 16  # Gemma full layer


def _t(fn, reps=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1000.0


def cg_time(call, N=50):
    """Capture N calls in a CUDA graph, replay, amortize the per-replay floor."""
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(N):
            call()
    return _t(g.replay, reps=50) / N


_buf = {}
def scratch(name, shape, dtype, device):
    if name not in _buf:
        _buf[name] = torch.empty(shape, dtype=dtype, device=device)
    return _buf[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=str, default="1,4,16,32,64")
    ap.add_argument("--md", type=str, default="")
    a = ap.parse_args()
    batches = [int(x) for x in a.batches.split(",")]
    scale = 1.0 / hs ** 0.5
    ppr = L // page

    rows = []
    for R in batches:
        numpg = R * ppr
        torch.manual_seed(0)
        lp = torch.full((R,), page, dtype=torch.int32, device=dev)
        q = torch.randn(R, Hq, hs, dtype=dt, device=dev)
        res = {}

        # --- DENSE attend via FlashInfer paged decode (all pages) ---
        try:
            k_c = torch.randn(numpg, page, Hk, hs, dtype=dt, device=dev)
            v_c = torch.randn(numpg, page, Hk, hs, dtype=dt, device=dev)
            ib = torch.zeros(R + 1, dtype=torch.int32, device=dev)
            xb = torch.zeros(numpg, dtype=torch.int32, device=dev)
            lb = torch.zeros(R, dtype=torch.int32, device=dev)
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=dev),
                kv_layout="NHD", use_cuda_graph=True,
                paged_kv_indptr_buffer=ib, paged_kv_indices_buffer=xb,
                paged_kv_last_page_len_buffer=lb)
            w.plan(torch.arange(0, (R + 1) * ppr, ppr, dtype=torch.int32, device=dev),
                   torch.arange(numpg, dtype=torch.int32, device=dev), lp,
                   Hq, Hk, hs, page, q_data_type=dt, kv_data_type=dt)
            res["dense"] = cg_time(lambda: w.run(q, (k_c, v_c)))
            del k_c, v_c, w
            torch.cuda.empty_cache()
        except Exception as e:
            res["dense"] = None
            res["dense_err"] = f"{type(e).__name__}: {str(e)[:80]}"

        # --- JSSA components (combined-slot kv + fp8 contig proj_K) ---
        slot = 2 * hs + cs_h
        kv = torch.randn(R * ppr + 1, page, Hk, slot, dtype=dt, device=dev)
        bt = torch.arange(R * ppr, device=dev, dtype=torch.int32).reshape(R, ppr)
        seq = torch.full((R,), L, dtype=torch.int32, device=dev)
        proj_q = torch.randn(R, Hk, cs_h, dtype=dt, device=dev)
        scores_buf = torch.empty(R, Hk, ppr * page, dtype=torch.float32, device=dev)
        projk = torch.zeros(R * ppr + 1, page, Hk, cs_h,
                            dtype=torch.float8_e4m3fn, device=dev)
        pscale = torch.full((Hk, cs_h), 0.05, dtype=torch.float32, device=dev)
        M = torch.randn(Hk, cs_h, hs, dtype=dt, device=dev)
        key = torch.randn(R, Hk, hs, dtype=dt, device=dev)
        val = torch.randn(R, Hk, hs, dtype=dt, device=dev)
        slotmap = torch.arange(R, device=dev, dtype=torch.long)
        tidx = torch.stack([
            torch.stack([torch.randperm(L, device=dev)[:n_fac] for _ in range(Hk)])
            for _ in range(R)]).to(torch.int32)
        tio = torch.empty(R, Hk, n_fac, dtype=torch.int64, device=dev)
        tso = torch.empty(R, Hk, n_fac, dtype=torch.float32, device=dev)
        o = torch.empty(R, Hq, hs, dtype=dt, device=dev)
        lrosa_score_topk(proj_q, kv, bt, seq, n_fac, hs, cs_h, use_radix=True,
                         projk_cache=projk); torch.cuda.synchronize()

        def _cg(fn):
            try:
                return cg_time(fn)
            except Exception as e:
                print(f"  [R={R} {fn} FAIL: {type(e).__name__}: {str(e)[:80]}]")
                return None

        res["proj_k"] = _cg(lambda: lrosa_project_and_store_contig_fp8(
            key, val, kv, projk, pscale, slotmap, M))
        res["score"] = _cg(lambda: lrosa_score(
            proj_q, kv, bt, seq, hs, cs_h, projk_cache=projk, scores_out=scores_buf))
        res["comb"] = _cg(lambda: lrosa_score_topk(
            proj_q, kv, bt, seq, n_fac, hs, cs_h, use_radix=True, projk_cache=projk,
            scores_out=scores_buf, top_idx_out=tio, top_scores_out=tso))
        res["idx"] = _cg(lambda: lrosa_indexed_attend(
            q, kv, bt, tidx, hs, scale, o, scratch=scratch))
        res["topk"] = (max(res["comb"] - res["score"], 0.0)
                       if (res["comb"] and res["score"]) else None)
        rows.append((R, res))
        del kv, projk, scores_buf, tidx, tio, tso, M
        torch.cuda.empty_cache(); _buf.clear()

    # --- report ---
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else ("—" if x is None else str(x))
    lines = ["# JSSA decode component breakdown — Gemma-4-26B-A4B (CUDA-GRAPH)", "",
             f"head_dim {hs}, num_kv_heads {Hk}, num_q_heads {Hq}, cs_h {cs_h}, "
             f"page {page}, ctx {L}, n_fac {n_fac}. All ops cudagraph-captured "
             f"(launch floor amortized). Dense = FlashInfer paged decode (all pages).", "",
             "| batch | proj_k | score-scan | top-k | indexed-attend | JSSA-sel(sc+tk) | dense-attend(FI) |",
             "|---|---|---|---|---|---|---|"]
    for R, res in rows:
        sel = (res["score"] + res["topk"]) if (res["score"] and res["topk"]) else None
        lines.append(f"| {R} | {f(res['proj_k'])} | {f(res['score'])} | {f(res['topk'])} "
                     f"| {f(res['idx'])} | {f(sel)} | {f(res['dense'])} |")
    if any(r[1].get("dense_err") for r in rows):
        lines.append("")
        lines.append(f"- dense (FlashInfer) error: "
                     f"{[r[1].get('dense_err') for r in rows if r[1].get('dense_err')][0]}")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    if a.md:
        open(a.md, "w").write(report + "\n")
        print(f"[written] {a.md}")


if __name__ == "__main__":
    main()
