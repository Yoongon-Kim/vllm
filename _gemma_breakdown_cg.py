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
    lrosa_score, lrosa_score_topk, _radix_topk,
)
from vllm.v1.attention.ops.triton_lrosa_indexed_attend import (  # noqa: E402
    lrosa_indexed_attend,
)
from vllm.v1.attention.ops.triton_quest import (  # noqa: E402
    quest_minmax_update, quest_page_score, quest_blocksparse_attn, quest_num_splits,
)
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides as _canon  # noqa: E402

dev, dt = "cuda", torch.bfloat16
import os as _os  # noqa: E402
# dims default to Gemma-4-26B full layer; override via env for other models
# (Qwen3-8B: HS=128 HK=8 HQ=32 CSH=32).
L, n_fac, page = 131072, 2048, 16
hs = int(_os.environ.get("HS", 512))
Hk = int(_os.environ.get("HK", 2))
Hq = int(_os.environ.get("HQ", 16))
cs_h = int(_os.environ.get("CSH", 64))


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
        del kv, projk, scores_buf, tidx, tio, tso, M, o
        torch.cuda.empty_cache(); _buf.clear()

        # --- QUEST components (block/page scoring): combined-slot kv 2*hs + page
        # min/max buffer. token_budget = n_fac (2048) → same attended-token budget
        # as JSSA, so the two are directly comparable. Components:
        # minmax_update / page_score (block, full-head min|max) / page top-k / blocksparse_attend
        pb = n_fac // page - 1                          # page_budget (127 @ n_fac2048/page16)
        qkv = torch.randn(R * ppr + 1, page, Hk, 2 * hs, dtype=dt, device=dev)
        minmax = torch.randn(R * ppr + 1, Hk, 2 * hs, dtype=dt, device=dev)
        q_kv = q.view(R, Hk, Hq // Hk, hs).mean(2).contiguous()
        qsc = torch.empty(R, Hk, ppr, dtype=torch.float32, device=dev)
        num_full = ((seq.to(torch.int64) - 1) // page).to(torch.int32)
        pidx = torch.empty(R, Hk, pb, dtype=torch.int32, device=dev)
        oq = torch.empty(R, Hq, hs, dtype=dt, device=dev)
        ns = quest_num_splits(pb)
        pacc = torch.empty(R, Hq, ns, hs, dtype=torch.float32, device=dev)
        pm = torch.empty(R, Hq, ns, dtype=torch.float32, device=dev)
        pl = torch.empty(R, Hq, ns, dtype=torch.float32, device=dev)
        quest_page_score(q_kv, minmax, bt, seq, page, hs, scores_out=qsc)
        pidx.copy_(qsc.topk(pb, dim=-1).indices.to(torch.int32)); torch.cuda.synchronize()
        res["q_minmax"] = _cg(lambda: quest_minmax_update(key, minmax, bt, seq, hs, page))
        res["q_score"] = _cg(lambda: quest_page_score(q_kv, minmax, bt, seq, page, hs, scores_out=qsc))
        res["q_topk"] = _cg(lambda: _radix_topk(qsc, num_full, pb, idx_out=pidx))
        res["q_attn"] = _cg(lambda: quest_blocksparse_attn(
            query=q, kv_cache=qkv, page_idx=pidx, block_table=bt, seq_lens=seq, output=oq,
            scale=scale, page_size=page, head_size=hs, num_kv_groups=Hq // Hk,
            partial_acc=pacc, partial_m=pm, partial_l=pl))

        # --- QUEST attend via the NEW per-head FlashInfer path (matches the vLLM
        # backend _fi_attend exactly): per kv-head decode (num_kv_heads=1,
        # num_qo_heads=G) over the pb selected pages + trailing, on a single 5D
        # NHD view [nb,2,page,1,hs] of the combined-slot cache + canonicalised
        # singleton strides; q pre-scaled (sm_scale=1.0). Faithful per-head
        # selection (NOT the earlier per-request head-0 approximation). ---
        try:
            G = Hq // Hk
            nb = qkv.shape[0]
            scale_q = q * scale
            trail = torch.gather(bt, 1, num_full.long().view(R, 1))       # (R,1)
            ind = torch.arange(0, (R + 1) * (pb + 1), pb + 1,
                               dtype=torch.int32, device=dev)
            last = (seq - num_full * page).to(torch.int32)                # trailing_len
            wrappers, kv_views = [], []
            for h in range(Hk):
                idx_buf = torch.zeros(R * (pb + 1), dtype=torch.int32, device=dev)
                sel = torch.gather(bt, 1, pidx[:, h].long())              # (R, pb)
                idx_buf.copy_(torch.cat([sel, trail], dim=1).reshape(-1).to(torch.int32))
                w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
                    "NHD", use_cuda_graph=True,
                    paged_kv_indptr_buffer=ind, paged_kv_indices_buffer=idx_buf,
                    paged_kv_last_page_len_buffer=last)
                w.plan(ind, idx_buf, last, G, 1, hs, page,
                       q_data_type=dt, kv_data_type=dt, sm_scale=1.0)
                kv_view = _canon(qkv[:, :, h, :].view(nb, page, 2, hs)
                                 .permute(0, 2, 1, 3).unsqueeze(3))       # [nb,2,page,1,hs]
                wrappers.append(w); kv_views.append(kv_view)

            def _fi_run():
                for h in range(Hk):
                    oq[:, h * G:(h + 1) * G, :] = wrappers[h].run(
                        scale_q[:, h * G:(h + 1) * G, :].contiguous(), kv_views[h])
            res["q_fi_attend"] = cg_time(_fi_run)
            del wrappers, kv_views
            torch.cuda.empty_cache()
        except Exception as e:
            res["q_fi_attend"] = None
            res["q_fi_err"] = f"{type(e).__name__}: {str(e)[:100]}"

        rows.append((R, res))
        del qkv, minmax, qsc, pidx, oq, pacc, pm, pl, bt, seq, q, key, val, q_kv, proj_q
        torch.cuda.empty_cache(); _buf.clear()

    # --- report ---
    def f(x):
        return f"{x:.4f}" if isinstance(x, float) else ("—" if x is None else str(x))
    _model = _os.environ.get("MODEL_TAG",
                             f"head{hs}-Hk{Hk}-Hq{Hq}-cs{cs_h}")
    lines = [f"# JSSA vs QUEST decode component breakdown — {_model} (CUDA-GRAPH)", "",
             f"head_dim {hs}, num_kv_heads {Hk}, num_q_heads {Hq}, cs_h {cs_h}, "
             f"page {page}, ctx {L}, n_fac {n_fac}. All ops cudagraph-captured "
             f"(launch floor amortized). Dense = FlashInfer paged decode (all pages).", "",
             "## JSSA (proj_k / score-scan / top-k / indexed-attend), ms/layer",
             "| batch | proj_k | score-scan | top-k | indexed-attend | JSSA-total | JSSA-sel |",
             "|---|---|---|---|---|---|---|"]
    def _tot(res, keys):
        vs = [res.get(k) for k in keys]
        return sum(vs) if all(isinstance(v, float) for v in vs) else None
    for R, res in rows:
        jt = _tot(res, ("proj_k", "score", "topk", "idx"))
        jsel = _tot(res, ("score", "topk"))
        lines.append(f"| {R} | {f(res['proj_k'])} | {f(res['score'])} | {f(res['topk'])} "
                     f"| {f(res['idx'])} | {f(jt)} | {f(jsel)} |")
    lines += ["", "## QUEST (minmax-update / page-score / page-topk / FI-per-head-attend), ms/layer",
              "attend = the NEW per-head FlashInfer paged decode (the deployed backend path), "
              "NOT the old custom Triton blocksparse. QUEST-total uses the FI attend.", "",
              "| batch | minmax | page-score | page-topk | FI-attend | QUEST-total(FI) | QUEST-sel | (custom-attend) |",
              "|---|---|---|---|---|---|---|---|"]
    for R, res in rows:
        qt = _tot(res, ("q_minmax", "q_score", "q_topk", "q_fi_attend"))
        qsel = _tot(res, ("q_score", "q_topk"))
        lines.append(f"| {R} | {f(res.get('q_minmax'))} | {f(res.get('q_score'))} | {f(res.get('q_topk'))} "
                     f"| {f(res.get('q_fi_attend'))} | {f(qt)} | {f(qsel)} | {f(res.get('q_attn'))} |")
    lines += ["", "## JSSA vs QUEST(FI) vs dense (ms/layer)",
              "| batch | JSSA-total | QUEST-total(FI) | dense(FI) | JSSA score-scan | QUEST page-score |",
              "|---|---|---|---|---|---|"]
    for R, res in rows:
        jt = _tot(res, ("proj_k", "score", "topk", "idx"))
        qt = _tot(res, ("q_minmax", "q_score", "q_topk", "q_fi_attend"))
        lines.append(f"| {R} | {f(jt)} | {f(qt)} | {f(res.get('dense'))} "
                     f"| {f(res.get('score'))} | {f(res.get('q_score'))} |")
    lines += ["", "## ATTEND-only (ms/layer, same ~2048-token budget): the NEW per-head FI",
              "JSSA idx = custom token-indexed Triton; QUEST custom = old quest_blocksparse_attn Triton; "
              "QUEST FI = the NEW per-head FlashInfer paged decode (deployed); dense = FI all pages.", "",
              "| batch | JSSA idx | QUEST custom | QUEST FI (new) | dense (FI all) | custom/FI | FI/JSSA-idx |",
              "|---|---|---|---|---|---|---|"]
    for R, res in rows:
        idx, qc, qfi = res.get("idx"), res.get("q_attn"), res.get("q_fi_attend")
        r1 = f"{qc / qfi:.2f}×" if (isinstance(qc, float) and isinstance(qfi, float) and qfi) else "—"
        r2 = f"{qfi / idx:.2f}×" if (isinstance(qfi, float) and isinstance(idx, float) and idx) else "—"
        lines.append(f"| {R} | {f(idx)} | {f(qc)} | {f(qfi)} | {f(res.get('dense'))} | {r1} | {r2} |")
    if any(r[1].get("dense_err") for r in rows):
        lines.append(f"\n- dense (FlashInfer) error: "
                     f"{[r[1].get('dense_err') for r in rows if r[1].get('dense_err')][0]}")
    if any(r[1].get("q_fi_err") for r in rows):
        lines.append(f"\n- QUEST FI-attend error: "
                     f"{[r[1].get('q_fi_err') for r in rows if r[1].get('q_fi_err')][0]}")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    if a.md:
        open(a.md, "w").write(report + "\n")
        print(f"[written] {a.md}")


if __name__ == "__main__":
    main()
