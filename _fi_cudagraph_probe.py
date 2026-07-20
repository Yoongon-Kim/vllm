"""Phase 3a: validate the ONE novel cudagraph mechanic for QUEST-FI:
  - plan() done OUTSIDE the graph (in "build"), using indptr/last_page_len from seq_lens.
  - the page INDICES are written INSIDE the captured graph (QUEST computes page_idx in
    forward), then wrapper.run() (also captured) must read those fresh indices.
  - between replays, update page_idx (selection changes) + re-plan for a new seq_len;
    the replayed graph must produce the correct output.

vLLM writes indices OUT of the graph (build); QUEST must write them IN-graph. This
checks that a captured run() reflects in-graph buffer writes across replays.
"""
import sys
import torch
sys.path.insert(0, "/NHNHOME/jiwonsong/vllm")
import flashinfer
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides

dev, dt = "cuda", torch.bfloat16
import os
hs, G, page = int(os.environ.get("HS", 512)), int(os.environ.get("GG", 8)), 16
R, pb = 4, 127                    # 4 requests, page_budget 127
scale = 1.0 / hs ** 0.5


def ref(q, cache, sel_phys, trail_phys, trail_len):
    # q [G,hs]; attend pages sel_phys (full) + trail_phys (trail_len tokens)
    ks, vs = [], []
    for p in sel_phys.tolist():
        ks.append(cache[p, :, 0, :hs]); vs.append(cache[p, :, 0, hs:2 * hs])
    ks.append(cache[trail_phys, :trail_len, 0, :hs])
    vs.append(cache[trail_phys, :trail_len, 0, hs:2 * hs])
    k = torch.cat(ks).float(); v = torch.cat(vs).float()
    s = (q.float() @ k.T) * scale
    p = torch.softmax(s, -1)
    return (p @ v).to(q.dtype)


def main():
    torch.manual_seed(0)
    max_cols = 520                     # >= num_full+1 for seq up to ~8300 (nf<=512)
    num_blocks = R * max_cols
    cache = torch.randn(num_blocks, page, 1, 2 * hs, dtype=dt, device=dev)  # combined slot K|V
    block_table = torch.arange(num_blocks, device=dev).reshape(R, max_cols).int()

    # static buffers (persistent, mark_static_address for graph safety)
    indptr_buf = torch.zeros(R + 1, dtype=torch.int32, device=dev)
    indices_buf = torch.zeros(R * (pb + 1), dtype=torch.int32, device=dev)
    last_buf = torch.zeros(R, dtype=torch.int32, device=dev)
    for b in (indptr_buf, indices_buf, last_buf):
        torch._dynamo.mark_static_address(b)
    # dynamic inputs the graph reads (updated between replays)
    page_idx_buf = torch.zeros(R, pb, dtype=torch.int64, device=dev)     # selection cols
    numfull_buf = torch.zeros(R, 1, dtype=torch.int64, device=dev)       # trailing col
    q_buf = torch.zeros(R, G, hs, dtype=dt, device=dev)
    out_buf = torch.zeros(R, G, hs, dtype=dt, device=dev)
    torch._dynamo.mark_static_address(page_idx_buf)
    torch._dynamo.mark_static_address(numfull_buf)
    torch._dynamo.mark_static_address(q_buf)

    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev), "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=indptr_buf,
        paged_kv_indices_buffer=indices_buf,
        paged_kv_last_page_len_buffer=last_buf)

    # Single combined 5D KV in FlashInfer NHD layout (num_blocks, 2, page, n_kv=1, hs),
    # built as a strided VIEW of the combined-slot cache[nb,page,Hk,2hs] for head 0.
    # canonicalize_singleton_dim_strides fixes the num_kv_heads=1 singleton stride that
    # FlashInfer otherwise misreads as head_dim 0 (the "vo=0" bug). Mirrors vLLM.
    h = 0
    kv_view = (cache[:, :, h, :]                    # [nb, page, 2hs]  strided
               .view(num_blocks, page, 2, hs)       # [nb, page, 2, hs] (last dim contig)
               .permute(0, 2, 1, 3)                 # [nb, 2, page, hs]
               .unsqueeze(3))                        # [nb, 2, page, 1, hs]
    kv_view = canonicalize_singleton_dim_strides(kv_view)

    def plan_for(seq_len):
        nf = (seq_len - 1) // page
        indptr_buf.copy_(torch.arange(0, (R + 1) * (pb + 1), pb + 1, device=dev).int())
        last_buf.fill_(seq_len - nf * page)
        w.plan(indptr_buf, indices_buf, last_buf, G, 1, hs, page,
               q_data_type=dt, kv_data_type=dt, sm_scale=scale)
        return nf

    def write_indices_and_run():
        # IN-GRAPH: gather physical pages from block_table by selection cols, + trailing
        sel = torch.gather(block_table, 1, page_idx_buf.int())          # (R, pb)
        trail = torch.gather(block_table, 1, numfull_buf.int())          # (R, 1)
        idx = torch.cat([sel, trail], dim=1).reshape(-1)                 # (R*(pb+1),)
        indices_buf.copy_(idx)
        o = w.run(q_buf, kv_view)
        out_buf.copy_(o)

    # ---- capture at seq_len0 ----
    seq0 = 8192
    nf0 = plan_for(seq0)
    # warmup selection
    for r in range(R):
        page_idx_buf[r] = torch.randperm(nf0, device=dev)[:pb]
        numfull_buf[r, 0] = nf0
    q_buf.copy_(torch.randn(R, G, hs, dtype=dt, device=dev))
    for _ in range(3):
        write_indices_and_run()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        write_indices_and_run()

    # ---- replay 1: new selection + new query, same seq ----
    torch.manual_seed(1)
    for r in range(R):
        page_idx_buf[r] = torch.randperm(nf0, device=dev)[:pb]
    q_buf.copy_(torch.randn(R, G, hs, dtype=dt, device=dev))
    plan_for(seq0)                          # re-plan (fast path); schedule same
    g.replay(); torch.cuda.synchronize()
    err1 = 0.0
    for r in range(R):
        for gi in range(G):
            rf = ref(q_buf[r, gi], cache, block_table[r, page_idx_buf[r]],
                     int(block_table[r, nf0]), seq0 - nf0 * page)
            err1 = max(err1, (rf.float() - out_buf[r, gi].float()).abs().max().item())
    print(f"replay1 (new sel+q, seq={seq0}): max err = {err1:.4f} "
          f"{'OK' if err1 < 0.05 else '*** FAIL ***'}")

    # ---- replay 2: different seq_len (re-plan changes last_page_len) ----
    seq1 = 8203                              # non-page-aligned -> trailing_len != page
    nf1 = (seq1 - 1) // page
    for r in range(R):
        page_idx_buf[r] = torch.randperm(nf1, device=dev)[:pb]
        numfull_buf[r, 0] = nf1
    q_buf.copy_(torch.randn(R, G, hs, dtype=dt, device=dev))
    plan_for(seq1)
    g.replay(); torch.cuda.synchronize()
    err2 = 0.0
    for r in range(R):
        for gi in range(G):
            rf = ref(q_buf[r, gi], cache, block_table[r, page_idx_buf[r]],
                     int(block_table[r, nf1]), seq1 - nf1 * page)
            err2 = max(err2, (rf.float() - out_buf[r, gi].float()).abs().max().item())
    print(f"replay2 (new seq={seq1}, trail_len={seq1 - nf1 * page}): max err = {err2:.4f} "
          f"{'OK' if err2 < 0.05 else '*** FAIL ***'}")
    print("VERDICT:", "in-graph index write + captured run WORKS across replays"
          if max(err1, err2) < 0.05 else "*** cudagraph mechanic BROKEN ***")


if __name__ == "__main__":
    main()
