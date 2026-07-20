"""Phase 1: does a FlashInfer per-head page-list attend reproduce
quest_blocksparse_attn's output? Validates the trailing-page + last_page_len +
selected-page-set logic in isolation (no vLLM backend / no cudagraph).

For each kv-head h: page list per request = [valid selected full pages] + [trailing
page], last_page_len = trailing_len. Run Hk FlashInfer decodes (num_kv_heads=1,
num_qo_heads=num_kv_groups) on the STRIDED per-head view of the combined-slot cache.
Compare vs quest_blocksparse_attn (the current Triton reference).
"""
import sys
import torch
sys.path.insert(0, "/NHNHOME/jiwonsong/vllm")
import flashinfer
from vllm.v1.attention.ops.triton_quest import quest_blocksparse_attn, quest_num_splits

dev, dt = "cuda", torch.bfloat16


def run_case(hs, Hk, Hq, page, n_fac, seq_lens_list, tag):
    R = len(seq_lens_list)
    G = Hq // Hk
    pb = n_fac // page - 1
    scale = 1.0 / hs ** 0.5
    torch.manual_seed(0)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=dev)
    max_blocks = (int(seq_lens.max()) + page - 1) // page + 2
    num_blocks = R * max_blocks + 1
    cache = torch.randn(num_blocks, page, Hk, 2 * hs, dtype=dt, device=dev)
    block_table = torch.arange(R * max_blocks, device=dev,
                               dtype=torch.int32).reshape(R, max_blocks)
    q = torch.randn(R, Hq, hs, dtype=dt, device=dev)
    num_full = (seq_lens.to(torch.int64) - 1) // page                       # (R,)

    # distinct valid page_idx per (r,h); pad short with col=nf (invalid -> masked)
    page_idx = torch.zeros(R, Hk, pb, dtype=torch.int32, device=dev)
    for r in range(R):
        nf = int(num_full[r].item())
        for h in range(Hk):
            perm = torch.randperm(max(nf, 1), device=dev)[:pb]
            if perm.numel() < pb:
                pad = torch.full((pb - perm.numel(),), nf, device=dev, dtype=perm.dtype)
                perm = torch.cat([perm, pad])
            page_idx[r, h] = perm.to(torch.int32)

    # --- reference: quest_blocksparse_attn ---
    out_ref = torch.empty(R, Hq, hs, dtype=dt, device=dev)
    ns = quest_num_splits(pb)
    pacc = pm = pl = None
    if ns > 1:
        pacc = torch.empty(R, Hq, ns, hs, dtype=torch.float32, device=dev)
        pm = torch.empty(R, Hq, ns, dtype=torch.float32, device=dev)
        pl = torch.empty(R, Hq, ns, dtype=torch.float32, device=dev)
    quest_blocksparse_attn(
        query=q, kv_cache=cache, page_idx=page_idx, block_table=block_table,
        seq_lens=seq_lens, output=out_ref, scale=scale, page_size=page,
        head_size=hs, num_kv_groups=G, partial_acc=pacc, partial_m=pm, partial_l=pl)

    # --- FlashInfer per-head ---
    out_fi = torch.empty(R, Hq, hs, dtype=dt, device=dev)
    for h in range(Hk):
        k_view = cache[:, :, h, :hs].unsqueeze(2)          # [nb, page, 1, hs] strided
        v_view = cache[:, :, h, hs:2 * hs].unsqueeze(2)
        indptr_l, idx_l, last_l = [0], [], []
        for r in range(R):
            nf = int(num_full[r].item()); sl = int(seq_lens[r].item())
            cols = page_idx[r, h]
            valid = cols[cols < nf]                        # valid selected columns
            sel_phys = block_table[r, valid.long()]
            trail_phys = block_table[r, nf].view(1)        # trailing page LAST
            pages = torch.cat([sel_phys, trail_phys])
            idx_l.append(pages)
            indptr_l.append(indptr_l[-1] + pages.numel())
            last_l.append(sl - nf * page)                  # trailing_len in [1,page]
        indices = torch.cat(idx_l).to(torch.int32)
        indptr = torch.tensor(indptr_l, dtype=torch.int32, device=dev)
        last_len = torch.tensor(last_l, dtype=torch.int32, device=dev)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
            kv_layout="NHD")
        try:
            w.plan(indptr, indices, last_len, G, 1, hs, page,
                   q_data_type=dt, kv_data_type=dt, sm_scale=scale)
            o = w.run(q[:, h * G:(h + 1) * G, :].contiguous(), (k_view, v_view))
        except TypeError:
            w.plan(indptr, indices, last_len, G, 1, hs, page,
                   q_data_type=dt, kv_data_type=dt)
            o = w.run(q[:, h * G:(h + 1) * G, :].contiguous(), (k_view, v_view),
                      sm_scale=scale)
        out_fi[:, h * G:(h + 1) * G, :] = o

    err = (out_ref.float() - out_fi.float()).abs().max().item()
    denom = out_ref.float().abs().max().item() + 1e-6
    rel = err / denom
    print(f"{tag} (seq={seq_lens_list}): max|ref-fi|={err:.4f} rel={rel:.4f} "
          f"{'OK' if rel < 0.02 else '*** MISMATCH ***'}")


if __name__ == "__main__":
    print("=== Gemma-full head512 (Hk2 Hq16 n_fac2048 pb127) ===")
    run_case(512, 2, 16, 16, 2048, [8192, 8192, 8192], "long-uniform")
    run_case(512, 2, 16, 16, 2048, [8192, 40000, 16384], "long-varying")
    run_case(512, 2, 16, 16, 2048, [1024, 1024, 1024], "short(dense)")
    run_case(512, 2, 16, 16, 2048, [1024, 8192, 40000], "mixed")
    print("=== Qwen3-8B head128 (Hk8 Hq32 n_fac2048 pb127) ===")
    run_case(128, 8, 32, 16, 2048, [8192, 8192, 8192], "long-uniform")
    run_case(128, 8, 32, 16, 2048, [8192, 40000, 16384], "long-varying")
    run_case(128, 8, 32, 16, 2048, [1024, 8192, 40000], "mixed")
    print("[done]")
