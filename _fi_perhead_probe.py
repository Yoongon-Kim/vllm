"""Feasibility probe: can FlashInfer paged decode read a per-kv-head STRIDED view
of QUEST's combined-slot cache [num_blocks, page, Hk, 2*hs] (K=[..,:hs], V=[..,hs:2hs])
WITHOUT a contiguous copy, and produce correct GQA output?

If yes -> faithful per-head FlashInfer attend = Hk decode calls, no per-step copy.
If it silently requires contiguous -> per-step copy needed (dealbreaker).

Tests head_dim 512 (Gemma full) and 128 (Qwen3), num_kv_heads=1 per call,
num_qo_heads = num_kv_groups. Verifies output vs a manual reference.
"""
import sys
import torch
sys.path.insert(0, "/NHNHOME/jiwonsong/vllm")
import flashinfer

dev = "cuda"
dt = torch.bfloat16


def ref_attn(q, k, v, scale):
    # q [G, hs], k/v [T, hs]  -> [G, hs]
    s = (q.float() @ k.float().T) * scale         # [G, T]
    p = torch.softmax(s, dim=-1)
    return (p @ v.float()).to(q.dtype)


def probe(hs, Hk, Hq, page, nb_per_req, R, tag):
    print(f"\n=== {tag}: hs={hs} Hk={Hk} Hq={Hq} page={page} pages/req={nb_per_req} R={R} ===")
    G = Hq // Hk
    scale = 1.0 / hs ** 0.5
    torch.manual_seed(0)
    num_blocks = R * nb_per_req + 1
    # combined-slot cache exactly like QUEST: [nb, page, Hk, 2*hs]
    cache = torch.randn(num_blocks, page, Hk, 2 * hs, dtype=dt, device=dev)
    q = torch.randn(R, Hq, hs, dtype=dt, device=dev)

    # per-request page table: request r owns pages [r*nb_per_req, (r+1)*nb_per_req)
    # (all pages full -> last_page_len = page)
    out_fi = torch.zeros(R, Hq, hs, dtype=dt, device=dev)
    ok_all = True
    for h in range(Hk):
        # STRIDED per-head views (no .contiguous())
        k_view = cache[:, :, h, :hs]          # [nb, page, hs]  strided
        v_view = cache[:, :, h, hs:2 * hs]    # [nb, page, hs]  strided
        k_nhd = k_view.unsqueeze(2)           # [nb, page, 1, hs] num_kv_heads=1
        v_nhd = v_view.unsqueeze(2)
        print(f"  head{h}: k_view contig={k_view.is_contiguous()} "
              f"strides={tuple(k_nhd.stride())} shape={tuple(k_nhd.shape)}")
        indptr = torch.arange(0, (R + 1) * nb_per_req, nb_per_req,
                              dtype=torch.int32, device=dev)
        indices = torch.arange(R * nb_per_req, dtype=torch.int32, device=dev)
        last = torch.full((R,), page, dtype=torch.int32, device=dev)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev),
            kv_layout="NHD")
        try:
            w.plan(indptr, indices, last, G, 1, hs, page,
                   q_data_type=dt, kv_data_type=dt)
            qh = q[:, h * G:(h + 1) * G, :].contiguous()   # [R, G, hs]
            o = w.run(qh, (k_nhd, v_nhd))                  # [R, G, hs]
            out_fi[:, h * G:(h + 1) * G, :] = o
        except Exception as e:
            print(f"    !! FlashInfer FAILED on strided view: {type(e).__name__}: {str(e)[:160]}")
            ok_all = False
            # retry with contiguous to see if that is the requirement
            try:
                o = w.run(qh, (k_nhd.contiguous(), v_nhd.contiguous()))
                print(f"    (contiguous copy WORKS -> FlashInfer requires contiguous per-head cache)")
            except Exception as e2:
                print(f"    (even contiguous failed: {type(e2).__name__}: {str(e2)[:120]})")
            continue

    if not ok_all:
        print("  VERDICT: strided per-head view NOT accepted -> per-step copy needed.")
        return

    # correctness vs manual reference (request 0, all heads)
    max_err = 0.0
    for r in range(min(R, 2)):
        for h in range(Hk):
            pg0 = r * nb_per_req
            k_tok = cache[pg0:pg0 + nb_per_req, :, h, :hs].reshape(-1, hs)
            v_tok = cache[pg0:pg0 + nb_per_req, :, h, hs:2 * hs].reshape(-1, hs)
            for g in range(G):
                qh = q[r, h * G + g, :]
                ref = ref_attn(qh.unsqueeze(0), k_tok, v_tok, scale).squeeze(0)
                got = out_fi[r, h * G + g, :]
                err = (ref.float() - got.float()).abs().max().item()
                max_err = max(max_err, err)
    print(f"  strided view ACCEPTED. max |ref-fi| = {max_err:.4f} "
          f"({'OK' if max_err < 0.05 else 'MISMATCH'})")
    print("  VERDICT: faithful per-head FlashInfer attend feasible WITHOUT copy." if max_err < 0.05
          else "  VERDICT: runs but WRONG output on strided view.")


if __name__ == "__main__":
    probe(hs=512, Hk=2, Hq=16, page=16, nb_per_req=4, R=3, tag="Gemma-full head512")
    probe(hs=128, Hk=8, Hq=32, page=16, nb_per_req=4, R=3, tag="Qwen3-8B head128")
    print("\n[done]")
