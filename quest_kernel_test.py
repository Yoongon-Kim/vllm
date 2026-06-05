"""Standalone numerical correctness test for the Quest vLLM kernels.

Builds a small paged combined-slot cache by hand, exercises the three Quest
kernels (prefill min/max build, decode running min/max update, page
upper-bound score) + the token-expansion helper, and checks each against a
pure-PyTorch reference that mirrors pca's quest/quest.py math. No engine,
no model — just the ops.

Run:
  CUDA_VISIBLE_DEVICES=2 /home/jiwonsong/.conda/envs/gaep_vllm/bin/python \
      quest_kernel_test.py
"""
import torch

from vllm.v1.attention.ops.triton_quest import (
    quest_blocksparse_attn,
    quest_build_page_minmax_prefill,
    quest_minmax_update,
    quest_num_splits,
    quest_page_score,
    quest_pages_to_token_idx,
)
from vllm.v1.attention.ops.triton_lrosa_store import lrosa_store
from vllm.v1.attention.ops.triton_lrosa_gather import lrosa_gather

torch.manual_seed(0)
dev = "cuda"
HS = 128
PS = 16            # page_size == block_size
H_KV = 2
H_Q = 8            # GQA group 4
GQA = H_Q // H_KV
SLOT = 4 * HS      # [K|V|min|max]


def make_cache(num_blocks):
    return torch.zeros(num_blocks, PS, H_KV, SLOT, dtype=torch.bfloat16, device=dev)


def store_seq(kv_cache, block_ids, K, V):
    """Store a full sequence's K/V token-by-token via lrosa_store + slot_mapping."""
    L = K.shape[0]
    slot_mapping = torch.empty(L, dtype=torch.int64, device=dev)
    for t in range(L):
        b = block_ids[t // PS]
        slot_mapping[t] = b * PS + (t % PS)
    lrosa_store(K, V, kv_cache, slot_mapping)
    return slot_mapping


def ref_page_minmax(K, page_size):
    """K: (L, H_kv, hs) → per full-page min/max (n_full, H_kv, hs)."""
    L = K.shape[0]
    nf = L // page_size
    Kp = K[: nf * page_size].reshape(nf, page_size, K.shape[1], K.shape[2])
    return Kp.min(dim=1).values, Kp.max(dim=1).values, nf


def read_repr_minmax(kv_cache, block_ids, nf):
    """Read back the min/max stored at each page's representative slot."""
    mins, maxs = [], []
    for p in range(nf):
        b = block_ids[p]
        mins.append(kv_cache[b, 0, :, 2 * HS:3 * HS].float())
        maxs.append(kv_cache[b, 0, :, 3 * HS:4 * HS].float())
    return torch.stack(mins, 0), torch.stack(maxs, 0)  # (nf, H_kv, hs)


def test_prefill_minmax():
    print("\n=== test 1: prefill page min/max build ===")
    L = 100                      # 6 full pages + 4 trailing
    # Pre-allocate blocks for the full sequence + 20 decode steps (test 2 grows
    # to ~120 tokens → up to 8 pages). Build a block-table covering all of them
    # so store + kernel agree on physical blocks throughout.
    max_pages = 9
    nblk = max_pages + 4
    kv = make_cache(nblk)
    block_ids = list(range(2, 2 + max_pages))  # arbitrary physical blocks
    K = torch.randn(L, H_KV, HS, dtype=torch.bfloat16, device=dev)
    V = torch.randn(L, H_KV, HS, dtype=torch.bfloat16, device=dev)
    # store only the L prefill tokens (first ceil(L/PS) pages)
    store_seq(kv, block_ids[: (L + PS - 1) // PS], K, V)

    bt = torch.zeros(64, dtype=torch.int32, device=dev)
    bt[: len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32, device=dev)
    quest_build_page_minmax_prefill(K, kv, bt, PS, HS, prefill_start_pos=0)

    rmin, rmax, nf = ref_page_minmax(K, PS)
    gmin, gmax = read_repr_minmax(kv, block_ids, nf)
    emin = (rmin - gmin).abs().max().item()
    emax = (rmax - gmax).abs().max().item()
    print(f"  full pages={nf} min err={emin:.4e} max err={emax:.4e}")
    # trailing partial page (positions 96..99) repr check
    tb = block_ids[nf]
    tmin = kv[tb, 0, :, 2 * HS:3 * HS].float()
    rtmin = K[nf * PS:].float().min(dim=0).values
    et = (tmin - rtmin).abs().max().item()
    print(f"  trailing partial-page min err={et:.4e}")
    assert emin < 1e-2 and emax < 1e-2 and et < 1e-2, "prefill min/max mismatch"
    print("  PASS")
    return kv, block_ids, K, V, bt


def test_decode_update(kv, block_ids, K, V, bt):
    print("\n=== test 2: decode running min/max update ===")
    L0 = K.shape[0]               # 100
    # Decode 20 more tokens; after each, the current partial page's repr min/max
    # must equal the reference min/max over its tokens so far.
    Kfull = K.clone()
    errs = []
    for step in range(20):
        seq_len = L0 + step + 1   # length INCLUDING the new token
        newK = torch.randn(1, H_KV, HS, dtype=torch.bfloat16, device=dev)
        Kfull = torch.cat([Kfull, newK], 0)
        # store the new token (block_ids pre-allocated in test 1, bt matches)
        t = seq_len - 1
        b = block_ids[t // PS]
        sm = torch.tensor([b * PS + t % PS], dtype=torch.int64, device=dev)
        lrosa_store(newK, newK, kv, sm)
        # run update kernel
        seq_t = torch.tensor([seq_len], dtype=torch.int32, device=dev)
        quest_minmax_update(newK, kv, bt.view(1, -1), seq_t, HS, PS)
        # check the current (partial) block repr
        page = t // PS
        lo = page * PS
        ref_min = Kfull[lo:seq_len].float().min(dim=0).values  # (H_kv,hs)
        got_min = kv[block_ids[page], 0, :, 2 * HS:3 * HS].float()
        errs.append((ref_min - got_min).abs().max().item())
    print(f"  max running-min err over 20 decode steps={max(errs):.4e}")
    assert max(errs) < 1e-2, "decode running update mismatch"
    print("  PASS")
    return Kfull


def _grow(block_ids, kv):
    nb = max(block_ids) + 1
    block_ids.append(nb)
    return nb


def test_page_score():
    print("\n=== test 3: page upper-bound score (vs pca reference) ===")
    L = 8000                      # long ctx, 500 pages
    nf_pages = L // PS
    nblk = nf_pages + 8
    kv = make_cache(nblk)
    block_ids = list(range(nblk - nf_pages, nblk))  # arbitrary contiguous
    K = torch.randn(L, H_KV, HS, dtype=torch.bfloat16, device=dev)
    V = torch.randn(L, H_KV, HS, dtype=torch.bfloat16, device=dev)
    store_seq(kv, block_ids, K, V)
    bt = torch.zeros(1, 600, dtype=torch.int32, device=dev)
    bt[0, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32, device=dev)
    quest_build_page_minmax_prefill(K, kv, bt[0], PS, HS, prefill_start_pos=0)

    # group-mean query (per kv-head)
    q = torch.randn(1, H_Q, HS, dtype=torch.bfloat16, device=dev)
    q_kv = q.view(1, H_KV, GQA, HS).mean(2)                      # (1,H_kv,hs)
    seq = torch.tensor([L], dtype=torch.int32, device=dev)
    scores = quest_page_score(q_kv, kv, bt, seq, PS, HS)         # (1,H_kv,600)

    # reference: score[h,p] = Σ_c max(q[c]*min, q[c]*max) over SELECTABLE pages.
    # The kernel reserves the last full page (idx (seq_len-1)//PS) as the
    # always-attended trailing page, so only pages [0, num_full) are scored.
    rmin, rmax, _ = ref_page_minmax(K, PS)                       # (500,H_kv,hs)
    num_full = (L - 1) // PS                                     # 499 selectable
    rmin, rmax = rmin[:num_full], rmax[:num_full]
    qf = q_kv[0].float()                                         # (H_kv,hs)
    prod = torch.maximum(qf[None] * rmin, qf[None] * rmax)       # (nf,H_kv,hs)
    ref = prod.sum(-1).transpose(0, 1)                           # (H_kv,nf)
    got = scores[0, :, :num_full]
    err = (ref - got).abs().max().item()
    rel = err / ref.abs().max().item()
    nf = num_full
    print(f"  selectable pages={nf}  score abs err={err:.4e}  rel={rel:.2e}")
    # pages >= num_full must be -inf (trailing + beyond-seq)
    inf_ok = torch.isinf(scores[0, :, nf:nf_pages]).all().item()
    print(f"  pages>=num_full are -inf: {inf_ok}")
    assert rel < 1e-2 and inf_ok, "page score mismatch"

    # top-K page selection agreement vs reference argsort
    pb = 16
    ref_top = ref.topk(pb, -1).indices.sort(-1).values            # (H_kv,pb)
    got_top = got.topk(pb, -1).indices.sort(-1).values
    agree = (ref_top == got_top).float().mean().item()
    print(f"  top-{pb} page agreement: {agree*100:.1f}%")
    assert agree > 0.95, "top-K disagreement"
    print("  PASS")


def test_token_expand():
    print("\n=== test 4: pages → token indices + seqused ===")
    nd, hk, pb = 1, H_KV, 15
    n_fac = (pb + 1) * PS  # 256
    seq_len = 8003
    seq = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    num_full = torch.tensor([(seq_len - 1) // PS], dtype=torch.int64, device=dev)
    # pick some valid pages
    page_idx = torch.randint(0, int(num_full[0]), (nd, hk, pb),
                             dtype=torch.int32, device=dev)
    tok, seqused = quest_pages_to_token_idx(page_idx, seq, PS, n_fac, num_full)
    print(f"  token_idx shape={tuple(tok.shape)} (want (1,{hk},{n_fac}))")
    # selected token block i should be page_idx[i]*PS + [0..PS)
    for h in range(hk):
        for j in range(pb):
            base = int(page_idx[0, h, j]) * PS
            seg = tok[0, h, j * PS:(j + 1) * PS].tolist()
            assert seg == list(range(base, base + PS)), f"page {j} expand wrong"
    # trailing block = [num_full*PS .. seq_len) then clamp
    nfp = int(num_full[0])
    trail = tok[0, 0, pb * PS:]
    expect0 = nfp * PS
    trailing_len = seq_len - nfp * PS
    assert int(trail[0]) == expect0, "trailing start wrong"
    assert int(seqused[0]) == pb * PS + trailing_len, \
        f"seqused {int(seqused[0])} != {pb*PS+trailing_len}"
    print(f"  trailing_len={trailing_len}  seqused={int(seqused[0])} (=pb*16+trail)")
    print("  PASS")


def _run_blocksparse_case(page_budget, L=8003):
    """Run quest_blocksparse_attn at a given page_budget (forces single-pass
    when num_splits==1, split-KV when >1) and return max rel err vs a
    dense-attention-on-selected-tokens reference."""
    nf_pages = (L + PS - 1) // PS
    nblk = nf_pages + 8
    kv = make_cache(nblk)
    block_ids = list(range(nblk - nf_pages, nblk))
    K = torch.randn(L, H_KV, HS, dtype=torch.bfloat16, device=dev) * 0.5
    V = torch.randn(L, H_KV, HS, dtype=torch.bfloat16, device=dev) * 0.5
    store_seq(kv, block_ids, K, V)
    bt = torch.zeros(1, nf_pages + 8, dtype=torch.int32, device=dev)
    bt[0, : len(block_ids)] = torch.tensor(block_ids, dtype=torch.int32, device=dev)

    num_full = (L - 1) // PS
    page_idx = torch.zeros(1, H_KV, page_budget, dtype=torch.int32, device=dev)
    for h in range(H_KV):
        perm = torch.randperm(num_full, device=dev)[:page_budget].to(torch.int32)
        page_idx[0, h] = perm

    q = torch.randn(1, H_Q, HS, dtype=torch.bfloat16, device=dev)
    out = torch.zeros(1, H_Q, HS, dtype=torch.bfloat16, device=dev)
    scale = HS ** -0.5
    ns = quest_num_splits(page_budget)
    pa = pm = pl = None
    if ns > 1:
        pa = torch.zeros(1, H_Q, ns, HS, dtype=torch.float32, device=dev)
        pm = torch.zeros(1, H_Q, ns, dtype=torch.float32, device=dev)
        pl = torch.zeros(1, H_Q, ns, dtype=torch.float32, device=dev)
    quest_blocksparse_attn(q, kv, page_idx, bt,
                           torch.tensor([L], dtype=torch.int32, device=dev),
                           out, scale, PS, HS, GQA,
                           partial_acc=pa, partial_m=pm, partial_l=pl)

    trail_start = num_full * PS
    errs = []
    for hq in range(H_Q):
        h = hq // GQA
        toks = []
        for j in range(page_budget):
            p = int(page_idx[0, h, j])
            toks.extend(range(p * PS, p * PS + PS))
        toks.extend(range(trail_start, L))
        toks = torch.tensor(sorted(set(toks)), device=dev)
        Ksel = K[toks, h].float()
        Vsel = V[toks, h].float()
        qh = q[0, hq].float()
        w = torch.softmax((Ksel @ qh) * scale, dim=0)
        ref = (w[:, None] * Vsel).sum(0)
        got = out[0, hq].float()
        errs.append((ref - got).abs().max().item() / (ref.abs().max().item() + 1e-6))
    return max(errs), ns


def test_blocksparse_attn():
    print("\n=== test 5: block-sparse attention (single-pass + split-KV) ===")
    for pb in (15, 127):           # budget 256 (single-pass) + 2048 (split-KV)
        err, ns = _run_blocksparse_case(pb)
        print(f"  page_budget={pb} num_splits={ns} max rel err={err:.3e}")
        assert err < 1e-2, f"block-sparse mismatch at page_budget={pb}"
    print("  PASS")


if __name__ == "__main__":
    kv, block_ids, K, V, bt = test_prefill_minmax()
    test_decode_update(kv, block_ids, K, V, bt)
    test_page_score()
    test_token_expand()
    test_blocksparse_attn()
    print("\nALL QUEST KERNEL TESTS PASSED")
