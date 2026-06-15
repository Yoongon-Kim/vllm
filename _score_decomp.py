"""Decompose the LRoSA score-scan into score-compute vs radix-topk, across ctx.

Both are O(ctx). Knowing the split tells us what to attack: if score-compute
dominates, a coarse/hierarchical pre-filter or kernel tuning; if topk dominates,
a cheaper/approximate selection. fp8/contig only sped the whole thing ~1.16-1.4x.
"""
import argparse, time
import torch
from vllm.v1.attention.ops.triton_lrosa_score_topk import (
    lrosa_score, _radix_topk, lrosa_score_topk,
)

ap = argparse.ArgumentParser()
ap.add_argument("--num_reqs", type=int, default=32)
ap.add_argument("--ctx", type=int, default=131072)
ap.add_argument("--n_fac", type=int, default=2048)
ap.add_argument("--cs_h", type=int, default=32)
ap.add_argument("--head_size", type=int, default=128)
ap.add_argument("--n_kv", type=int, default=8)
ap.add_argument("--block_size", type=int, default=16)
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()

dev, dt = "cuda", torch.bfloat16
R, L, Hk, hs, cs_h = a.num_reqs, a.ctx, a.n_kv, a.head_size, a.cs_h
n_fac = min(a.n_fac, L)
slot = 2 * hs + cs_h
bpr = (L + a.block_size - 1) // a.block_size
num_blocks = R * bpr + 1
torch.manual_seed(0)
kv = torch.randn(num_blocks, a.block_size, Hk, slot, dtype=dt, device=dev)
bt = torch.arange(R * bpr, device=dev, dtype=torch.int32).reshape(R, bpr)
proj_q = torch.randn(R, Hk, cs_h, dtype=dt, device=dev)
seq = torch.full((R,), L, dtype=torch.int32, device=dev)
scores_buf = torch.empty(R, Hk, L, dtype=torch.float32, device=dev)


def _t(fn, reps=a.reps, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts) * 1000.0


# precompute a scores buffer for the topk-only timing
lrosa_score(proj_q, kv, bt, seq, hs, cs_h, scores_out=scores_buf)
torch.cuda.synchronize()


def score_only():
    lrosa_score(proj_q, kv, bt, seq, hs, cs_h, scores_out=scores_buf)


def combined():
    lrosa_score_topk(proj_q, kv, bt, seq, n_fac, hs, cs_h, use_radix=True)


# warm + trigger _C op registration via the combined path before timing
combined()
torch.cuda.synchronize()
s = _t(score_only)
c = _t(combined)
t = max(c - s, 0.0)  # radix-topk = combined - score-compute (op-registration safe)
NL = 36
print(f"R={R} ctx={L} n_fac={n_fac} cs_h={cs_h} head={hs} kv={Hk}")
print(f"  per-layer: score-compute={s:.4f}  radix-topk={t:.4f}  combined={c:.4f}ms"
      f"  (compute {s/c*100:.0f}%, topk {t/c*100:.0f}%)")
print(f"  per-step (x{NL}): score={s*NL:.2f}  topk={t*NL:.2f}  combined={c*NL:.2f}ms")
