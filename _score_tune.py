"""Exact score-kernel tuning: sweep BLOCK_T for the contiguous proj_K score
kernel (same output, just faster). Reports per-layer ms at each BLOCK_T so we
can pick the default. proj_K read is the bandwidth-bound cost; BLOCK_T trades
program count vs per-program coalescing/work."""
import argparse, time
import torch
from vllm.v1.attention.ops.triton_lrosa_score_topk import lrosa_score

ap = argparse.ArgumentParser()
ap.add_argument("--num_reqs", type=int, default=32)
ap.add_argument("--ctx", type=int, default=131072)
ap.add_argument("--cs_h", type=int, default=32)
ap.add_argument("--head_size", type=int, default=128)
ap.add_argument("--n_kv", type=int, default=8)
ap.add_argument("--block_size", type=int, default=16)
ap.add_argument("--fp8", action="store_true")
ap.add_argument("--inslot", action="store_true", help="in-slot proj_K (projk_cache=None)")
ap.add_argument("--reps", type=int, default=30)
a = ap.parse_args()

dev, dt = "cuda", torch.bfloat16
R, L, Hk, hs, cs_h = a.num_reqs, a.ctx, a.n_kv, a.head_size, a.cs_h
slot = 2 * hs + cs_h
bpr = (L + a.block_size - 1) // a.block_size
num_blocks = R * bpr + 1
torch.manual_seed(0)
kv = torch.randn(num_blocks, a.block_size, Hk, slot, dtype=dt, device=dev)
bt = torch.arange(R * bpr, device=dev, dtype=torch.int32).reshape(R, bpr)
proj_q = torch.randn(R, Hk, cs_h, dtype=dt, device=dev)
seq = torch.full((R,), L, dtype=torch.int32, device=dev)
pk = torch.randn(num_blocks, a.block_size, Hk, cs_h, dtype=dt, device=dev)
if a.fp8:
    pk = pk.to(torch.float8_e4m3fn)
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


print(f"R={R} ctx={L} cs_h={cs_h} fp8={a.fp8}  (proj_K read ideal @8TB/s ~ "
      f"{cs_h*Hk*L*R*(1 if a.fp8 else 2)/8e12*1e3:.3f}ms/layer)")
for BT in [32, 64, 128, 256, 512]:
    def run():
        lrosa_score(proj_q, kv, bt, seq, hs, cs_h, block_t=BT,
                    scores_out=scores_buf,
                    projk_cache=(None if a.inslot else pk))
    try:
        ms = _t(run)
        print(f"  BLOCK_T={BT:4d}: {ms:.4f} ms/layer  ({ms*36:.2f} ms/step x36)")
    except Exception as e:
        print(f"  BLOCK_T={BT:4d}: FAIL {type(e).__name__}: {str(e)[:60]}")
