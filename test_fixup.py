#!/usr/bin/env python3
"""Bit-identical validation: fused Triton fixup kernel vs the original torch chain.

Reproduces the exact invalid-index remap semantics of `_radix_topk` and asserts
`torch.equal` between:
  (a) the ORIGINAL torch chain (arange/clamp/remainder/lt/ge/or/where), and
  (b) the FUSED Triton kernel (_fixup_idx_fused).

This is the GATE: no perf measurement until every case is bit-identical.

Run:
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  /NHNHOME/jiwonsong/miniconda3/envs/vllm/bin/python test_fixup.py
"""
import sys
import torch

from vllm.v1.attention.ops.triton_lrosa_score_topk import _fixup_idx_fused

DEV = "cuda"


def torch_chain(topk_idx: torch.Tensor, seq_lens_rows: torch.Tensor) -> torch.Tensor:
    """The ORIGINAL chain, recomputed with fresh tensors (semantics ref)."""
    rows, k = topk_idx.shape
    slot_off = torch.arange(k, dtype=torch.int32, device=topk_idx.device)
    sl = torch.clamp(seq_lens_rows.view(rows, 1), min=1)              # max(seq_len,1)
    fill = torch.remainder(slot_off.view(1, k).expand(rows, k), sl)  # j % sl
    neg = torch.lt(topk_idx, 0)
    oob = torch.ge(topk_idx, seq_lens_rows.view(rows, 1))            # >= RAW seq_len
    invalid = torch.logical_or(neg, oob)
    res = torch.where(invalid, fill, topk_idx)
    return res.to(torch.int32)


def make_case(name, topk_idx, seq_lens_rows):
    topk_idx = topk_idx.to(torch.int32).contiguous().to(DEV)
    seq_lens_rows = seq_lens_rows.to(torch.int32).contiguous().to(DEV)
    return name, topk_idx, seq_lens_rows


def build_cases():
    cases = []
    g = torch.Generator(device="cpu").manual_seed(1234)

    # --- hand-crafted edge cases -------------------------------------------
    # all-valid: every idx in [0, seq_len)
    rows, k = 8, 256
    sl = torch.full((rows,), 300)
    idx = torch.randint(0, 300, (rows, k), generator=g)
    cases.append(make_case("all-valid (sl=300>k)", idx, sl))

    # seq_len < k → radix -1 padding in the tail; mix of valid + (-1) pad
    rows, k = 8, 256
    sl = torch.tensor([10, 10, 10, 10, 10, 10, 10, 10])
    idx = torch.randint(0, 10, (rows, k), generator=g)
    idx[:, 10:] = -1  # radix pads unfilled slots with -1
    cases.append(make_case("seq_len(10) < k, -1 pad tail", idx, sl))

    # seq_len == 0 (clamp→1, fill=j%1=0, everything invalid since idx>=0 always
    # and idx<seq_len impossible)
    rows, k = 4, 256
    sl = torch.zeros(rows)
    idx = torch.full((rows, k), -1)
    cases.append(make_case("seq_len=0 (all -1)", idx, sl))

    # seq_len == 0 but with some stale positive indices (must all be remapped)
    rows, k = 4, 256
    sl = torch.zeros(rows)
    idx = torch.randint(0, 5000, (rows, k), generator=g)
    cases.append(make_case("seq_len=0 (stale positives)", idx, sl))

    # seq_len == 1 (fill=j%1=0). idx 0 valid, anything else invalid
    rows, k = 4, 256
    sl = torch.ones(rows)
    idx = torch.randint(-2, 4, (rows, k), generator=g)
    cases.append(make_case("seq_len=1", idx, sl))

    # all-invalid: every idx >= seq_len (stale OOB positives)
    rows, k = 8, 256
    sl = torch.full((rows,), 50)
    idx = torch.randint(50, 9999, (rows, k), generator=g)  # all >= 50
    cases.append(make_case("all-invalid (stale OOB positive)", idx, sl))

    # all-invalid: all -1
    rows, k = 8, 256
    sl = torch.full((rows,), 128)
    idx = torch.full((rows, k), -1)
    cases.append(make_case("all-invalid (all -1)", idx, sl))

    # k % seq_len == 0  (k=256, seq_len=64 → 256%64==0)
    rows, k = 8, 256
    sl = torch.full((rows,), 64)
    idx = torch.randint(0, 64, (rows, k), generator=g)
    idx[:, 64:] = -1
    cases.append(make_case("k%%seq_len==0 (sl=64)", idx, sl))

    # k % seq_len != 0  (k=256, seq_len=70 → 256%70==46)
    rows, k = 8, 256
    sl = torch.full((rows,), 70)
    idx = torch.randint(0, 70, (rows, k), generator=g)
    idx[:, 70:] = -1
    cases.append(make_case("k%%seq_len!=0 (sl=70)", idx, sl))

    # mixed per-row seq_lens incl 0,1, <k, >k, plus stale OOB positives
    rows, k = 64, 256
    sl = torch.randint(0, 600, (rows,), generator=g)
    sl[0] = 0; sl[1] = 1; sl[2] = 256; sl[3] = 255; sl[4] = 257
    idx = torch.randint(-1, 700, (rows, k), generator=g)
    # sprinkle exact-boundary indices (idx == seq_len-1 valid, idx == seq_len OOB)
    for r in range(rows):
        s = int(sl[r])
        idx[r, 0] = s - 1   # last valid (or -1 if s==0)
        idx[r, 1] = s       # first OOB
        idx[r, 2] = -1      # radix pad
    cases.append(make_case("mixed rows=64 + boundaries", idx, sl))

    # --- assorted shapes ----------------------------------------------------
    for (rows, k) in [(8, 256), (64, 256), (1, 256), (16, 256), (32, 256),
                      (8, 128), (64, 64), (128, 256), (8, 1)]:
        sl = torch.randint(0, 2 * k + 50, (rows,), generator=g)
        idx = torch.randint(-1, 3 * k, (rows, k), generator=g)
        cases.append(make_case(f"random rows={rows} k={k}", idx, sl))

    # --- fuzz: many random cases -------------------------------------------
    for t in range(40):
        rows = int(torch.randint(1, 80, (1,), generator=g).item())
        k = 256
        # bias seq_len to span 0..1..<k..>k
        maxsl = int(torch.randint(1, 4 * k, (1,), generator=g).item())
        sl = torch.randint(0, maxsl, (rows,), generator=g)
        idx = torch.randint(-3, 3 * k, (rows, k), generator=g)
        cases.append(make_case(f"fuzz#{t} rows={rows} k={k} maxsl={maxsl}", idx, sl))

    return cases


def main():
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        sys.exit(2)
    cases = build_cases()
    n_pass = n_fail = 0
    for name, idx, sl in cases:
        ref = torch_chain(idx, sl)
        # fused writes into a caller buffer (persistent scratch in prod; here a
        # plain tensor — wrapper itself does no allocation in the hot path).
        res = torch.empty_like(idx)
        _fixup_idx_fused(idx, sl, res)
        ok = torch.equal(ref, res)
        if ok:
            n_pass += 1
            print(f"  PASS  {name}")
        else:
            n_fail += 1
            mism = (ref != res)
            nmis = int(mism.sum().item())
            # show a couple mismatching slots
            r, c = torch.nonzero(mism, as_tuple=True)
            ex = []
            for i in range(min(5, nmis)):
                rr, cc = int(r[i]), int(c[i])
                ex.append(f"[{rr},{cc}] idx={int(idx[rr,cc])} sl={int(sl[rr])} "
                          f"ref={int(ref[rr,cc])} got={int(res[rr,cc])}")
            print(f"  FAIL  {name}  ({nmis} mismatches): " + "; ".join(ex))
    print(f"\n==== {n_pass} pass / {n_fail} fail / {len(cases)} total ====")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
