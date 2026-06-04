# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness test: streaming top-K kernel vs the 2-pass score+topk
implementation, on identical paged inputs. Threshold: ≥0.99 Jaccard
(precision-level mismatches at the tail of the top-K rank may flip 1-2
positions; same threshold as the existing test_lrosa_score_topk).
"""

import pytest
import torch

from vllm.v1.attention.ops.triton_lrosa_score_topk import lrosa_score_topk
from vllm.v1.attention.ops.triton_lrosa_store import lrosa_project_and_store
from vllm.v1.attention.ops.triton_lrosa_streaming_topk import (
    alloc_candidates_buf,
    lrosa_streaming_topk,
)


def _build(num_reqs, H_kv, head_size, cs_h, seq_len, block_size, dtype, device):
    torch.manual_seed(0)
    num_blocks_per_req = (seq_len + block_size - 1) // block_size
    num_blocks = num_reqs * num_blocks_per_req + 2
    slot_size = 2 * head_size + cs_h

    M_raw = torch.randn(H_kv, cs_h, head_size, dtype=torch.float32, device=device)
    Q, _ = torch.linalg.qr(M_raw.transpose(1, 2))
    M = Q.transpose(1, 2).contiguous().to(dtype)

    K_flat = torch.randn(
        num_reqs * seq_len, H_kv, head_size, dtype=dtype, device=device
    )
    V_flat = torch.randn(
        num_reqs * seq_len, H_kv, head_size, dtype=dtype, device=device
    )

    kv_cache = torch.zeros(
        num_blocks, block_size, H_kv, slot_size, dtype=dtype, device=device
    )
    slot_mapping = torch.empty(num_reqs * seq_len, dtype=torch.int64, device=device)
    block_table = torch.zeros(
        num_reqs, num_blocks_per_req, dtype=torch.int32, device=device
    )
    bc = 1
    for r in range(num_reqs):
        for b in range(num_blocks_per_req):
            block_table[r, b] = bc
            bc += 1
        for t in range(seq_len):
            bid = int(block_table[r, t // block_size].item())
            slot_mapping[r * seq_len + t] = bid * block_size + (t % block_size)

    lrosa_project_and_store(K_flat, V_flat, kv_cache, slot_mapping, M)
    torch.accelerator.synchronize()

    proj_q = torch.randn(num_reqs, H_kv, cs_h, dtype=dtype, device=device)
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)
    return kv_cache, block_table, seq_lens, proj_q


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seq_len", [512, 4096])
def test_streaming_matches_two_pass(dtype, seq_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = torch.device("cuda:0")
    num_reqs, H_kv = 2, 4
    head_size, cs_h = 128, 32
    block_size, n_fac = 16, 256

    kv_cache, block_table, seq_lens, proj_q = _build(
        num_reqs,
        H_kv,
        head_size,
        cs_h,
        seq_len,
        block_size,
        dtype,
        device,
    )

    top_idx_2p, _ = lrosa_score_topk(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        n_fac=n_fac,
        head_size=head_size,
        cs_h=cs_h,
    )
    chunk_size = 4096
    max_num_chunks = max(1, (seq_len + chunk_size - 1) // chunk_size)
    # Round up to power of 2 so MAX_NUM_CHUNKS * N_FAC stays power-of-2 for tl.topk.
    p2 = 1
    while p2 < max_num_chunks:
        p2 *= 2
    candidates_buf = alloc_candidates_buf(
        num_reqs,
        H_kv,
        p2,
        n_fac,
        device,
    )
    top_idx_stream = lrosa_streaming_topk(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        n_fac=n_fac,
        head_size=head_size,
        cs_h=cs_h,
        candidates_buf=candidates_buf,
        chunk_size=chunk_size,
    )

    # Compare as sets per (r, h). Both should pick the same positions
    # modulo ≤1-2 tail flips from bf16 vs fp32 score precision.
    n_eff = min(n_fac, seq_len)
    for r in range(num_reqs):
        for h in range(H_kv):
            two_pass = set(top_idx_2p[r, h].tolist())
            streaming = set(top_idx_stream[r, h].tolist())
            common = len(two_pass & streaming)
            j = common / n_eff
            assert j >= 0.99, (
                f"r={r} h={h} dtype={dtype} seq_len={seq_len}: "
                f"Jaccard {j:.4f} below 0.99 ({common}/{n_eff})"
            )


if __name__ == "__main__":
    for dtype in (torch.bfloat16, torch.float16):
        for seq_len in (512, 4096):
            test_streaming_matches_two_pass(dtype, seq_len)
            print(f"PASS dtype={dtype} seq_len={seq_len}")
