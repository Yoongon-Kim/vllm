# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical test for the LRoSA gather kernel (Step 3b-3).

Builds a synthetic combined-slot KV cache + random top_idx, runs the Triton
gather kernel, and compares against an eager PyTorch reference built from
``torch.gather``.
"""

import pytest
import torch

from vllm.v1.attention.ops.triton_lrosa_gather import lrosa_gather
from vllm.v1.attention.ops.triton_lrosa_store import lrosa_project_and_store


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gather_matches_reference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    num_reqs, H_kv, head_size, cs_h = 2, 4, 128, 32
    seq_len, block_size, n_fac = 96, 16, 20
    blocks_per_req = (seq_len + block_size - 1) // block_size
    num_blocks = num_reqs * blocks_per_req
    slot_size = 2 * head_size + cs_h

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        H_kv,
        slot_size,
        dtype=dtype,
        device=device,
    )
    K = torch.randn(num_reqs * seq_len, H_kv, head_size, dtype=dtype, device=device)
    V = torch.randn(num_reqs * seq_len, H_kv, head_size, dtype=dtype, device=device)
    M = torch.randn(H_kv, cs_h, head_size, dtype=dtype, device=device)
    slot_mapping = torch.arange(num_reqs * seq_len, dtype=torch.int64, device=device)
    lrosa_project_and_store(K, V, kv_cache, slot_mapping, M)

    block_table = torch.arange(
        num_reqs * blocks_per_req,
        dtype=torch.int32,
        device=device,
    ).view(num_reqs, blocks_per_req)

    # Per (req, head) random selection of n_fac unique positions within seq_len.
    top_idx = torch.empty(num_reqs, H_kv, n_fac, dtype=torch.int32, device=device)
    for r in range(num_reqs):
        for h in range(H_kv):
            perm = torch.randperm(seq_len)[:n_fac]
            top_idx[r, h] = perm.to(device=device, dtype=torch.int32)

    K_sel, V_sel = lrosa_gather(
        kv_cache, block_table, top_idx, head_size=head_size, dtype=dtype
    )

    # Build eager reference from the original K, V tensors (request r has
    # tokens at K[r*seq_len:(r+1)*seq_len, :, :]).
    K_per_req = K.view(num_reqs, seq_len, H_kv, head_size)
    V_per_req = V.view(num_reqs, seq_len, H_kv, head_size)

    for r in range(num_reqs):
        for h in range(H_kv):
            for i in range(n_fac):
                t = int(top_idx[r, h, i])
                expected_K = K_per_req[r, t, h]
                expected_V = V_per_req[r, t, h]
                got_K = K_sel[r * n_fac + i, h]
                got_V = V_sel[r * n_fac + i, h]
                assert torch.equal(got_K, expected_K), (
                    f"K mismatch at r={r} h={h} i={i} t={t}: "
                    f"max abs diff {(got_K - expected_K).abs().max()}"
                )
                assert torch.equal(got_V, expected_V), (
                    f"V mismatch at r={r} h={h} i={i} t={t}"
                )


if __name__ == "__main__":
    test_gather_matches_reference(torch.bfloat16)
    test_gather_matches_reference(torch.float16)
    print("All gather tests passed.")
