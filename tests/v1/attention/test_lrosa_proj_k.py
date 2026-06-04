# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical test for the LRoSA fused project + store kernel.

Builds a small random (K, V, M) input, allocates a combined-slot KV cache,
runs `lrosa_project_and_store`, reads the proj_K region back from each slot,
and asserts it matches the reference `einsum("nhd,hcd->nhc", K, M)`.
"""

import pytest
import torch

from vllm.v1.attention.ops.triton_lrosa_store import lrosa_project_and_store


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_size,cs_h", [(128, 32), (64, 16)])
def test_lrosa_project_and_store(dtype, head_size, cs_h):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    num_tokens = 64
    num_kv_heads = 8
    num_blocks = 8
    block_size = 16
    slot_size = 2 * head_size + cs_h

    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    M = torch.randn(num_kv_heads, cs_h, head_size, dtype=dtype, device=device)

    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        slot_size,
        dtype=dtype,
        device=device,
    )

    # Map token i → slot i (dense, contiguous).
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    lrosa_project_and_store(key, value, kv_cache, slot_mapping, M)

    # Read back K, V, proj_K from each token's slot and compare with refs.
    K_ref = key
    V_ref = value
    proj_K_ref = torch.einsum("nhd,hcd->nhc", K_ref.float(), M.float()).to(dtype)

    K_back = torch.empty_like(K_ref)
    V_back = torch.empty_like(V_ref)
    proj_K_back = torch.empty(
        num_tokens, num_kv_heads, cs_h, dtype=dtype, device=device
    )

    for t in range(num_tokens):
        slot = int(slot_mapping[t])
        block = slot // block_size
        pos = slot % block_size
        K_back[t] = kv_cache[block, pos, :, :head_size]
        V_back[t] = kv_cache[block, pos, :, head_size : 2 * head_size]
        proj_K_back[t] = kv_cache[block, pos, :, 2 * head_size : 2 * head_size + cs_h]

    # K and V should be exact (bit-identical copy).
    assert torch.equal(K_back, K_ref), "K readback mismatch"
    assert torch.equal(V_back, V_ref), "V readback mismatch"

    # proj_K uses fp32 accumulation in the kernel vs einsum reference; allow
    # some tolerance for bf16/fp16 cast at the end.
    atol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 5e-3
    assert torch.allclose(proj_K_back, proj_K_ref, atol=atol, rtol=rtol), (
        f"proj_K mismatch: max abs diff {(proj_K_back - proj_K_ref).abs().max().item()}"
    )


if __name__ == "__main__":
    # Direct execution mode for quick smoke (no pytest harness).
    test_lrosa_project_and_store(torch.bfloat16, 128, 32)
    test_lrosa_project_and_store(torch.bfloat16, 64, 16)
    test_lrosa_project_and_store(torch.float16, 128, 32)
    print("All proj_K tests passed.")
