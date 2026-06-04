# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical test for the LRoSA score + top-K kernel (Step 3b-2).

Builds a synthetic combined-slot KV cache + paged block_table, runs the
Triton score kernel and ``torch.topk``, then compares against an eager
PyTorch reference that gathers proj_K via ``index_select`` and computes
the dot product directly.
"""

import pytest
import torch

from vllm.v1.attention.ops.triton_lrosa_score_topk import (
    lrosa_score,
    lrosa_score_topk,
)
from vllm.v1.attention.ops.triton_lrosa_store import lrosa_project_and_store


def _build_synthetic_cache(
    num_reqs,
    H_kv,
    head_size,
    cs_h,
    seq_len,
    block_size,
    num_blocks,
    dtype,
    device,
):
    """Allocate combined-slot cache, populate it with `lrosa_project_and_store`
    using fresh random K, V, M. Returns (kv_cache, M, K, V, block_table, seq_lens)."""
    torch.manual_seed(0)
    slot_size = 2 * head_size + cs_h
    kv_cache = torch.zeros(
        num_blocks, block_size, H_kv, slot_size, dtype=dtype, device=device
    )

    K = torch.randn(num_reqs * seq_len, H_kv, head_size, dtype=dtype, device=device)
    V = torch.randn(num_reqs * seq_len, H_kv, head_size, dtype=dtype, device=device)
    M = torch.randn(H_kv, cs_h, head_size, dtype=dtype, device=device)

    # Dense slot mapping: request r occupies slots [r*seq_len, (r+1)*seq_len)
    slot_mapping = torch.arange(num_reqs * seq_len, dtype=torch.int64, device=device)
    lrosa_project_and_store(K, V, kv_cache, slot_mapping, M)

    # Build block table: request r owns blocks
    # [r * blocks_per_req, (r+1) * blocks_per_req)
    blocks_per_req = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(
        num_reqs * blocks_per_req,
        dtype=torch.int32,
        device=device,
    ).view(num_reqs, blocks_per_req)
    seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int32, device=device)

    return kv_cache, M, K, V, block_table, seq_lens


def _eager_score_reference(K, M, proj_q, num_reqs, seq_len):
    """Pure-PyTorch reference: project K on-the-fly, score per (r,h,t)."""
    # K shape (num_reqs*seq_len, H_kv, d), reshape to (num_reqs, seq_len, H_kv, d)
    K_reshaped = K.view(num_reqs, seq_len, K.shape[1], K.shape[2])
    proj_K_ref = torch.einsum("rthd,hcd->rthc", K_reshaped.float(), M.float())
    # proj_q (num_reqs, H_kv, cs_h)
    scores = torch.einsum("rhc,rthc->rht", proj_q.float(), proj_K_ref)
    return scores  # (num_reqs, H_kv, seq_len)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_score_matches_reference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda:0")
    num_reqs, H_kv, head_size, cs_h = 4, 8, 128, 32
    seq_len, block_size = 64, 16
    num_blocks = num_reqs * ((seq_len + block_size - 1) // block_size)

    kv_cache, M, K, V, block_table, seq_lens = _build_synthetic_cache(
        num_reqs,
        H_kv,
        head_size,
        cs_h,
        seq_len,
        block_size,
        num_blocks,
        dtype,
        device,
    )

    proj_q = torch.randn(num_reqs, H_kv, cs_h, dtype=dtype, device=device)

    scores = lrosa_score(
        proj_q, kv_cache, block_table, seq_lens, head_size=head_size, cs_h=cs_h
    )
    # Slice to actual seq_len; rest should be -inf.
    scores_slice = scores[..., :seq_len]
    scores_beyond = scores[..., seq_len:]
    assert torch.isinf(scores_beyond).all() and (scores_beyond < 0).all(), (
        "positions beyond seq_len must be -inf"
    )

    ref = _eager_score_reference(K, M, proj_q, num_reqs, seq_len)

    # Allow generous tolerance: kernel uses fp32 accum then stores fp32;
    # reference is full fp32. proj_K through bf16 round-trip is the main source
    # of error.
    atol = 5e-1 if dtype == torch.bfloat16 else 5e-2
    rtol = 5e-2
    max_diff = (scores_slice - ref).abs().max().item()
    assert torch.allclose(scores_slice, ref, atol=atol, rtol=rtol), (
        f"score mismatch: max abs diff {max_diff}"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_topk_indices_match_reference(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    device = torch.device("cuda:0")
    num_reqs, H_kv, head_size, cs_h = 2, 4, 128, 32
    seq_len, block_size = 128, 16
    n_fac = 16
    num_blocks = num_reqs * ((seq_len + block_size - 1) // block_size)

    kv_cache, M, K, V, block_table, seq_lens = _build_synthetic_cache(
        num_reqs,
        H_kv,
        head_size,
        cs_h,
        seq_len,
        block_size,
        num_blocks,
        dtype,
        device,
    )

    proj_q = torch.randn(num_reqs, H_kv, cs_h, dtype=dtype, device=device)

    top_idx, top_scores = lrosa_score_topk(
        proj_q,
        kv_cache,
        block_table,
        seq_lens,
        n_fac=n_fac,
        head_size=head_size,
        cs_h=cs_h,
    )

    ref_scores = _eager_score_reference(K, M, proj_q, num_reqs, seq_len)
    _, ref_idx = torch.topk(ref_scores, k=n_fac, dim=-1)

    # We allow that ties may flip ordering between implementations. Compare
    # as *sets*: the chosen positions should match within tolerance.
    for r in range(num_reqs):
        for h in range(H_kv):
            ours = set(top_idx[r, h].tolist())
            ref = set(ref_idx[r, h].tolist())
            common = len(ours & ref)
            assert common >= int(0.9 * n_fac), (
                f"req {r} head {h}: only {common}/{n_fac} top-K positions match"
            )


if __name__ == "__main__":
    test_score_matches_reference(torch.bfloat16)
    test_score_matches_reference(torch.float16)
    test_topk_indices_match_reference(torch.bfloat16)
    test_topk_indices_match_reference(torch.float16)
    print("All score+topk tests passed.")
