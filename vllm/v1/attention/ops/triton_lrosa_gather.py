# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRoSA paged → contiguous K/V gather (Step 3b-3).

Given top-K token indices per (request, kv_head) — different heads pick
different positions per the LRoSA paper — gather the corresponding K and V
slices out of the paged combined-slot cache into varlen-flat buffers that
``flash_attn_varlen_func`` can consume directly (no block_table on the
attention side).

Output layout:
    K_sel[r * n_fac + i, h, :] = kv_cache[block_id, pos, h, 0 : head_size]
    V_sel[r * n_fac + i, h, :] = kv_cache[block_id, pos, h, head_size : 2*head_size]
where (block_id, pos) is decoded from top_idx[r, h, i] via block_table[r].

flash_attn handles per-head differing positions naturally: it never inspects
the "real" position of a K row, only its value.

Buffers are sourced from ``WorkspaceManager`` when available so the
allocation cost is amortized across the 36 layers of one forward pass.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)


@triton.jit
def _lrosa_gather_kernel(
    kv_cache_ptr,  # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,  # [num_reqs, max_blocks]
    top_idx_ptr,  # [num_reqs, H_kv, n_fac]  int32 or int64
    K_sel_ptr,  # [num_reqs * n_fac, H_kv, head_size]
    V_sel_ptr,  # [num_reqs * n_fac, H_kv, head_size]
    n_fac,
    head_size,
    block_size,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    top_stride_r,
    top_stride_h,
    ksel_stride_t,
    ksel_stride_h,
    vsel_stride_t,
    vsel_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)  # which of n_fac

    if pid_i >= n_fac:
        return

    # Position in paged cache
    t = tl.load(top_idx_ptr + pid_r * top_stride_r + pid_h * top_stride_h + pid_i)
    block_idx_in_table = t // block_size
    pos_in_block = t % block_size
    # int64: physical block ids overflow int32 cache-offset arithmetic once the
    # KV cache fills large memory -> intermittent OOB gather at long context.
    block_id = tl.load(
        block_table_ptr + pid_r * bt_stride_r + block_idx_in_table
    ).to(tl.int64)

    # Load K, V from slot
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size
    src_base = (
        block_id * cache_stride_block
        + pos_in_block * cache_stride_pos
        + pid_h * cache_stride_head
    )
    K = tl.load(kv_cache_ptr + src_base + d_offs, mask=d_mask, other=0.0)
    V = tl.load(kv_cache_ptr + src_base + head_size + d_offs, mask=d_mask, other=0.0)

    # Store into varlen-flat K_sel/V_sel
    out_row = pid_r * n_fac + pid_i
    dst_K = K_sel_ptr + out_row * ksel_stride_t + pid_h * ksel_stride_h
    dst_V = V_sel_ptr + out_row * vsel_stride_t + pid_h * vsel_stride_h
    tl.store(dst_K + d_offs, K, mask=d_mask)
    tl.store(dst_V + d_offs, V, mask=d_mask)


def lrosa_gather(
    kv_cache: torch.Tensor,  # (num_blocks, block_size, H_kv, slot_size)
    block_table: torch.Tensor,  # (num_reqs, max_blocks) int32
    top_idx: torch.Tensor,  # (num_reqs, H_kv, n_fac) int32 or int64
    head_size: int,
    dtype: torch.dtype | None = None,
    K_sel_out: torch.Tensor | None = None,
    V_sel_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K and V slices for selected positions into varlen-flat buffers.

    Returns (K_sel, V_sel) each shaped (num_reqs * n_fac, H_kv, head_size).

    Buffer source priority:
      1. Caller-provided ``K_sel_out`` / ``V_sel_out`` (preferred for CG
         capture — see the docstring on
         ``LRoSAMetadataBuilder._K_sel_buf`` for why we can't use the
         shared ``WorkspaceManager`` on this path).
      2. ``WorkspaceManager`` if initialized.
      3. Plain ``torch.empty`` fallback.
    """
    num_reqs, H_kv, n_fac = top_idx.shape
    if dtype is None:
        dtype = kv_cache.dtype

    flat_len = num_reqs * n_fac
    shape = (flat_len, H_kv, head_size)

    if K_sel_out is not None and V_sel_out is not None:
        # Caller owns the buffers (LRoSA backend path). Slice to this
        # step's flat_len. The caller pre-allocated for the engine's
        # ``max_num_reqs * n_fac``, so the slice always fits.
        assert K_sel_out.shape[0] >= flat_len and V_sel_out.shape[0] >= flat_len
        assert K_sel_out.dtype == dtype and V_sel_out.dtype == dtype
        K_sel = K_sel_out[:flat_len]
        V_sel = V_sel_out[:flat_len]
    elif is_workspace_manager_initialized():
        K_sel, V_sel = current_workspace_manager().get_simultaneous(
            (shape, dtype),
            (shape, dtype),
        )
    else:
        device = kv_cache.device
        K_sel = torch.empty(shape, dtype=dtype, device=device)
        V_sel = torch.empty(shape, dtype=dtype, device=device)

    BLOCK_D = triton.next_power_of_2(head_size)
    block_size = kv_cache.shape[1]
    grid = (num_reqs, H_kv, n_fac)

    _lrosa_gather_kernel[grid](
        kv_cache,
        block_table,
        top_idx,
        K_sel,
        V_sel,
        n_fac,
        head_size,
        block_size,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        top_idx.stride(0),
        top_idx.stride(1),
        K_sel.stride(0),
        K_sel.stride(1),
        V_sel.stride(0),
        V_sel.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return K_sel, V_sel
