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
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # One program copies a TILE of BLOCK_N selected tokens for one (req, kv-head)
    # as a 2D [BLOCK_N, BLOCK_D] load/store. The old kernel launched one program
    # per (req, head, token) — n_fac× more tiny blocks — and was launch/occupancy
    # bound (a contiguous-index gather ran at the same speed as a scattered one,
    # i.e. the access pattern wasn't the limiter, the block count was). Tiling
    # cuts the grid by BLOCK_N× and vectorizes the copy.
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)  # tile of BLOCK_N tokens within n_fac

    i_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    i_mask = i_offs < n_fac

    t = tl.load(
        top_idx_ptr + pid_r * top_stride_r + pid_h * top_stride_h + i_offs,
        mask=i_mask, other=0,
    )
    block_idx_in_table = t // block_size
    pos_in_block = t % block_size
    # int64: physical block ids overflow int32 cache-offset arithmetic once the
    # KV cache fills large memory -> intermittent OOB gather at long context.
    block_id = tl.load(
        block_table_ptr + pid_r * bt_stride_r + block_idx_in_table,
        mask=i_mask, other=0,
    ).to(tl.int64)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size
    m = i_mask[:, None] & d_mask[None, :]
    # Per-token slot base (K region); V is +head_size within the same slot.
    src_base = (
        block_id * cache_stride_block
        + pos_in_block * cache_stride_pos
        + pid_h * cache_stride_head
    )  # [BLOCK_N]
    src = kv_cache_ptr + src_base[:, None] + d_offs[None, :]  # [BLOCK_N, BLOCK_D]
    K = tl.load(src, mask=m, other=0.0)
    V = tl.load(src + head_size, mask=m, other=0.0)

    out_row = pid_r * n_fac + i_offs  # [BLOCK_N]
    dst_K = (
        K_sel_ptr + out_row[:, None] * ksel_stride_t
        + pid_h * ksel_stride_h + d_offs[None, :]
    )
    dst_V = (
        V_sel_ptr + out_row[:, None] * vsel_stride_t
        + pid_h * vsel_stride_h + d_offs[None, :]
    )
    tl.store(dst_K, K, mask=m)
    tl.store(dst_V, V, mask=m)


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
    BLOCK_N = 16  # tokens copied per program (grid shrinks by this factor)
    block_size = kv_cache.shape[1]
    grid = (num_reqs, H_kv, triton.cdiv(n_fac, BLOCK_N))

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
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    return K_sel, V_sel


@triton.jit
def _lrosa_gather_layer_kernel(
    kv_cache_ptr,  # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,  # [num_reqs, max_blocks]
    top_idx_ptr,  # [num_reqs, n_fac]  int32 or int64  (SHARED across heads)
    K_sel_ptr,  # [num_reqs * n_fac, H_kv, head_size]
    V_sel_ptr,  # [num_reqs * n_fac, H_kv, head_size]
    n_fac,
    head_size,
    block_size,
    H_kv,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    top_stride_r,
    ksel_stride_t,
    ksel_stride_h,
    vsel_stride_t,
    vsel_stride_h,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Per-layer LRoSA: ONE shared top-K per request (same tokens for all kv
    # heads). One program copies a tile of BLOCK_N tokens × ALL H_kv heads, so
    # the block_id/pos decode happens once per token (not H_kv×) and the grid
    # is (num_reqs, n_fac/BLOCK_N) — H_kv× fewer programs than the per-head
    # gather, whose limiter was the block count (see _lrosa_gather_kernel).
    pid_r = tl.program_id(0)
    pid_n = tl.program_id(1)

    i_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    i_mask = i_offs < n_fac

    t = tl.load(
        top_idx_ptr + pid_r * top_stride_r + i_offs, mask=i_mask, other=0,
    )
    block_idx_in_table = t // block_size
    pos_in_block = t % block_size
    block_id = tl.load(
        block_table_ptr + pid_r * bt_stride_r + block_idx_in_table,
        mask=i_mask, other=0,
    ).to(tl.int64)

    h_offs = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    d_offs = tl.arange(0, BLOCK_D)  # [BLOCK_D]
    h_mask = h_offs < H_kv
    d_mask = d_offs < head_size
    m = i_mask[:, None, None] & h_mask[None, :, None] & d_mask[None, None, :]

    # [BLOCK_N, BLOCK_H, BLOCK_D] gather: token (block_id,pos) shared across the
    # H_kv heads, which are contiguous in the slot's head dimension.
    src = (
        block_id[:, None, None] * cache_stride_block
        + pos_in_block[:, None, None] * cache_stride_pos
        + h_offs[None, :, None] * cache_stride_head
        + d_offs[None, None, :]
    )
    K = tl.load(kv_cache_ptr + src, mask=m, other=0.0)
    V = tl.load(kv_cache_ptr + src + head_size, mask=m, other=0.0)

    out_row = pid_r * n_fac + i_offs  # [BLOCK_N]
    dst_K = (
        K_sel_ptr + out_row[:, None, None] * ksel_stride_t
        + h_offs[None, :, None] * ksel_stride_h + d_offs[None, None, :]
    )
    dst_V = (
        V_sel_ptr + out_row[:, None, None] * vsel_stride_t
        + h_offs[None, :, None] * vsel_stride_h + d_offs[None, None, :]
    )
    tl.store(dst_K, K, mask=m)
    tl.store(dst_V, V, mask=m)


def lrosa_gather_layer(
    kv_cache: torch.Tensor,  # (num_blocks, block_size, H_kv, slot_size)
    block_table: torch.Tensor,  # (num_reqs, max_blocks) int32
    top_idx: torch.Tensor,  # (num_reqs, n_fac) int32/int64 — SHARED across heads
    head_size: int,
    dtype: torch.dtype | None = None,
    K_sel_out: torch.Tensor | None = None,
    V_sel_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-layer (shared-index) gather: one top-K per request, gathered for all
    H_kv heads in a single H_kv×-smaller grid. Returns (K_sel, V_sel) shaped
    (num_reqs * n_fac, H_kv, head_size) — identical layout to ``lrosa_gather``."""
    num_reqs, n_fac = top_idx.shape
    H_kv = kv_cache.shape[2]
    if dtype is None:
        dtype = kv_cache.dtype
    flat_len = num_reqs * n_fac
    shape = (flat_len, H_kv, head_size)

    if K_sel_out is not None and V_sel_out is not None:
        assert K_sel_out.shape[0] >= flat_len and V_sel_out.shape[0] >= flat_len
        assert K_sel_out.dtype == dtype and V_sel_out.dtype == dtype
        K_sel = K_sel_out[:flat_len]
        V_sel = V_sel_out[:flat_len]
    elif is_workspace_manager_initialized():
        K_sel, V_sel = current_workspace_manager().get_simultaneous(
            (shape, dtype), (shape, dtype),
        )
    else:
        K_sel = torch.empty(shape, dtype=dtype, device=kv_cache.device)
        V_sel = torch.empty(shape, dtype=dtype, device=kv_cache.device)

    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_H = triton.next_power_of_2(H_kv)
    BLOCK_N = 16
    block_size = kv_cache.shape[1]
    grid = (num_reqs, triton.cdiv(n_fac, BLOCK_N))

    _lrosa_gather_layer_kernel[grid](
        kv_cache, block_table, top_idx, K_sel, V_sel,
        n_fac, head_size, block_size, H_kv,
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), top_idx.stride(0),
        K_sel.stride(0), K_sel.stride(1),
        V_sel.stride(0), V_sel.stride(1),
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H, BLOCK_D=BLOCK_D,
    )
    return K_sel, V_sel
