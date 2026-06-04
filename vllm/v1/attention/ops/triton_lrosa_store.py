# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRoSA combined-slot KV cache store kernels.

Cache layout per slot:
    elems:    0───────head_size───────2*head_size─────2*head_size + cs_h
    content:  [        K          |        V         |      proj_K       ]

Public entries:
    lrosa_store(...)              — Step 2 fallback (K + V only,
                                    proj_K bytes uninitialized).
    lrosa_project_and_store(...)  — Step 3 production path: also computes
                                    proj_K = M_layer @ K and writes it
                                    into the proj_K region of each slot.
"""

import torch

from vllm.triton_utils import tl, triton

# ---------------------------------------------------------------------------
# Step 2 kernel: K + V only (kept for tests and ablation)
# ---------------------------------------------------------------------------


@triton.jit
def _lrosa_store_kv_kernel(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    kv_cache_ptr,  # [num_blocks, block_size, num_kv_heads, slot_size]
    slot_mapping_ptr,  # [num_tokens]
    num_tokens,
    head_size,
    block_size,
    key_stride_t,
    key_stride_h,
    val_stride_t,
    val_stride_h,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    BLOCK_D: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if token_id >= num_tokens:
        return

    slot = tl.load(slot_mapping_ptr + token_id)
    if slot < 0:
        return

    block_id = slot // block_size
    pos = slot % block_size

    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < head_size

    k_offset = token_id * key_stride_t + head_id * key_stride_h
    k = tl.load(key_ptr + k_offset + d_offs, mask=mask, other=0.0)

    v_offset = token_id * val_stride_t + head_id * val_stride_h
    v = tl.load(value_ptr + v_offset + d_offs, mask=mask, other=0.0)

    cache_offset = (
        block_id * cache_stride_block
        + pos * cache_stride_pos
        + head_id * cache_stride_head
    )

    tl.store(kv_cache_ptr + cache_offset + d_offs, k, mask=mask)
    tl.store(kv_cache_ptr + cache_offset + head_size + d_offs, v, mask=mask)
    # proj_K region [2*head_size : 2*head_size + cs_h] left untouched.


def lrosa_store(
    key: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
    value: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
    kv_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, slot_size)
    slot_mapping: torch.Tensor,
) -> None:
    """Write K and V into the combined LRoSA slot, leaving proj_K untouched."""
    num_tokens, num_kv_heads, head_size = key.shape
    if num_tokens == 0:
        return

    BLOCK_D = triton.next_power_of_2(head_size)
    grid = (num_tokens, num_kv_heads)

    _lrosa_store_kv_kernel[grid](
        key,
        value,
        kv_cache,
        slot_mapping,
        num_tokens,
        head_size,
        kv_cache.shape[1],
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# Step 3 kernel: fused project + store (K + V + proj_K = M @ K)
# ---------------------------------------------------------------------------


@triton.jit
def _lrosa_project_store_kernel(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    m_ptr,  # [num_kv_heads, cs_h, head_size]
    kv_cache_ptr,  # [num_blocks, block_size, num_kv_heads, slot_size]
    slot_mapping_ptr,  # [num_tokens]
    num_tokens,
    head_size,
    cs_h,
    block_size,
    key_stride_t,
    key_stride_h,
    val_stride_t,
    val_stride_h,
    m_stride_h,
    m_stride_c,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    if token_id >= num_tokens:
        return

    slot = tl.load(slot_mapping_ptr + token_id)
    if slot < 0:
        return

    block_id = slot // block_size
    pos = slot % block_size

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h

    # Load K and V vectors for this (token, head).
    k_offset = token_id * key_stride_t + head_id * key_stride_h
    k = tl.load(key_ptr + k_offset + d_offs, mask=d_mask, other=0.0)
    v_offset = token_id * val_stride_t + head_id * val_stride_h
    v = tl.load(value_ptr + v_offset + d_offs, mask=d_mask, other=0.0)

    # Compute proj_K = M[head] @ K. Accumulate in fp32 for accuracy
    # (matches pca's tensor-core convention even though this is a GEMV).
    # M[head] shape: (cs_h, head_size); loaded as a (BLOCK_C, BLOCK_D) tile.
    m_block_ptr = m_ptr + head_id * m_stride_h
    m_offs = c_offs[:, None] * m_stride_c + d_offs[None, :]
    m_mask = c_mask[:, None] & d_mask[None, :]
    m_tile = tl.load(m_block_ptr + m_offs, mask=m_mask, other=0.0)
    proj_f32 = tl.sum(m_tile * k[None, :].to(tl.float32), axis=1)
    proj = proj_f32.to(k.dtype)

    # Slot base for this (block, pos, head).
    cache_offset = (
        block_id * cache_stride_block
        + pos * cache_stride_pos
        + head_id * cache_stride_head
    )

    # Write K → slot[0 : head_size]
    tl.store(kv_cache_ptr + cache_offset + d_offs, k, mask=d_mask)
    # Write V → slot[head_size : 2*head_size]
    tl.store(kv_cache_ptr + cache_offset + head_size + d_offs, v, mask=d_mask)
    # Write proj_K → slot[2*head_size : 2*head_size + cs_h]
    tl.store(
        kv_cache_ptr + cache_offset + 2 * head_size + c_offs,
        proj,
        mask=c_mask,
    )


def lrosa_project_and_store(
    key: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
    value: torch.Tensor,  # (num_tokens, num_kv_heads, head_size)
    kv_cache: torch.Tensor,  # (num_blocks, block_size, num_kv_heads, slot_size)
    slot_mapping: torch.Tensor,
    M: torch.Tensor,  # (num_kv_heads, cs_h, head_size)
) -> None:
    """Write K, V, and proj_K = M @ K into the combined LRoSA slot."""
    num_tokens, num_kv_heads, head_size = key.shape
    if num_tokens == 0:
        return

    M_h, cs_h, M_d = M.shape
    assert M_h == num_kv_heads and M_d == head_size, (
        f"M shape mismatch: got {tuple(M.shape)}, expected "
        f"({num_kv_heads}, cs_h, {head_size})"
    )
    assert kv_cache.shape[-1] >= 2 * head_size + cs_h, (
        f"slot_size {kv_cache.shape[-1]} too small for "
        f"2 * head_size ({2 * head_size}) + cs_h ({cs_h})"
    )

    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_C = triton.next_power_of_2(cs_h)
    grid = (num_tokens, num_kv_heads)

    _lrosa_project_store_kernel[grid](
        key,
        value,
        M,
        kv_cache,
        slot_mapping,
        num_tokens,
        head_size,
        cs_h,
        kv_cache.shape[1],
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        M.stride(0),
        M.stride(1),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        BLOCK_D=BLOCK_D,
        BLOCK_C=BLOCK_C,
    )
