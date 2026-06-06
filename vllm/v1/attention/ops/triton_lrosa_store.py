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


# ---------------------------------------------------------------------------
# Per-layer CONCAT store: single basis over concatenated per-head K.
#   proj_K_layer = M_layer @ concat_h(K[h]),  M_layer [cs_h_layer, H_kv*d]
# cs_h_layer is spread across the H_kv slots' proj_K regions
# (cs_h_slot = cs_h_layer // H_kv each), reusing the combined-slot layout
# unchanged. We do the projection in PyTorch (small GEMM: num_tokens ×
# H_kv*d → cs_h_layer) and reuse the per-head K/V store kernel plus a tiny
# proj-scatter kernel — far simpler and safer than an in-register Triton
# concat/scatter.
# ---------------------------------------------------------------------------


@triton.jit
def _lrosa_store_proj_kernel(
    proj_ptr,        # [num_tokens, H_kv, cs_h_slot]  (already split per head)
    kv_cache_ptr,    # [num_blocks, block_size, H_kv, slot_size]
    slot_mapping_ptr,
    num_tokens,
    head_size,
    cs_h_slot,
    block_size,
    proj_stride_t, proj_stride_h,
    cache_stride_block, cache_stride_pos, cache_stride_head,
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
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h_slot
    p_off = token_id * proj_stride_t + head_id * proj_stride_h
    p = tl.load(proj_ptr + p_off + c_offs, mask=c_mask, other=0.0)
    cache_off = (block_id * cache_stride_block + pos * cache_stride_pos
                 + head_id * cache_stride_head + 2 * head_size)
    tl.store(kv_cache_ptr + cache_off + c_offs,
             p.to(kv_cache_ptr.dtype.element_ty), mask=c_mask)


def lrosa_project_and_store_layer(
    key: torch.Tensor,    # (num_tokens, H_kv, head_size)
    value: torch.Tensor,  # (num_tokens, H_kv, head_size)
    kv_cache: torch.Tensor,  # (num_blocks, block_size, H_kv, slot_size)
    slot_mapping: torch.Tensor,
    M_layer: torch.Tensor,   # (cs_h_layer, H_kv*head_size)  single per-layer basis
) -> None:
    """Per-layer CONCAT store: K, V (per head) + proj_K_layer scattered across
    the H_kv slots' proj_K regions."""
    num_tokens, H_kv, head_size = key.shape
    if num_tokens == 0:
        return
    cs_h_layer = M_layer.shape[0]
    assert M_layer.shape[1] == H_kv * head_size, (
        f"M_layer {tuple(M_layer.shape)} expected (cs_h_layer, {H_kv*head_size})"
    )
    assert cs_h_layer % H_kv == 0, (
        f"cs_h_layer {cs_h_layer} must be divisible by H_kv {H_kv}"
    )
    cs_h_slot = cs_h_layer // H_kv
    assert kv_cache.shape[-1] >= 2 * head_size + cs_h_slot

    # 1) Store K + V per head (proj region untouched) via the existing kernel.
    lrosa_store(key, value, kv_cache, slot_mapping)

    # 2) proj_K_layer = M_layer @ concat_h(K[h]). Small GEMM in fp32.
    K_concat = key.reshape(num_tokens, H_kv * head_size).to(torch.float32)
    proj_layer = K_concat @ M_layer.to(torch.float32).T       # [T, cs_h_layer]
    # split into per-head slot chunks: [T, H_kv, cs_h_slot]
    proj_split = proj_layer.reshape(num_tokens, H_kv, cs_h_slot).contiguous()

    # 3) Scatter into proj_K regions.
    BLOCK_C = triton.next_power_of_2(cs_h_slot)
    _lrosa_store_proj_kernel[(num_tokens, H_kv)](
        proj_split, kv_cache, slot_mapping,
        num_tokens, head_size, cs_h_slot, kv_cache.shape[1],
        proj_split.stride(0), proj_split.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        BLOCK_C=BLOCK_C,
    )


# ---------------------------------------------------------------------------
# Contiguous-proj_K store: K,V into the [K|V] slot; proj_K = M@K into a SEPARATE
# contiguous cache [num_blocks, block_size, H_kv, cs_h]. The separate cache lets
# the score kernel scan proj_K coalesced (vs the strided read when proj_K is
# interleaved inside the 2*head+cs_h slot) — ~1.4x faster score at long context.
# ---------------------------------------------------------------------------


@triton.jit
def _lrosa_project_store_contig_kernel(
    key_ptr,
    value_ptr,
    m_ptr,
    kv_cache_ptr,    # [num_blocks, block_size, H_kv, slot_size]  ([K|V] used)
    projk_ptr,       # [num_blocks, block_size, H_kv, cs_h]  contiguous proj_K
    slot_mapping_ptr,
    num_tokens,
    head_size,
    cs_h,
    block_size,
    key_stride_t, key_stride_h,
    val_stride_t, val_stride_h,
    m_stride_h, m_stride_c,
    cache_stride_block, cache_stride_pos, cache_stride_head,
    pk_stride_block, pk_stride_pos, pk_stride_head,
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
    block_id = (slot // block_size).to(tl.int64)
    pos = slot % block_size

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < cs_h

    k_offset = token_id * key_stride_t + head_id * key_stride_h
    k = tl.load(key_ptr + k_offset + d_offs, mask=d_mask, other=0.0)
    v_offset = token_id * val_stride_t + head_id * val_stride_h
    v = tl.load(value_ptr + v_offset + d_offs, mask=d_mask, other=0.0)

    m_block_ptr = m_ptr + head_id * m_stride_h
    m_offs = c_offs[:, None] * m_stride_c + d_offs[None, :]
    m_mask = c_mask[:, None] & d_mask[None, :]
    m_tile = tl.load(m_block_ptr + m_offs, mask=m_mask, other=0.0)
    proj = tl.sum(m_tile * k[None, :].to(tl.float32), axis=1).to(k.dtype)

    cache_offset = (
        block_id * cache_stride_block
        + pos * cache_stride_pos
        + head_id * cache_stride_head
    )
    tl.store(kv_cache_ptr + cache_offset + d_offs, k, mask=d_mask)
    tl.store(kv_cache_ptr + cache_offset + head_size + d_offs, v, mask=d_mask)
    # proj_K → its own contiguous cache (cs_h-wide slot).
    pk_offset = (
        block_id * pk_stride_block
        + pos * pk_stride_pos
        + head_id * pk_stride_head
    )
    tl.store(projk_ptr + pk_offset + c_offs, proj, mask=c_mask)


def lrosa_project_and_store_contig(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,    # [num_blocks, block_size, H_kv, slot_size]
    projk_cache: torch.Tensor,  # [num_blocks, block_size, H_kv, cs_h]
    slot_mapping: torch.Tensor,
    M: torch.Tensor,           # (num_kv_heads, cs_h, head_size)
) -> None:
    """K,V → combined slot; proj_K = M@K → separate contiguous projk_cache."""
    num_tokens, num_kv_heads, head_size = key.shape
    if num_tokens == 0:
        return
    M_h, cs_h, M_d = M.shape
    assert M_h == num_kv_heads and M_d == head_size
    assert projk_cache.shape[2] == num_kv_heads and projk_cache.shape[3] == cs_h, (
        f"projk_cache {tuple(projk_cache.shape)} expected (.,.,{num_kv_heads},{cs_h})"
    )
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_C = triton.next_power_of_2(cs_h)
    _lrosa_project_store_contig_kernel[(num_tokens, num_kv_heads)](
        key, value, M, kv_cache, projk_cache, slot_mapping,
        num_tokens, head_size, cs_h, kv_cache.shape[1],
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        M.stride(0), M.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        projk_cache.stride(0), projk_cache.stride(1), projk_cache.stride(2),
        BLOCK_D=BLOCK_D, BLOCK_C=BLOCK_C,
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
