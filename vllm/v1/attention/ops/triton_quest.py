# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quest (Tang et al., ICML 2024) page-level sparse-attention kernels.

Quest scores each KV *page* (= one paged-cache block) by a per-channel
min/max upper bound on the max-token attention score within the page:

    score_upper(page) = Σ_c max( q[c]·K_min[page,c], q[c]·K_max[page,c] )

then keeps the top ``page_budget = token_budget // page_size`` pages and
attends only the tokens in those pages (plus an always-attended trailing
partial page that holds the current decode token). This is the production
analog of pca's ``quest/quest.py`` reference, ported to vLLM's paged
combined-slot cache so LRoSA and Quest can be compared in the *same*
CUDA-graph + flash-attn stack.

Cache layout (block == page; block_size == page_size):
    slot per (token, kv-head):
      elems:   0────hs────2*hs────3*hs────4*hs
      content: [   K   |   V   | K_min | K_max ]
    K_min / K_max are the page's per-channel min/max, stored ONCE at the
    block's representative slot (position 0). The other 15 positions'
    min/max bytes are slack — this trades cache memory for backend
    drop-in simplicity (a single cache tensor, no model edits). Acceptable
    for the latency/quality comparison this backend exists to run.

Three pieces:
  1. ``quest_minmax_update`` — decode running min/max into the repr slot.
     pos==0 (new block) initializes; pos>0 folds the new K in. Race-free
     at decode (one token per request → distinct blocks).
  2. ``quest_build_page_minmax_prefill`` — prefill reduction over each full
     page (PyTorch on the contiguous ``key`` tensor + a scatter kernel).
  3. ``quest_page_score`` — per-(request, kv-head, page) upper-bound score,
     reading only the two min/max vectors per page (NOT the 16 K rows) so
     Quest's bandwidth advantage over a full-K scan is real.

Token gather + flash-attn reuse LRoSA's ``lrosa_gather`` (page indices are
expanded to token indices first) and ``_radix_topk`` (page top-K).
"""

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# 1. Decode running min/max update (into the block's representative slot)
# ---------------------------------------------------------------------------
@triton.jit
def _quest_minmax_update_kernel(
    key_ptr,            # [num_decodes, H_kv, head_size]  new decode K (post-RoPE)
    kv_cache_ptr,       # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,    # [num_decodes, max_blocks]
    seq_lens_ptr,       # [num_decodes]  int32 — length INCLUDING the new token
    head_size,
    page_size,
    key_stride_t,
    key_stride_h,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + r)
    last = seq_len - 1                      # 0-based position of the new token
    logical_block = last // page_size
    pos = last % page_size
    phys_block = tl.load(block_table_ptr + r * bt_stride_r + logical_block).to(tl.int64)

    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size
    k = tl.load(key_ptr + r * key_stride_t + h * key_stride_h + d, mask=dmask,
                other=0.0)

    repr_base = (phys_block * cache_stride_block
                 + h * cache_stride_head)   # block position 0 ⇒ no pos term
    min_off = repr_base + 2 * head_size + d
    max_off = repr_base + 3 * head_size + d

    if pos == 0:
        # First token of a fresh block: min = max = K.
        tl.store(kv_cache_ptr + min_off, k, mask=dmask)
        tl.store(kv_cache_ptr + max_off, k, mask=dmask)
    else:
        cur_min = tl.load(kv_cache_ptr + min_off, mask=dmask, other=0.0)
        cur_max = tl.load(kv_cache_ptr + max_off, mask=dmask, other=0.0)
        tl.store(kv_cache_ptr + min_off, tl.minimum(cur_min, k), mask=dmask)
        tl.store(kv_cache_ptr + max_off, tl.maximum(cur_max, k), mask=dmask)


def quest_minmax_update(
    key: torch.Tensor,          # (num_decodes, H_kv, head_size)
    kv_cache: torch.Tensor,     # (num_blocks, block_size, H_kv, slot_size)
    block_table: torch.Tensor,  # (num_decodes, max_blocks)
    seq_lens: torch.Tensor,     # (num_decodes,)  length including the new token
    head_size: int,
    page_size: int,
) -> None:
    """Fold each request's newly-decoded K into its current block's min/max."""
    num_decodes, H_kv, _ = key.shape
    if num_decodes == 0:
        return
    BLOCK_D = triton.next_power_of_2(head_size)
    _quest_minmax_update_kernel[(num_decodes, H_kv)](
        key, kv_cache, block_table, seq_lens,
        head_size, page_size,
        key.stride(0), key.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# 2. Prefill page min/max: PyTorch reduction over the contiguous K + scatter
# ---------------------------------------------------------------------------
@triton.jit
def _quest_scatter_minmax_kernel(
    minmax_ptr,         # [num_pages_total, H_kv, 2, head_size]  (min,max packed)
    kv_cache_ptr,       # [num_blocks, block_size, H_kv, slot_size]
    page_block_ptr,     # [num_pages_total]  physical block id of each page row
    head_size,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    mm_stride_p,
    mm_stride_h,
    mm_stride_x,
    BLOCK_D: tl.constexpr,
):
    p = tl.program_id(0)
    h = tl.program_id(1)
    phys_block = tl.load(page_block_ptr + p).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size

    kmin = tl.load(minmax_ptr + p * mm_stride_p + h * mm_stride_h
                   + 0 * mm_stride_x + d, mask=dmask, other=0.0)
    kmax = tl.load(minmax_ptr + p * mm_stride_p + h * mm_stride_h
                   + 1 * mm_stride_x + d, mask=dmask, other=0.0)

    repr_base = phys_block * cache_stride_block + h * cache_stride_head
    tl.store(kv_cache_ptr + repr_base + 2 * head_size + d, kmin, mask=dmask)
    tl.store(kv_cache_ptr + repr_base + 3 * head_size + d, kmax, mask=dmask)


def quest_build_page_minmax_prefill(
    key: torch.Tensor,          # (prefill_len, H_kv, head_size)  one request
    kv_cache: torch.Tensor,
    block_table_row: torch.Tensor,  # (max_blocks,) physical block ids for this req
    page_size: int,
    head_size: int,
    prefill_start_pos: int = 0,
) -> None:
    """Compute per-full-page min/max from a request's prefill K and scatter
    them into the representative slots.

    Only FULL pages formed by the prefill are written. The trailing partial
    page (``prefill_len % page_size`` tokens) is initialized too, so that the
    decode running-update can keep extending it — its repr slot holds the
    min/max over the partial tokens written so far.

    ``prefill_start_pos`` is the absolute position of ``key[0]`` in the
    sequence (0 for a from-scratch prefill; >0 for chunked prefill). Pages
    are aligned to absolute position, so a chunk that doesn't start on a page
    boundary contributes to a partial leading page — handled by the same
    running-update semantics (min/max fold, never re-init mid-page).
    """
    prefill_len, H_kv, _ = key.shape
    if prefill_len == 0:
        return
    device = key.device
    abs_end = prefill_start_pos + prefill_len           # exclusive
    # Pages whose ENTIRE [p*ps, (p+1)*ps) range lies within [start, end) can be
    # reduced in one shot; leading/trailing partial pages are folded per-token
    # via the running-update kernel to stay consistent with decode.
    first_full = (prefill_start_pos + page_size - 1) // page_size
    last_full = abs_end // page_size                     # exclusive
    if last_full > first_full:
        # Slice the contiguous K covering the full pages.
        lo = first_full * page_size - prefill_start_pos
        hi = last_full * page_size - prefill_start_pos
        n_full = last_full - first_full
        K_full = key[lo:hi].reshape(n_full, page_size, H_kv, head_size)
        kmin = K_full.min(dim=1).values                  # (n_full, H_kv, hs)
        kmax = K_full.max(dim=1).values
        minmax = torch.stack([kmin, kmax], dim=2).contiguous()  # (n_full,H_kv,2,hs)
        page_ids = block_table_row[first_full:last_full].contiguous()
        BLOCK_D = triton.next_power_of_2(head_size)
        _quest_scatter_minmax_kernel[(n_full, H_kv)](
            minmax, kv_cache, page_ids,
            head_size,
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            minmax.stride(0), minmax.stride(1), minmax.stride(2),
            BLOCK_D=BLOCK_D,
        )
    # Leading partial page (chunked prefill not starting on a boundary) and the
    # trailing partial page are handled by folding each of their tokens via the
    # running-update path. For the common from-scratch prefill, only a trailing
    # partial page exists; fold it here so decode can extend it.
    _fold_partial_pages(key, kv_cache, block_table_row, page_size, head_size,
                        prefill_start_pos, first_full, last_full)


def _fold_partial_pages(key, kv_cache, block_table_row, page_size, head_size,
                        prefill_start_pos, first_full, last_full):
    """Fold leading/trailing partial-page tokens into their repr slots with a
    per-token running min/max (PyTorch; prefill is not perf-critical)."""
    prefill_len, H_kv, _ = key.shape
    abs_end = prefill_start_pos + prefill_len
    # Trailing partial page: [last_full*ps, abs_end)
    trail_lo_abs = last_full * page_size
    if abs_end > trail_lo_abs and abs_end > prefill_start_pos:
        lo = max(trail_lo_abs, prefill_start_pos) - prefill_start_pos
        K_part = key[lo:]                                # (t, H_kv, hs)
        kmin = K_part.min(dim=0).values                 # (H_kv, hs)
        kmax = K_part.max(dim=0).values
        phys = int(block_table_row[last_full].item())
        kv_cache[phys, 0, :, 2 * head_size:3 * head_size] = kmin.to(kv_cache.dtype)
        kv_cache[phys, 0, :, 3 * head_size:4 * head_size] = kmax.to(kv_cache.dtype)
    # Leading partial page (only for chunked prefill mid-page starts).
    lead_hi_abs = first_full * page_size
    if prefill_start_pos < lead_hi_abs and prefill_start_pos < abs_end:
        hi = min(lead_hi_abs, abs_end) - prefill_start_pos
        if hi > 0:
            page = prefill_start_pos // page_size
            K_part = key[:hi]
            kmin = K_part.min(dim=0).values
            kmax = K_part.max(dim=0).values
            phys = int(block_table_row[page].item())
            # Fold with any existing repr (a previous chunk may have written it).
            ex_min = kv_cache[phys, 0, :, 2 * head_size:3 * head_size]
            ex_max = kv_cache[phys, 0, :, 3 * head_size:4 * head_size]
            kv_cache[phys, 0, :, 2 * head_size:3 * head_size] = torch.minimum(
                ex_min, kmin.to(kv_cache.dtype))
            kv_cache[phys, 0, :, 3 * head_size:4 * head_size] = torch.maximum(
                ex_max, kmax.to(kv_cache.dtype))


# ---------------------------------------------------------------------------
# 3. Page upper-bound score  (reads only the 2 min/max vectors per page)
# ---------------------------------------------------------------------------
@triton.jit
def _quest_page_score_kernel(
    q_ptr,              # [num_reqs, H_kv, head_size]  group-mean query (per kv-head)
    kv_cache_ptr,       # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,    # [num_reqs, max_blocks]
    seq_lens_ptr,       # [num_reqs]
    scores_ptr,         # [num_reqs, H_kv, max_pages]  fp32
    max_pages,
    head_size,
    page_size,
    q_stride_r,
    q_stride_h,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    sc_stride_r,
    sc_stride_h,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    h = tl.program_id(1)
    p = tl.program_id(2)

    out_off = r * sc_stride_r + h * sc_stride_h + p
    seq_len = tl.load(seq_lens_ptr + r)
    num_full = (seq_len - 1) // page_size       # full (selectable) pages
    if p >= num_full:
        tl.store(scores_ptr + out_off, float("-inf"))
        return

    phys_block = tl.load(block_table_ptr + r * bt_stride_r + p).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size
    q = tl.load(q_ptr + r * q_stride_r + h * q_stride_h + d, mask=dmask,
                other=0.0).to(tl.float32)
    base = phys_block * cache_stride_block + h * cache_stride_head
    kmin = tl.load(kv_cache_ptr + base + 2 * head_size + d, mask=dmask,
                   other=0.0).to(tl.float32)
    kmax = tl.load(kv_cache_ptr + base + 3 * head_size + d, mask=dmask,
                   other=0.0).to(tl.float32)
    # Upper bound: per-channel max of q·K over the page extremes.
    prod = tl.maximum(q * kmin, q * kmax)
    score = tl.sum(tl.where(dmask, prod, 0.0), axis=0)
    tl.store(scores_ptr + out_off, score)


def quest_page_score(
    q_kv: torch.Tensor,         # (num_reqs, H_kv, head_size)  group-mean query
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,  # (num_reqs, max_blocks)
    seq_lens: torch.Tensor,     # (num_reqs,)
    page_size: int,
    head_size: int,
    scores_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-(request, kv-head, page) upper-bound score. Returns
    ``(num_reqs, H_kv, max_pages)`` fp32; pages ≥ num_full_pages are -inf."""
    num_reqs, H_kv, _ = q_kv.shape
    max_pages = block_table.shape[1]
    if scores_out is not None:
        scores = scores_out[:num_reqs, :, :max_pages]
    else:
        scores = torch.empty((num_reqs, H_kv, max_pages), dtype=torch.float32,
                             device=q_kv.device)
    BLOCK_D = triton.next_power_of_2(head_size)
    _quest_page_score_kernel[(num_reqs, H_kv, max_pages)](
        q_kv, kv_cache, block_table, seq_lens, scores,
        max_pages, head_size, page_size,
        q_kv.stride(0), q_kv.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        scores.stride(0), scores.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return scores


# ---------------------------------------------------------------------------
# 4. Page indices → token indices (+ trailing partial page), for lrosa_gather
# ---------------------------------------------------------------------------
def quest_pages_to_token_idx(
    page_idx: torch.Tensor,     # (num_reqs, H_kv, page_budget)  int (selected pages)
    seq_lens: torch.Tensor,     # (num_reqs,)
    page_size: int,
    n_fac: int,
    num_full_pages: torch.Tensor,  # (num_reqs,)  = (seq_len-1)//page_size
    token_idx_out: torch.Tensor | None = None,  # (num_reqs, H_kv, n_fac) int32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand selected page ids → absolute token indices and append the
    always-attended trailing partial page.

    Layout per (req, head):  [ page_budget*page_size selected-page tokens |
                               page_size trailing tokens ]  == n_fac.
    Trailing tokens beyond seq_len are clamped to the last valid token; the
    returned ``seqused_k`` excludes that padding so flash never attends it.

    Returns ``(token_idx, seqused_k)``:
      token_idx: (num_reqs, H_kv, n_fac) int32
      seqused_k: (num_reqs,) int32 = page_budget*page_size + trailing_len
    """
    num_reqs, H_kv, page_budget = page_idx.shape
    device = page_idx.device
    assert page_budget * page_size + page_size == n_fac, (
        f"page_budget({page_budget})*page_size({page_size}) + page_size "
        f"!= n_fac({n_fac})"
    )
    off = torch.arange(page_size, device=device, dtype=torch.int64)
    seq64 = seq_lens.to(torch.int64).view(num_reqs, 1)
    nf64 = num_full_pages.to(torch.int64)                       # (nr,)
    # Long-context (num_full > page_budget): exactly seq_len > n_fac, so the
    # fixed [page_budget pages | trailing page] layout fills all n_fac slots.
    # Short-context (num_full <= page_budget, i.e. seq_len <= n_fac): attend
    # densely over [0, seq_len) — selection would be degenerate / mis-aligned.
    is_long = (nf64 > page_budget).view(num_reqs, 1, 1)

    # --- sparse (long) layout ---
    pi = page_idx.to(torch.int64)
    nf_clamp = nf64.clamp(min=1).view(num_reqs, 1, 1)
    pi = torch.minimum(pi, nf_clamp - 1).clamp(min=0)
    sel_tok = (pi.unsqueeze(-1) * page_size + off).reshape(
        num_reqs, H_kv, page_budget * page_size)
    trail_start = nf64 * page_size                              # (nr,)
    trail_tok = trail_start.view(num_reqs, 1) + off.view(1, page_size)
    last_valid = (seq64 - 1)
    trail_tok = torch.minimum(trail_tok, last_valid).clamp(min=0)
    trail_tok = trail_tok.unsqueeze(1).expand(num_reqs, H_kv, page_size)
    tok_long = torch.cat([sel_tok, trail_tok], dim=-1)          # (nr,H_kv,n_fac)

    # --- dense (short) layout: [0, 1, ..., n_fac) clamped to seq_len-1 ---
    dense = torch.arange(n_fac, device=device, dtype=torch.int64)
    tok_short = torch.minimum(dense.view(1, 1, n_fac), (seq64 - 1).view(num_reqs, 1, 1))
    tok_short = tok_short.clamp(min=0).expand(num_reqs, H_kv, n_fac)

    token_idx = torch.where(is_long, tok_long, tok_short).to(torch.int32)
    if token_idx_out is not None:
        token_idx_out[:num_reqs].copy_(token_idx)
        token_idx = token_idx_out[:num_reqs]

    # counts: long → page_budget*page_size + trailing_len; short → seq_len.
    trailing_len = (seq_lens.to(torch.int32)
                    - num_full_pages.to(torch.int32) * page_size)  # in [1, ps]
    counts_long = page_budget * page_size + trailing_len
    counts_short = seq_lens.to(torch.int32)
    counts = torch.where(nf64.to(torch.int32) > page_budget,
                         counts_long, counts_short).to(torch.int32)
    return token_idx, counts


# ---------------------------------------------------------------------------
# 5b. Block-sparse decode attention (Quest deployment-best, NO gather)
# ---------------------------------------------------------------------------
# Reads the selected pages' K/V directly from the paged cache and runs an
# online-softmax flash attention over them + the always-attended trailing
# partial page. Per-q-head program; the kv-head's selected pages are shared
# across its GQA group (per-kv-head selection, per-q-head attention). This is
# what the official Quest custom kernel does — no gather buffer, no full mask.
# A per-page validity check (page_col < num_full) makes short context attend
# all real pages + trailing == dense, with zero extra branches.
@triton.jit
def _quest_blocksparse_attn_kernel(
    q_ptr,              # (num_decodes, H_q, head_size)
    kv_cache_ptr,       # (num_blocks, block_size, H_kv, slot_size)
    page_idx_ptr,       # (num_decodes, H_kv, page_budget)  selected block columns
    block_table_ptr,    # (num_decodes, max_blocks)
    seq_lens_ptr,       # (num_decodes,)
    out_ptr,            # (num_decodes, H_q, head_size)
    scale,
    head_size,
    page_size,
    num_kv_groups,
    q_stride_r, q_stride_h,
    cache_stride_block, cache_stride_pos, cache_stride_head,
    pi_stride_r, pi_stride_h,
    bt_stride_r,
    page_budget,             # runtime loop bound (real loop, not unrolled — so
                             # large reasoning budgets e.g. 2048→127 pages don't
                             # blow up the kernel via constexpr unrolling)
    out_stride_r, out_stride_h,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,   # >= page_size (power of 2)
):
    r = tl.program_id(0)
    hq = tl.program_id(1)
    h = hq // num_kv_groups

    seq_len = tl.load(seq_lens_ptr + r)
    num_full = (seq_len - 1) // page_size       # selectable full pages

    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size
    q = tl.load(q_ptr + r * q_stride_r + hq * q_stride_h + d, mask=dmask,
                other=0.0).to(tl.float32)

    p_off = tl.arange(0, BLOCK_P)
    p_in = p_off < page_size

    # --- always-attended trailing partial page (block column == num_full) ---
    # Processed FIRST so m_i becomes finite before the selected-page loop:
    # online softmax is order-independent, and a finite running max makes
    # fully-invalid pages (all -inf logits, e.g. short-context padding)
    # contribute exactly 0 instead of producing exp(-inf-(-inf))=NaN.
    # The trailing page always has >= 1 valid token (trailing_len in [1, ps]).
    trail_start = num_full * page_size
    trailing_len = seq_len - trail_start                          # in [1, page_size]
    phys_t = tl.load(block_table_ptr + r * bt_stride_r + num_full).to(tl.int64)
    kv_base_t = (phys_t * cache_stride_block
                 + p_off[:, None] * cache_stride_pos
                 + h * cache_stride_head)
    t_valid = p_off < trailing_len
    tile_mask_t = t_valid[:, None] & dmask[None, :]
    k_t = tl.load(kv_cache_ptr + kv_base_t + d[None, :], mask=tile_mask_t,
                  other=0.0).to(tl.float32)
    v_t = tl.load(kv_cache_ptr + kv_base_t + head_size + d[None, :],
                  mask=tile_mask_t, other=0.0).to(tl.float32)
    logits_t = tl.sum(k_t * q[None, :], axis=1) * scale
    logits_t = tl.where(t_valid, logits_t, float("-inf"))
    m_i = tl.max(logits_t, axis=0)                                # finite (>=1 valid)
    p = tl.exp(logits_t - m_i)
    l_i = tl.sum(p, axis=0)
    acc = tl.sum(p[:, None] * v_t, axis=0)

    # --- selected full pages ---
    for j in range(page_budget):
        page_col = tl.load(page_idx_ptr + r * pi_stride_r + h * pi_stride_h + j)
        valid = (page_col >= 0) & (page_col < num_full)
        # block-table read clamped so an invalid (skipped) column never reads OOB
        page_col_safe = tl.where(valid, page_col, 0)
        phys = tl.load(block_table_ptr + r * bt_stride_r + page_col_safe).to(tl.int64)
        kv_base = (phys * cache_stride_block
                   + p_off[:, None] * cache_stride_pos
                   + h * cache_stride_head)
        tile_mask = p_in[:, None] & dmask[None, :]
        k_tile = tl.load(kv_cache_ptr + kv_base + d[None, :], mask=tile_mask,
                         other=0.0).to(tl.float32)
        v_tile = tl.load(kv_cache_ptr + kv_base + head_size + d[None, :],
                         mask=tile_mask, other=0.0).to(tl.float32)
        logits = tl.sum(k_tile * q[None, :], axis=1) * scale     # (BLOCK_P,)
        logits = tl.where(p_in & valid, logits, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(logits, axis=0))          # m_i already finite
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(logits - m_new)                               # (BLOCK_P,)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + r * out_stride_r + hq * out_stride_h + d,
             out.to(out_ptr.dtype.element_ty), mask=dmask)


# ---------------------------------------------------------------------------
# Split-KV block-sparse (flash-decoding style) for LARGE budgets.
# The single-pass kernel above launches only num_decodes*H_q programs, each
# looping ALL selected pages serially — fine at budget<=256 (16 pages) but
# under-parallelized at the reasoning budget (2048 → 127 pages: only 32
# programs at bsz=1, each doing ~2K tokens serially → slower than FKV). The
# split-KV variant fans the pages across NUM_SPLITS CTAs per (req, q-head),
# each computing a partial online-softmax, then a combine kernel merges them
# — the standard flash-decoding pattern, still GATHER-FREE (reads the paged
# cache in place; page granularity is exactly what lets us avoid gather).
@triton.jit
def _quest_blocksparse_partial_kernel(
    q_ptr,              # (num_decodes, H_q, head_size)
    kv_cache_ptr,
    page_idx_ptr,       # (num_decodes, H_kv, page_budget)
    block_table_ptr,
    seq_lens_ptr,
    acc_ptr,            # (num_decodes, H_q, NUM_SPLITS, head_size) fp32
    m_ptr,              # (num_decodes, H_q, NUM_SPLITS) fp32
    l_ptr,              # (num_decodes, H_q, NUM_SPLITS) fp32
    scale,
    head_size,
    page_size,
    num_kv_groups,
    page_budget,
    items_per_split,
    q_stride_r, q_stride_h,
    cache_stride_block, cache_stride_pos, cache_stride_head,
    pi_stride_r, pi_stride_h,
    bt_stride_r,
    acc_stride_r, acc_stride_h, acc_stride_s,
    ml_stride_r, ml_stride_h,
    BLOCK_D: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    r = tl.program_id(0)
    hq = tl.program_id(1)
    s = tl.program_id(2)
    h = hq // num_kv_groups

    seq_len = tl.load(seq_lens_ptr + r)
    num_full = (seq_len - 1) // page_size
    total_items = page_budget + 1          # selected pages + trailing page
    trailing_len = seq_len - num_full * page_size

    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size
    q = tl.load(q_ptr + r * q_stride_r + hq * q_stride_h + d, mask=dmask,
                other=0.0).to(tl.float32)
    p_off = tl.arange(0, BLOCK_P)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k in range(items_per_split):
        idx = s * items_per_split + k
        active = idx < total_items
        is_trail = idx == page_budget
        # selected-page column (clamped so idx==page_budget doesn't read OOB)
        col_load_idx = tl.where(idx < page_budget, idx, 0)
        page_col = tl.load(page_idx_ptr + r * pi_stride_r + h * pi_stride_h
                           + col_load_idx)
        sel_ok = active & (~is_trail) & (page_col >= 0) & (page_col < num_full)
        trail_ok = active & is_trail
        block_col = tl.where(is_trail, num_full, page_col)
        block_col = tl.where(active, block_col, 0)
        phys = tl.load(block_table_ptr + r * bt_stride_r + block_col).to(tl.int64)
        tok_ok = tl.where(is_trail, p_off < trailing_len, p_off < page_size)
        valid = tok_ok & (sel_ok | trail_ok)            # (BLOCK_P,)

        kv_base = (phys * cache_stride_block
                   + p_off[:, None] * cache_stride_pos
                   + h * cache_stride_head)
        tile_mask = valid[:, None] & dmask[None, :]
        k_tile = tl.load(kv_cache_ptr + kv_base + d[None, :], mask=tile_mask,
                         other=0.0).to(tl.float32)
        v_tile = tl.load(kv_cache_ptr + kv_base + head_size + d[None, :],
                         mask=tile_mask, other=0.0).to(tl.float32)
        logits = tl.sum(k_tile * q[None, :], axis=1) * scale
        logits = tl.where(valid, logits, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(logits, axis=0))
        do = m_new > float("-inf")
        alpha = tl.where(do, tl.where(m_i > float("-inf"),
                                      tl.exp(m_i - m_new), 0.0), 1.0)
        p = tl.where(do, tl.exp(logits - m_new), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v_tile, axis=0)
        m_i = tl.where(do, m_new, m_i)

    tl.store(acc_ptr + r * acc_stride_r + hq * acc_stride_h + s * acc_stride_s + d,
             acc, mask=dmask)
    tl.store(m_ptr + r * ml_stride_r + hq * ml_stride_h + s, m_i)
    tl.store(l_ptr + r * ml_stride_r + hq * ml_stride_h + s, l_i)


@triton.jit
def _quest_blocksparse_combine_kernel(
    acc_ptr, m_ptr, l_ptr, out_ptr,
    head_size,
    acc_stride_r, acc_stride_h, acc_stride_s,
    ml_stride_r, ml_stride_h,
    out_stride_r, out_stride_h,
    NUM_SPLITS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    r = tl.program_id(0)
    hq = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    dmask = d < head_size

    gm = float("-inf")
    for s in range(NUM_SPLITS):
        gm = tl.maximum(gm, tl.load(m_ptr + r * ml_stride_r + hq * ml_stride_h + s))
    l_tot = 0.0
    acc_tot = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in range(NUM_SPLITS):
        ms = tl.load(m_ptr + r * ml_stride_r + hq * ml_stride_h + s)
        ls = tl.load(l_ptr + r * ml_stride_r + hq * ml_stride_h + s)
        accs = tl.load(acc_ptr + r * acc_stride_r + hq * acc_stride_h
                       + s * acc_stride_s + d, mask=dmask, other=0.0)
        sc = tl.where(ms > float("-inf"), tl.exp(ms - gm), 0.0)
        l_tot += ls * sc
        acc_tot += accs * sc
    out = acc_tot / l_tot
    tl.store(out_ptr + r * out_stride_r + hq * out_stride_h + d,
             out.to(out_ptr.dtype.element_ty), mask=dmask)


def quest_num_splits(page_budget: int, items_per_split_target: int = 16) -> int:
    """NUM_SPLITS for the split-KV path. 1 → use the single-pass kernel.
    Target ~16 items (256 tokens) per split so each CTA's serial work matches
    the fast budget<=256 single-pass regime."""
    total_items = page_budget + 1
    return (total_items + items_per_split_target - 1) // items_per_split_target


def quest_blocksparse_attn(
    query: torch.Tensor,        # (num_decodes, H_q, head_size)
    kv_cache: torch.Tensor,
    page_idx: torch.Tensor,     # (num_decodes, H_kv, page_budget)  selected columns
    block_table: torch.Tensor,  # (num_decodes, max_blocks)
    seq_lens: torch.Tensor,     # (num_decodes,)
    output: torch.Tensor,       # (num_decodes, H_q, head_size)  written in place
    scale: float,
    page_size: int,
    head_size: int,
    num_kv_groups: int,
    partial_acc: torch.Tensor | None = None,  # (>=nd, H_q, NUM_SPLITS, hs) fp32
    partial_m: torch.Tensor | None = None,    # (>=nd, H_q, NUM_SPLITS) fp32
    partial_l: torch.Tensor | None = None,    # (>=nd, H_q, NUM_SPLITS) fp32
) -> None:
    """Quest block-sparse decode attention (no gather). Writes ``output``.

    Selected pages with column >= num_full (padding / short context) are
    skipped, so short sequences attend all real pages + the trailing page
    == dense, and long sequences attend exactly page_budget pages + trailing.

    Single-pass at small budget (NUM_SPLITS==1); flash-decoding split-KV at
    large budget (needs the partial_* scratch buffers).
    """
    num_decodes, H_q, _ = query.shape
    page_budget = page_idx.shape[-1]
    BLOCK_D = triton.next_power_of_2(head_size)
    BLOCK_P = triton.next_power_of_2(page_size)
    num_splits = quest_num_splits(page_budget)

    if num_splits <= 1 or partial_acc is None:
        _quest_blocksparse_attn_kernel[(num_decodes, H_q)](
            query, kv_cache, page_idx, block_table, seq_lens, output,
            float(scale), head_size, page_size, num_kv_groups,
            query.stride(0), query.stride(1),
            kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
            page_idx.stride(0), page_idx.stride(1),
            block_table.stride(0),
            page_budget,
            output.stride(0), output.stride(1),
            BLOCK_D=BLOCK_D,
            BLOCK_P=BLOCK_P,
        )
        return

    total_items = page_budget + 1
    ips = (total_items + num_splits - 1) // num_splits
    pa = partial_acc[:num_decodes]
    pm = partial_m[:num_decodes]
    pl = partial_l[:num_decodes]
    _quest_blocksparse_partial_kernel[(num_decodes, H_q, num_splits)](
        query, kv_cache, page_idx, block_table, seq_lens, pa, pm, pl,
        float(scale), head_size, page_size, num_kv_groups, page_budget, ips,
        query.stride(0), query.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        page_idx.stride(0), page_idx.stride(1),
        block_table.stride(0),
        pa.stride(0), pa.stride(1), pa.stride(2),
        pm.stride(0), pm.stride(1),
        BLOCK_D=BLOCK_D,
        BLOCK_P=BLOCK_P,
    )
    _quest_blocksparse_combine_kernel[(num_decodes, H_q)](
        pa, pm, pl, output,
        head_size,
        pa.stride(0), pa.stride(1), pa.stride(2),
        pm.stride(0), pm.stride(1),
        output.stride(0), output.stride(1),
        NUM_SPLITS=num_splits,
        BLOCK_D=BLOCK_D,
    )


# ---------------------------------------------------------------------------
# 5. Tight-packed page→token gather (per-request prefix-sum offset)
# ---------------------------------------------------------------------------
@triton.jit
def _quest_gather_kernel(
    kv_cache_ptr,       # [num_blocks, block_size, H_kv, slot_size]
    block_table_ptr,    # [num_reqs, max_blocks]
    token_idx_ptr,      # [num_reqs, H_kv, n_fac]  selected absolute token positions
    counts_ptr,         # [num_reqs]  attended count per request (<= n_fac)
    cu_k_ptr,           # [num_reqs+1]  prefix sum of counts (row offsets)
    K_sel_ptr,          # [total_rows, H_kv, head_size]
    V_sel_ptr,
    n_fac,
    head_size,
    block_size,
    cache_stride_block,
    cache_stride_pos,
    cache_stride_head,
    bt_stride_r,
    ti_stride_r,
    ti_stride_h,
    ksel_stride_t,
    ksel_stride_h,
    vsel_stride_t,
    vsel_stride_h,
    BLOCK_D: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_i = tl.program_id(2)

    count = tl.load(counts_ptr + pid_r)
    if pid_i >= count:
        return
    t = tl.load(token_idx_ptr + pid_r * ti_stride_r + pid_h * ti_stride_h + pid_i)
    block_idx_in_table = t // block_size
    pos_in_block = t % block_size
    block_id = tl.load(block_table_ptr + pid_r * bt_stride_r + block_idx_in_table).to(tl.int64)

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < head_size
    src_base = (block_id * cache_stride_block
                + pos_in_block * cache_stride_pos
                + pid_h * cache_stride_head)
    K = tl.load(kv_cache_ptr + src_base + d_offs, mask=d_mask, other=0.0)
    V = tl.load(kv_cache_ptr + src_base + head_size + d_offs, mask=d_mask, other=0.0)

    out_row = tl.load(cu_k_ptr + pid_r) + pid_i
    tl.store(K_sel_ptr + out_row * ksel_stride_t + pid_h * ksel_stride_h + d_offs,
             K, mask=d_mask)
    tl.store(V_sel_ptr + out_row * vsel_stride_t + pid_h * vsel_stride_h + d_offs,
             V, mask=d_mask)


def quest_gather(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,   # (num_reqs, max_blocks)
    token_idx: torch.Tensor,     # (num_reqs, H_kv, n_fac)  absolute token positions
    counts: torch.Tensor,        # (num_reqs,)  attended count per request
    cu_k: torch.Tensor,          # (num_reqs+1,)  prefix sum of counts (int32)
    head_size: int,
    K_sel_out: torch.Tensor,
    V_sel_out: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected K/V into a TIGHTLY-PACKED varlen buffer.

    Request r's tokens land in rows [cu_k[r], cu_k[r]+counts[r]) of K_sel_out;
    ``cu_k`` is the exclusive prefix sum of ``counts`` (so ``cu_k`` doubles as
    flash-attn's ``cu_seqlens_k``). All H_kv heads share the same row range
    (same count) but gather different token positions per head. No padding /
    duplicate rows — flash attends exactly ``counts[r]`` keys per request.
    """
    num_reqs, H_kv, n_fac = token_idx.shape
    BLOCK_D = triton.next_power_of_2(head_size)
    block_size = kv_cache.shape[1]
    K_sel = K_sel_out
    V_sel = V_sel_out
    _quest_gather_kernel[(num_reqs, H_kv, n_fac)](
        kv_cache, block_table, token_idx, counts, cu_k, K_sel, V_sel,
        n_fac, head_size, block_size,
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        token_idx.stride(0), token_idx.stride(1),
        K_sel.stride(0), K_sel.stride(1),
        V_sel.stride(0), V_sel.stride(1),
        BLOCK_D=BLOCK_D,
    )
    return K_sel, V_sel
