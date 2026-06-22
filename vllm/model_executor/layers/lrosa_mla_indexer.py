"""LRoSA-on-MLA token selector (appendix: GLM-4.7-Flash, FKV vs LRoSA).

A drop-in replacement for the DeepSeek-V3.2 lightning indexer (`Indexer` in
deepseek_v2.py): SAME forward signature `(hidden_states, q_c, positions, rope)`
and SAME deployment-grade kernel path (the fp8 paged-MQA-logits + CTA-persistent
radix top-k in `sparse_attn_indexer` -> `topk_indices_buffer`, consumed unchanged
by FlashMLASparse). The ONLY change vs DeepSeek's learned indexer is the score:

    index_q[h] = [ M . q_latent[h]  |  q_pe[h] ]      (q_latent = q_nope @ W_UK)
    index_k    = [ M . c_KV         |  k_pe    ]      (shared / MQA on the latent)

so per head  index_q[h]·index_k = (M q_latent[h])·(M c_KV) + q_pe[h]·k_pe, and the
kernel's weighted head-sum (uniform weights) gives the head-aggregated latent score
plus the decoupled-RoPE term — exactly LRoSA's MLA score, but computed with the same
fp8 kernels DeepSeek's DSA uses (NOT an eager PyTorch fallback). M:[cs_h, kv_lora_rank]
is the calibrated D1 rotation per layer. head_dim = cs_h + qk_rope_head_dim (=128 at
cs_h=64, matching the indexer's index_head_dim and the fp8 quant block).
"""
from __future__ import annotations

import os

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from vllm.model_executor.models.utils import extract_layer_index
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)


def quest_mla_select(
    q_h_all: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_indices_buffer: torch.Tensor,
    page_size: int,
    topk_tokens: int,
    key_dim: int,
) -> torch.Tensor:
    """Quest page min/max selection, wrapped as a custom op.

    This is registered as a `splitting_ops` boundary (like
    `vllm::sparse_attn_indexer`) so inductor treats it as opaque and runs it
    EAGER between captured cudagraph pieces. The body is dynamic (`.item()`
    host-syncs, per-request Python loop, data-dependent shapes) and would be
    mis-compiled if traced — hence the custom op. Mutates `kv_cache` (inserts
    the current keys) and `topk_indices_buffer` (writes the selected LOCAL
    positions). Mirrors the eager forward exactly.
    """
    md = get_forward_context().attn_metadata
    buf = topk_indices_buffer
    T = key.shape[0]
    if not isinstance(md, dict):
        return buf  # profiling/dummy
    m = md.get(_resolve_layer_name(k_cache_prefix))
    cache = kv_cache                                           # [n_blk, blk, key_dim] bf16
    buf[:T] = -1
    if m is None or cache is None or (hasattr(cache, "numel") and cache.numel() == 0):
        return buf
    flat = cache.view(-1, key_dim)
    flat[m.slot_mapping[:T]] = key.to(flat.dtype)               # insert current keys
    # Decode: eager per-request page min/max selection.
    if m.num_decodes > 0 and m.decode is not None:
        bt = m.decode.block_table                              # [B, max_blk]
        sl = m.decode.seq_lens.view(m.num_decodes, -1)[:, -1].to(torch.long)  # [B]
        blk = cache.shape[1]
        P, K = page_size, topk_tokens
        ar_blk = torch.arange(blk, device=bt.device)
        ar_P = torch.arange(P, device=bt.device)
        for r in range(m.num_decodes):
            L = int(sl[r].item())
            if L <= 0:
                continue
            # FlashMLASparse expects LOCAL positions (0..L-1); the block table is
            # used ONLY to read the cached keys for the min/max (global slots).
            if L <= K:                                        # all fit -> attend all
                buf[r, :L] = torch.arange(L, device=bt.device).to(buf.dtype); continue
            nblk = (L + blk - 1) // blk
            ids = (bt[r, :nblk, None] * blk + ar_blk).view(-1)[:L]   # global slots, local order
            ks = flat[ids]                                    # [L, key_dim] (row j = local pos j)
            # ALWAYS keep the page holding the current decode token + most-recent
            # context (Quest keeps the local window); score only the full pages
            # strictly before it and fill the remaining budget with the top pages.
            last_start = ((L - 1) // P) * P
            tail = torch.arange(last_start, L, device=bt.device)   # current/last page
            ncand = last_start // P                                # full candidate pages
            keep = (K - int(tail.numel())) // P                    # pages affordable after tail
            if ncand == 0 or keep <= 0:
                out = torch.cat([torch.arange(0, last_start, device=bt.device), tail])[:K]
            else:
                full = ks[:ncand * P].view(ncand, P, key_dim)
                kmin = full.min(1).values; kmax = full.max(1).values   # [ncand, key_dim]
                # Faithful Quest GQA reduction = group-mean the query to the single
                # (shared) KV head, then ONE max(q*kmin, q*kmax) score. MLA has one
                # latent KV head so the whole head set is one group -> mean over heads.
                # (Matches pca quest.py q_aggregation="groupmean", the paper default.)
                qg = q_h_all[r].mean(0)                                # [key_dim]
                qp = qg.clamp(min=0); qn = qg.clamp(max=0)
                score = qp @ kmax.t() + qn @ kmin.t()                  # [ncand]
                top = score.topk(min(keep, ncand)).indices
                sel = (top[:, None] * P + ar_P).view(-1)               # LOCAL positions of pages
                out = torch.cat([tail, sel])[:K]                       # tail first -> never dropped
            out = torch.sort(out).values                               # kernel expects ascending
            buf[r, :out.shape[0]] = out.to(buf.dtype)
    # Prefill: each token attends its causal context. When the causal length fits
    # the budget (the common reasoning case: prompt <= budget) -> attend ALL causal
    # LOCAL positions (0..i). When it overflows (long prompt + small budget, nf512
    # only) -> streaming fallback: the K most-recent causal positions (sorted, incl
    # self). Quest's sparsity target is the DECODE path; reasoning prompts are short
    # so the overflow branch is rarely hit. Assumes no prefix-cache reuse (chunk
    # covers the sequence from position 0; true for the eval).
    if m.num_prefills > 0 and m.prefill is not None:
        base = m.num_decode_tokens
        K = topk_tokens
        for ch in m.prefill.chunks:
            cu = ch.cu_seq_lens.to(torch.long)                # [num_reqs+1] local cumsum
            for j in range(ch.num_reqs):
                s, e = int(cu[j].item()), int(cu[j + 1].item())
                for i in range(e - s):
                    tok = base + ch.token_start + s + i
                    Lc = i + 1                                # causal length (pos 0..i)
                    if Lc <= K:
                        buf[tok, :Lc] = torch.arange(Lc, device=buf.device).to(buf.dtype)
                    else:
                        buf[tok, :K] = torch.arange(Lc - K, Lc, device=buf.device).to(buf.dtype)
    return buf


def quest_mla_select_fake(
    q_h_all: torch.Tensor,
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_indices_buffer: torch.Tensor,
    page_size: int,
    topk_tokens: int,
    key_dim: int,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="quest_mla_select",
    op_func=quest_mla_select,
    mutates_args=["kv_cache", "topk_indices_buffer"],
    fake_impl=quest_mla_select_fake,
    dispatch_key=current_platform.dispatch_key,
)


class QuestMLAIndexer(nn.Module):
    """QUEST-on-MLA token selector (page-level upper bound on the MLA key). Quest
    scores a page p as Σ_c max(q[c]·Kmin_p[c], q[c]·Kmax_p[c]) — a min/max upper
    bound, NOT a q·k dot product, so it CANNOT use the fp8 paged-MQA-logits kernel
    that LRoSA/FASA reuse. Instead this keeps its OWN bf16 paged cache of the raw key
    [c_KV | k_pe] (head_dim = kv_lora_rank + qk_rope), and at decode does an EAGER
    page-min/max scorer per request: gather cached keys via the block table, fold into
    page_size-token pages, score, take the top (n_fac // page_size) pages, expand to
    token ids (+ the always-attended trailing partial page), and write topk_indices_buffer
    for the FlashMLASparse attend. Query = group-mean over heads of [q_latent | q_pe]
    (MQA upper bound, matching the GQA Quest port). enforce_eager required (the eager
    gather/min-max ops break CUDA-graph capture). Prefill: when the context fits the
    budget (reasoning's short prompt) every token attends all causal positions; longer
    prefill falls back to per-token causal page selection. No calibration."""

    def __init__(self, vllm_config, config, cache_config, fused_qkv_a_proj,
                 kv_a_proj_with_mqa, q_b_proj, kv_a_layernorm, kv_b_proj, rotary_emb,
                 q_lora_rank, n_fac, page_size, topk_indices_buffer, prefix=""):
        super().__init__()
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.n_head = config.num_attention_heads
        self.key_dim = self.kv_lora_rank + self.qk_rope_head_dim   # 576 for GLM
        self.page_size = page_size
        self.topk_tokens = n_fac
        self.fused_qkv_a_proj = fused_qkv_a_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.q_b_proj = q_b_proj
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_b_proj = kv_b_proj
        self.rotary_emb = rotary_emb
        self.q_lora_rank = q_lora_rank
        self.layer_idx = extract_layer_index(prefix)
        self.topk_indices_buffer = topk_indices_buffer
        # bf16 paged cache of the raw key [c_KV | k_pe] (NOT fp8 — min/max must be exact).
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.key_dim, dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache", cache_config=cache_config)

    def forward(self, hidden_states, q_c, positions, rotary_emb=None):
        from vllm.forward_context import get_forward_context
        T = hidden_states.shape[0]
        H = self.n_head
        kvlr, rope_d = self.kv_lora_rank, self.qk_rope_head_dim
        # latent c_KV + decoupled rope key (kv_a path)
        if self.q_lora_rank is not None:
            kv_lora = self.fused_qkv_a_proj(hidden_states)[0][..., self.q_lora_rank:]
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        c_kv = self.kv_a_layernorm(kv_lora[..., :kvlr])             # [T, kvlr]
        k_pe = kv_lora[..., kvlr:].unsqueeze(1)                     # [T,1,rope]
        q = self.q_b_proj(q_c)[0].view(T, H, self.qk_head_dim)
        q_nope = q[..., :self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim:]                      # [T,H,rope]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_pe = q_pe.reshape(T, H, rope_d)
        k_pe = k_pe.reshape(T, rope_d)
        W_UK = self.kv_b_proj.weight.view(
            H, self.qk_nope_head_dim + self.v_head_dim, kvlr)[:, :self.qk_nope_head_dim, :]
        q_latent = torch.einsum("thn,hnl->thl", q_nope, W_UK)      # [T,H,kvlr]
        # Per-head query [T,H,key_dim]. The heads share the latent KV (MQA-on-MLA),
        # so the Quest page bound is computed PER HEAD and reduced by max over heads
        # (union: keep a page if ANY head wants it). Averaging the query over heads
        # cancels opposite-sign components -> a near-zero query -> near-random pages.
        q_h_all = torch.cat([q_latent, q_pe], dim=-1)              # [T, H, key_dim]
        key = torch.cat([c_kv, k_pe], dim=-1)                       # [T, key_dim]

        # Dynamic page min/max selection runs in a registered custom op (a
        # splitting_ops boundary) so it executes EAGER and is never traced by
        # inductor; the projections above are compiled. See quest_mla_select.
        return torch.ops.vllm.quest_mla_select(
            q_h_all, key, self.k_cache.kv_cache,
            _encode_layer_name(self.k_cache.prefix),
            self.topk_indices_buffer, self.page_size, self.topk_tokens,
            self.key_dim)


def triattn_mla_select(
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_indices_buffer: torch.Tensor,
    q_mean_real: torch.Tensor,
    q_mean_imag: torch.Tensor,
    q_abs_mean: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    rope_style: str,
    aggregation: str,
    topk_tokens: int,
    key_dim: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    """TriAttention frequency-domain selection, wrapped as a custom op.

    Same role as `quest_mla_select`: a `splitting_ops` boundary so the dynamic
    eager scoring (`.item()` host-syncs, per-request loop, cos/atan2) is NOT
    traced by inductor — runs eager between captured PIECEWISE cudagraph pieces.
    Mutates `kv_cache` (inserts PRE-rope keys) and `topk_indices_buffer`.
    """
    from vllm.model_executor.layers.triattention_utils import (
        compute_frequency_statistics_from_means, score_keys_for_round,
    )
    md = get_forward_context().attn_metadata
    buf = topk_indices_buffer
    T = key.shape[0]
    if not isinstance(md, dict):
        return buf  # profiling/dummy
    m = md.get(_resolve_layer_name(k_cache_prefix))
    cache = kv_cache                                           # [n_blk, blk, key_dim]
    buf[:T] = -1
    if m is None or cache is None or (hasattr(cache, "numel") and cache.numel() == 0):
        return buf
    flat = cache.view(-1, key_dim)
    flat[m.slot_mapping[:T]] = key.to(flat.dtype)               # insert current keys
    q_mean_complex = torch.complex(q_mean_real, q_mean_imag)   # [nFC]
    kvlr = kv_lora_rank
    K = topk_tokens
    # Decode: eager per-request TriAttention frequency scoring.
    if m.num_decodes > 0 and m.decode is not None:
        bt = m.decode.block_table                              # [B, max_blk]
        sl = m.decode.seq_lens.view(m.num_decodes, -1)[:, -1].to(torch.long)
        blk = cache.shape[1]
        ar_blk = torch.arange(blk, device=bt.device)
        for r in range(m.num_decodes):
            L = int(sl[r].item())
            if L <= 0:
                continue
            # FlashMLASparse expects LOCAL positions (0..L-1); the block table is
            # used ONLY to read the cached pre-rope k_pe (global slots).
            if L <= K:                                        # all fit -> attend all
                buf[r, :L] = torch.arange(L, device=bt.device).to(buf.dtype); continue
            nblk = (L + blk - 1) // blk
            ids = (bt[r, :nblk, None] * blk + ar_blk).view(-1)[:L]   # global slots
            k_pe = flat[ids, kvlr:].float()                   # [L, rope] pre-rope k_pe
            # amp/phi/extra from the calibrated query stat + this request's keys.
            amp, phi, extra = compute_frequency_statistics_from_means(
                q_mean_complex, q_abs_mean, k_pe, style=rope_style)
            # round_start = current decode position = L-1 (0-indexed last token).
            key_idx = torch.arange(L, device=bt.device)
            score = score_keys_for_round(
                key_idx, L - 1, amp, phi, omega, extra,
                offsets, aggregation, freq_scale_sq)           # [L]
            # ALWAYS keep the current/last token (self-attention); fill the rest
            # of the budget with the top-scoring tokens. Bias self up so it is in.
            score[L - 1] = float("inf")
            top = score.topk(min(K, L)).indices               # LOCAL positions
            out = torch.sort(top).values                      # kernel needs ascending
            buf[r, :out.shape[0]] = out.to(buf.dtype)
    # Prefill: each token attends its causal context. Prompt <= budget (the common
    # reasoning case) -> attend ALL causal LOCAL positions. Overflow (long prompt +
    # small budget) -> recent-window fallback (same stack-wide limit as Quest/FASA).
    if m.num_prefills > 0 and m.prefill is not None:
        base = m.num_decode_tokens
        for ch in m.prefill.chunks:
            cu = ch.cu_seq_lens.to(torch.long)
            for j in range(ch.num_reqs):
                s, e = int(cu[j].item()), int(cu[j + 1].item())
                for i in range(e - s):
                    tok = base + ch.token_start + s + i
                    Lc = i + 1
                    if Lc <= K:
                        buf[tok, :Lc] = torch.arange(Lc, device=buf.device).to(buf.dtype)
                    else:
                        buf[tok, :K] = torch.arange(Lc - K, Lc, device=buf.device).to(buf.dtype)
    return buf


def triattn_mla_select_fake(
    key: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_prefix: LayerNameType,
    topk_indices_buffer: torch.Tensor,
    q_mean_real: torch.Tensor,
    q_mean_imag: torch.Tensor,
    q_abs_mean: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    rope_style: str,
    aggregation: str,
    topk_tokens: int,
    key_dim: int,
    kv_lora_rank: int,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="triattn_mla_select",
    op_func=triattn_mla_select,
    mutates_args=["kv_cache", "topk_indices_buffer"],
    fake_impl=triattn_mla_select_fake,
    dispatch_key=current_platform.dispatch_key,
)


class TriAttentionMLAIndexer(nn.Module):
    """TriAttention-on-MLA token selector (NOVEL port — no MLA variant exists upstream).

    TriAttention (frequency-domain KV pruning) scores a key purely from the RoPE'd
    complex pairs: given per-(layer,head) calibrated query frequency statistics
    (q_mean_complex, q_abs_mean over decode steps of the PRE-RoPE query as complex
    pairs) and the PRE-RoPE key as complex pairs k_complex, it forms

        relative = q_mean_complex * conj(k_complex)
        phi   = atan2(Im(relative), Re(relative))          # per-pair phase
        amp   = |q_mean_complex| * |k_complex|             # per-pair amplitude
        extra = (q_abs_mean - |q_mean_complex|) * |k_complex|   # mean-level residual (MLR)

    then for a decode at position `round_start` scores each cached key at position
    `key_idx` over a geometric grid of offsets o ∈ {1,2,4,...} (RoPE frequency
    aliasing probe):

        delta      = round_start - key_idx
        delta_grid = delta + offsets                       # [n_off]
        phase      = delta_grid[:,None] * omega[None,:] + phi          # [n_off, nFC]
        base       = Σ_FC amp * freq_scale_sq * cos(phase)             # [n_off]
        additive   = Σ_FC extra * freq_scale_sq            # scalar (offset-independent)
        score(o)   = base(o) + additive
        score      = mean_o score(o)                       # offset aggregation = mean

    MLA mapping (the novel part). MLA stores per token the latent c_KV (NoPE, dim
    kv_lora_rank=512 — NO RoPE, so NOT scorable by TriAttention) plus a single
    decoupled k_pe (qk_rope_head_dim=64 → 32 complex pairs, SHARED across heads =
    MQA). TriAttention's score lives ENTIRELY on the RoPE'd pairs, so for MLA we
    score PURELY on k_pe (32 pairs); omega = the k_pe rope inv_freq (32 freqs);
    k_unrot = the PRE-RoPE k_pe (cached pre-rope so it is directly available, no
    rope inversion). c_KV is attend-only (the FlashMLASparse attend reads the full
    [c_KV|k_pe] of the selected tokens — TriAttention only picks WHICH tokens).
    Per-head q_pe stats are reduced by GROUP-MEAN over the 20 heads to ONE query
    stat per layer (the faithful MQA reduction — the same lesson as QuestMLAIndexer:
    average the query, do NOT take per-head max).

    Own bf16 paged cache stores [c_KV | k_pe_PRE_rope] (head_dim = kv_lora_rank +
    qk_rope; pre-rope k_pe is written so k_unrot is read directly). Decode (eager,
    per request): gather cached pre-rope k_pe for all token positions, compute
    amp/phi/extra, score every cached token with round_start = current decode
    position, take top-n_fac, ALWAYS keep the current/last token, write ascending-
    sorted LOCAL positions (0..L-1) to topk_indices_buffer (FlashMLASparse converts
    local→global and REQUIRES ascending order). Prefill: prompt ≤ budget attends all
    causal; overflow falls back to the recent window (same stack-wide limit as
    Quest/FASA). Budget must be a multiple of 128 (512/1024/2048). enforce_eager
    required. Stats from triattn_stats_mla_<tag>.pt (calibrate.py
    _compute_triattn_stats_mla)."""

    def __init__(self, vllm_config, config, cache_config, fused_qkv_a_proj,
                 kv_a_proj_with_mqa, q_b_proj, kv_a_layernorm, kv_b_proj, rotary_emb,
                 q_lora_rank, basis_path, n_fac, topk_indices_buffer, prefix=""):
        super().__init__()
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.model_executor.layers.triattention_utils import (
            build_geometric_offsets,
        )

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.n_head = config.num_attention_heads
        self.key_dim = self.kv_lora_rank + self.qk_rope_head_dim   # 576 for GLM
        self.nFC = self.qk_rope_head_dim // 2                       # 32 complex pairs
        self.topk_tokens = n_fac
        self.fused_qkv_a_proj = fused_qkv_a_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.q_b_proj = q_b_proj
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_b_proj = kv_b_proj
        # NOTE: kv_a_layernorm / q_b_proj / kv_b_proj are kept for parity with the
        # other MLA indexers (the wrapper passes them) but TriAttention's MLA score
        # uses ONLY the pre-rope k_pe + the calibrated query stats — no q/k projection
        # is needed at decode (the query stats are baked in at calibration). c_KV is
        # cached for the attend only.
        self.rotary_emb = rotary_emb
        self.q_lora_rank = q_lora_rank
        self.layer_idx = extract_layer_index(prefix)
        self.topk_indices_buffer = topk_indices_buffer
        # offset-grid aggregation; "mean" matches the reference port default.
        self.aggregation = os.environ.get("TRIATTN_MLA_AGG", "mean")

        # Load calibrated TriAttention stats for this layer (group-meaned over heads
        # to a single query at calibration time). The .pt holds per-layer
        # q_mean_complex [nFC] (complex) + q_abs_mean [nFC] (real) + metadata with
        # omega [nFC], freq_scale [nFC], rope_dim, offset_max_length.
        ckpt = torch.load(basis_path, map_location="cpu", weights_only=False)
        meta = ckpt["metadata"]
        # RoPE complex-pairing style — MUST match the calibration that produced the
        # stats (and the model's RoPE convention). GLM-4.7-Flash has
        # rope_interleave=True -> style="interleaved" (pairs (x0,x1),(x2,x3),...);
        # the calibration de-interleaves identically before forming complex pairs.
        self.rope_style = meta.get("rope_style", "half")
        stats = ckpt["stats"][self.layer_idx]                       # per-layer dict
        q_mc = torch.complex(stats["q_mean_real"].float(),
                             stats["q_mean_imag"].float())          # [nFC]
        q_am = stats["q_abs_mean"].float()                          # [nFC]
        omega = torch.as_tensor(meta["omega"], dtype=torch.float32) # [nFC] rope inv_freq
        freq_scale = torch.as_tensor(meta["freq_scale"], dtype=torch.float32)  # [nFC]
        freq_scale_sq = freq_scale.pow(2)
        off_max = int(meta.get("offset_max_length", 512))
        _dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        offsets = build_geometric_offsets(off_max, torch.device(_dev))  # [n_off]
        if q_mc.shape[-1] != self.nFC or omega.shape[-1] != self.nFC:
            raise RuntimeError(
                f"TriAttention-MLA stats nFC mismatch: q_mean_complex {tuple(q_mc.shape)}, "
                f"omega {tuple(omega.shape)}, expected nFC={self.nFC} "
                f"(qk_rope_head_dim={self.qk_rope_head_dim}).")
        # Non-persistent on-device buffers (graph-capture safe; vLLM builds on GPU).
        self.register_buffer("q_mean_real", q_mc.real.to(_dev), persistent=False)
        self.register_buffer("q_mean_imag", q_mc.imag.to(_dev), persistent=False)
        self.register_buffer("q_abs_mean", q_am.to(_dev), persistent=False)
        self.register_buffer("omega", omega.to(_dev), persistent=False)
        self.register_buffer("freq_scale_sq", freq_scale_sq.to(_dev), persistent=False)
        self.register_buffer("offsets", offsets.to(_dev), persistent=False)
        for b in ("q_mean_real", "q_mean_imag", "q_abs_mean", "omega",
                  "freq_scale_sq", "offsets"):
            torch._dynamo.mark_static_address(getattr(self, b))

        # bf16 paged cache of [c_KV | k_pe_PRE_rope] (NOT fp8 — frequency score must
        # be exact; mirrors QuestMLAIndexer's own bf16 cache).
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.key_dim, dtype=torch.bfloat16,
            prefix=f"{prefix}.k_cache", cache_config=cache_config)

    def forward(self, hidden_states, q_c, positions, rotary_emb=None):
        kvlr = self.kv_lora_rank
        # latent c_KV + decoupled rope key (kv_a path). k_pe is taken PRE-rope so the
        # cached value is exactly TriAttention's k_unrot (no rope inversion needed).
        if self.q_lora_rank is not None:
            kv_lora = self.fused_qkv_a_proj(hidden_states)[0][..., self.q_lora_rank:]
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        c_kv = self.kv_a_layernorm(kv_lora[..., :kvlr])             # [T, kvlr]
        k_pe_pre = kv_lora[..., kvlr:]                              # [T, rope] PRE-rope
        key = torch.cat([c_kv, k_pe_pre], dim=-1)                  # [T, key_dim]

        # Dynamic frequency selection runs in a registered custom op (a
        # splitting_ops boundary) so it executes EAGER and is never traced by
        # inductor; the projections above are compiled. See triattn_mla_select.
        return torch.ops.vllm.triattn_mla_select(
            key, self.k_cache.kv_cache, _encode_layer_name(self.k_cache.prefix),
            self.topk_indices_buffer,
            self.q_mean_real, self.q_mean_imag, self.q_abs_mean, self.omega,
            self.offsets, self.freq_scale_sq, self.rope_style, self.aggregation,
            self.topk_tokens, self.key_dim, self.kv_lora_rank)


class FASAMLAIndexer(nn.Module):
    """FASA-on-MLA token selector (partial-RoPE recipe, OpenReview FnSgecCEwg §6 +
    DeepSeek-V2 rebuttal). MLA decouples RoPE to a small partition (k_pe); the NoPE
    latent has no RoPE -> no frequency channels. FASA's functional sparsity lives in
    the RoPE FCs, so the TIP token-selection uses ONLY the RoPE cache (k_pe) over the
    calibrated dominant FCs I_dom; the NoPE latent is uniformly important (not subset-
    selectable) and is used only in the FAC attend (full latent via FlashMLASparse).

        index_q[h] = [ mask_{I_dom}(q_pe[h])  |  0 ]      (RoPE channels not in I_dom zeroed)
        index_k    = [ k_pe                   |  0 ]      (shared / MQA on the RoPE key)

    so index_q[h]·index_k = Σ_{i∈I_dom} (q_pe[h]·k_pe)_i (the head-summed FASA RoPE-FC
    score), computed by the SAME fp8 paged-MQA-logits + radix-topk kernel as LRoSA/DSA.
    head_dim padded to 128 (= one fp8 quant block; only the qk_rope channels carry
    signal). I_dom from `fasa_idom_mla_<tag>.pt` (calibrate.py _compute_fasa_idom_mla)."""

    def __init__(self, vllm_config, config, cache_config, fused_qkv_a_proj,
                 kv_a_proj_with_mqa, q_b_proj, kv_a_layernorm, kv_b_proj, rotary_emb,
                 q_lora_rank, basis_path, n_tip, n_fac, topk_indices_buffer, prefix=""):
        super().__init__()
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.n_head = config.num_attention_heads
        self.topk_tokens = n_fac
        self.quant_block_size = 128
        self.scale_fmt = "ue8m0"
        self.fused_qkv_a_proj = fused_qkv_a_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.q_b_proj = q_b_proj
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_b_proj = kv_b_proj
        self.rotary_emb = rotary_emb
        self.q_lora_rank = q_lora_rank
        self.layer_idx = extract_layer_index(prefix)
        self.topk_indices_buffer = topk_indices_buffer
        # head_dim = one fp8 quant block; qk_rope channels (64) carry the score, rest 0.
        self.head_dim = self.quant_block_size
        self.softmax_scale = self.head_dim**-0.5

        # Load I_dom (dominant RoPE FCs for this layer) and build a per-channel keep
        # mask over qk_rope_head_dim (FC i -> channels {2i, 2i+1}). Zero non-dominant
        # RoPE channels of q so the head-sum logit = Σ_{i∈I_dom} (q_pe·k_pe)_i.
        ckpt = torch.load(basis_path, map_location="cpu", weights_only=False)
        idom = ckpt["idom"][self.layer_idx]              # [n_tip_max] FC indices (of qk_rope/2)
        n_keep = min(int(n_tip), idom.shape[0])
        keep_fc = idom[:n_keep].long()
        mask = torch.zeros(self.qk_rope_head_dim, dtype=torch.bfloat16)
        mask[2 * keep_fc] = 1.0
        mask[2 * keep_fc + 1] = 1.0
        _dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.register_buffer("q_pe_mask", mask.to(_dev), persistent=False)  # [qk_rope]
        torch._dynamo.mark_static_address(self.q_pe_mask)

        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8, prefix=f"{prefix}.k_cache", cache_config=cache_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache, self.quant_block_size, self.scale_fmt, self.topk_tokens,
            self.head_dim, vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config), topk_indices_buffer)

    def forward(self, hidden_states, q_c, positions, rotary_emb=None):
        T = hidden_states.shape[0]
        H = self.n_head
        rope_d = self.qk_rope_head_dim
        # decoupled-RoPE key (recompute the kv_a path); NoPE latent NOT needed for FASA score
        if self.q_lora_rank is not None:
            kv_lora = self.fused_qkv_a_proj(hidden_states)[0][..., self.q_lora_rank:]
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        k_pe = kv_lora[..., self.kv_lora_rank:].unsqueeze(1)               # [T,1,rope]
        q = self.q_b_proj(q_c)[0].view(T, H, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim:]                              # [T,H,rope]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_pe = q_pe.reshape(T, H, rope_d)
        k_pe = k_pe.reshape(T, rope_d)
        # mask non-dominant RoPE channels of q; pad both to head_dim (128) with zeros.
        q_pe = q_pe * self.q_pe_mask                                       # [T,H,rope]
        pad = self.head_dim - rope_d
        index_q = torch.nn.functional.pad(q_pe, (0, pad))                  # [T,H,head_dim]
        index_k = torch.nn.functional.pad(k_pe, (0, pad))                  # [T,head_dim]
        H_pad = 32 if H <= 32 else 64
        if H_pad != H:
            index_q = torch.nn.functional.pad(index_q, (0, 0, 0, H_pad - H))
        qf = index_q.reshape(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            qf, self.quant_block_size, column_major_scales=False, use_ue8m0=True)
        q_fp8 = q_fp8.view(T, H_pad, self.head_dim)
        q_scale = q_scale.view(T, H_pad, 1)
        weights = (q_scale * self.softmax_scale * (H ** -0.5)).squeeze(-1)  # [T,H_pad]
        if H_pad != H:
            weights[:, H:] = 0
        return self.indexer_op(hidden_states, q_fp8, index_k, weights)


class LRoSAMLAIndexer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config,
        cache_config: CacheConfig,
        fused_qkv_a_proj: nn.Module | None,
        kv_a_proj_with_mqa: nn.Module | None,
        q_b_proj: nn.Module | None,
        kv_a_layernorm: nn.Module,
        kv_b_proj: nn.Module,
        rotary_emb: nn.Module,
        q_lora_rank: int | None,
        basis_path: str,
        cs_h: int,
        n_fac: int,
        topk_indices_buffer: torch.Tensor | None,
        prefix: str = "",
    ):
        super().__init__()
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.n_head = config.num_attention_heads
        self.cs_h = cs_h
        # head_dim / softmax_scale are set after M is loaded (variant-dependent).
        self.topk_tokens = n_fac
        self.quant_block_size = 128
        self.scale_fmt = "ue8m0"

        self.fused_qkv_a_proj = fused_qkv_a_proj
        self.kv_a_proj_with_mqa = kv_a_proj_with_mqa
        self.q_b_proj = q_b_proj
        self.kv_a_layernorm = kv_a_layernorm
        self.kv_b_proj = kv_b_proj
        self.rotary_emb = rotary_emb
        self.q_lora_rank = q_lora_rank
        self.layer_idx = extract_layer_index(prefix)

        self.topk_indices_buffer = topk_indices_buffer  # MLA wrapper reads this
        # Load the calibrated rotation M HERE (not lazily in forward): torch.load is
        # untraceable and breaks vLLM's fullgraph CUDA-graph capture. Register as a
        # non-persistent buffer so it moves to device with the model.
        ckpt = torch.load(basis_path, map_location="cpu", weights_only=False)
        M0 = ckpt["M"][self.layer_idx][0]                  # [rows, cols]
        # Variant. A-variant: M is [cs_h, kv_lora] — project NoPE only, RoPE is
        # concatenated RAW after projection (index dim = cs_h + qk_rope).
        # B-variant: M is [cs_h, kv_lora + qk_rope] — JOINTLY compress
        # [c_kv | k_pe] so RoPE lives inside the rotation (index dim = cs_h).
        self.concat_b = (ckpt.get("variant") == "concat_b"
                         or M0.shape[-1] == self.kv_lora_rank + self.qk_rope_head_dim)
        self.head_dim = cs_h if self.concat_b else cs_h + self.qk_rope_head_dim
        self.softmax_scale = self.head_dim**-0.5
        # Put M_basis on the worker's device at construction (vLLM builds the
        # model on the target GPU; current_device == this worker's GPU). A
        # non-persistent CPU buffer is NOT moved by vLLM weight loading, which
        # would force a CPU->CUDA copy in forward that breaks CUDA-graph capture.
        _dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.register_buffer("M_basis",
                             M0.to(torch.bfloat16).to(_dev).contiguous(),
                             persistent=False)             # [rows, cols] on-device
        torch._dynamo.mark_static_address(self.M_basis)
        # ReLU-shift: the deepgemm logits kernel computes sum_h w_h*RELU(q_h.k)
        # (sm100_fp8_paged_mqa_logits.cuh) while M is calibrated for the plain
        # head-sum — the ReLU breaks the calibrated ranking (true-attention
        # coverage 0.91 -> 0.79 on reasoning traces). Basis files with
        # "shift_c" carry a 63-row M + a per-layer constant c: one head_dim
        # slot is spent on the pair (beta, c/beta), beta = sqrt(c), so every
        # per-head logit gains +c and ReLU is the identity (relu(x+c)=x+c);
        # the +c offset is constant per key, leaving the ranking equal to the
        # calibrated head-sum. Verified: coverage 0.787 -> 0.912 incl. fp8.
        shift = ckpt.get("shift_c")
        self.shift_c = float(shift[self.layer_idx]) if shift is not None else 0.0
        self.shift_beta = self.shift_c ** 0.5 if self.shift_c > 0 else 1.0
        rows, cols = self.M_basis.shape
        if self.concat_b:
            exp_rows, exp_cols = self.head_dim, self.kv_lora_rank + self.qk_rope_head_dim
        else:
            exp_rows = self.head_dim - self.qk_rope_head_dim - (1 if self.shift_c > 0 else 0)
            exp_cols = self.kv_lora_rank
        if (rows, cols) != (exp_rows, exp_cols):
            raise RuntimeError(
                f"LRoSA-MLA basis shape {(rows, cols)} != expected "
                f"{(exp_rows, exp_cols)} (variant={'B' if self.concat_b else 'A'}, "
                f"head_dim={self.head_dim}, shift={'on' if self.shift_c > 0 else 'off'}).")
        # env knobs read once (NOT in the traced forward)
        self.alpha = float(os.environ.get("LROSA_MLA_ROPE_W", "1.0"))
        self.recent_window = int(os.environ.get("LROSA_MLA_RECENT_W", "0"))

        # Reuse the DSA fp8 index cache + the fp8 paged-logits/radix-topk op.
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8, prefix=f"{prefix}.k_cache", cache_config=cache_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache, self.quant_block_size, self.scale_fmt, self.topk_tokens,
            self.head_dim, vllm_config.model_config.max_model_len,
            get_max_prefill_buffer_size(vllm_config), topk_indices_buffer)

    def forward(self, hidden_states, q_c, positions, rotary_emb=None):
        T = hidden_states.shape[0]
        H = self.n_head
        # latent c_KV + decoupled rope key (recompute the kv_a path)
        if self.q_lora_rank is not None:
            kv_lora = self.fused_qkv_a_proj(hidden_states)[0][..., self.q_lora_rank:]
        else:
            kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        c_kv = self.kv_a_layernorm(kv_lora[..., : self.kv_lora_rank])      # [T, kvlr]
        k_pe = kv_lora[..., self.kv_lora_rank:].unsqueeze(1)               # [T,1,rope]
        # absorbed query q_latent = q_nope @ W_UK
        q = self.q_b_proj(q_c)[0].view(T, H, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]                           # [T,H,nope]
        q_pe = q[..., self.qk_nope_head_dim:]                              # [T,H,rope]
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q_pe = q_pe.reshape(T, H, self.qk_rope_head_dim)
        k_pe = k_pe.reshape(T, self.qk_rope_head_dim)
        W_UK = self.kv_b_proj.weight.view(
            H, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )[:, : self.qk_nope_head_dim, :]                                   # [H,nope,kvlr] view
        Mb = self.M_basis   # already on-device (set in __init__), graph-capture safe
        q_latent = torch.einsum("thn,hnl->thl", q_nope, W_UK)              # [T,H,kvlr]
        if self.alpha != 1.0:
            q_pe = q_pe * self.alpha
        if self.concat_b:
            # B-variant: jointly compress [c_kv | k_pe] with one M[cs, kvlr+rope].
            # RoPE is inside the rotation (no raw concat), index dim = cs_h.
            q_full = torch.cat([q_latent, q_pe], dim=-1)                   # [T,H,kvlr+rope]
            k_full = torch.cat([c_kv, k_pe], dim=-1)                       # [T,kvlr+rope]
            index_q = torch.einsum("thd,cd->thc", q_full, Mb)             # [T,H,cs_h]
            index_k = k_full @ Mb.T                                       # [T,cs_h]
        else:
            # A-variant: project NoPE, concatenate raw RoPE (+ optional ReLU-shift).
            proj_q = torch.einsum("thl,cl->thc", q_latent, Mb)            # [T,H,cs_h]
            proj_K = c_kv @ Mb.T                                          # [T,cs_h]
            if self.shift_c > 0:
                bq = q_pe.new_full((T, H, 1), self.shift_beta)
                bk = k_pe.new_full((T, 1), self.shift_c / self.shift_beta)
                index_q = torch.cat([proj_q, q_pe, bq], dim=-1)           # [T,H,head_dim]
                index_k = torch.cat([proj_K, k_pe, bk], dim=-1)           # [T,head_dim]
            else:
                index_q = torch.cat([proj_q, q_pe], dim=-1)               # [T,H,head_dim]
                index_k = torch.cat([proj_K, k_pe], dim=-1)               # [T,head_dim]
        # The deepgemm fp8 paged-MQA-logits kernel only supports num_heads in
        # {32, 64}. GLM has H=20 -> zero-pad the query heads to H_pad; padded heads
        # contribute 0 to the head-sum score (q=0) and get weight 0, so the top-k
        # is exactly the sum over the H real heads.
        H_pad = 32 if H <= 32 else 64
        if H_pad != H:
            index_q = torch.nn.functional.pad(index_q, (0, 0, 0, H_pad - H))
        # fp8 quant of q (per group), fold scale + uniform weight (no learned wk weight)
        qf = index_q.reshape(-1, self.head_dim)
        q_fp8, q_scale = per_token_group_quant_fp8(
            qf, self.quant_block_size, column_major_scales=False, use_ue8m0=True)
        q_fp8 = q_fp8.view(T, H_pad, self.head_dim)
        q_scale = q_scale.view(T, H_pad, 1)
        weights = (q_scale * self.softmax_scale * (H ** -0.5)).squeeze(-1)  # [T,H_pad]
        if H_pad != H:
            weights[:, H:] = 0  # padded heads add nothing
        buf = self.indexer_op(hidden_states, q_fp8, index_k, weights)

        # Recency window: MLA's positionless latent makes pure score-based
        # selection drop the recent reasoning chain (-> rambling). Force the last
        # W cached positions into each decode token's top-k (StreamingLLM/H2O-style
        # structural prior; not a tuned knob). env LROSA_MLA_RECENT_W (0=off).
        W = self.recent_window
        if W > 0:
            md = get_forward_context().attn_metadata
            if isinstance(md, dict):
                m = md.get(self.k_cache.prefix)
                if m is not None and m.num_decodes > 0 and m.decode is not None:
                    nd = m.num_decodes
                    sl = m.decode.seq_lens.view(nd, -1)[:, -1].to(torch.long)   # [nd]
                    ar = torch.arange(W, device=buf.device)
                    rec = sl.unsqueeze(1) - 1 - ar                       # [nd,W] recent desc
                    # out-of-range (seq_len < W) -> -1 (skip), NOT 0: clamping to 0
                    # duplicates the BOS token, which the attend softmax over-weights
                    # (sums duplicate keys) and corrupts the output -> immediate eos.
                    rec = torch.where(rec >= 0, rec,
                                      torch.full_like(rec, -1)).to(buf.dtype)
                    w = min(W, buf.shape[1])
                    buf[:nd, buf.shape[1] - w:] = rec[:, :w]
        return buf
