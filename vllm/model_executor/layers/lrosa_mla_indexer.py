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
        self.head_dim = cs_h + self.qk_rope_head_dim       # index score dim (=128 @ cs_h64)
        self.topk_tokens = n_fac
        self.softmax_scale = self.head_dim**-0.5
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
        M = ckpt["M"][self.layer_idx]
        # Put M_basis on the worker's device at construction (vLLM builds the
        # model on the target GPU; current_device == this worker's GPU). A
        # non-persistent CPU buffer is NOT moved by vLLM weight loading, which
        # would force a CPU->CUDA copy in forward that breaks CUDA-graph capture.
        _dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        self.register_buffer("M_basis",
                             M[0].to(torch.bfloat16).to(_dev).contiguous(),
                             persistent=False)             # [rows, kv_lora_rank] on-device
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
        rows = self.M_basis.shape[0]
        expect = self.head_dim - self.qk_rope_head_dim - (1 if self.shift_c > 0 else 0)
        if rows != expect:
            raise RuntimeError(
                f"LRoSA-MLA basis rows {rows} != expected {expect} "
                f"(head_dim={self.head_dim}, rope={self.qk_rope_head_dim}, "
                f"shift={'on' if self.shift_c > 0 else 'off'}).")
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
        proj_q = torch.einsum("thl,cl->thc", q_latent, Mb)                 # [T,H,cs_h]
        proj_K = c_kv @ Mb.T                                               # [T,cs_h]
        # MLA's latent c_KV is positionless; recency lives only in the decoupled
        # RoPE term q_pe.k_pe. Up-weight it (alpha, diagnostic) — recency window is
        # the real fix (below).
        if self.alpha != 1.0:
            q_pe = q_pe * self.alpha
        if self.shift_c > 0:
            # ReLU-shift constant pair (see __init__): logit += beta*(c/beta)=c.
            bq = q_pe.new_full((T, H, 1), self.shift_beta)
            bk = k_pe.new_full((T, 1), self.shift_c / self.shift_beta)
            index_q = torch.cat([proj_q, q_pe, bq], dim=-1)                # [T,H,head_dim]
            index_k = torch.cat([proj_K, k_pe, bk], dim=-1)                # [T,head_dim]
        else:
            index_q = torch.cat([proj_q, q_pe], dim=-1)                    # [T,H,head_dim]
            index_k = torch.cat([proj_K, k_pe], dim=-1)                    # [T,head_dim]
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
