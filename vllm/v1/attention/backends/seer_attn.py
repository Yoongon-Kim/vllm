# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SeerAttention-R (Gao et al., 2025, arXiv:2506.08889) decode-sparse backend.

Port of the *decode* branch of microsoft/SeerAttention onto vLLM's paged
combined-slot cache, so SeerAttention can be compared in the SAME stack as
QUEST / LRoSA / FASA. SeerAttention-R attaches a small **trained AttnGate** to
every decoder layer: it pools the K cache into ``gate_block_size`` (=64) blocks,
projects pooled-K and the (GQA-grouped) query into a low-dim gate space, scores
each K-block by a softmax over q·k_block, and attends only the top
``token_budget // gate_block_size`` blocks (+ the always-attended recent /
trailing tokens). The gate weights live in a tiny adapter checkpoint
(``attn_gate_weights.pth``) composed on top of the frozen base model.

Design (Qwen3-8B target; head_size 128, 32 Q / 8 KV heads, gate_hidden 128):
  * Cache layout — block == page (page_size 16), per-token slot ``[ K | V ]``
    (2*head_size), identical to the QUEST backend. ``gate_block_size`` (64) is
    a multiple of the page size, so one gate-block == 4 paged blocks.
  * The gate's **compressed-K** (one ``gate_hidden`` vector per 64-token block,
    per kv-head) is kept in a SEPARATE backend-managed buffer
    ``layer._seer_kc`` of shape ``(num_blocks, H_kv, gate_hidden)``, indexed by
    the PHYSICAL block id of the gate-block's first page — stable across decode
    steps, the same trick QUEST uses for its per-page min/max. A gate-block is
    compressed once, when it fills (prefill: all complete blocks; decode: the
    block that completes when ``seq_len % 64 == 0``).
  * Decode forward: build the gate query, score the compressed K-blocks, pick
    the top blocks, **expand each selected 64-block into its 4 page columns**,
    always include the current partial gate-block's pages (recency), then reuse
    the QUEST block-sparse paged attention kernel
    (:func:`quest_blocksparse_attn`) — no new attention kernel. When
    ``seq_len <= token_budget`` all blocks are selected ⇒ exactly dense.
  * Prefill is dense (SeerAttention-R does not sparsify prefill); the prefill
    branch only compresses the completed K-blocks so subsequent decodes have
    valid gate metadata.

Limitations (first integration, verification = build + smoke): runs
**eager-only** (``AttentionCGSupport.NEVER``) — the per-step gate compression is
data-dependent (fires every 64 tokens) and the per-request gather is a PyTorch
loop. CUDA-graph capture + a fused gate kernel are a follow-up optimisation; the
attended-token *selection* (the accuracy-relevant part) is already faithful.
"""

import math
import os
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_lrosa_store import lrosa_store
from vllm.v1.attention.ops.triton_quest import (
    quest_blocksparse_attn,
    quest_num_splits,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()

# Gate-blocks map onto an integer number of paged pages (page 16 | block 64).
_SEER_PAGE_SIZE = 16


def _load_gate_state(path: str) -> dict:
    """Load the AttnGate state_dict from a local ``.pth`` file, a local adapter
    snapshot dir, or an HF repo id. Returns the raw ``{param_name: tensor}``
    dict (keys ``model.layers.{i}.self_attn.attn_gate.*``)."""
    fname = "attn_gate_weights.pth"
    if os.path.isfile(path):
        ckpt_path = path
    elif os.path.isdir(path):
        ckpt_path = os.path.join(path, fname)
    else:
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id=path, filename=fname)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise RuntimeError(f"SeerAttention: {ckpt_path} is not a state_dict.")
    return sd


class SeerAttentionBackend(AttentionBackend):
    """SeerAttention-R decode-sparse backend (combined-slot cache)."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["seer"]
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "seer"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [_SEER_PAGE_SIZE]

    @staticmethod
    def get_name() -> str:
        return "SEER"

    @staticmethod
    def get_impl_cls() -> type["SeerImpl"]:
        return SeerImpl

    @staticmethod
    def get_builder_cls() -> type["SeerMetadataBuilder"]:
        return SeerMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size != _SEER_PAGE_SIZE:
            raise ValueError(
                f"SeerAttention: block_size must equal page_size "
                f"({_SEER_PAGE_SIZE}); got {block_size}."
            )
        slot_size = 2 * head_size
        return (num_blocks, block_size, num_kv_heads, slot_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128, 256]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


@dataclass
class SeerMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    num_decodes: int = 0
    num_decode_tokens: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    causal: bool = True


class SeerMetadataBuilder(AttentionMetadataBuilder[SeerMetadata]):
    # Eager-only: the gate compression is data-dependent (every 64 tokens) and
    # the per-request gather is a Python loop — not CUDA-graph capturable yet.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config: VllmConfig, kv_cache_spec: AttentionSpec
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_size = kv_cache_spec.head_size
        self.block_size = kv_cache_spec.block_size

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SeerMetadata:
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, _np, num_decode_tokens, _npt = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold)
        return SeerMetadata(
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            query_start_loc=cam.query_start_loc,
            seq_lens=cam.seq_lens,
            block_table=cam.block_table_tensor,
            slot_mapping=cam.slot_mapping,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
            causal=cam.causal,
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class SeerImpl(AttentionImpl[SeerMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        vllm_config = get_current_vllm_config()
        ac = vllm_config.attention_config
        self.gate_path = getattr(ac, "seer_gate_path", None)
        self.token_budget = int(getattr(ac, "seer_token_budget", 4096))
        self.gate_block_size = int(getattr(ac, "seer_gate_block_size", 64))
        self.gate_hidden = int(getattr(ac, "seer_gate_hidden_size", 128))
        self.sparsity_method = str(getattr(ac, "seer_sparsity_method", "token_budget"))
        self.threshold = float(getattr(ac, "seer_threshold", 0.0))
        self.start_layer = int(getattr(ac, "seer_start_layer", 0))
        self.rope_theta = float(
            getattr(vllm_config.model_config.hf_config, "rope_theta", 1e6))
        # gate-block <-> page geometry
        self.pages_per_block = self.gate_block_size // _SEER_PAGE_SIZE
        self.block_budget = max(self.token_budget // self.gate_block_size, 1)
        # selectable page columns: top blocks (+ forced-last recency block +
        # one current partial gate-block of recent pages), expanded to pages.
        self.page_budget = (self.block_budget + 2) * self.pages_per_block

        if self.gate_block_size % _SEER_PAGE_SIZE != 0:
            raise ValueError(
                f"SeerAttention: gate_block_size ({self.gate_block_size}) must "
                f"be a multiple of page_size ({_SEER_PAGE_SIZE}).")
        if self.gate_path is None:
            raise RuntimeError(
                "SeerAttention requires `-ac.seer_gate_path=<adapter>` "
                "(attn_gate_weights.pth / adapter dir / HF repo id).")

        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = (-1, -1) if sliding_window is None else (
            sliding_window - 1, 0)
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self._gate_state_cpu: dict | None = None
        self._inv_freq: torch.Tensor | None = None  # (gate_hidden/2,)

    # ---- rotary (gate head_dim == gate_hidden, NeoX rotate_half) ----
    def _rope(self, positions: torch.Tensor, device, dtype):
        """cos/sin for `positions` (1D) → (P, gate_hidden) each."""
        if self._inv_freq is None or self._inv_freq.device != device:
            d = self.gate_hidden
            self._inv_freq = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, d, 2, device=device, dtype=torch.float32) / d))
        freqs = positions.to(torch.float32)[:, None] * self._inv_freq[None, :]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rope(self, x, cos, sin):
        # x: (..., H_kv, gate_hidden); cos/sin: (P, gate_hidden) broadcast over H_kv
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        return x * cos + _rotate_half(x) * sin

    # ---- per-layer gate weights (lazy, cached on the layer) ----
    def _ensure_gate(self, layer: torch.nn.Module, device, dtype):
        cached = getattr(layer, "_seer_gate", None)
        if cached is not None:
            return cached
        if self._gate_state_cpu is None:
            self._gate_state_cpu = _load_gate_state(self.gate_path)
        from vllm.model_executor.models.utils import extract_layer_index

        li = extract_layer_index(layer.layer_name)
        pre = f"model.layers.{li}.self_attn.attn_gate."
        sd = self._gate_state_cpu
        try:
            wq = sd[pre + "attngate_linear_q.weight"]  # (H_kv, gqa, hs, gate_h)
            wk = sd[pre + "attngate_linear_k.weight"]  # (H_kv, 3*hs, gate_h)
        except KeyError as e:
            raise RuntimeError(
                f"SeerAttention: missing gate weight {e} for layer {li}.")
        gate = {
            "wq": wq.to(device=device, dtype=dtype).contiguous(),
            "wk": wk.to(device=device, dtype=dtype).contiguous(),
            "qnorm": sd.get(pre + "attngate_qnorm.weight"),
            "knorm": sd.get(pre + "attngate_knorm.weight"),
            "layer_idx": li,
        }
        if gate["qnorm"] is not None:
            gate["qnorm"] = gate["qnorm"].to(device=device, dtype=torch.float32)
        if gate["knorm"] is not None:
            gate["knorm"] = gate["knorm"].to(device=device, dtype=torch.float32)
        layer._seer_gate = gate
        return gate

    @staticmethod
    def _rmsnorm(x, weight, eps=1e-6):
        if weight is None:
            return x
        dt = x.dtype
        xf = x.to(torch.float32)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        return (xf * weight).to(dt)

    def do_kv_cache_update(
        self, layer, key, value, kv_cache, slot_mapping
    ) -> None:
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        lrosa_store(k, v, kv_cache, slot_mapping)

    def _ensure_kc(self, layer, kv_cache):
        kc = getattr(layer, "_seer_kc", None)
        if kc is None:
            kc = torch.zeros(
                kv_cache.shape[0], self.num_kv_heads, self.gate_hidden,
                dtype=kv_cache.dtype, device=kv_cache.device)
            layer._seer_kc = kc
        return kc

    def _gather_block_k(self, kv_cache, block_table_r, g, head_size):
        """Gather the 64 raw-K vectors of gate-block g for one request.
        Returns (gate_block_size, H_kv, head_size)."""
        bs = self.gate_block_size
        ps = _SEER_PAGE_SIZE
        cols = block_table_r[g * self.pages_per_block:(g + 1) * self.pages_per_block]
        phys = cols.to(torch.int64)
        # kv_cache[phys] : (pages_per_block, ps, H_kv, 2*hs) → K part
        k = kv_cache[phys, :, :, :head_size]  # (ppb, ps, H_kv, hs)
        return k.reshape(bs, self.num_kv_heads, head_size)

    def _compress_block(self, gate, k_raw, block_pos, device, dtype):
        """k_raw: (64, H_kv, hs) → compressed gate-K (H_kv, gate_hidden)."""
        kmax = k_raw.amax(dim=0)  # (H_kv, hs)
        kmin = k_raw.amin(dim=0)
        kavg = k_raw.mean(dim=0)
        pooled = torch.cat([kmax, kmin, kavg], dim=-1)  # (H_kv, 3*hs)
        # MultiHeadLinear: einsum('hi,hio->ho')
        kc = torch.einsum("hi,hio->ho", pooled.to(dtype), gate["wk"])  # (H_kv,gh)
        kc = self._rmsnorm(kc, gate["knorm"])
        cos, sin = self._rope(
            torch.tensor([block_pos], device=device), device, dtype)
        kc = self._apply_rope(kc.unsqueeze(0), cos, sin).squeeze(0)
        return kc

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata: SeerMetadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "SeerAttention: fused output quant not supported.")
        if attn_metadata is None:
            return output.fill_(0)

        head_size = self.head_size
        num_actual = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        key_cache = kv_cache[..., :head_size]
        value_cache = kv_cache[..., head_size : 2 * head_size]
        device = query.device
        dtype = query.dtype

        gate = self._ensure_gate(layer, device, dtype)
        kc_buf = self._ensure_kc(layer, kv_cache)
        is_dense_layer = gate["layer_idx"] < self.start_layer

        # ---- prefill / mixed ----
        if num_decodes == 0 or num_decode_tokens < num_actual:
            if num_decodes > 0:
                self._decode(layer, gate, kc_buf, query, kv_cache, attn_metadata,
                             output, head_size, is_dense_layer)
            # compress all completed K-blocks for each prefill request
            if not is_dense_layer:
                self._compress_prefill(gate, kc_buf, kv_cache, attn_metadata,
                                       head_size, num_decodes, num_decode_tokens,
                                       device, dtype)
            # dense flash over the prefill rows
            if num_decodes == 0:
                q_pf, out_pf = query[:num_actual], output[:num_actual]
                qsl = attn_metadata.query_start_loc
                seqk = attn_metadata.seq_lens
                bt = attn_metadata.block_table
            else:
                q_pf = query[num_decode_tokens:num_actual]
                out_pf = output[num_decode_tokens:num_actual]
                qsl = attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
                seqk = attn_metadata.seq_lens[num_decodes:]
                bt = attn_metadata.block_table[num_decodes:]
            if head_size > 256:
                unified_attention(
                    q=q_pf, k=key_cache, v=value_cache, out=out_pf,
                    cu_seqlens_q=qsl, max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=seqk, max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale, causal=True,
                    window_size=self.sliding_window, block_table=bt,
                    softcap=self.logits_soft_cap,
                    q_descale=None, k_descale=None, v_descale=None,
                    alibi_slopes=self.alibi_slopes)
            else:
                flash_attn_varlen_func(
                    q=q_pf, k=key_cache, v=value_cache, out=out_pf,
                    cu_seqlens_q=qsl, max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=seqk, max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale, causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=list(self.sliding_window), block_table=bt,
                    softcap=self.logits_soft_cap)
            return output

        # ---- all-decode ----
        self._decode(layer, gate, kc_buf, query, kv_cache, attn_metadata,
                     output, head_size, is_dense_layer)
        return output

    def _compress_prefill(self, gate, kc_buf, kv_cache, md, head_size,
                          num_decodes, num_decode_tokens, device, dtype):
        qsl_cpu = md.query_start_loc_cpu
        seq_cpu = md.seq_lens_cpu
        if qsl_cpu is None or seq_cpu is None:
            return
        bt = md.block_table
        bs = self.gate_block_size
        for r in range(num_decodes, bt.shape[0]):
            seq_len = int(seq_cpu[r].item())
            nblocks = seq_len // bs
            for g in range(nblocks):
                k_raw = self._gather_block_k(kv_cache, bt[r], g, head_size)
                kc = self._compress_block(gate, k_raw, g * bs, device, dtype)
                kc_buf[int(bt[r, g * self.pages_per_block].item())] = kc

    def _decode(self, layer, gate, kc_buf, query, kv_cache, md, output,
                head_size, is_dense_layer):
        nd = md.num_decodes
        device = query.device
        dtype = query.dtype
        bs = self.gate_block_size
        ppb = self.pages_per_block
        ps = _SEER_PAGE_SIZE
        block_table = md.block_table[:nd]
        seq_lens = md.seq_lens[:nd]
        seq_cpu = md.seq_lens_cpu
        q_dec = query[:nd]  # (nd, H_q, hs)
        out_view = output[:nd].view(nd, self.num_heads, head_size)

        # 1. compress the gate-block that just completed (seq_len % bs == 0)
        if not is_dense_layer:
            for r in range(nd):
                sl = int(seq_cpu[r].item())
                if sl % bs == 0 and sl >= bs:
                    g = sl // bs - 1
                    k_raw = self._gather_block_k(kv_cache, block_table[r], g,
                                                 head_size)
                    kc = self._compress_block(gate, k_raw, g * bs, device, dtype)
                    kc_buf[int(block_table[r, g * ppb].item())] = kc

        # dense layer (layer_idx < start_layer): plain paged flash decode
        if is_dense_layer:
            flash_attn_varlen_func(
                q=q_dec, k=kv_cache[..., :head_size],
                v=kv_cache[..., head_size : 2 * head_size], out=out_view,
                cu_seqlens_q=md.query_start_loc[:nd + 1],
                max_seqlen_q=1, seqused_k=seq_lens,
                max_seqlen_k=md.max_seq_len, softmax_scale=self.scale,
                causal=True, alibi_slopes=self.alibi_slopes,
                window_size=[-1, -1], block_table=block_table,
                softcap=self.logits_soft_cap)
            return

        # 2. gate query: HeadPoolingLinear over GQA group + qnorm + RoPE@cur_pos
        qg = q_dec.view(nd, self.num_kv_heads, self.num_kv_groups, head_size)
        qg = torch.einsum("nkgi,kgio->nko", qg.to(dtype), gate["wq"])  # (nd,H_kv,gh)
        qg = self._rmsnorm(qg, gate["qnorm"])
        cur_pos = (seq_lens.to(torch.int64) - 1)
        cos, sin = self._rope(cur_pos, device, dtype)  # (nd, gh)
        qg = self._apply_rope(qg, cos, sin)  # (nd, H_kv, gh)

        # 3. block scoring + page-column selection — fully vectorized (no
        #    per-(req, head) Python loop, no host syncs in the hot path).
        H_kv = self.num_kv_heads
        sl = seq_lens.to(torch.int64)                # (nd,)
        nb = sl // bs                                # completed gate-blocks (nd,)
        num_full = (sl - 1) // ps                    # selectable full pages (nd,)
        max_nb = int(nb.max().item()) if nd > 0 else 0  # 1 cheap sync/step
        inv_sqrt = 1.0 / math.sqrt(self.gate_hidden)
        max_pages = block_table.shape[1]
        p_ar = torch.arange(max_pages, device=device)        # (max_pages,)

        keep = None
        if max_nb > 0:
            # gather compressed-K at each gate-block's first physical page
            bstart = block_table[:, ::ppb][:, :max_nb].to(torch.int64)  # (nd,max_nb)
            kc = kc_buf[bstart]                          # (nd, max_nb, H_kv, gh)
            attn = torch.einsum("nhd,nshd->nhs", qg, kc) * inv_sqrt  # (nd,H_kv,max_nb)
            g_ar = torch.arange(max_nb, device=device)
            valid_blk = g_ar[None, :] < nb[:, None]      # (nd, max_nb)
            attn = attn.masked_fill(~valid_blk[:, None, :], float("-inf"))
            attn = torch.softmax(attn.float(), dim=-1)
            if self.sparsity_method == "threshold":
                keep = attn > self.threshold
            else:
                kbud = min(self.block_budget, max_nb)
                topk = attn.topk(kbud, dim=-1).indices   # (nd,H_kv,kbud)
                keep = torch.zeros_like(attn, dtype=torch.bool)
                keep.scatter_(-1, topk, True)
            keep &= valid_blk[:, None, :]
            # recency: force the last completed block (g = nb-1) per req w/ nb>0
            last = (nb - 1).clamp(min=0)
            keep.scatter_(
                -1, last[:, None, None].expand(nd, H_kv, 1),
                (nb > 0)[:, None, None].expand(nd, H_kv, 1))

        # per-(req, kv-head) page-keep mask over all page columns
        nbp = (nb * ppb)[:, None, None]              # (nd,1,1) block-region end
        nf = num_full[:, None, None]                 # (nd,1,1)
        pe = p_ar[None, None, :]                     # (1,1,max_pages)
        in_recent = pe >= nbp                        # current partial-block pages
        if keep is not None:
            blk_of_p = (p_ar // ppb).clamp(max=max_nb - 1)   # (max_pages,)
            bk_exp = keep[:, :, blk_of_p]            # (nd,H_kv,max_pages)
            page_keep = ((bk_exp & (pe < nbp)) | in_recent) & (pe < nf)
        else:
            page_keep = in_recent & (pe < nf)        # short ctx (nb==0): dense
        # pad selected columns ascending → fixed width; sentinel BIG → -1
        BIG = max_pages + 1
        p_full = p_ar.view(1, 1, -1).expand(nd, H_kv, max_pages)
        key = torch.where(page_keep, p_full, torch.full_like(p_full, BIG))
        sel = key.sort(dim=-1).values[:, :, : self.page_budget]
        page_idx = torch.where(
            sel >= BIG, torch.full_like(sel, -1), sel).to(torch.int32)

        # 4. block-sparse paged attention over the selected page columns (+ the
        #    always-attended trailing partial page, handled by the kernel).
        ns = quest_num_splits(self.page_budget)
        partial_acc = partial_m = partial_l = None
        if ns > 1:
            partial_acc = torch.empty(
                (nd, self.num_heads, ns, head_size),
                dtype=torch.float32, device=device)
            partial_m = torch.empty(
                (nd, self.num_heads, ns), dtype=torch.float32, device=device)
            partial_l = torch.empty(
                (nd, self.num_heads, ns), dtype=torch.float32, device=device)
        quest_blocksparse_attn(
            query=q_dec, kv_cache=kv_cache, page_idx=page_idx,
            block_table=block_table, seq_lens=seq_lens, output=out_view,
            scale=self.scale, page_size=ps, head_size=head_size,
            num_kv_groups=self.num_kv_groups,
            partial_acc=partial_acc, partial_m=partial_m, partial_l=partial_l)
