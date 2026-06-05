# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quest (Tang et al., ICML 2024) page-level sparse-attention backend.

Production port of pca's ``quest/quest.py`` onto vLLM's paged combined-slot
cache, so Quest and LRoSA can be compared in the SAME stack (CUDA graph +
flash-attn + radix top-K). The research question this backend exists to
answer: Quest scores T/page_size *pages* (reading only 2 min/max vectors
each) vs LRoSA's T *tokens* (reading a cs_h-dim proj). Quest's scoring is
structurally cheaper — does that translate to an e2e decode-latency edge
that offsets LRoSA's +10.8 AIME reasoning win? Only a same-stack
measurement settles it.

Cache layout — block == page (block_size forced to page_size):
    slot per (token, kv-head): [ K(hs) | V(hs) | K_min(hs) | K_max(hs) ]
    slot_size = 4 * head_size. K_min/K_max are the page's per-channel
    min/max, kept ONLY at the block's representative slot (position 0).

Decode path (CG-captured, all-decode batch):
  1. fold the new decode K into its block's running min/max
     (``quest_minmax_update``)
  2. per-(req, kv-head, page) upper-bound score reading only min/max
     (``quest_page_score``)
  3. page top-K (radix, reused from the LRoSA ops)
  4. expand selected pages → token indices + always-attended trailing
     partial page (``quest_pages_to_token_idx``)
  5. gather (reused ``lrosa_gather``) + flash-attn varlen with
     ``seqused_k`` capping each request's variable attended count.

Prefill is dense (Quest does not sparsify prefill); the prefill branch only
builds per-full-page min/max so the subsequent decodes have valid metadata.

Correctness regime: exact for ``seq_len >= token_budget`` (the deployment /
long-context regime, and the only one the LRoSA-vs-Quest comparison runs
in). For very short sequences the fixed-shape gather may double-count a few
padding tokens — same pragmatic stance as the LRoSA backend's short-context
path, and irrelevant to the long-context / long-decode benchmarks.
"""

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
from vllm.v1.attention.ops.triton_lrosa_score_topk import (
    _radix_topk,
    _radix_topk_available,
)
from vllm.v1.attention.ops.triton_lrosa_store import lrosa_store
from vllm.v1.attention.ops.triton_quest import (
    quest_blocksparse_attn,
    quest_build_page_minmax_prefill,
    quest_minmax_update,
    quest_num_splits,
    quest_page_score,
)
from vllm.v1.kv_cache_interface import AttentionSpec

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()

# Quest pages map 1:1 onto paged-cache blocks, so block_size must equal the
# page size. 16 matches the Quest paper + pca reference; the only supported
# value (raising the engine block_size would put multiple pages per block).
_QUEST_PAGE_SIZE = 16


class QuestAttentionBackend(AttentionBackend):
    """Quest page-level sparse-attention backend (combined-slot cache)."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # "quest" kv_cache_dtype routes through LRoSAFullAttentionSpec with a
    # 4*head_size slot (see model_executor/layers/attention/attention.py).
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["quest"]
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "quest"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        # block == page; force the engine to use the Quest page size.
        return [_QUEST_PAGE_SIZE]

    @staticmethod
    def get_name() -> str:
        return "QUEST"

    @staticmethod
    def get_impl_cls() -> type["QuestImpl"]:
        return QuestImpl

    @staticmethod
    def get_builder_cls() -> type["QuestMetadataBuilder"]:
        return QuestMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Combined slot: [ K | V | K_min | K_max ].
        if block_size != _QUEST_PAGE_SIZE:
            raise ValueError(
                f"Quest: block_size must equal page_size ({_QUEST_PAGE_SIZE}); "
                f"got {block_size}."
            )
        slot_size = 4 * head_size
        return (num_blocks, block_size, num_kv_heads, slot_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [64, 128]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


@dataclass
class QuestMetadata(AttentionMetadata):
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
    # Static buffers (CG-safe), sliced to this step's decode batch. The
    # block-sparse attention reads K/V straight from the paged cache, so the
    # only buffers needed are the page scores and the selected page columns.
    scores_buf: torch.Tensor | None = None
    page_idx_buf: torch.Tensor | None = None
    # Split-KV flash-decoding scratch (only when num_splits > 1, i.e. large
    # budget); None → block-sparse uses the single-pass kernel.
    partial_acc: torch.Tensor | None = None
    partial_m: torch.Tensor | None = None
    partial_l: torch.Tensor | None = None


class QuestMetadataBuilder(AttentionMetadataBuilder[QuestMetadata]):
    # Decode-only CG capture (UNIFORM_SINGLE_TOKEN_DECODE), same engineering
    # as the LRoSA backend: all static buffers are pre-allocated and
    # mark_static_address'd so replays see stable pointers; the all-decode
    # branch always runs the sparse page path (path-stable).
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
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

        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_size = kv_cache_spec.head_size
        self.block_size = kv_cache_spec.block_size
        self.page_size = self.block_size
        self._device = device

        attn_cfg = vllm_config.attention_config
        token_budget = int(getattr(attn_cfg, "quest_token_budget", 256))
        if token_budget % self.page_size != 0:
            raise ValueError(
                f"Quest: quest_token_budget ({token_budget}) must be a "
                f"multiple of page_size ({self.page_size})."
            )
        # n_fac = buffer width per (req, kv-head). page_budget selected full
        # pages + 1 always-attended trailing page = token_budget tokens.
        self.n_fac = token_budget
        self.page_budget = token_budget // self.page_size - 1
        if self.page_budget < 1:
            raise ValueError(
                f"Quest: token_budget ({token_budget}) too small for "
                f"page_size ({self.page_size}); need >= 2 pages."
            )
        self.use_radix_topk = getattr(attn_cfg, "lrosa_use_radix_topk", True)

        H_kv = self.num_kv_heads
        pb = self.page_budget
        # Selected page columns (per kv-head). int32 for radix, int64 for the
        # torch.topk fallback. No gather buffers — block-sparse attends the
        # paged cache in place.
        self._page_idx_buf = torch.empty(
            (self.max_num_reqs, H_kv, pb), dtype=torch.int32, device=device)
        self._page_idx_buf_i64 = torch.empty(
            (self.max_num_reqs, H_kv, pb), dtype=torch.int64, device=device)

        # Lazily sized score buffer (max_pages depends on runtime block_table).
        self._scores_buf: torch.Tensor | None = None
        self._scores_max_pages = 0

        # Split-KV flash-decoding scratch for large budgets. num_splits depends
        # only on page_budget (config-fixed). For budget<=256 (num_splits==1)
        # the block-sparse single-pass kernel runs and no scratch is needed.
        self.num_splits = quest_num_splits(self.page_budget)
        self._partial_acc = None
        self._partial_m = None
        self._partial_l = None
        if self.num_splits > 1:
            H_q = vllm_config.model_config.get_num_attention_heads(
                vllm_config.parallel_config)
            ns = self.num_splits
            self._partial_acc = torch.empty(
                (self.max_num_reqs, H_q, ns, self.head_size),
                dtype=torch.float32, device=device)
            self._partial_m = torch.empty(
                (self.max_num_reqs, H_q, ns), dtype=torch.float32, device=device)
            self._partial_l = torch.empty(
                (self.max_num_reqs, H_q, ns), dtype=torch.float32, device=device)

        bufs = [self._page_idx_buf, self._page_idx_buf_i64]
        if self.num_splits > 1:
            bufs += [self._partial_acc, self._partial_m, self._partial_l]
        for buf in bufs:
            torch._dynamo.mark_static_address(buf)

    def _ensure_scores_buf(self, max_pages: int) -> torch.Tensor:
        if self._scores_buf is not None and self._scores_max_pages >= max_pages:
            return self._scores_buf
        aligned = ((max_pages + 63) // 64) * 64
        self._scores_buf = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, aligned),
            dtype=torch.float32, device=self._device)
        self._scores_max_pages = aligned
        torch._dynamo.mark_static_address(self._scores_buf)
        return self._scores_buf

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QuestMetadata:
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, _np, num_decode_tokens, _npt = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold)

        if num_decodes > 0:
            max_pages = cam.block_table_tensor.shape[1]
            full_scores = self._ensure_scores_buf(max_pages)
            scores_buf = full_scores[:num_decodes]
            page_idx_buf = (self._page_idx_buf[:num_decodes]
                            if self.use_radix_topk
                            else self._page_idx_buf_i64[:num_decodes])
            partial_acc = (self._partial_acc[:num_decodes]
                           if self._partial_acc is not None else None)
            partial_m = (self._partial_m[:num_decodes]
                         if self._partial_m is not None else None)
            partial_l = (self._partial_l[:num_decodes]
                         if self._partial_l is not None else None)
        else:
            scores_buf = None
            page_idx_buf = None
            partial_acc = None
            partial_m = None
            partial_l = None

        return QuestMetadata(
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
            scores_buf=scores_buf,
            page_idx_buf=page_idx_buf,
            partial_acc=partial_acc,
            partial_m=partial_m,
            partial_l=partial_l,
        )


class QuestImpl(AttentionImpl[QuestMetadata]):
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
        self.slot_size = 4 * head_size
        self.kv_cache_dtype = kv_cache_dtype

        vllm_config = get_current_vllm_config()
        attn_cfg = vllm_config.attention_config
        token_budget = int(getattr(attn_cfg, "quest_token_budget", 256))
        self.n_fac = token_budget
        self.use_radix_topk = getattr(attn_cfg, "lrosa_use_radix_topk", True)

        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        self.attn_type = attn_type
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Write K, V into the combined slot ([K|V|min|max]); min/max regions
        # are filled separately in forward (decode running update / prefill
        # reduction), which is where the metadata (seq_lens, block_table) lives.
        N = slot_mapping.shape[0]
        if N <= 0:
            return
        k = key[:N].view(N, self.num_kv_heads, self.head_size)
        v = value[:N].view(N, self.num_kv_heads, self.head_size)
        lrosa_store(k, v, kv_cache, slot_mapping)

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: QuestMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "Quest: fused output quantization not supported yet.")
        if attn_metadata is None:
            return output.fill_(0)

        head_size = self.head_size
        page_size = kv_cache.shape[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        key_cache = kv_cache[..., :head_size]
        value_cache = kv_cache[..., head_size : 2 * head_size]

        # Mixed / all-prefill: build prefill page min/max, then dense flash.
        if num_decodes == 0 or num_decode_tokens < num_actual_tokens:
            if num_decodes > 0:
                # Decode portion: fold the new K + run the sparse page path.
                self._sparse_decode_forward(
                    query=query, key=key, kv_cache=kv_cache,
                    attn_metadata=attn_metadata, output=output,
                    head_size=head_size, page_size=page_size,
                )

            # Build per-full-page min/max for each prefill request so its
            # later decode steps have valid metadata.
            self._build_prefill_minmax(
                key, kv_cache, attn_metadata, head_size, page_size,
                num_decodes, num_decode_tokens)

            # Dense flash over the prefill rows.
            if num_decodes == 0:
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens], k=key_cache, v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale, causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=list(self.sliding_window),
                    block_table=attn_metadata.block_table,
                    softcap=self.logits_soft_cap,
                )
            else:
                prefill_qsl = (attn_metadata.query_start_loc[num_decodes:]
                               - num_decode_tokens)
                flash_attn_varlen_func(
                    q=query[num_decode_tokens:num_actual_tokens],
                    k=key_cache, v=value_cache,
                    out=output[num_decode_tokens:num_actual_tokens],
                    cu_seqlens_q=prefill_qsl,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens[num_decodes:],
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale, causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=list(self.sliding_window),
                    block_table=attn_metadata.block_table[num_decodes:],
                    softcap=self.logits_soft_cap,
                )
            return output

        # All-decode (CG-captured): sparse page path always runs.
        self._sparse_decode_forward(
            query=query, key=key, kv_cache=kv_cache,
            attn_metadata=attn_metadata, output=output,
            head_size=head_size, page_size=page_size,
        )
        return output

    def _build_prefill_minmax(
        self, key, kv_cache, attn_metadata, head_size, page_size,
        num_decodes, num_decode_tokens,
    ) -> None:
        """Per-request prefill page min/max build (PyTorch loop; not CG /
        perf-critical). Uses CPU-side start/seq lengths to avoid GPU syncs."""
        qsl_cpu = attn_metadata.query_start_loc_cpu
        seq_cpu = attn_metadata.seq_lens_cpu
        if qsl_cpu is None or seq_cpu is None:
            return
        block_table = attn_metadata.block_table
        n_reqs = block_table.shape[0]
        for i in range(num_decodes, n_reqs):
            q0 = int(qsl_cpu[i].item())
            q1 = int(qsl_cpu[i + 1].item())
            plen = q1 - q0
            if plen <= 0:
                continue
            seq_len = int(seq_cpu[i].item())
            start_pos = seq_len - plen  # absolute pos of key[q0] (chunked prefill)
            key_i = key[q0:q1].view(plen, self.num_kv_heads, head_size)
            quest_build_page_minmax_prefill(
                key_i, kv_cache, block_table[i], page_size, head_size,
                prefill_start_pos=max(start_pos, 0),
            )

    def _sparse_decode_forward(
        self, query, key, kv_cache, attn_metadata, output, head_size, page_size,
    ) -> None:
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert num_decode_tokens == num_decodes, (
            "Quest sparse decode assumes q_len=1 per decode request")

        block_table_dec = attn_metadata.block_table[:num_decodes]
        seq_lens_dec = attn_metadata.seq_lens[:num_decodes]

        # 1. Fold the newly-decoded K into its block's running min/max.
        key_dec = key[:num_decode_tokens].view(
            num_decodes, self.num_kv_heads, head_size)
        quest_minmax_update(
            key_dec, kv_cache, block_table_dec, seq_lens_dec,
            head_size, page_size)

        # 2. GQA group-mean query → per-kv-head, then page upper-bound score.
        q_decode = query[:num_decode_tokens]  # (nd, H_q, hs)
        q_kv = q_decode.view(
            num_decodes, self.num_kv_heads, self.num_kv_groups, head_size
        ).mean(dim=2)  # (nd, H_kv, hs)
        scores = quest_page_score(
            q_kv, kv_cache, block_table_dec, seq_lens_dec,
            page_size, head_size, scores_out=attn_metadata.scores_buf)

        # 3. Page top-K. num_full = (seq_len-1)//page_size selectable pages.
        num_full = (seq_lens_dec.to(torch.int64) - 1) // page_size  # (nd,)
        page_budget = attn_metadata.page_idx_buf.shape[-1]
        if self.use_radix_topk and _radix_topk_available():
            page_idx = _radix_topk(
                scores, num_full.to(torch.int32), page_budget,
                idx_out=attn_metadata.page_idx_buf)  # (nd,H_kv,pb) int32
        else:
            page_idx = scores.topk(page_budget, dim=-1).indices.to(torch.int32)

        # 4. Block-sparse attention directly on the selected pages (NO gather):
        #    read each selected block's K/V from the paged cache + the trailing
        #    partial page, online-softmax. Per-q-head program; the kv-head's
        #    selected pages are shared across its GQA group (per-kv-head select,
        #    per-q-head attend). Pages with column >= num_full are skipped, so
        #    short context (seq_len <= n_fac) attends all real pages + trailing
        #    == dense — no separate dense path needed.
        out_view = output[:num_decode_tokens].view(
            num_decodes, self.num_heads, head_size)
        quest_blocksparse_attn(
            query=q_decode, kv_cache=kv_cache, page_idx=page_idx,
            block_table=block_table_dec, seq_lens=seq_lens_dec, output=out_view,
            scale=self.scale, page_size=page_size, head_size=head_size,
            num_kv_groups=self.num_kv_groups,
            partial_acc=attn_metadata.partial_acc,
            partial_m=attn_metadata.partial_m,
            partial_l=attn_metadata.partial_l)
