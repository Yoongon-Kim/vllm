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
    per-token slot per (token, kv-head): [ K(hs) | V(hs) ], slot_size = 2*hs.
    Per-page K_min/K_max live in a SEPARATE backend-managed buffer
    ``layer._quest_minmax`` of shape (num_blocks, H_kv, 2*head_size)
    ([:hs]=min, [hs:]=max), indexed by physical block id. This matches the
    original MIT Quest layout and cuts Quest KV memory from ~2.0x dense
    (old 4*hs slot, 15/16 of the min/max region wasted) to ~1.06x dense.

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
# vLLM Triton flash for head_dim>256 (Gemma 4 full layers) prefill — FA2 caps
# at 256; unified_attention is head-dim agnostic and O(L) memory.
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.attention.ops.triton_quest import (
    quest_blocksparse_attn,
    quest_build_page_minmax_prefill,
    quest_fi_build_indices,
    quest_minmax_update,
    quest_num_splits,
    quest_page_score,
)
from vllm.v1.kv_cache_interface import AttentionSpec

# FlashInfer vendor-kernel attend for the QUEST baseline (per-head paged decode).
# Optional: only imported/used when quest_fi_attend is on (default) and available.
try:
    import flashinfer as _flashinfer
    from vllm.v1.attention.backends.flashinfer import fast_plan_decode as _fi_fast_plan
    from vllm.utils.torch_utils import canonicalize_singleton_dim_strides as _fi_canon
    _HAS_FLASHINFER = True
except Exception:  # pragma: no cover - FlashInfer optional
    _flashinfer = None
    _HAS_FLASHINFER = False

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
        # Per-token slot: [ K | V ]. Page min/max live in a SEPARATE
        # backend-managed per-page buffer (see QuestImpl), not in the slot.
        if block_size != _QUEST_PAGE_SIZE:
            raise ValueError(
                f"Quest: block_size must equal page_size ({_QUEST_PAGE_SIZE}); "
                f"got {block_size}."
            )
        slot_size = 2 * head_size
        return (num_blocks, block_size, num_kv_heads, slot_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_sink(cls) -> bool:
        # gpt-oss sink: QUEST prefill/decode route to flash_attn_varlen_func
        # (s_aux on FA3/4, B200).
        from vllm.v1.attention.backends.fa_utils import flash_attn_supports_sinks
        return flash_attn_supports_sinks()

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 256/512: Gemma 4 (sliding head_dim=256 + full head_dim=512). The
        # head>256 prefill routes through vLLM's unified_attention (FA2 caps at
        # 256); decode uses the page path (head-dim agnostic kernels).
        return [64, 128, 256, 512]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Gemma 4 is multimodal; its full-attention layers are flagged with
        # use_mm_prefix. Quest targets TEXT decoding (the mm-prefix path reduces
        # to ordinary full attention there), so accept it. (Sparse page
        # selection over actual image tokens is out of scope.)
        return True

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
    # FlashInfer per-head decode (vendor-kernel attend, default). When set, the
    # impl builds per-head page lists into fi_indices and runs the Hk planned
    # wrappers instead of the custom quest_blocksparse_attn Triton kernel.
    fi_wrappers: list | None = None          # Hk BatchDecodeWithPagedKVCacheWrapper
    fi_indices: torch.Tensor | None = None   # (Hk, max_pages) int32 paged_kv_indices
    fi_indptr: torch.Tensor | None = None    # (num_decodes+1,) int32 page offsets


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

        # ---- FlashInfer per-head decode (vendor-kernel attend, default ON) ----
        # QUEST's page selection is per-kv-head; FlashInfer paged decode uses one
        # page list per request, so we run one decode PER kv-head (num_kv_heads=1,
        # num_qo_heads=group). plan() lives here in build() (eager, out of graph);
        # only .run() is captured. sm_scale=1.0 (the impl pre-scales q) so this
        # builder needs no per-layer scale. Gated off for sinks models in the impl.
        import os as _os
        _fi_env = _os.environ.get("QUEST_FI_ATTEND")
        _fi_flag = (bool(int(_fi_env)) if _fi_env is not None
                    else bool(getattr(attn_cfg, "quest_fi_attend", True)))
        self.quest_fi_attend = _HAS_FLASHINFER and _fi_flag
        H_q = vllm_config.model_config.get_num_attention_heads(
            vllm_config.parallel_config)
        self.num_qo_heads_per_kv = H_q // H_kv
        self._model_dtype = vllm_config.model_config.dtype
        self._fi_wrappers: dict[int, list] = {}
        if self.quest_fi_attend:
            max_pages = self.max_num_reqs * (self.page_budget + 1)
            self._fi_workspace = torch.empty(
                256 * 1024 * 1024, dtype=torch.uint8, device=device)
            self._fi_indptr = torch.zeros(
                self.max_num_reqs + 1, dtype=torch.int32, device=device)
            self._fi_indptr_cpu = torch.zeros(
                self.max_num_reqs + 1, dtype=torch.int32, pin_memory=True)
            self._fi_last = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=device)
            self._fi_last_cpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, pin_memory=True)
            self._fi_indices = torch.zeros(
                (H_kv, max_pages), dtype=torch.int32, device=device)
            for b in (self._fi_indptr, self._fi_last, self._fi_indices):
                torch._dynamo.mark_static_address(b)

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

    def _get_fi_wrappers(self, bs: int) -> list:
        """Hk FlashInfer decode wrappers for batch size ``bs`` (one per kv-head,
        num_kv_heads=1). Cached per bs; all share the workspace + indptr/last_page
        buffers, each binds its own per-head indices buffer."""
        ws = self._fi_wrappers.get(bs)
        if ws is None:
            ws = [
                _flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    self._fi_workspace, "NHD", use_cuda_graph=True,
                    paged_kv_indptr_buffer=self._fi_indptr[: bs + 1],
                    paged_kv_indices_buffer=self._fi_indices[h],
                    paged_kv_last_page_len_buffer=self._fi_last[:bs])
                for h in range(self.num_kv_heads)
            ]
            self._fi_wrappers[bs] = ws
        return ws

    def _plan_fi(self, num_decodes: int, seq_lens_cpu: torch.Tensor) -> list | None:
        """Plan the Hk decode wrappers from seq_lens (page counts are head-
        independent). Runs in build() (eager, out of graph); only .run() is
        captured. sm_scale=1.0 — the impl pre-scales q. Returns the wrappers."""
        nd = num_decodes
        sc = seq_lens_cpu[:nd].to(torch.int64)
        nfull = (sc - 1) // self.page_size
        vcount = nfull.clamp(max=self.page_budget)                # valid selected pages
        counts = (vcount + 1)                                      # + trailing page
        self._fi_indptr_cpu[0] = 0
        self._fi_indptr_cpu[1:nd + 1] = torch.cumsum(counts, 0).to(torch.int32)
        self._fi_last_cpu[:nd] = (sc - nfull * self.page_size).to(torch.int32)
        # Populate the GPU buffers explicitly: the compaction kernel (forward)
        # reads self._fi_indptr for per-request page offsets, and the wrappers'
        # last_page_len buffer is self._fi_last (fast_plan also copies these, but
        # we don't rely on its internals for the kernel-visible indptr).
        self._fi_indptr[:nd + 1].copy_(self._fi_indptr_cpu[:nd + 1], non_blocking=True)
        self._fi_last[:nd].copy_(self._fi_last_cpu[:nd], non_blocking=True)
        wrappers = self._get_fi_wrappers(nd)
        # plain plan() on GPU buffers (CUDA-cores decode; supports head 512).
        # fast_plan_decode passes fixed_split_size which the non-tensor-core path
        # rejects, so we re-plan directly each step (build() is out of graph).
        for h, w in enumerate(wrappers):
            w.plan(
                self._fi_indptr[:nd + 1], self._fi_indices[h], self._fi_last[:nd],
                self.num_qo_heads_per_kv, 1, self.head_size, self.page_size,
                q_data_type=self._model_dtype, kv_data_type=self._model_dtype,
                sm_scale=1.0)
        return wrappers

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
            fi_wrappers = None
            fi_indices = None
            fi_indptr = None
            if self.quest_fi_attend and cam.seq_lens_cpu_upper_bound is not None:
                fi_wrappers = self._plan_fi(
                    num_decodes, cam.seq_lens_cpu_upper_bound)
                fi_indices = self._fi_indices
                fi_indptr = self._fi_indptr[:num_decodes + 1]
        else:
            scores_buf = None
            page_idx_buf = None
            partial_acc = None
            partial_m = None
            partial_l = None
            fi_wrappers = None
            fi_indices = None
            fi_indptr = None

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
            fi_wrappers=fi_wrappers,
            fi_indices=fi_indices,
            fi_indptr=fi_indptr,
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
        self.slot_size = 2 * head_size
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
        # gpt-oss attention sink (same handling as LRoSA): s_aux on FA3/4 (B200).
        self.sinks = kwargs.get("sinks")
        from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
        from vllm.vllm_flash_attn.flash_attn_interface import DEFAULT_FA_VERSION
        if self.sinks is not None:
            assert self.sinks.shape[0] == num_heads, (
                f"sinks {tuple(self.sinks.shape)} vs num_heads {num_heads}")
            self._fa_version = get_flash_attn_version(has_sinks=True)
        else:
            self._fa_version = DEFAULT_FA_VERSION

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

        # Sliding-attention layer (hybrid models, e.g. Gemma 4: 25/30 layers are
        # sliding head_dim=256): Quest's page selection/attention does NOT honor
        # the sliding window, so applying it here attends tokens outside the
        # window and corrupts the output. Fall back to plain windowed flash over
        # the combined-slot K/V (head<=256 ⇒ FA2 OK) — one varlen call covers
        # prefill, mixed, and the CG-captured all-decode path. Full-attention
        # layers (window == (-1,-1)) skip this and use the page path.
        if self.sliding_window != (-1, -1):
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
                s_aux=self.sinks, fa_version=self._fa_version,
            )
            return output

        # Per-page min/max buffer (backend-managed, separate from the 2*hs
        # per-token slot). Indexed by physical block id, so it must have one row
        # per KV-cache block (kv_cache.shape[0]). Lazily (re)allocated when
        # missing or too small: the first forward may be a profiling/dummy run
        # with a tiny placeholder cache (e.g. shape[0]==max-cudagraph-size for a
        # hybrid model's full-attention layers), so sizing off that first cache
        # under-allocates and the scatter kernel writes out of bounds at real
        # inference. Growing to the real block count converges after the first
        # real-sized forward (which precedes CUDA-graph capture), then stays
        # stable; mark_static_address keeps the (final) address fixed for graph
        # replay.
        minmax = getattr(layer, "_quest_minmax", None)
        if minmax is None or minmax.shape[0] < kv_cache.shape[0]:
            minmax = torch.zeros(
                kv_cache.shape[0], self.num_kv_heads, 2 * head_size,
                dtype=kv_cache.dtype, device=kv_cache.device)
            layer._quest_minmax = minmax
            torch._dynamo.mark_static_address(minmax)

        # Mixed / all-prefill: build prefill page min/max, then dense flash.
        if num_decodes == 0 or num_decode_tokens < num_actual_tokens:
            if num_decodes > 0:
                # Decode portion: fold the new K + run the sparse page path.
                self._sparse_decode_forward(
                    query=query, key=key, kv_cache=kv_cache, minmax=minmax,
                    attn_metadata=attn_metadata, output=output,
                    head_size=head_size, page_size=page_size,
                )

            # Build per-full-page min/max for each prefill request so its
            # later decode steps have valid metadata.
            self._build_prefill_minmax(
                key, kv_cache, minmax, attn_metadata, head_size, page_size,
                num_decodes, num_decode_tokens)

            # Dense flash over the prefill rows. head_dim>256 (Gemma 4 full
            # layers) can't use FA2 → vLLM's head-agnostic Triton flash.
            if num_decodes == 0:
                q_pf = query[:num_actual_tokens]
                out_pf = output[:num_actual_tokens]
                qsl_pf = attn_metadata.query_start_loc
                seqk_pf = attn_metadata.seq_lens
                bt_pf = attn_metadata.block_table
            else:
                q_pf = query[num_decode_tokens:num_actual_tokens]
                out_pf = output[num_decode_tokens:num_actual_tokens]
                qsl_pf = (attn_metadata.query_start_loc[num_decodes:]
                          - num_decode_tokens)
                seqk_pf = attn_metadata.seq_lens[num_decodes:]
                bt_pf = attn_metadata.block_table[num_decodes:]
            if head_size > 256:
                unified_attention(
                    q=q_pf, k=key_cache, v=value_cache, out=out_pf,
                    cu_seqlens_q=qsl_pf, max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=seqk_pf, max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale, causal=True,
                    window_size=self.sliding_window, block_table=bt_pf,
                    softcap=self.logits_soft_cap,
                    q_descale=None, k_descale=None, v_descale=None,
                    alibi_slopes=self.alibi_slopes,
                )
            else:
                flash_attn_varlen_func(
                    q=q_pf, k=key_cache, v=value_cache, out=out_pf,
                    cu_seqlens_q=qsl_pf, max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=seqk_pf, max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale, causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=list(self.sliding_window),
                    block_table=bt_pf,
                    softcap=self.logits_soft_cap,
                    s_aux=self.sinks, fa_version=self._fa_version,
                )
            return output

        # All-decode (CG-captured): sparse page path always runs.
        self._sparse_decode_forward(
            query=query, key=key, kv_cache=kv_cache, minmax=minmax,
            attn_metadata=attn_metadata, output=output,
            head_size=head_size, page_size=page_size,
        )
        return output

    def _build_prefill_minmax(
        self, key, kv_cache, minmax, attn_metadata, head_size, page_size,
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
                key_i, kv_cache, minmax, block_table[i], page_size, head_size,
                prefill_start_pos=max(start_pos, 0),
            )

    def _sparse_decode_forward(
        self, query, key, kv_cache, minmax, attn_metadata, output, head_size,
        page_size,
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
            key_dec, minmax, block_table_dec, seq_lens_dec,
            head_size, page_size)

        # 2. GQA group-mean query → per-kv-head, then page upper-bound score.
        # NOTE: sliding-window layers never reach here — forward() routes them
        # to dense windowed flash before any _sparse_decode_forward call, so
        # self.sliding_window == (-1, -1) is guaranteed at this point.
        q_decode = query[:num_decode_tokens]  # (nd, H_q, hs)
        q_kv = q_decode.view(
            num_decodes, self.num_kv_heads, self.num_kv_groups, head_size
        ).mean(dim=2)  # (nd, H_kv, hs)
        scores = quest_page_score(
            q_kv, minmax, block_table_dec, seq_lens_dec,
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

        # 4. Attend the selected pages + trailing page (NO gather). Two paths,
        #    identical selection/semantics (pages with column >= num_full skipped,
        #    so short seqs attend all real pages + trailing == dense):
        #    (a) DEFAULT: per-kv-head FlashInfer paged decode (vendor kernel) — a
        #        fair, GQA-group-optimised baseline so QUEST isn't penalised by an
        #        un-GQA-fused custom kernel. Requires no attention sink.
        #    (b) FALLBACK: the custom quest_blocksparse_attn Triton kernel (used
        #        for sinks models / when FlashInfer is disabled).
        out_view = output[:num_decode_tokens].view(
            num_decodes, self.num_heads, head_size)
        if attn_metadata.fi_wrappers is not None and self.sinks is None:
            self._fi_attend(q_decode, kv_cache, page_idx, block_table_dec,
                            seq_lens_dec, attn_metadata, head_size, page_size,
                            out_view)
        else:
            quest_blocksparse_attn(
                query=q_decode, kv_cache=kv_cache, page_idx=page_idx,
                block_table=block_table_dec, seq_lens=seq_lens_dec, output=out_view,
                scale=self.scale, page_size=page_size, head_size=head_size,
                num_kv_groups=self.num_kv_groups,
                partial_acc=attn_metadata.partial_acc,
                partial_m=attn_metadata.partial_m,
                partial_l=attn_metadata.partial_l,
                sinks=self.sinks)

    def _fi_attend(self, q_decode, kv_cache, page_idx, block_table_dec,
                   seq_lens_dec, attn_metadata, head_size, page_size,
                   out_view) -> None:
        """Per-kv-head FlashInfer paged decode over the selected pages + trailing.
        Builds each head's page list (compaction kernel) into the static indices
        buffer, then runs the Hk planned wrappers. q is pre-scaled by self.scale
        (wrappers planned with sm_scale=1.0). num_kv_heads=1 per call; the combined
        [K|V] slot is passed as a single 5D NHD view [nb, 2, page, 1, hs] with
        canonicalised singleton strides (FlashInfer misreads the n_kv=1 stride
        otherwise)."""
        Hk = self.num_kv_heads
        G = self.num_kv_groups
        nb = kv_cache.shape[0]
        # in-graph: compact valid selected pages + trailing into fi_indices
        quest_fi_build_indices(
            attn_metadata.fi_indices, page_idx, block_table_dec, seq_lens_dec,
            attn_metadata.fi_indptr, page_size)
        q_scaled = q_decode * self.scale                       # sm_scale=1.0 in plan
        for h in range(Hk):
            kv_view = (kv_cache[:, :, h, :]                     # [nb, page, 2*hs]
                       .view(nb, page_size, 2, head_size)       # [nb, page, 2, hs]
                       .permute(0, 2, 1, 3)                     # [nb, 2, page, hs]
                       .unsqueeze(3))                           # [nb, 2, page, 1, hs]
            kv_view = _fi_canon(kv_view)
            qh = q_scaled[:, h * G:(h + 1) * G, :].contiguous()  # [nd, G, hs]
            oh = attn_metadata.fi_wrappers[h].run(qh, kv_view)
            out_view[:, h * G:(h + 1) * G, :] = oh
