# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LRoSA (Learned Rotation Sparse Attention) backend.

Step 1: capability declarations + flash_attn fallback (no sparse logic yet).
        Standard split KV cache layout (2, num_blocks, block_size, H_kv, d).
Step 2: combined-slot KV cache (K + V + proj_K), proj_K store kernel.
Step 3: sparse decode forward (score + topk + gather + SDPA).
Step 4b: CUDA-Graph-friendly decode path
         - ``_cudagraph_support = UNIFORM_SINGLE_TOKEN_DECODE``
         - score / top-K / cu_seqlens buffers pre-allocated in builder
           with ``mark_static_address`` so the CG path doesn't see a
           mutating data pointer between replays.
         - the "all decode" branch always runs the sparse kernels (no
           runtime ``sparse_active`` dispatch) so the captured kernel
           sequence is path-stable. Dense fallback only applies to mixed-
           or all-prefill batches, neither of which is captured under
           ``UNIFORM_SINGLE_TOKEN_DECODE``.
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
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.triton_lrosa_gather import lrosa_gather
from vllm.v1.attention.ops.triton_lrosa_score_topk import lrosa_score_topk
from vllm.v1.attention.ops.triton_fasa_score_topk import fasa_score_topk
from vllm.v1.attention.ops.triton_lrosa_store import (
    lrosa_project_and_store,
    lrosa_project_and_store_layer,
    lrosa_store,
)
from vllm.v1.attention.ops.triton_lrosa_streaming_topk import (
    alloc_candidates_buf,
    lrosa_streaming_topk,
)
from vllm.v1.kv_cache_interface import AttentionSpec

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()


def _cs_h_for(head_size: int) -> int:
    """LRoSA convention: cs_h = head_size // 4 (paper default for d=128)."""
    return head_size // 4


def _resolve_slot_cs_h(head_size: int) -> int:
    """Width of the proj_K region in the combined slot for this run.

    - FASA-fc (``kv_cache_dtype == "fasa"``): 0 — the slot is just [K|V]; the
      paper-faithful path re-reads the I_dom channels from full K each step,
      so nothing extra is cached.
    - LRoSA: the ``lrosa_cs_h`` override (e.g. Gemma 4 cs_h=64) or head_size//4.
    """
    cfg = get_current_vllm_config()
    if cfg.cache_config.cache_dtype == "fasa":
        return 0
    cs_override = getattr(cfg.attention_config, "lrosa_cs_h", None)
    return cs_override if cs_override else _cs_h_for(head_size)


class LRoSAAttentionBackend(AttentionBackend):
    """LRoSA backend.

    Step 1 status: standard split KV cache; forward delegates to
    FlashAttention varlen. No sparse selection yet. proj_K cache will
    be added in Step 2 (combined-slot layout).
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # LRoSA owns its combined-slot cache layout. The "lrosa" tag routes through
    # TQFullAttentionSpec (see attention.py) so memory budgeting uses our
    # slot_size = 2*head_size + cs_h rather than the standard 2*head_size.
    # "fasa" is the paper-faithful FASA-fc mode (no proj_K cache; reads the
    # I_dom channels from full K each step). Shares this backend's gather /
    # attend / metadata infra; only the slot width (no proj_K) and the score
    # kernel differ.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["lrosa", "fasa"]
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype in ("lrosa", "fasa")

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "LROSA"

    @staticmethod
    def get_impl_cls() -> type["LRoSAImpl"]:
        return LRoSAImpl

    @staticmethod
    def get_builder_cls() -> type["LRoSAMetadataBuilder"]:
        return LRoSAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # Combined slot: [ K (head_size) | V (head_size) | proj_K (cs_h) ]
        if block_size % 16 != 0:
            raise ValueError("LRoSA: block_size must be a multiple of 16.")
        # Honor fasa (cs_h=0, [K|V] slot) and the lrosa_cs_h override (e.g.
        # Gemma 4 cs_h=64) so the allocated slot matches get_kv_cache_spec and
        # the Impl. Default = head_size//4.
        slot_size = 2 * head_size + _resolve_slot_cs_h(head_size)
        return (num_blocks, block_size, num_kv_heads, slot_size)

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 64/128: GPT-OSS / Llama-Qwen-Mistral family. 256/512: Gemma 4 26B-A4B
        # (sliding layers head_dim=256, full-attention layers head_dim=512).
        return [64, 128, 256, 512]

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Gemma 4 is multimodal; vLLM flags its full-attention layers with
        # use_mm_prefix. LRoSA targets TEXT decoding (no image tokens in the
        # reasoning/long-context evals), where the mm-prefix path reduces to
        # ordinary combined-slot full attention — so accept it. (Sparse
        # selection over actual image tokens is out of scope.)
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability >= DeviceCapability(8, 0)


@dataclass
class LRoSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    # Decode / prefill split. With reorder_batch_threshold=1, the scheduler
    # places decode requests at the head of the batch; these fields give the
    # boundary so the sparse decode path (added in 3b-4) can operate on
    # `[:num_decodes]` requests / `[:num_decode_tokens]` tokens without
    # disturbing prefill rows.
    num_decodes: int = 0
    num_decode_tokens: int = 0
    # CPU-resident copies surface here so the forward path can derive per-
    # request lengths without forcing GPU→CPU syncs.
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    causal: bool = True
    # Step 4b: views into the builder's static buffers, sized to this step's
    # decode batch. ``None`` when ``num_decodes == 0`` (all-prefill batch);
    # the sparse decode path is then skipped.
    scores_buf: torch.Tensor | None = None
    top_idx_buf: torch.Tensor | None = None
    top_scores_buf: torch.Tensor | None = None
    cu_seqlens_q_dec: torch.Tensor | None = None
    cu_seqlens_k_dec: torch.Tensor | None = None
    # Step 4a: when True, the impl picks the streaming Triton kernel
    # (V2 chunk-parallel two-stage) over the two-pass score + topk.
    use_streaming_topk: bool = False
    candidates_buf: torch.Tensor | None = None
    chunk_size: int = 0
    # Pre-allocated gather output buffers (Step 4b CG fix). Live for the
    # life of the builder, ``mark_static_address``'d so the FULL CG records
    # stable pointers. See the builder docstring for why we can't reuse the
    # shared ``WorkspaceManager`` here.
    K_sel_buf: torch.Tensor | None = None
    V_sel_buf: torch.Tensor | None = None


class LRoSAMetadataBuilder(AttentionMetadataBuilder[LRoSAMetadata]):
    # Step 4b: decode-only CG capture is supported. Mixed-batch (prefill +
    # decode) is not — the forward dispatch still has a Python-side branch
    # that the captured graph wouldn't tolerate. ``UNIFORM_SINGLE_TOKEN_DECODE``
    # gates capture to all-q_len==1 batches, which is exactly the path that's
    # been engineered for stability here.
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    @classmethod
    def get_cudagraph_support(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,
    ) -> AttentionCGSupport:
        """Both V2 streaming and the 2-pass path support FULL CG.

        Historical context (2026-05-23): an earlier version returned
        ``NEVER`` for V2 streaming because of a ``cudaErrorIllegalAddress``
        at the first ``run_fullgraph(...).replay()``. Root cause was that
        ``lrosa_gather`` allocated K_sel / V_sel via the shared
        ``WorkspaceManager`` (``vllm/v1/worker/workspace.py``), and the
        v2 model-runner's post-capture ``warmup_kernels`` ran a decode
        with ``num_reqs ≈ max_num_seqs`` (≈1024) that grew the workspace
        from the small per-capture size to ~1 GiB at a new base
        address — the captured FULL graphs still held the old base in
        their kernel-arg pointers, so replay dereferenced freed memory.
        The two-pass path tripped the same bug under ``[64, 256]`` but
        survived the default 51-size capture by allocator coincidence
        rather than design.

        Fix: ``__init__`` pre-allocates ``_K_sel_buf`` / ``_V_sel_buf``
        for the worst-case decode batch and ``mark_static_address``'s
        them; ``lrosa_gather`` slices those buffers via the new
        ``K_sel_out`` / ``V_sel_out`` args and never touches the shared
        ``WorkspaceManager`` on the LRoSA path.
        """
        return cls._cudagraph_support

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        # Threshold=1 → scheduler reorders so all q_len==1 requests come first.
        # Mirrors TurboQuantMetadataBuilder.
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)

        # Static buffers for the sparse decode path. The scores buffer is
        # sized lazily on the first ``build()`` call because the runtime
        # block_table column count (= max_kv_len / block_size) depends on
        # vLLM's allocator alignment, which isn't visible here at __init__.
        # cu_seqlens are size-independent and allocated up front.
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.num_kv_heads = kv_cache_spec.num_kv_heads
        self.head_size = kv_cache_spec.head_size
        self.cs_h = _cs_h_for(self.head_size)
        self.block_size = kv_cache_spec.block_size
        self._device = device

        attn_cfg = vllm_config.attention_config
        self.n_fac = attn_cfg.lrosa_n_fac
        self.use_streaming_topk = attn_cfg.lrosa_use_streaming_topk
        self.use_radix_topk = getattr(attn_cfg, "lrosa_use_radix_topk", True)
        max_model_len = vllm_config.model_config.max_model_len

        # ``torch.topk(out=)`` requires int64 indices for the 2-pass path; the
        # streaming kernel writes int32. We pre-allocate one buffer per dtype
        # and the impl picks at forward time.
        self._top_idx_buf = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, self.n_fac),
            dtype=torch.int64,
            device=device,
        )
        self._top_idx_buf_i32 = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, self.n_fac),
            dtype=torch.int32,
            device=device,
        )
        self._top_scores_buf = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, self.n_fac),
            dtype=torch.float32,
            device=device,
        )
        # cu_seqlens_q for decode is the fixed sequence [0, 1, 2, ..., N];
        # the matching cu_seqlens_k for the gathered K_sel/V_sel layout (n_fac
        # rows per request) is just that arange scaled by n_fac.
        cu_q = torch.arange(self.max_num_reqs + 1, dtype=torch.int32, device=device)
        self._cu_seqlens_q_dec = cu_q
        self._cu_seqlens_k_dec = (cu_q * self.n_fac).to(torch.int32)

        # Lazily filled on the first build() call (see _ensure_scores_buf).
        self._scores_buf: torch.Tensor | None = None
        self._scores_max_kv_len = 0

        # Pre-allocated gather output buffers (K_sel, V_sel).
        #
        # Why we can't use the shared ``WorkspaceManager`` here:
        # ``vllm/v1/worker/gpu_worker.py`` runs ``warmup_kernels`` AFTER
        # ``capture_model``, and that post-capture warmup executes a full
        # decode forward with ``num_reqs ≈ scheduler.max_num_seqs`` (≈1024).
        # That call grows ``WorkspaceManager._current_workspaces`` from the
        # small per-CG size we captured with (e.g. 12 reqs ⇒ ~12 MiB) to
        # ~1 GiB at a *new* base address — the v2 gpu/ runner has no
        # ``lock_workspace()`` call before warmup. The previously captured
        # FULL graphs still hold the *old* base address inside the gather
        # kernel's params, so the first ``run_fullgraph(...).replay()``
        # dereferences freed memory and the device fires
        # ``cudaErrorIllegalAddress``.
        #
        # Fix: bypass ``WorkspaceManager`` entirely for LRoSA gather. We
        # allocate buffers sized for the engine's worst-case decode batch
        # at builder init and ``mark_static_address`` them; ``lrosa_gather``
        # slices the views to the current step's ``num_decodes * n_fac``.
        # Worst case is ``max_num_reqs * n_fac * H_kv * head_size`` per
        # K/V — ≈1 GiB total for Qwen3-8B at max_num_seqs=1024, n_fac=256,
        # which is what ``warmup_kernels`` would have grown the workspace
        # to anyway (and the warmup happens regardless of which backend is
        # in use), so net memory is unchanged.
        kv_dtype = vllm_config.model_config.dtype
        sel_shape = (
            self.max_num_reqs * self.n_fac,
            self.num_kv_heads,
            self.head_size,
        )
        self._K_sel_buf = torch.empty(sel_shape, dtype=kv_dtype, device=device)
        self._V_sel_buf = torch.empty(sel_shape, dtype=kv_dtype, device=device)

        # V2 streaming candidates buffer. Sized to cover the engine's
        # max_model_len in power-of-2 chunks. Only allocated when streaming
        # is enabled; mark_static_address makes it CG-safe.
        #
        # Stage 2's ``tl.topk`` runs over ``MERGE_SIZE = max_num_chunks *
        # n_fac`` packed uint64 entries, all kept live in shared memory at
        # once. At MERGE_SIZE=16384 (8B × 16384 = 128 KB) the kernel exceeds
        # the H100/Blackwell SM smem limit (~99 KB) and fails to launch
        # with ``triton.runtime.errors.OutOfResources: shared memory``.
        # Budget MERGE_SIZE ≤ 8192 (64 KB) to leave headroom for sort
        # scratch, matching pca's reference defaults. To stay inside the
        # budget at long context, auto-scale ``chunk_size`` (4096 → 8192 →
        # 16384 → ...) until ``next_pow2(ceil(max_model_len / chunk_size))
        # * n_fac ≤ 8192``.
        self._candidates_buf: torch.Tensor | None = None
        MAX_MERGE_SIZE = 8192
        if self.use_streaming_topk:
            chunk_size = 4096
            while True:
                chunks_needed = (max_model_len + chunk_size - 1) // chunk_size
                chunks_pow2 = 1
                while chunks_pow2 < chunks_needed:
                    chunks_pow2 *= 2
                if chunks_pow2 * self.n_fac <= MAX_MERGE_SIZE:
                    break
                chunk_size *= 2
            self.streaming_chunk_size = chunk_size
            self.streaming_max_num_chunks = chunks_pow2
            self._candidates_buf = alloc_candidates_buf(
                self.max_num_reqs,
                self.num_kv_heads,
                self.streaming_max_num_chunks,
                self.n_fac,
                device,
            )
            torch._dynamo.mark_static_address(self._candidates_buf)
        else:
            self.streaming_chunk_size = 4096
            self.streaming_max_num_chunks = 0

        for buf in (
            self._top_idx_buf,
            self._top_idx_buf_i32,
            self._top_scores_buf,
            self._cu_seqlens_q_dec,
            self._cu_seqlens_k_dec,
            self._K_sel_buf,
            self._V_sel_buf,
        ):
            # Tell inductor's CG path the address is stable so it doesn't
            # log "skipping cudagraphs because of mutated inputs" and bail
            # to eager on every replay.
            torch._dynamo.mark_static_address(buf)

    def _ensure_scores_buf(self, max_kv_len: int) -> torch.Tensor:
        """Allocate (or grow) the static scores buffer. Called from build()
        on the first decode batch we see — runs before CG capture so
        reallocation is safe; after capture the buffer is locked at its
        peak size and never moves."""
        if self._scores_buf is not None and self._scores_max_kv_len >= max_kv_len:
            return self._scores_buf
        # Round up to alignment of 128 tokens so subsequent steps with
        # slightly larger sequences don't trigger a reallocation that would
        # invalidate the captured CG address.
        aligned = ((max_kv_len + 127) // 128) * 128
        self._scores_buf = torch.empty(
            (self.max_num_reqs, self.num_kv_heads, aligned),
            dtype=torch.float32,
            device=self._device,
        )
        self._scores_max_kv_len = aligned
        torch._dynamo.mark_static_address(self._scores_buf)
        return self._scores_buf

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LRoSAMetadata:
        cam = common_attn_metadata
        assert self.reorder_batch_threshold is not None
        num_decodes, _num_prefills, num_decode_tokens, _num_prefill_tokens = (
            split_decodes_and_prefills(
                cam, decode_threshold=self.reorder_batch_threshold
            )
        )

        # Slice the static buffers to this step's decode batch. Slicing
        # produces views with the same base data pointer as the parent
        # buffer, so addresses stay stable across replays.
        if num_decodes > 0:
            # block_table is (num_reqs, max_blocks_runtime). Ensure the
            # scores buffer is large enough; first call (warmup) allocates.
            runtime_max_kv_len = cam.block_table_tensor.shape[1] * self.block_size
            # The streaming kernel walks proj_K directly and never touches the
            # scores buffer; skip the allocation in that mode.
            if self.use_streaming_topk:
                scores_buf = None
                top_idx_buf = self._top_idx_buf_i32[:num_decodes]
            elif self.use_radix_topk:
                # Radix path still needs the scores buffer (2-pass score
                # kernel), but writes int32 indices.
                full_scores = self._ensure_scores_buf(runtime_max_kv_len)
                scores_buf = full_scores[:num_decodes]
                top_idx_buf = self._top_idx_buf_i32[:num_decodes]
            else:
                full_scores = self._ensure_scores_buf(runtime_max_kv_len)
                scores_buf = full_scores[:num_decodes]
                top_idx_buf = self._top_idx_buf[:num_decodes]
            candidates_buf = (
                self._candidates_buf[:num_decodes]
                if self.use_streaming_topk and self._candidates_buf is not None
                else None
            )
            top_scores_buf = self._top_scores_buf[:num_decodes]
            cu_seqlens_q_dec = self._cu_seqlens_q_dec[: num_decodes + 1]
            cu_seqlens_k_dec = self._cu_seqlens_k_dec[: num_decodes + 1]
            # Pass the full pre-allocated K_sel/V_sel buffers to the impl;
            # ``lrosa_gather`` slices internally to this step's
            # ``num_decodes * n_fac`` rows.
            K_sel_buf = self._K_sel_buf
            V_sel_buf = self._V_sel_buf
        else:
            scores_buf = None
            top_idx_buf = None
            top_scores_buf = None
            cu_seqlens_q_dec = None
            cu_seqlens_k_dec = None
            candidates_buf = None
            K_sel_buf = None
            V_sel_buf = None

        return LRoSAMetadata(
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
            top_idx_buf=top_idx_buf,
            top_scores_buf=top_scores_buf,
            cu_seqlens_q_dec=cu_seqlens_q_dec,
            cu_seqlens_k_dec=cu_seqlens_k_dec,
            use_streaming_topk=self.use_streaming_topk,
            candidates_buf=candidates_buf,
            chunk_size=self.streaming_chunk_size,
            K_sel_buf=K_sel_buf,
            V_sel_buf=V_sel_buf,
        )


class LRoSAImpl(AttentionImpl[LRoSAMetadata]):
    """LRoSA per-layer attention implementation.

    All-decode batches always run the sparse path (score + top-K + gather +
    flash_attn varlen on selected K/V). The dense fallback only handles
    mixed / all-prefill batches; under ``UNIFORM_SINGLE_TOKEN_DECODE`` CG
    capture, the captured graph is decode-only and never sees that branch.
    """

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

        # LRoSA calibration (lazy on-device load via _ensure_on_device).
        vllm_config = get_current_vllm_config()
        attn_cfg = vllm_config.attention_config
        self._lrosa_basis_path = attn_cfg.lrosa_basis_path
        self.n_fac = attn_cfg.lrosa_n_fac
        # FASA-fc mode (kv_cache_dtype="fasa"): [K|V] slot, score by reading the
        # I_dom channels from full K each step (no proj_K). n_tip = #FC pairs.
        self.is_fasa = kv_cache_dtype == "fasa"
        self.n_tip = getattr(attn_cfg, "lrosa_n_tip", 16)
        # cs_h: 0 for fasa ([K|V] only); else the lrosa_cs_h override (e.g.
        # Gemma 4 cs_h=64) or head_size//4. Matches get_kv_cache_spec/_shape.
        self.cs_h = _resolve_slot_cs_h(head_size)
        self.slot_size = 2 * head_size + self.cs_h
        self.use_radix_topk = getattr(attn_cfg, "lrosa_use_radix_topk", True)
        # Per-layer CONCAT: single basis over concatenated per-head K. cs_h
        # (the per-slot proj_K width) stays head_size//4, and cs_h_layer =
        # cs_h * H_kv is spread across the H_kv slots — so slot_size is
        # unchanged and the same combined-slot cache is reused.
        self.per_layer_concat = getattr(attn_cfg, "lrosa_per_layer_concat", False)
        self.cs_h_slot = self.cs_h
        self.cs_h_layer = self.cs_h_slot * self.num_kv_heads
        self._M_dict_cpu: dict[int, torch.Tensor] | None = None  # lazy load
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

    def _ensure_on_device(
        self, layer: torch.nn.Module, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Lazy-load M for this layer.

        Returns the layer-local M tensor [H_kv, cs_h, head_size] on `device`
        with `dtype`. Cached on the layer object as `layer._lrosa_M` so
        subsequent calls are a single attribute access.
        """
        cached = getattr(layer, "_lrosa_M", None)
        if cached is not None:
            return cached
        # Resolved as a dense (no-basis) layer on a previous call — e.g. Gemma 4
        # sliding-attention layers, which LRoSA does not select over (the basis
        # is calibrated on full-attention layers only). Return None so forward /
        # do_kv_cache_update take the dense path.
        if getattr(layer, "_lrosa_dense", False):
            return None

        if self._lrosa_basis_path is None:
            raise RuntimeError(
                "LRoSA backend requires `-ac.lrosa_basis_path=<path>`; the "
                "calibration .pt file ({'M': {layer_idx: tensor}, ...}) is "
                "not optional."
            )

        if self._M_dict_cpu is None:
            ckpt = torch.load(
                self._lrosa_basis_path,
                map_location="cpu",
                weights_only=False,
            )
            # FASA reads the I_dom FC-index dict (fasa_idom_*.pt key 'idom');
            # LRoSA reads the rotation-matrix dict (key 'M').
            key = "idom" if self.is_fasa else "M"
            self._M_dict_cpu = (
                ckpt[key] if isinstance(ckpt, dict) and key in ckpt else ckpt
            )
            if not isinstance(self._M_dict_cpu, dict):
                raise RuntimeError(
                    f"LRoSA basis at {self._lrosa_basis_path} did not "
                    f"contain a {{layer_idx: tensor}} dict at key '{key}'."
                )

        from vllm.model_executor.models.utils import extract_layer_index

        layer_idx = extract_layer_index(layer.layer_name)
        layer._lrosa_layer_idx = layer_idx
        M_cpu = self._M_dict_cpu.get(layer_idx)
        if M_cpu is None:
            # No basis for this layer → dense-attention layer. For hybrid models
            # (Gemma 4: 25 sliding head_dim=256 + 5 full head_dim=512) the basis
            # holds only the full layers; sliding layers run dense (no selection)
            # with their own sliding window. Mark and return None.
            layer._lrosa_dense = True
            return None

        if self.is_fasa:
            # idom_layer: [H_kv, n_tip_max] FC (RoPE-pair) indices. Take the top
            # n_tip and expand each pair fc -> raw channels {2*fc, 2*fc+1}, so
            # the score reads 2*n_tip channels from full K. Cached as the
            # layer's "M" (an int32 [H_kv, 2*n_tip] channel-offset tensor).
            fcs = M_cpu[:, : self.n_tip].to(torch.int64)  # [H_kv, n_tip]
            ch = (
                torch.stack([fcs * 2, fcs * 2 + 1], dim=-1)
                .reshape(self.num_kv_heads, 2 * self.n_tip)
                .to(device=device, dtype=torch.int32)
                .contiguous()
            )
            torch._dynamo.mark_static_address(ch)
            layer._lrosa_M = ch
            return ch

        if self.per_layer_concat:
            # Per-layer basis: [cs_h_layer, H_kv*head_size] (or
            # [1, cs_h_layer, H_kv*head_size] from pca's per-layer dict).
            if M_cpu.dim() == 3 and M_cpu.shape[0] == 1:
                M_cpu = M_cpu[0]
            expected_shape = (self.cs_h_layer, self.num_kv_heads * self.head_size)
            if tuple(M_cpu.shape) != expected_shape:
                raise RuntimeError(
                    f"LRoSA per-layer basis M[{layer_idx}] shape "
                    f"{tuple(M_cpu.shape)} != expected {expected_shape} "
                    f"(cs_h_layer={self.cs_h_layer}, "
                    f"H_kv*d={self.num_kv_heads * self.head_size})."
                )
        else:
            expected_shape = (self.num_kv_heads, self.cs_h, self.head_size)
            if tuple(M_cpu.shape) != expected_shape:
                raise RuntimeError(
                    f"LRoSA basis M[{layer_idx}] shape {tuple(M_cpu.shape)} != "
                    f"expected {expected_shape} (H_kv={self.num_kv_heads}, "
                    f"cs_h={self.cs_h}, d={self.head_size})."
                )

        M = M_cpu.to(device=device, dtype=dtype).contiguous()
        # Calibration M is layer-static, so its address never moves once
        # loaded. Marking it lets inductor keep `M @ K` inside the captured
        # decode graph.
        torch._dynamo.mark_static_address(M)
        layer._lrosa_M = M
        return M

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: LRoSAMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "LRoSA: fused output quantization not supported yet."
            )
        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        head_size = self.head_size
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        # Combined slot layout: kv_cache shape
        # (num_blocks, block_size, H_kv, slot_size).
        # K occupies slot[:head_size], V occupies slot[head_size:2*head_size].
        # The slices are non-contiguous views (last-dim stride 1, head-dim stride
        # slot_size) — flash_attn_varlen_func is stride-aware so this is OK.
        key_cache = kv_cache[..., :head_size]
        value_cache = kv_cache[..., head_size : 2 * head_size]

        # Dense (no-basis) layer — e.g. Gemma 4 sliding-attention layers: no
        # LRoSA selection, just paged flash-attention over the combined-slot
        # K/V with this layer's sliding window. One varlen call covers prefill,
        # mixed, and the CG-captured all-decode path uniformly.
        M = self._ensure_on_device(layer, query.device, query.dtype)
        if M is None:
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=attn_metadata.query_start_loc,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens,
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=list(self.sliding_window),
                block_table=attn_metadata.block_table,
                softcap=self.logits_soft_cap,
            )
            return output

        # Mixed-batch or all-prefill: dense flash_attn over the full batch.
        # Under ``UNIFORM_SINGLE_TOKEN_DECODE`` CG capture this branch is
        # never reached (capture runs all-decode batches), so the Python
        # branch is safe.
        if num_decodes == 0 or num_decode_tokens < num_actual_tokens:
            if num_decodes == 0:
                if head_size > 256:
                    # head_dim>256 (Gemma 4 full layers): FA2 caps at 256 and
                    # the Triton context kernel blows shared memory at
                    # BLOCK_DMODEL=512. Reasoning prefills are short, so a
                    # per-request manual causal GQA attention over the
                    # freshly-passed contiguous K/V is cheap and head-dim
                    # agnostic. (Only the 5 full-attention layers hit this;
                    # sliding layers use the dense FA2 path at head_dim=256.)
                    self._manual_prefill(
                        query, key, value, output, attn_metadata, head_size
                    )
                    return output
                flash_attn_varlen_func(
                    q=query[:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_actual_tokens],
                    cu_seqlens_q=attn_metadata.query_start_loc,
                    max_seqlen_q=attn_metadata.max_query_len,
                    seqused_k=attn_metadata.seq_lens,
                    max_seqlen_k=attn_metadata.max_seq_len,
                    softmax_scale=self.scale,
                    causal=attn_metadata.causal,
                    alibi_slopes=self.alibi_slopes,
                    window_size=list(self.sliding_window),
                    block_table=attn_metadata.block_table,
                    softcap=self.logits_soft_cap,
                )
                return output

            # Mixed: sparse decode for the first `num_decodes` requests,
            # dense flash_attn for the trailing prefill rows.
            M = self._ensure_on_device(layer, query.device, query.dtype)
            self._sparse_decode_forward(
                query=query,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                M=M,
                head_size=head_size,
            )

            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            flash_attn_varlen_func(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_qsl,
                max_seqlen_q=attn_metadata.max_query_len,
                seqused_k=attn_metadata.seq_lens[num_decodes:],
                max_seqlen_k=attn_metadata.max_seq_len,
                softmax_scale=self.scale,
                causal=attn_metadata.causal,
                alibi_slopes=self.alibi_slopes,
                window_size=list(self.sliding_window),
                block_table=attn_metadata.block_table[num_decodes:],
                softcap=self.logits_soft_cap,
            )
            return output

        # All-decode (the CG-captured path). Sparse selection always runs;
        # for short sequences the score kernel masks invalid positions with
        # -inf and torch.topk falls back to whatever fits, which is harmless
        # because (a) deployment-relevant kv_len ≫ n_fac, and (b) the
        # dummy-capture seq_len=1 case never inspects the resulting tokens.
        M = self._ensure_on_device(layer, query.device, query.dtype)
        self._sparse_decode_forward(
            query=query,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            M=M,
            head_size=head_size,
        )
        return output

    def _sparse_decode_forward(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: LRoSAMetadata,
        output: torch.Tensor,
        M: torch.Tensor,
        head_size: int,
    ) -> None:
        """LRoSA sparse decode: GQA-reduce Q → proj_q → score+topk → gather → SDPA."""
        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens
        # supports_spec_as_decode=False guarantees this equality.
        assert num_decode_tokens == num_decodes, (
            "LRoSA sparse decode assumes q_len=1 per decode request "
            f"(num_decode_tokens={num_decode_tokens}, num_decodes={num_decodes})"
        )

        q_decode = query[:num_decode_tokens]  # (num_decodes, H_q, head_size)
        # GQA group-mean: H_q → H_kv. View groups Q heads by their parent kv-head.
        q_kv = q_decode.view(
            num_decodes, self.num_kv_heads, self.num_kv_groups, head_size
        ).mean(dim=2)  # (num_decodes, H_kv, head_size)

        block_table_dec = attn_metadata.block_table[:num_decodes]
        seq_lens_dec = attn_metadata.seq_lens[:num_decodes]

        # Sliding-window layers (hybrid models, e.g. Ministral): restrict the
        # LRoSA candidate set to the last ``window`` tokens before top-K, so a
        # decode at a position past the window matches the model's trained
        # sliding attention. self.sliding_window is (window-1, 0) for sliding
        # layers and (-1, -1) for full attention. window=0 → no restriction.
        # When window <= n_fac the masked region has <= n_fac valid tokens, so
        # the top-K selects them all → exact dense-windowed attention.
        win_left = self.sliding_window[0]
        window = (win_left + 1) if win_left >= 0 else 0

        if self.per_layer_concat:
            # Per-layer CONCAT: one projection over concatenated per-head Q,
            # one score row per request, one shared top-K broadcast to all
            # kv-heads. M is [cs_h_layer, H_kv*head_size].
            from vllm.v1.attention.ops.triton_lrosa_score_topk import (
                lrosa_score_layer, _radix_topk_available,
            )
            q_concat = q_kv.reshape(num_decodes, self.num_kv_heads * head_size)
            proj_q_layer = (q_concat.to(torch.float32) @ M.to(torch.float32).T)
            scores = lrosa_score_layer(
                proj_q_layer, kv_cache, block_table_dec, seq_lens_dec,
                head_size=head_size, cs_h_slot=self.cs_h_slot,
                H_kv=self.num_kv_heads,
            )  # (num_decodes, max_kv_len) fp32
            k = min(self.n_fac, scores.shape[-1])
            if _radix_topk_available():
                # reuse radix via (num_reqs, 1, max_kv) shaping
                from vllm.v1.attention.ops.triton_lrosa_score_topk import _radix_topk
                ti = _radix_topk(scores.unsqueeze(1), seq_lens_dec, k)  # (nd,1,k) i32
            else:
                ti = scores.topk(k, dim=-1).indices.to(torch.int32).unsqueeze(1)
            # broadcast the single selection to all kv-heads for the gather
            top_idx = ti.expand(num_decodes, self.num_kv_heads, k).contiguous()
            K_sel, V_sel = lrosa_gather(
                kv_cache, block_table_dec, top_idx, head_size=head_size,
                dtype=q_decode.dtype, K_sel_out=attn_metadata.K_sel_buf,
                V_sel_out=attn_metadata.V_sel_buf,
            )
            n_fac_eff = top_idx.shape[-1]
            cu_seqlens_q = attn_metadata.cu_seqlens_q_dec
            cu_seqlens_k = (attn_metadata.cu_seqlens_k_dec
                            if n_fac_eff == self.n_fac
                            else cu_seqlens_q * n_fac_eff)
            flash_attn_varlen_func(
                q=q_decode, k=K_sel, v=V_sel, out=output[:num_decode_tokens],
                cu_seqlens_q=cu_seqlens_q, max_seqlen_q=1,
                cu_seqlens_k=cu_seqlens_k, max_seqlen_k=n_fac_eff,
                softmax_scale=self.scale, causal=False,
                alibi_slopes=self.alibi_slopes, softcap=self.logits_soft_cap,
            )
            return

        if self.is_fasa:
            # FASA-fc (paper-faithful): gather the I_dom channels of the
            # group-mean query, then score by reading those same channels from
            # full K each step — no proj_K cache. ``M`` here is the int32
            # [H_kv, 2*n_tip] channel-offset tensor built in _ensure_on_device.
            ch = M
            idx = ch.to(torch.int64).unsqueeze(0).expand(num_decodes, -1, -1)
            q_sub = torch.gather(q_kv, 2, idx)  # (num_decodes, H_kv, 2*n_tip)
            top_idx, _ = fasa_score_topk(
                q_sub,
                kv_cache,
                block_table_dec,
                seq_lens_dec,
                ch,
                n_fac=self.n_fac,
                scores_out=attn_metadata.scores_buf,
                top_idx_out=attn_metadata.top_idx_buf,
                top_scores_out=attn_metadata.top_scores_buf,
                use_radix=self.use_radix_topk,
                window=window,
            )
        else:
            proj_q = torch.einsum("rhd,hcd->rhc", q_kv, M)  # (nd, H_kv, cs_h)
            if attn_metadata.use_streaming_topk:
                # Step 4a: V2 chunk-parallel streaming kernel — no full
                # scores buffer; tiny candidates buffer per (req, kv-head, chunk).
                top_idx = lrosa_streaming_topk(
                    proj_q,
                    kv_cache,
                    block_table_dec,
                    seq_lens_dec,
                    n_fac=self.n_fac,
                    head_size=head_size,
                    cs_h=self.cs_h,
                    candidates_buf=attn_metadata.candidates_buf,
                    chunk_size=attn_metadata.chunk_size,
                    top_idx_out=attn_metadata.top_idx_buf,
                )  # (num_decodes, H_kv, n_fac) int32
            elif self.use_radix_topk:
                # DSA radix top-K: O(seq) selection, int32 output. The builder
                # routed the int32 buffer into top_idx_buf for this path, so the
                # gather kernel (int32/int64 agnostic) is unaffected.
                top_idx, _ = lrosa_score_topk(
                    proj_q,
                    kv_cache,
                    block_table_dec,
                    seq_lens_dec,
                    n_fac=self.n_fac,
                    head_size=head_size,
                    cs_h=self.cs_h,
                    scores_out=attn_metadata.scores_buf,
                    top_idx_out=attn_metadata.top_idx_buf,
                    top_scores_out=attn_metadata.top_scores_buf,
                    use_radix=True,
                    window=window,
                )  # (num_decodes, H_kv, n_fac_eff) int32
            else:
                top_idx, _ = lrosa_score_topk(
                    proj_q,
                    kv_cache,
                    block_table_dec,
                    seq_lens_dec,
                    n_fac=self.n_fac,
                    head_size=head_size,
                    cs_h=self.cs_h,
                    scores_out=attn_metadata.scores_buf,
                    top_idx_out=attn_metadata.top_idx_buf,
                    top_scores_out=attn_metadata.top_scores_buf,
                    window=window,
                )  # (nd, H_kv, n_fac_eff) int64 CG path; int32 fallback

        K_sel, V_sel = lrosa_gather(
            kv_cache,
            block_table_dec,
            top_idx,
            head_size=head_size,
            dtype=q_decode.dtype,
            K_sel_out=attn_metadata.K_sel_buf,
            V_sel_out=attn_metadata.V_sel_buf,
        )  # (num_decodes * n_fac_eff, H_kv, head_size)

        n_fac_eff = top_idx.shape[-1]
        # Pre-allocated cu_seqlens cover up to ``max_num_reqs`` decodes;
        # builder.build() already sliced them to ``num_decodes + 1`` rows.
        # On the rare path where the score-kernel had to clip n_fac because
        # max_kv_len < n_fac, recompute the cu_seqlens_k scaling factor (else
        # use the pre-scaled constant).
        cu_seqlens_q = attn_metadata.cu_seqlens_q_dec
        if n_fac_eff == self.n_fac:
            cu_seqlens_k = attn_metadata.cu_seqlens_k_dec
        else:
            cu_seqlens_k = cu_seqlens_q * n_fac_eff

        if head_size > 256:
            # FlashAttention caps head_dim at 256; Gemma 4 full-attention
            # layers are head_dim=512. The gathered set is small (n_fac tokens,
            # q_len=1), so a manual GQA attention is cheap and head-dim-agnostic
            # (and CUDA-Graph safe — pure matmul + softmax). Selected K is a set
            # (no causal mask), matching the flash path's causal=False.
            nd = num_decodes
            nf = n_fac_eff
            Hkv = self.num_kv_heads
            G = self.num_kv_groups
            d = head_size
            K = K_sel.view(nd, nf, Hkv, d).permute(0, 2, 1, 3)  # (nd,Hkv,nf,d)
            V = V_sel.view(nd, nf, Hkv, d).permute(0, 2, 1, 3)
            q = q_decode.view(nd, Hkv, G, d).to(torch.float32)  # GQA-grouped
            scores = torch.einsum("nhgd,nhfd->nhgf", q, K.to(torch.float32))
            scores = scores * self.scale
            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum("nhgf,nhfd->nhgd", attn, V.to(torch.float32))
            output[:num_decode_tokens] = out.reshape(
                nd, Hkv * G, d
            ).to(output.dtype)
            return

        flash_attn_varlen_func(
            q=q_decode,
            k=K_sel,
            v=V_sel,
            out=output[:num_decode_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=n_fac_eff,
            softmax_scale=self.scale,
            causal=False,  # selected K is a set, not a sequence
            alibi_slopes=self.alibi_slopes,
            softcap=self.logits_soft_cap,
        )

    def _manual_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: LRoSAMetadata,
        head_size: int,
    ) -> None:
        """Per-request manual causal GQA attention for head_dim>256 full layers
        (Gemma 4) at all-prefill. Head-dim agnostic — no FA2 256 cap, no Triton
        shared-memory limit. Short reasoning prefills only (O(L^2) per request);
        not CUDA-Graph captured (prefill is the Python branch). Operates on the
        freshly-passed contiguous K/V (full sequence when chunked-prefill is off).
        """
        qsl_cpu = attn_metadata.query_start_loc_cpu
        if qsl_cpu is None:
            qsl_cpu = attn_metadata.query_start_loc.cpu()
        n_actual = attn_metadata.num_actual_tokens
        Hkv, G, d = self.num_kv_heads, self.num_kv_groups, head_size
        qf = query[:n_actual]
        kf = key[:n_actual]
        vf = value[:n_actual]
        for r in range(qsl_cpu.shape[0] - 1):
            s = int(qsl_cpu[r])
            e = int(qsl_cpu[r + 1])
            L = e - s
            if L <= 0:
                continue
            # (Hkv, G, L, d) / (Hkv, L, d) — GQA-grouped, fp32 for stability.
            qr = qf[s:e].view(L, Hkv, G, d).permute(1, 2, 0, 3).to(torch.float32)
            kr = kf[s:e].permute(1, 0, 2).to(torch.float32)
            vr = vf[s:e].permute(1, 0, 2).to(torch.float32)
            scores = torch.einsum("hgld,hmd->hglm", qr, kr) * self.scale
            causal = torch.triu(
                torch.ones(L, L, device=qf.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            out = torch.einsum("hglm,hmd->hgld", attn, vr)  # (Hkv, G, L, d)
            output[s:e] = (
                out.permute(2, 0, 1, 3).reshape(L, Hkv * G, d).to(output.dtype)
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        # Fused write: K, V, and proj_K = M @ K into the combined slot.
        # M is loaded/materialized on this layer's device lazily on first call.
        M = self._ensure_on_device(layer, key.device, key.dtype)
        if M is None or self.is_fasa:
            # Write K|V only, leave any proj_K region untouched. M is None for
            # dense (no-basis) layers; FASA-fc never stores proj_K (it re-reads
            # I_dom channels from full K each step) so its [K|V] slot is K|V.
            lrosa_store(key, value, kv_cache, slot_mapping)
        elif self.per_layer_concat:
            lrosa_project_and_store_layer(key, value, kv_cache, slot_mapping, M)
        else:
            lrosa_project_and_store(key, value, kv_cache, slot_mapping, M)
