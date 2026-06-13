# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Literal

from pydantic import field_validator

from vllm.config.utils import config
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@config
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. Use "auto" or None for automatic selection."""

    flash_attn_version: Literal[2, 3, 4] | None = None
    """Force vllm to use a specific flash-attention version (2, 3, or 4).
    Only valid when using the flash-attention backend."""

    use_prefill_decode_attention: bool = False
    """Use separate prefill and decode kernels for attention instead of
    the unified triton kernel."""

    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    tq_max_kv_splits_for_cuda_graph: int = 32
    """TurboQuant max NUM_KV_SPLITS for cuda graph decode.
    Fixes the split count so grid dimensions are constant across captures,
    and buffers can be pre-allocated to avoid inflating the memory estimate."""

    lrosa_basis_path: str | None = None
    """Path to LRoSA calibration .pt file. Expected format:
    ``{'M': {layer_idx: Tensor[H_kv, cs_h, d] fp32}, 'cs_h_max': int, ...}``.
    Loaded once on first decode step, projected onto each layer's device/dtype."""

    lrosa_n_fac: int = 256
    """LRoSA top-K token budget per (decode step, kv-head). Sparse decode
    attends to this many tokens; n_fac >= max_kv_len falls back to dense."""

    lrosa_n_tip: int = 16
    """FASA-fc only (kv_cache_dtype="fasa"): number of dominant frequency
    components (RoPE pairs) per kv-head selected at calibration. The per-step
    score reads ``2*n_tip`` raw channels {2*fc, 2*fc+1} from full K. Paper
    default 16 for d=128 (iso-byte with LRoSA cs_h=32). The basis file
    (``lrosa_basis_path``) is the fasa_idom_*.pt {'idom': {layer: [H_kv,
    n_tip_max]}} dict; the first ``n_tip`` columns are used."""

    lrosa_cs_h: int | None = None
    """Override the projected-score width cs_h. Default (None) derives it as
    ``head_size // 4`` (paper convention: 32 for d=128, 16 for GPT-OSS d=64).
    Set explicitly when the basis was calibrated at a non-default cs_h — e.g.
    Gemma 4 26B-A4B uses cs_h=64 on its head_dim=512 full-attention layers
    (head_size//4 would be 128). Must match the calibrated basis' cs_h."""

    lrosa_use_streaming_topk: bool = False
    """Use the single-stage streaming Triton kernel (Step 4a) instead of the
    two-pass score + ``torch.topk`` path. Streaming avoids the
    ``(num_reqs, H_kv, max_kv_len)`` fp32 score buffer; pre-PR profiling
    showed it pays off at long context (>~16K). Default False until the
    full benchmark sweep lands.

    NOTE (2026-06): on Blackwell sm_120 the streaming kernel is actually
    *slower* than the 2-pass + radix path at every measured context length
    (8K/32K/128K). Prefer ``lrosa_use_radix_topk`` over streaming."""

    lrosa_use_radix_topk: bool = True
    """Use DSA's CTA-persistent radix top-K (csrc/topk.cu,
    ``torch.ops._C.top_k_per_row_decode``) for the 2-pass selection instead
    of ``torch.topk`` (full sort). O(seq) vs O(seq log seq); measured 2-3x
    faster top-K and 100%% identical selection set. Falls back to torch.topk
    automatically if the binding is absent. Only applies when
    ``lrosa_use_streaming_topk`` is False."""

    lrosa_per_layer_concat: bool = False
    """Use the per-layer CONCAT LRoSA variant: a single basis M_layer
    [cs_h_layer, H_kv*head_size] projects the CONCATENATED per-head K (not a
    per-kv-head basis), and one shared top-K selection per layer feeds all
    kv-heads. cs_h_layer is spread across the H_kv slots' proj_K regions
    (cs_h_slot = cs_h_layer // H_kv), reusing the combined-slot layout. The
    basis .pt must hold M as [1, cs_h_layer, H_kv*head_size] (or
    [cs_h_layer, H_kv*head_size]). Trades a per-head selection for H_kv× fewer
    block-table lookups + lower metadata at iso-quality (pca: -0.53 F1 on
    16-task LongBench at cs_h_layer=256 vs per-head cs_h=32)."""

    lrosa_contig_projk: bool = False
    """Store proj_K in a SEPARATE contiguous cache [num_blocks, block_size,
    H_kv, cs_h] instead of interleaved in the [K|V|proj_K] slot. The score
    kernel then scans proj_K coalesced (no strided cache-line waste), ~1.4x
    faster at long context on high-bandwidth GPUs (B200). When on, the combined
    slot shrinks to [K|V]; the separate proj_K cache is an extra (~cs_h/2*head ≈
    11%) allocation NOT tracked by vLLM's memory profiler, so lower
    ``gpu_memory_utilization`` by ~that fraction to avoid OOM (true memory
    neutrality would need proj_K registered as a second KV cache group).
    OFF by default for that reason. Only the per-kv-head radix path honors it;
    streaming / per_layer_concat keep the interleaved slot. Numerically
    identical selection (Qasper F1 within radix-tie-break noise)."""

    lrosa_fp8_projk: bool = False
    """Store proj_K as FP8 (e4m3) in the contiguous proj_K cache (implies
    lrosa_contig_projk). Halves the proj_K read bandwidth of the score scan and
    halves the proj_K cache footprint. A per-(head,channel) scale (the row-norm
    of the basis M) is folded into proj_q so the score dot recovers
    proj_q·proj_K up to fp8 rounding — quantization affects only top-k SELECTION,
    not the attention output (K/V stay bf16), so it is near-lossless (LRoSA's
    proj_K quantizes far better than FASA's K_sel; see paper). Mirrors DSA's fp8
    index logits. ~1.16x on the score kernel alone."""

    lrosa_indexed_attend: bool = False
    """Gather-free fused indexed attention for decode: stream the
    top-``lrosa_n_fac`` selected paged slots directly through an online-softmax
    kernel instead of materializing a contiguous K_sel/V_sel buffer and running
    a dense flash_attn over it. The buffer scales with num_decodes * n_fac, so
    removing it cuts the attention-side per-step fixed overhead ~2.3x at large
    batch (the throughput operating point); negligible at batch 1. Only the
    plain path (head_size<=256, set attention, no sinks/softcap/alibi,
    steady-state non-partial) takes it — gemma full layers (head 512), gpt-oss
    (sinks), and the eager seq<k_eff partial tail fall back to gather+flash.
    Reads only K|V from the slot, so independent of the fp8/contig proj_K score
    path. OFF by default."""

    lrosa_indexed_min_batch: int = 1
    """Minimum decode batch (num_decodes) for the gather-free indexed kernel to
    engage; below this, fall back to gather+flash. The v3 split-N path occupies
    the SMs at small batch too (grid num_decodes*H_kv*S), so the kernel is a net
    win at every batch and the default floor is 1. Raise it only on hardware
    where the split heuristic underperforms the gather+flash path at tiny batch."""

    lrosa_mla: bool = False
    """Apply LRoSA token selection to an MLA model (e.g. GLM-4.7-Flash) by
    scoring the latent c_KV with the calibrated rotation M (M:[1, cs_h,
    kv_lora_rank]) instead of per-head K. Routes the MLA layer through the
    existing FLASHMLA_SPARSE attend: an LRoSAMLAIndexer writes the top-``lrosa_n_fac``
    token indices into the shared topk_indices_buffer. Reuses ``lrosa_basis_path``
    / ``lrosa_n_fac`` / ``lrosa_cs_h`` (cs_h ≈ 25%% of the latent's K-share =
    kv_lora_rank/2; 64 for GLM-4.7-Flash's 512 latent, iso-overhead with the
    KV cache). Appendix / eager+bf16 only; FKV is plain dense MLA (this flag off)."""

    quest_token_budget: int = 256
    """Quest backend: total per-(decode step, kv-head) token budget. The
    page budget is ``quest_token_budget // page_size`` selected full pages;
    one additional page is always-attended for the trailing partial page
    (which holds the current decode token). Iso-budget with LRoSA's
    ``lrosa_n_fac`` when set equal."""

    quest_page_size: int = 16
    """Quest backend: page size in tokens. Quest pages map 1:1 onto paged-
    cache blocks, so the engine block_size is forced to this value. Default
    16 matches the Quest paper and pca's ``quest/quest.py`` reference."""

    seer_gate_path: str | None = None
    """SeerAttention-R backend (kv_cache_dtype="seer"): path to the AttnGate
    weights (``attn_gate_weights.pth`` state_dict with keys
    ``model.layers.{i}.self_attn.attn_gate.attngate_{linear_q,linear_k,qnorm,
    knorm}.weight``) or the HF adapter repo dir (e.g.
    ``SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates``). Required."""

    seer_token_budget: int = 4096
    """SeerAttention-R: total per-(decode step, kv-head) token budget. The
    block budget is ``seer_token_budget // seer_gate_block_size`` selected
    64-token K-blocks; the trailing partial block is always attended."""

    seer_gate_block_size: int = 64
    """SeerAttention-R: AttnGate K-block size in tokens (must be a multiple of
    the paged page size 16). Default 64 matches the released AttnGates."""

    seer_gate_hidden_size: int = 128
    """SeerAttention-R: AttnGate projection dim. Overridden by the adapter's
    config.json ``seerattn_gate_hidden_size`` when present."""

    seer_sparsity_method: str = "token_budget"
    """SeerAttention-R: "token_budget" (top-k blocks) or "threshold" (blocks
    with softmax gate-score > ``seer_threshold``)."""

    seer_threshold: float = 0.0
    """SeerAttention-R: gate-score threshold when ``seer_sparsity_method`` is
    "threshold"."""

    seer_start_layer: int = 0
    """SeerAttention-R: first decoder layer (inclusive) to apply gate sparsity;
    earlier layers run dense."""

    use_trtllm_attention: bool | None = None
    """If set to True/False, use or don't use the TRTLLM attention backend
    in flashinfer. If None, auto-detect the attention backend in flashinfer."""

    disable_flashinfer_q_quantization: bool = False
    """If set, when using fp8 kv, do not quantize Q to fp8."""

    mla_prefill_backend: MLAPrefillBackendEnum | None = None
    """MLA prefill backend to use. If None, will be selected automatically.
    Valid options: FLASH_ATTN (FA3/FA4), FLASHINFER, TRTLLM_RAGGED."""

    use_prefill_query_quantization: bool = False
    """If set, quantize query for attention in prefill."""

    use_fp4_indexer_cache: bool = False
    """If set, use fp4 indexer cache for dsv32 family model (not support yet)"""

    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""

    flex_attn_block_m: int | None = None
    """Triton kernel BLOCK_M tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_block_n: int | None = None
    """Triton kernel BLOCK_N tile size for flex attention.
    Must be a power of 2 >= 16. If None and VLLM_BATCH_INVARIANT=1,
    defaults to 16."""

    flex_attn_q_block_size: int | None = None
    """Logical Q block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_m.
    If None, uses the default (16 on PyTorch >= 2.9, 128 otherwise)."""

    flex_attn_kv_block_size: int | None = None
    """Logical KV block size for the flex attention block mask.
    Must be a power of 2 and divisible by flex_attn_block_n.
    If None, uses the default (kv_cache_block_size on PyTorch >= 2.9,
    128 otherwise)."""

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        ignored_factors: set[str] = set()
        factors = get_hash_factors(self, ignored_factors)
        return hash_factors(factors)

    @field_validator("backend", mode="before")
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `backend` enum type from string.

        The special value "auto" is treated as None, which triggers
        automatic backend selection.
        """
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value

    @field_validator("mla_prefill_backend", mode="before")
    @classmethod
    def validate_mla_prefill_backend_before(cls, value: Any) -> Any:
        """Enable parsing of the `mla_prefill_backend` enum type from string."""
        if isinstance(value, str):
            return MLAPrefillBackendEnum[value.upper()]
        return value
