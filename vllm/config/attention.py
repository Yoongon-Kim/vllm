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
