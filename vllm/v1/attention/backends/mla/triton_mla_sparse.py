# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Arch-portable sparse-MLA attend backend (sm_120 / RTX PRO 6000 Blackwell).

FlashMLA / FlashInfer-TRT-LLM sparse-MLA decode kernels are sm_90 / sm_100 only
and are NOT built on sm_120. This backend provides an *arch-portable* sparse MLA
attend for sm_120 by reusing the Triton MQA decode kernel (`decode_attention_fwd`,
the same one `TritonMLAImpl.forward_mqa` uses for DENSE MLA) over a GATHERED,
contiguous latent buffer of the selected (top-k) tokens.

It is purely ADDITIVE and sm_120-GATED: `supports_compute_capability` returns
True ONLY for `capability.major == 12`. The sm_90 / sm_100 FlashMLA(Sparse)
code paths and numerics are byte-for-byte untouched; FlashMLA still owns 9/10.

The selection (which tokens) is produced upstream by the LRoSA-MLA indexer into
`indexer.topk_indices_buffer` (int32 [T, n_fac], LOCAL token positions, left-
packed ascending + -1 pad). This backend only does the latent ATTEND over those
selected tokens, mirroring `FlashMLASparseImpl.forward_mqa` but with a portable
Triton kernel instead of the FlashMLA sparse kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    CommonAttentionMetadata,
    SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)

# Cap the gathered-latent buffer to ~this many rows per chunk (rows = tokens ×
# topk). 512K rows × 576 bf16 ≈ 0.56 GiB; keeps the prefill all-tokens gather
# bounded. Decode batches are tiny -> a single chunk.
_GATHER_CHUNK_ROWS = 512 * 1024


@dataclass
class TritonMLASparseMetadata(MLACommonMetadata):
    # Per-token request id (np.repeat(arange(num_reqs), per-req token counts)).
    # Used by triton_convert_req_index_to_global_index to map LOCAL token
    # positions -> GLOBAL cache slot ids via the block table.
    req_id_per_token: torch.Tensor | None = None
    # Full-batch block table + seq lens (NOT decode-only): a sparse MLA impl
    # routes ALL tokens (prefill + decode) through forward_mqa, so it cannot
    # rely on the decode-only `decode.block_table`. Mirrors FlashMLASparse.
    sparse_block_table: torch.Tensor | None = None
    sparse_seq_lens: torch.Tensor | None = None
    block_size: int = 64
    topk_tokens: int = 2048


class TritonMLASparseMetadataBuilder(
    MLACommonMetadataBuilder[TritonMLASparseMetadata]
):
    # The gather is dynamic-shape per request; capture under PIECEWISE only.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls=TritonMLASparseMetadata,
        )
        # LRoSA-on-MLA models have no native DSA `index_topk`; the selection
        # budget is the configured lrosa_n_fac (the indexer writes that many).
        # Mirrors FlashMLASparseMetadataBuilder.
        self.topk_tokens = getattr(
            vllm_config.model_config.hf_config, "index_topk", None
        )
        if self.topk_tokens is None:
            self.topk_tokens = vllm_config.attention_config.lrosa_n_fac
        self.req_id_per_token_buffer = torch.empty(
            (vllm_config.scheduler_config.max_num_batched_tokens,),
            dtype=torch.int32,
            device=device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonMLASparseMetadata:
        metadata = super().build(
            common_prefix_len, common_attn_metadata, fast_build
        )

        # req_id_per_token: copy the FlashMLASparse builder verbatim.
        cm = common_attn_metadata
        num_tokens = cm.num_actual_tokens
        starts = np.asarray(cm.query_start_loc_cpu, dtype=np.int32)
        seg_lengths = np.diff(starts)
        req_id_per_token = np.repeat(
            np.arange(seg_lengths.shape[0], dtype=np.int32), seg_lengths
        )
        # Zero-fill for cudagraphs
        self.req_id_per_token_buffer.fill_(0)
        self.req_id_per_token_buffer[: req_id_per_token.shape[0]].copy_(
            torch.from_numpy(req_id_per_token), non_blocking=True
        )

        metadata.req_id_per_token = self.req_id_per_token_buffer[:num_tokens]
        metadata.sparse_block_table = cm.block_table_tensor
        metadata.sparse_seq_lens = cm.seq_lens
        metadata.block_size = self.kv_cache_spec.block_size
        metadata.topk_tokens = self.topk_tokens
        return metadata


class TritonMLASparseBackend(AttentionBackend):
    # sm_120 only supports the bf16 latent cache (no fp8_ds_mla custom layout).
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        # sm_120 ONLY. Do NOT claim 9/10 — FlashMLA(Sparse) owns those and its
        # numerics/code path must stay byte-for-byte unchanged.
        return capability.major == 12

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # bf16 latent layout: (num_blocks, block_size, head_size=576).
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder


class TritonMLASparseImpl(
    MLACommonImpl[TritonMLASparseMetadata],
    SparseMLAAttentionImpl[TritonMLASparseMetadata],
):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: object | None = None,
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            **mla_args,
        )
        assert indexer is not None, (
            "TritonMLASparse requires a sparse indexer "
            "(LRoSAMLAIndexer) producing topk_indices_buffer."
        )
        self.topk_indices_buffer = indexer.topk_indices_buffer
        # self.kv_lora_rank is set by MLACommonImpl.__init__ from mla_args.
        self._sm_count = current_platform.num_compute_units()

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: TritonMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert self.topk_indices_buffer is not None
        assert attn_metadata.sparse_block_table is not None

        # 1. q arrives as (ql_nope[B,N,512], q_pe[B,N,64]) for the bf16 path.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert isinstance(q, torch.Tensor)
        B = q.shape[0]  # all mqa tokens (prefill + decode for a sparse impl)
        q_num_heads = q.shape[1]
        head_dim = kv_c_and_k_pe_cache.shape[-1]  # 576 for GLM/DSV3
        device = q.device

        # 2. LOCAL top-k positions per token, [B, topk], left-packed + -1 pad.
        topk_local = self.topk_indices_buffer[:B]
        topk = topk_local.shape[1]

        # 3. Map per-token LOCAL positions -> GLOBAL cache slot ids (preserving
        # -1 padding) via the FULL block table. Reuses the same converter
        # FlashMLASparse uses.
        global_slots = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:B],
            attn_metadata.sparse_block_table,
            topk_local,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk,
        )  # [B, topk], -1 = invalid/padding

        # valid count per token = #non-(-1) slots (indexer left-packs valid then
        # -1 pads the tail).
        b_seq_len_all = (global_slots >= 0).sum(dim=1).to(torch.int32)  # [B]

        # 4-7. Outputs + split bookkeeping (mirror TritonMLAImpl.forward_mqa).
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=device)

        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            min_work_per_split = 512
            # at most `topk` valid positions per request
            ideal_splits = max(1, topk // min_work_per_split)
            ideal_splits = triton.next_power_of_2(ideal_splits)
            occupancy_multiplier = 2
            max_splits = self._sm_count * occupancy_multiplier
            num_kv_splits = min(ideal_splits, max_splits)

        flat = kv_c_and_k_pe_cache.view(-1, head_dim)  # [num_slots, 576]
        # Chunk over tokens so the gathered latent buffer (B_chunk * topk * 576)
        # stays bounded — a prefill batch can be thousands of tokens × topk=2048,
        # which would be tens of GiB if materialized at once. Decode batches are
        # tiny so this is a single chunk there.
        chunk = max(1, _GATHER_CHUNK_ROWS // max(1, topk))
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            bc = end - start
            slots = global_slots[start:end]  # [bc, topk]
            safe = slots.clamp(min=0)  # -1 -> slot 0 (masked out via b_seq_len)
            k_gathered = flat[safe.reshape(-1)].view(bc * topk, 1, head_dim)
            # MLA: v is the kv_lora_rank prefix of the same buffer (is_mla=True
            # transposes k for the value path); pass it for shape inference.
            v_gathered = k_gathered[..., : self.kv_lora_rank]
            # Synthetic req_to_token: row r -> gathered rows [r*topk : (r+1)*topk].
            ar = torch.arange(topk, device=device, dtype=torch.int32)
            req_to_token = (
                torch.arange(bc, device=device, dtype=torch.int32).unsqueeze(1)
                * topk
                + ar.unsqueeze(0)
            )  # [bc, topk]
            attn_logits = torch.empty(
                (bc, q_num_heads, num_kv_splits, self.kv_lora_rank + 1),
                dtype=torch.float32,
                device=device,
            )
            # Attend over the gathered latent buffer with PAGE_SIZE=1 (logical
            # pos j -> gathered row req_to_token[r, j]).
            decode_attention_fwd(
                q[start:end],
                k_gathered,
                v_gathered,
                o[start:end],
                lse[start:end],
                req_to_token,
                b_seq_len_all[start:end].contiguous(),
                attn_logits,
                num_kv_splits,
                self.scale,
                1,  # page_size
                k_scale=layer._k_scale,
                v_scale=layer._k_scale,
                is_mla=True,
            )

        return o, lse
