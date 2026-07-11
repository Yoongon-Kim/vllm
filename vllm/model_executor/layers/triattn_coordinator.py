"""Cross-layer coordinator for a FAITHFUL TriAttention vLLM runner.

TriAttention (default per_layer_perhead_pruning=False) is CROSS-LAYER GLOBAL
eviction: one keep set computed from ALL layers' K, applied to every layer. vLLM
attention is per-layer, so we coordinate: at a compression trigger, gather every
layer's post-RoPE K for a sequence out of the paged combined-slot cache, assemble
the `pkv_tuple` the reference `TriAttention.compute_keep_indices` expects, and
call it UNCHANGED (faithful by construction — see stage-1 validation in
[[triattention-vllm-faithful-port]]). The per-seq keep set is then shared with
every layer's select-at-attend (full cache kept; only the keep set is attended —
faithful for ACCURACY, vLLM can't compact the paged cache).

This module is the new piece; the per-layer attend reuses the lrosa_attn
select-at-attend machinery. Multi-batch = call the (batch=1) compute_keep_indices
per sequence.
"""
from __future__ import annotations

import torch


class TriAttnCoordinator:
    """Holds the calibrated TriAttention compressor + refs to every layer's K
    cache, and computes the per-sequence cross-layer keep set at trigger steps.

    Cache layout (combined slot, same as lrosa_attn): each layer's kv_cache is
    ``(num_blocks, block_size, H_kv, slot_size)`` with K at ``slot[:head_size]``.
    """

    def __init__(self, compressor, *, head_size: int, num_kv_heads: int,
                 block_size: int, budget: int, divide_length: int = 128):
        self.comp = compressor                 # reference TriAttention (stats loaded)
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.budget = budget
        self.divide_length = divide_length
        # Per-sequence state.
        self.keep: dict = {}                  # req_key -> LOCAL keep indices
        self._last_trigger_len: dict = {}     # req_key -> cache_len at last trigger
        self._prefix: dict = {}               # req_key -> prompt length (always kept)
        self._seen_len: dict = {}             # req_key -> last seq_len (reset detect)
        # Per-step decode attend positions (computed once at the first layer).
        self._dec_gblk = None
        self._dec_goff = None
        self._dec_cu = None
        self._dec_maxn = 0
        # All-layer K-cache refs (filled by register()).
        self._layer_caches: dict[int, torch.Tensor] = {}
        self.first_layer: int = 1 << 30

    def reset_seq(self, seq_id: int) -> None:
        self.keep.pop(seq_id, None)
        self._last_trigger_len.pop(seq_id, None)

    def _gather_layer_K(self, k_cache: torch.Tensor, block_table_row: torch.Tensor,
                        positions: torch.Tensor) -> torch.Tensor:
        """Gather one layer's post-RoPE K at the given ABSOLUTE token `positions`
        (the logical compacted cache) → ``[1, H_kv, n, head_size]`` (the contract
        compute_keep_indices consumes). block_table_row: physical block ids."""
        bs = self.block_size
        blk = block_table_row[positions // bs].to(torch.long)  # [n] physical block
        off = (positions % bs).to(torch.long)                  # [n] within-block
        k = k_cache[blk, off, :, : self.head_size]             # [n, H_kv, hs]
        return k.permute(1, 0, 2).unsqueeze(0).contiguous()    # [1, H_kv, n, hs]

    # --- registration + per-step driver (vLLM wiring) ------------------- #
    def register(self, layer_idx: int, kv_cache: torch.Tensor) -> None:
        """Each layer's attention forward registers its combined-slot kv_cache
        (K at slot[:head_size]). Refs are persistent across steps, so by the
        first decode step the coordinator holds every layer's cache."""
        self._layer_caches[layer_idx] = kv_cache
        self.first_layer = min(self.first_layer, layer_idx)

    def step(self, block_tables, seq_lens, req_keys) -> None:
        """Driver invoked once per decode batch (from the first layer): for each
        decode sequence, gather all registered layers' K → maybe (re)compute the
        cross-layer keep set. `seq_lens`/`req_keys` are PYTHON int lists (the
        caller does ONE .tolist() each — avoids per-seq host syncs). prefix_length
        (always-kept prompt) is tracked per key on first sight (first decode step
        → seq_len-1 = prompt length)."""
        if not self._layer_caches:
            return
        k_caches = [self._layer_caches[i] for i in sorted(self._layer_caches)]
        for i, key in enumerate(req_keys):
            L = seq_lens[i]
            # Detect a NEW request on this key: either unseen, or seq_len is not
            # the previous+1 (a finished request's block reused by a new one, OR
            # a profiling/warmup dummy step polluted the key with a tiny L). In
            # normal decode seq_len increments by exactly 1 each step, so any
            # other delta means a fresh prefill happened → reset state and set
            # the prompt length = L-1 (this is the FIRST real decode step, so
            # L = prompt_len + 1). The old `L < seen_len` check missed warmup
            # pollution (warmup L=3 → prefix=2 stuck, since a real L≫3 never
            # "drops"), which left the prompt almost unprotected.
            prev_L = self._seen_len.get(key)
            if prev_L is None or L != prev_L + 1:
                self.keep.pop(key, None)
                self._last_trigger_len.pop(key, None)
                self._prefix[key] = max(0, L - 1)   # prompt length at first real decode
            self._seen_len[key] = L
            self.maybe_compute_keep(key, k_caches, block_tables[i], L, self._prefix[key])

    def cache_positions(self, block_tables, seq_lens, req_keys) -> None:
        """Compute the attended positions (keep ∪ tokens-since-trigger) for ALL
        decode seqs ONCE per step and cache the concatenated paged (block, offset)
        indices + cu_seqlens for the batched varlen flash. The keep is CROSS-LAYER,
        so every layer attends the SAME positions — computing this here (at the
        first layer) instead of in every layer's attend removes ~36x redundant
        work + host syncs. `seq_lens`/`req_keys` are python int lists."""
        bs = self.block_size
        dev = block_tables.device
        blk_parts, off_parts, lens = [], [], []
        for i, key in enumerate(req_keys):
            L = seq_lens[i]
            keep = self.keep.get(key)
            if keep is None:
                pos = torch.arange(L, device=dev)
            else:
                last = self._last_trigger_len.get(key, 0)
                pos = torch.cat([keep, torch.arange(last, L, device=dev)]).unique()
            blk_parts.append(block_tables[i, (pos // bs)].to(torch.long))
            off_parts.append((pos % bs).to(torch.long))
            lens.append(pos.numel())
        self._dec_gblk = torch.cat(blk_parts)
        self._dec_goff = torch.cat(off_parts)
        cu = torch.zeros(len(req_keys) + 1, dtype=torch.int32, device=dev)
        cu[1:] = torch.tensor(lens, dtype=torch.int32, device=dev).cumsum(0)
        self._dec_cu = cu
        self._dec_maxn = max(lens) if lens else 0

    def maybe_compute_keep(self, seq_id, k_caches: list[torch.Tensor],
                           block_table_row: torch.Tensor, seq_len: int,
                           prefix_length: int) -> torch.Tensor | None:
        """At a trigger, score the LOGICAL COMPACTED CACHE (previous keep ∪ tokens
        since the last trigger) — NOT the full history — and select the new keep.
        This mirrors TriAttention's actual compaction: after each compression the
        cache holds only `keep`, new tokens append, and the next compression scores
        that compacted set. Scoring the full 0..L history instead would (1) be
        O(L^2) -> deep-context slowdown, and (2) re-admit evicted tokens -> NOT
        faithful past the first compression. Returns the new keep (ABSOLUTE token
        positions, sorted) or None if no trigger this step."""
        # Official slack trigger (triattention/vllm/runtime/hook_runtime_context
        # .py:188 — local_length_threshold = budget + max(1, divide_length)):
        # compress once the logical cache reaches budget + divide_length, then
        # prune to top-budget. The cache oscillates budget ↔ budget+divide_length,
        # compressing every divide_length generated tokens (exactly the user's
        # "B+128 -> back to B" model). Purely length-thresholded — NO modular or
        # prefix-relative alignment — so the prefix tracking is irrelevant here.
        dev = block_table_row.device
        prev_keep = self.keep.get(seq_id)
        last = None
        if prev_keep is None:
            cache_len = seq_len   # no prune yet — full 0..L history
        else:
            last = self._last_trigger_len[seq_id]
            cache_len = prev_keep.numel() + (seq_len - last)
        import os as _os
        if _os.environ.get("TRIATTN_TRIGGER") == "modular":
            # Official TRANSFORMERS benchmark trigger (triattention_forward,
            # use_slack_trigger=False, count_prompt_tokens=False): compress at
            # ABSOLUTE positions that are multiples of divide_length, once the
            # GENERATED count has reached budget. This is the config that yields
            # the published 32.9 — different from the vLLM-runtime slack trigger.
            if seq_len % self.divide_length != 0 or (seq_len - prefix_length) < self.budget:
                return None
        else:
            # slack (default, official vLLM runtime hook_runtime_context:188):
            # compress when logical cache reaches budget+divide_length.
            if cache_len < self.budget + self.divide_length:
                return None
        # Threshold reached — assemble the logical compacted set to score.
        if prev_keep is None:
            compacted = torch.arange(seq_len, device=dev)
        else:
            recent = torch.arange(last, seq_len, device=dev)
            compacted = torch.cat([prev_keep.to(dev), recent]).unique()  # sorted
        # Assemble pkv (K per layer at the compacted positions; V unused by scoring).
        pkv = tuple((self._gather_layer_K(kc, block_table_row, compacted),) * 2
                    for kc in k_caches)
        self.comp.cache_positions = compacted.tolist()    # ABSOLUTE positions (RoPE)
        self.comp.absolute_position = seq_len
        keep_local = self.comp.compute_keep_indices(pkv, prefix_length=prefix_length)
        # keep_local indexes the compacted sequence → map back to absolute tokens.
        new_keep = torch.sort(compacted[keep_local.to(dev)]).values
        self.keep[seq_id] = new_keep
        self._last_trigger_len[seq_id] = seq_len
        import os as _os
        if _os.environ.get("TRIATTN_KEEPDUMP"):
            ws = self.comp.window_size
            nk = new_keep
            n_recent = int((nk >= (seq_len - ws)).sum())
            print(f"[keep] L={seq_len} |compacted|={compacted.numel()} |keep|={nk.numel()} "
                  f"budget={self.budget} recent_in_keep={n_recent}/{ws} "
                  f"min={int(nk.min())} max={int(nk.max())} "
                  f"old(<L-2000)={int((nk < seq_len-2000).sum())}", flush=True)
        return new_keep


_COORDINATOR: "TriAttnCoordinator | None" = None
_LROSA_DEV = "/NHNHOME/jiwonsong/LRoSA-dev"


def get_or_create_coordinator(*, stats_path: str, model_path: str, budget: int,
                              divide_length: int, head_size: int, num_kv_heads: int,
                              block_size: int, device,
                              window_size: int = 128) -> "TriAttnCoordinator":
    """Lazily build the single shared coordinator: construct the reference
    `TriAttention` compressor from the GQA stats + model config (same defaults as
    LRoSA-dev `apply_triattention_patch`; the compressor builds its own rotary from
    the model config — correct for Qwen3 no-yarn). One coordinator per process
    (one model per eval process)."""
    global _COORDINATOR
    if _COORDINATOR is not None:
        return _COORDINATOR
    import sys
    from pathlib import Path
    if _LROSA_DEV not in sys.path:
        sys.path.insert(0, _LROSA_DEV)
    from triattention.methods.triattention import TriAttention, TriAttentionConfig
    cfg = TriAttentionConfig(
        stats_path=Path(stats_path), model_path=Path(model_path), device=device,
        dtype=torch.float32, budget=budget, offset_max_length=65536,
        score_aggregation="mean", seed=0, metadata_expectations=None,
        normalize_scores=False, count_prompt_tokens=True,  # paper: budget gate on TOTAL cache
        # official default protect_prefill=False → prefill competes on score (no
        # pinning); recent window_size=128 always kept (forced +inf before top-B).
        # DIAG: TRIATTN_PROTECT_PREFILL=1 pins the prompt (allow_prefill_compression
        # =False) to test the long-reasoning prompt-eviction hypothesis.
        allow_prefill_compression=(__import__("os").environ.get("TRIATTN_PROTECT_PREFILL", "0") != "1"),
        window_size=int(__import__("os").environ.get("TRIATTN_WINDOW", str(window_size))),  # DIAG override
        divide_length=divide_length,
        use_slack_trigger=False, per_head_pruning=False,
        per_layer_perhead_pruning=False, layer_perhead_aggregation="max",
        disable_mlr=False, disable_trig=False,
    )
    comp = TriAttention(cfg)
    _COORDINATOR = TriAttnCoordinator(
        comp, head_size=head_size, num_kv_heads=num_kv_heads,
        block_size=block_size, budget=budget, divide_length=divide_length)
    return _COORDINATOR


# --------------------------------------------------------------------------- #
# Self-test (stage 2a): verify the paged→contiguous gather is CORRECT against a
# known cache, and that compute_keep_indices runs on the assembled pkv.
#   python -m vllm.model_executor.layers.triattn_coordinator
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    H_kv, hs, slot, bs = 8, 128, 256, 16        # slot = 2*hs (K|V), block_size 16
    num_layers, seq_len, num_blocks = 4, 100, 16

    # Build known per-token K per layer, scatter into a paged combined-slot cache.
    block_table = torch.arange(num_blocks, device=dev)          # identity mapping
    k_caches, known_K = [], []
    for _ in range(num_layers):
        kc = torch.zeros(num_blocks, bs, H_kv, slot, device=dev)
        kK = torch.randn(seq_len, H_kv, hs, device=dev)
        for t in range(seq_len):
            kc[t // bs, t % bs, :, :hs] = kK[t]
        k_caches.append(kc); known_K.append(kK)

    # Minimal coordinator (no compressor needed for the gather check).
    coord = TriAttnCoordinator.__new__(TriAttnCoordinator)
    coord.head_size, coord.num_kv_heads, coord.block_size = hs, H_kv, bs
    g0 = coord._gather_layer_K(k_caches[0], block_table, seq_len)   # [1,H_kv,seq,hs]
    ref = known_K[0].permute(1, 0, 2).unsqueeze(0)
    ok = torch.equal(g0, ref.contiguous())
    print(f"### gather correctness: shape={tuple(g0.shape)} matches-known-K={ok}")
    assert ok, "paged->contiguous gather mismatch"
    print("### STAGE 2a (gather plumbing) PASS — coordinator reconstructs "
          "per-layer [1,H_kv,seq,hs] K from the paged combined-slot cache.")
    sys.exit(0)
