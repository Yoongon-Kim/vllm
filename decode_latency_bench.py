"""Decode-latency benchmark: LRoSA vs Quest vs FKV in the SAME vLLM stack.

The deliverable for the Quest-vs-LRoSA comparison. Measures steady-state
decode ms/token at a fixed context length, with CUDA graph + the production
attention path for each backend:
  - fkv   : full attention (FlashAttention paged), the dense upper bound
  - lrosa : token-level learned-rotation selection (gather + dense attn)
  - quest : page-level min/max selection (block-sparse attn, no gather)

Method: single request, bsz=1, ignore_eos. Time a prefill-only generate
(max_tokens=1) and a prefill+D generate (max_tokens=1+D); decode ms/tok =
(T_{1+D} - T_1) / D. Prefix caching OFF so both include a real prefill.
Same random prompt across backends (fixed seed) for apples-to-apples.

Run (one backend per invocation):
  CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    /home/jiwonsong/.conda/envs/gaep_vllm/bin/python quest_latency_bench.py \
      --backend quest --prefill_len 65536 --decode_len 128
"""
import argparse
import os
import time

# flashinfer sampler JIT-compiles via ninja (absent in this env); force the
# native PyTorch sampler. Unrelated to the attention backends being benchmarked.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from _bench_common import lrosa_basis_path, fasa_idom_path, yarn_overrides

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def build_llm(backend, prefill_len, decode_len, n_fac, gpu_mem, batch_size,
              max_num_seqs=0, model=MODEL, basis=None, cs_h=32, n_tip=16,
              use_radix=True, streaming=False, contig_projk=False, fp8_projk=False, per_layer=False,
              indexed_attend=True,
              tp=1, mla_kv_dtype="fp8_ds_mla", mla_backend=None, mla_fkv_fp8=False,
              chunked_prefill=False, cudagraph_mode=None, dense_mla=False):
    # max_num_seqs caps concurrency → sizes the LRoSA/Quest static decode
    # buffers. 0 → batch_size (tight, for clean latency). >0 simulates online
    # serving concurrency (stresses the buffers; Quest's are tiny, LRoSA's
    # gather buffers grow with n_fac × max_num_seqs).
    mns = max_num_seqs if max_num_seqs > 0 else max(batch_size, 1)
    kw = dict(model=model, max_model_len=prefill_len + decode_len + 16,
              gpu_memory_utilization=gpu_mem, enforce_eager=os.environ.get("ENFORCE_EAGER", "0") == "1",
              tensor_parallel_size=tp,
              enable_prefix_caching=False, max_num_seqs=mns,
              # Steady-state decode latency: keep prefill and decode in
              # separate batches (no chunked-prefill mixed batches). Also avoids
              # the sparse backends' mixed-batch score-kernel bug at bsz>1.
              # --chunked_prefill opts in (caps peak prefill activation so long
              # ctx fits at the proper decode batch; decode timing is unaffected).
              enable_chunked_prefill=chunked_prefill)
    if os.environ.get("BLOCK_SIZE"):
        # Larger block_size -> fewer scattered physical blocks per score-kernel
        # tile (BLOCK_T=512) -> coalesced proj_K reads -> kills the chunked-
        # prefill fragmentation penalty. Fix for GQA LRoSA + chunked deployment.
        kw["block_size"] = int(os.environ["BLOCK_SIZE"])
    if chunked_prefill:
        # MNBT env overrides the chunk size (diagnostic: sweep to separate
        # prefill-fragmentation from chunked-mode as the decode-slowdown cause).
        kw["max_num_batched_tokens"] = int(os.environ.get("MNBT", "8192"))
    # Qwen3: enable YaRN so served rope matches the basis's calibration rope
    # (also lets prefill_len exceed the 32K native window).
    ov = yarn_overrides(model)
    if ov:
        kw["hf_overrides"] = ov
    if backend == "lrosa":
        basis = basis or lrosa_basis_path(model, cs_h=cs_h)
        kw["kv_cache_dtype"] = "lrosa"
        kw["attention_config"] = {"backend": "LROSA", "lrosa_basis_path": basis,
                                  "lrosa_n_fac": n_fac, "lrosa_cs_h": cs_h,
                                  "lrosa_use_radix_topk": use_radix,
                                  "lrosa_use_streaming_topk": streaming,
                                  "lrosa_contig_projk": contig_projk,
                                  "lrosa_fp8_projk": fp8_projk,
                                  "lrosa_indexed_attend": indexed_attend,
                                  "lrosa_per_layer_concat": per_layer}
    elif backend == "fasa":
        idom = basis or fasa_idom_path(model)
        kw["kv_cache_dtype"] = "fasa"
        kw["attention_config"] = {"backend": "LROSA", "lrosa_basis_path": idom,
                                  "lrosa_n_fac": n_fac, "lrosa_n_tip": n_tip,
                                  "lrosa_use_radix_topk": use_radix}
    elif backend == "quest":
        kw["kv_cache_dtype"] = "quest"
        kw["attention_config"] = {"backend": "QUEST", "quest_token_budget": n_fac}
    elif backend == "lrosa_mla":
        # LRoSA on an MLA model (GLM-4.7-Flash): score the latent c_KV via the
        # calibrated rotation M -> top-k drives the FlashMLASparse attend. MLA
        # keeps its own latent cache (no kv_cache_dtype="lrosa" override); the
        # gathered/attended unit is the compact latent (kv_lora_rank + rope),
        # which is the whole point of this DSA-stack comparison.
        basis = basis or lrosa_basis_path(model, cs_h=cs_h, variant="d1")
        # fp8_projk wires the fp8 score cache (proj_k) INDEPENDENTLY of the
        # attended latent dtype (kv_cache_dtype). GLM-4.7-Flash methodology =
        # bf16 latent attend (--mla_kv_dtype bfloat16) + fp8 proj_k score cache
        # (--fp8_projk): iso-bf16-latent vs dense, only the tiny selection cache
        # is fp8. Without this both were coupled (fp8_ds_mla latent + bf16 projk).
        kw["attention_config"] = {"lrosa_mla": True, "lrosa_basis_path": basis,
                                  "lrosa_n_fac": n_fac, "lrosa_cs_h": cs_h,
                                  "lrosa_fp8_projk": fp8_projk}
        kw["kv_cache_dtype"] = mla_kv_dtype
    elif backend == "dsa":
        # Native GLM-5.2 (DeepSeek-V3.2-style) lightning indexer. No
        # attention_config override -> is_v32 auto-builds the native indexer +
        # its 21 'full'-layer top-k with cross-layer sharing (indexer_types).
        # fp8_ds_mla latent KV (native default). cudagraph MUST be PIECEWISE
        # (the data-dependent indexer top-k garbles under a FULL graph) -- see
        # --cudagraph_mode; use the same PIECEWISE for the sparse baselines so
        # the comparison isn't a cudagraph-mode artifact.
        kw["kv_cache_dtype"] = mla_kv_dtype
    elif backend == "fasa_mla":
        # FASA on an MLA model (GLM): partial-RoPE — FASAMLAIndexer scores only the
        # dominant decoupled-RoPE FCs (basis = fasa_idom_mla_*.pt); cs_h = N_tip
        # (#dominant RoPE FCs of 32). Same FLASHMLA_SPARSE attend as lrosa_mla.
        assert basis, "fasa_mla needs --basis <fasa_idom_mla_*.pt>"
        # cs_h == N_tip == # RoPE FCs kept. n_tip=32 keeps ALL 32 decoupled-RoPE
        # FCs → full-RoPE scoring (idom-independent, no channel-subset selection).
        kw["attention_config"] = {"fasa_mla": True, "lrosa_basis_path": basis,
                                  "lrosa_n_fac": n_fac, "lrosa_cs_h": cs_h,
                                  "lrosa_n_tip": cs_h}
        kw["kv_cache_dtype"] = mla_kv_dtype
    # fkv: default backend. On a natively-sparse GLM-5.2 (is_v32) the model
    # AUTO-BUILDS the native DSA indexer unless overridden -> a plain "fkv" run
    # silently executes native DSA (identical throughput to --backend dsa), NOT
    # true dense. dense_mla=True sets _force_dense_mla -> skips the indexer +
    # is_v32=False -> genuine full dense MLA over all tokens (the real baseline).
    if backend == "fkv" and dense_mla:
        ac = kw.get("attention_config") or {}
        ac["dense_mla"] = True
        kw["attention_config"] = ac
        # dense MLA rejects fp8_ds_mla (sparse-only); use bf16 latent KV (robust).
        kw["kv_cache_dtype"] = mla_kv_dtype if mla_kv_dtype != "fp8_ds_mla" else "auto"
    elif backend == "fkv" and mla_fkv_fp8:
        kw["kv_cache_dtype"] = mla_kv_dtype
    if mla_backend:  # force a specific MLA backend (GLM head-count workaround)
        ac = kw.get("attention_config") or {}
        ac["backend"] = mla_backend
        kw["attention_config"] = ac
    if cudagraph_mode and os.environ.get("ENFORCE_EAGER", "0") != "1":
        # Force a cudagraph mode (PIECEWISE for GLM-5.2: DSA's data-dependent
        # top-k cannot ride a FULL graph; matching all sparse backends to
        # PIECEWISE removes cudagraph-mode as a confound).
        kw["compilation_config"] = {"cudagraph_mode": cudagraph_mode}
    if kw.get("kv_cache_dtype") in ("lrosa", "fasa", "quest", "seer"):
        # Hybrid models (Gemma 4, Ministral): window-bound the sliding layers'
        # KV cache instead of the combined-slot full-length cache — output-
        # invariant, only cuts KV memory / raises max batch. No-op for full-
        # attention models. Essential for a fair sparse-vs-FKV max-batch number.
        kw["kv_cache_dtype_skip_layers"] = ["sliding_window"]
    return LLM(**kw)


def time_generate(llm, prompts, max_tokens, reps=2):
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    tp = [TokensPrompt(prompt_token_ids=p) for p in prompts]
    # warmup
    llm.generate(tp, sp, use_tqdm=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate(tp, sp, use_tqdm=False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return min(ts)  # min = least noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["fkv", "lrosa", "quest", "fasa", "lrosa_mla", "fasa_mla", "dsa"], required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--basis", default=None,
                    help="LRoSA basis / FASA idom .pt; default = pca bases/<tag>/...")
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_tip", type=int, default=16,
                    help="FASA-fc only: # dominant FCs per kv-head.")
    ap.add_argument("--no_radix", action="store_true",
                    help="Disable DSA radix top-K (use torch.topk); needed at bsz>1 + long ctx.")
    ap.add_argument("--streaming", action="store_true",
                    help="LRoSA: use V2 chunk-parallel streaming top-K (no full fp32 "
                         "scores buffer) instead of score+radix.")
    ap.add_argument("--contig_projk", action="store_true",
                    help="LRoSA: store proj_K in a separate contiguous cache for a "
                         "coalesced score scan (faster at long ctx).")
    ap.add_argument("--fp8_projk", action="store_true",
                    help="LRoSA: store proj_K as FP8 e4m3 (implies contig); halves score read.")
    ap.add_argument("--per_layer", action="store_true",
                    help="LRoSA: per-layer CONCAT (one shared top-k/layer; needs layer_concat basis).")
    ap.add_argument("--indexed_attend", action=argparse.BooleanOptionalAction, default=True,
                    help="LRoSA: gather-free fused indexed attention (no K_sel/V_sel "
                         "buffer; ~2.3x less attention-side overhead at large batch). "
                         "ON by default (2026-06-17); --no-indexed_attend forces gather+flash.")
    ap.add_argument("--prefill_len", type=int, default=65536)
    ap.add_argument("--decode_len", type=int, default=128)
    ap.add_argument("--n_fac", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_num_seqs", type=int, default=0,
                    help="0 → use batch_size (tight); >0 → simulate online "
                         "serving with this concurrency cap (stresses the "
                         "decode static buffers).")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    ap.add_argument("--mla_kv_dtype", default="fp8_ds_mla",
                    help="lrosa_mla / fkv(--mla_fkv_fp8): MLA latent KV cache dtype.")
    ap.add_argument("--mla_backend", default=None,
                    help="Force a specific MLA attention backend (GLM head-count workaround).")
    ap.add_argument("--mla_fkv_fp8", action="store_true",
                    help="fkv on an MLA model: use the same fp8 latent KV as lrosa_mla "
                         "(apples-to-apples dense vs sparse attend).")
    ap.add_argument("--chunked_prefill", action="store_true",
                    help="Enable chunked prefill (caps peak prefill activation so long "
                         "ctx fits at the proper decode batch). Safe for fkv/fasa; sparse "
                         "backends may hit the mixed-batch score-kernel bug at bsz>1.")
    ap.add_argument("--cudagraph_mode", default=None,
                    help="Force compilation_config.cudagraph_mode (e.g. PIECEWISE). "
                         "GLM-5.2: DSA needs PIECEWISE; match all sparse backends to it.")
    ap.add_argument("--dense_mla", action="store_true",
                    help="fkv on GLM-5.2: force TRUE dense (dense_mla=True disables the "
                         "auto-built native DSA indexer). Without this, 'fkv' on a v32 "
                         "model silently runs native DSA, not dense.")
    a = ap.parse_args()

    torch.manual_seed(0)
    # B distinct random prompts (prefix caching is off, so each prefills fully).
    prompts = [torch.randint(1000, 30000, (a.prefill_len,)).tolist()
               for _ in range(a.batch_size)]

    llm = build_llm(a.backend, a.prefill_len, a.decode_len, a.n_fac, a.gpu_mem,
                    a.batch_size, a.max_num_seqs, model=a.model, basis=a.basis,
                    cs_h=a.cs_h, n_tip=a.n_tip, use_radix=not a.no_radix,
                    streaming=a.streaming, contig_projk=a.contig_projk, fp8_projk=a.fp8_projk,
                    indexed_attend=a.indexed_attend,
                    per_layer=a.per_layer, tp=a.tp, mla_kv_dtype=a.mla_kv_dtype,
                    mla_backend=a.mla_backend, mla_fkv_fp8=a.mla_fkv_fp8,
                    chunked_prefill=a.chunked_prefill, cudagraph_mode=a.cudagraph_mode,
                    dense_mla=a.dense_mla)

    t_prefill = time_generate(llm, prompts, 1)
    t_full = time_generate(llm, prompts, 1 + a.decode_len)
    # T_full - T_prefill == time for ``decode_len`` decode STEPS (each step
    # advances all B streams in parallel). Per-stream ms/tok = that / decode_len;
    # aggregate throughput = B * decode_len / (T_full - T_prefill).
    decode_ms = (t_full - t_prefill) / a.decode_len * 1000.0
    per_stream_tok_s = 1000.0 / decode_ms if decode_ms > 0 else float("nan")
    agg_tok_s = per_stream_tok_s * a.batch_size
    print(f"[LATENCY] model={a.model} backend={a.backend} prefill={a.prefill_len} "
          f"decode={a.decode_len} n_fac={a.n_fac} bsz={a.batch_size}")
    print(f"  prefill_time={t_prefill*1000:.1f}ms  full_time={t_full*1000:.1f}ms")
    print(f"  DECODE_MS_PER_TOK={decode_ms:.3f}  PER_STREAM_TOK_S={per_stream_tok_s:.1f}"
          f"  AGG_TOK_S={agg_tok_s:.1f}")


if __name__ == "__main__":
    main()
