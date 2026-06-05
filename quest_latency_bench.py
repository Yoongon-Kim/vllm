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
              use_radix=True):
    # max_num_seqs caps concurrency → sizes the LRoSA/Quest static decode
    # buffers. 0 → batch_size (tight, for clean latency). >0 simulates online
    # serving concurrency (stresses the buffers; Quest's are tiny, LRoSA's
    # gather buffers grow with n_fac × max_num_seqs).
    mns = max_num_seqs if max_num_seqs > 0 else max(batch_size, 1)
    kw = dict(model=model, max_model_len=prefill_len + decode_len + 16,
              gpu_memory_utilization=gpu_mem, enforce_eager=False,
              enable_prefix_caching=False, max_num_seqs=mns,
              # Steady-state decode latency: keep prefill and decode in
              # separate batches (no chunked-prefill mixed batches). Also avoids
              # the sparse backends' mixed-batch score-kernel bug at bsz>1.
              enable_chunked_prefill=False)
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
                                  "lrosa_use_radix_topk": use_radix}
    elif backend == "fasa":
        idom = basis or fasa_idom_path(model)
        kw["kv_cache_dtype"] = "fasa"
        kw["attention_config"] = {"backend": "LROSA", "lrosa_basis_path": idom,
                                  "lrosa_n_fac": n_fac, "lrosa_n_tip": n_tip,
                                  "lrosa_use_radix_topk": use_radix}
    elif backend == "quest":
        kw["kv_cache_dtype"] = "quest"
        kw["attention_config"] = {"backend": "QUEST", "quest_token_budget": n_fac}
    # fkv: default backend, no special kv_cache_dtype
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
    ap.add_argument("--backend", choices=["fkv", "lrosa", "quest", "fasa"], required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--basis", default=None,
                    help="LRoSA basis / FASA idom .pt; default = pca bases/<tag>/...")
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_tip", type=int, default=16,
                    help="FASA-fc only: # dominant FCs per kv-head.")
    ap.add_argument("--no_radix", action="store_true",
                    help="Disable DSA radix top-K (use torch.topk); needed at bsz>1 + long ctx.")
    ap.add_argument("--prefill_len", type=int, default=65536)
    ap.add_argument("--decode_len", type=int, default=128)
    ap.add_argument("--n_fac", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_num_seqs", type=int, default=0,
                    help="0 → use batch_size (tight); >0 → simulate online "
                         "serving with this concurrency cap (stresses the "
                         "decode static buffers).")
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    a = ap.parse_args()

    torch.manual_seed(0)
    # B distinct random prompts (prefix caching is off, so each prefills fully).
    prompts = [torch.randint(1000, 30000, (a.prefill_len,)).tolist()
               for _ in range(a.batch_size)]

    llm = build_llm(a.backend, a.prefill_len, a.decode_len, a.n_fac, a.gpu_mem,
                    a.batch_size, a.max_num_seqs, model=a.model, basis=a.basis,
                    cs_h=a.cs_h, n_tip=a.n_tip, use_radix=not a.no_radix)

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
