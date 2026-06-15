"""Sanity: indexed-attend ON vs OFF should produce near-identical greedy output.

Selection (top_idx) is identical between the two paths; only the attend numerics
differ (fused fp32-accum kernel vs gather+flash bf16). So greedy decode should
match to bf16 tie-break. Run with --indexed_attend on/off in separate processes
and diff the printed token ids.
"""
import argparse, os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
from vllm import LLM, SamplingParams
from _bench_common import lrosa_basis_path, yarn_overrides

ap = argparse.ArgumentParser()
ap.add_argument("--indexed_attend", action="store_true")
ap.add_argument("--ctx", type=int, default=6000)
a = ap.parse_args()

M = "Qwen/Qwen3-8B"
basis = lrosa_basis_path(M, cs_h=32)
ac = {"backend": "LROSA", "lrosa_basis_path": basis, "lrosa_n_fac": 2048,
      "lrosa_cs_h": 32, "lrosa_use_radix_topk": True, "lrosa_fp8_projk": True,
      "lrosa_indexed_attend": a.indexed_attend}
kw = dict(model=M, max_model_len=a.ctx + 128, gpu_memory_utilization=0.85,
          enforce_eager=False, enable_prefix_caching=False, max_num_seqs=4,
          enable_chunked_prefill=False, kv_cache_dtype="lrosa",
          attention_config=ac)
ov = yarn_overrides(M)
if ov:
    kw["hf_overrides"] = ov
llm = LLM(**kw)

# Long context (> n_fac so selection is active) + a concrete question whose
# answer lives near the end, so a wrong selection/attend would change the text.
filler = ("The quarterly logistics report covers warehouse throughput, fleet "
          "utilization, and regional demand. ") * 220
prompt = (filler + "\n\nIMPORTANT FACT: The access code for vault 7 is "
          "ALPHA-93472-ZULU. \n\nQuestion: What is the exact access code for "
          "vault 7? Answer concisely.")
sp = SamplingParams(temperature=0.0, max_tokens=48)
out = llm.generate([prompt], sp, use_tqdm=False)[0].outputs[0]
print(f"INDEXED={a.indexed_attend}")
print("TOKEN_IDS:", list(out.token_ids))
print("TEXT:", repr(out.text))
