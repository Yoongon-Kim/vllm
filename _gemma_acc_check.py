"""gemma-4 LRoSA: indexed-attend ON vs OFF token-identical greedy output?

gemma is hybrid (25 sliding head=256 + 5 full head=512) -> exercises BOTH the
new head=512 indexed path and the head=256 path. Multimodal config, so feed raw
token ids (TokensPrompt) to bypass the image preprocessor. Same random prompt;
selection is identical between paths, so greedy decode must match.
"""
import argparse, os
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from _bench_common import lrosa_basis_path

ap = argparse.ArgumentParser()
ap.add_argument("--indexed_attend", action="store_true")
ap.add_argument("--ctx", type=int, default=8192)
a = ap.parse_args()

M = "google/gemma-4-26B-A4B-it"
basis = lrosa_basis_path(M, cs_h=64)
ac = {"backend": "LROSA", "lrosa_basis_path": basis, "lrosa_n_fac": 2048,
      "lrosa_cs_h": 64, "lrosa_use_radix_topk": True, "lrosa_fp8_projk": True,
      "lrosa_indexed_attend": a.indexed_attend}
llm = LLM(model=M, max_model_len=a.ctx + 128, gpu_memory_utilization=0.85,
          enforce_eager=False, enable_prefix_caching=False, max_num_seqs=24,
          enable_chunked_prefill=False, kv_cache_dtype="lrosa", attention_config=ac)

# REAL tokens (not random): well-separated logits so greedy is not flipped by
# bf16-level noise. Tokenize text ourselves -> TokensPrompt bypasses multimodal.
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(M)
text = ("The quarterly logistics report covers warehouse throughput, fleet "
        "utilization, and regional demand. ") * 240
text += ("\n\nIMPORTANT FACT: The access code for vault 7 is ALPHA-93472-ZULU."
         "\n\nQuestion: What is the exact access code for vault 7? Answer: ")
toks = tok(text, add_special_tokens=True).input_ids[:a.ctx]
sp = SamplingParams(temperature=0.0, max_tokens=48)
out = llm.generate([TokensPrompt(prompt_token_ids=toks)], sp, use_tqdm=False)[0].outputs[0]
print(f"INDEXED={a.indexed_attend}")
print("TOKEN_IDS:", list(out.token_ids))
print("TEXT:", repr(out.text[:160]))
