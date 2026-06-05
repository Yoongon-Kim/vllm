"""LRoSA backend e2e smoke test (offline LLM).

Loads Llama-3.1-8B with the LRoSA attention backend + a pca-calibrated
per-kv-head cs_h=32 basis, runs a few prompts, and prints outputs. Verifies
the backend produces coherent text end-to-end (radix top-K selection path
active by default).

Run:
  VLLM_ATTENTION_BACKEND=LROSA \
  CUDA_VISIBLE_DEVICES=0 \
  /home/jiwonsong/.conda/envs/gaep_vllm/bin/python lrosa_smoke.py
"""
import os

# flashinfer sampling JIT-compiles a kernel on first use and fails to spawn
# ninja in this env; force the native PyTorch sampler instead. Unrelated to
# the LRoSA attention backend.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

BASIS = (
    "/home/jiwonsong/pca/bases/llama_3_1_8b_instruct/"
    "pca_d1_cs32_kv_head_llama_3_1_8b_instruct.pt"
)

prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "List three prime numbers:",
]

sampling = SamplingParams(temperature=0.0, max_tokens=64)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="lrosa",
    attention_config={
        "backend": "LROSA",
        "lrosa_basis_path": BASIS,
        "lrosa_n_fac": 256,
        "lrosa_use_radix_topk": True,
    },
    max_model_len=8192,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
)

outs = llm.generate(prompts, sampling)
for o in outs:
    print("=" * 60)
    print("PROMPT:", o.prompt)
    print("OUTPUT:", o.outputs[0].text)
