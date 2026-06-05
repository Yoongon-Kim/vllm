"""Quest backend e2e smoke test (offline LLM).

Loads Llama-3.1-8B with the Quest attention backend (page-level sparse
attention, combined-slot [K|V|min|max] cache), runs a few prompts, and prints
outputs. Verifies coherent text end-to-end. Quest has NO basis file (it scores
pages by per-channel K min/max, not a learned projection).

Run:
  CUDA_VISIBLE_DEVICES=0 \
  /home/jiwonsong/.conda/envs/gaep_vllm/bin/python quest_smoke.py [--eager]
"""
import os
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

eager = "--eager" in sys.argv
budget = 256
for i, a in enumerate(sys.argv):
    if a == "--budget":
        budget = int(sys.argv[i + 1])

prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "List three prime numbers:",
    "Write a haiku about autumn:",
]

sampling = SamplingParams(temperature=0.0, max_tokens=64)

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="quest",
    attention_config={
        "backend": "QUEST",
        "quest_token_budget": budget,
    },
    max_model_len=8192,
    gpu_memory_utilization=0.45,
    enforce_eager=eager,
)

outs = llm.generate(prompts, sampling)
for o in outs:
    print("=" * 60)
    print("PROMPT:", o.prompt)
    print("OUTPUT:", o.outputs[0].text)
