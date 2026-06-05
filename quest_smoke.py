"""Quest backend e2e smoke test (offline LLM).

Loads a model with the Quest attention backend (page-level sparse attention,
combined-slot [K|V|min|max] cache), runs a few prompts, and prints outputs.
Verifies coherent text end-to-end. Quest has NO basis file (it scores pages by
per-channel K min/max, not a learned projection). Model-agnostic: defaults to
Llama-3.1-8B but takes --model; for Qwen3 it auto-enables YaRN.

Run:
  CUDA_VISIBLE_DEVICES=0 python quest_smoke.py [--eager] [--budget 256]
  CUDA_VISIBLE_DEVICES=0 python quest_smoke.py --model Qwen/Qwen3-8B
"""
import argparse
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

from _bench_common import yarn_overrides

prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "List three prime numbers:",
    "Write a haiku about autumn:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()

    print(f"[smoke] model={a.model} quest_budget={a.budget}")
    kw = dict(
        model=a.model,
        kv_cache_dtype="quest",
        attention_config={"backend": "QUEST", "quest_token_budget": a.budget},
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.45,
        enforce_eager=a.eager,
    )
    ov = yarn_overrides(a.model)
    if ov:
        kw["hf_overrides"] = ov
        print(f"[smoke] YaRN enabled: {ov['rope_parameters']}")

    llm = LLM(**kw)
    sampling = SamplingParams(temperature=0.0, max_tokens=64)
    outs = llm.generate(prompts, sampling)
    for o in outs:
        print("=" * 60)
        print("PROMPT:", o.prompt)
        print("OUTPUT:", o.outputs[0].text)


if __name__ == "__main__":
    main()
