"""LRoSA backend e2e smoke test (offline LLM).

Loads a model with the LRoSA attention backend + a pca-calibrated per-kv-head
cs_h basis, runs a few prompts, and prints outputs. Verifies the backend
produces coherent text end-to-end (radix top-K selection path active by
default). Model-agnostic: defaults to Llama-3.1-8B but takes --model / --basis;
for Qwen3 it auto-enables YaRN to match the calibrated basis.

Run:
  CUDA_VISIBLE_DEVICES=0 python lrosa_smoke.py
  CUDA_VISIBLE_DEVICES=0 python lrosa_smoke.py --model Qwen/Qwen3-8B
"""
import argparse
import os

# flashinfer sampling JIT-compiles a kernel on first use and fails to spawn
# ninja in this env; force the native PyTorch sampler instead. Unrelated to
# the LRoSA attention backend.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from vllm import LLM, SamplingParams

from _bench_common import lrosa_basis_path, yarn_overrides

prompts = [
    "The capital of France is",
    "Explain in one sentence why the sky is blue:",
    "List three prime numbers:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--basis", default=None,
                    help="LRoSA basis .pt; default = pca bases/<tag>/pca_d1_cs<N>_kv_head.")
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_fac", type=int, default=256)
    ap.add_argument("--max_model_len", type=int, default=8192)
    ap.add_argument("--eager", action="store_true")
    a = ap.parse_args()

    basis = a.basis or lrosa_basis_path(a.model, cs_h=a.cs_h)
    print(f"[smoke] model={a.model}\n[smoke] basis={basis}")

    kw = dict(
        model=a.model,
        kv_cache_dtype="lrosa",
        attention_config={
            "backend": "LROSA",
            "lrosa_basis_path": basis,
            "lrosa_n_fac": a.n_fac,
            "lrosa_cs_h": a.cs_h,
            "lrosa_use_radix_topk": True,
        },
        max_model_len=a.max_model_len,
        gpu_memory_utilization=0.85,
        enforce_eager=a.eager,
    )
    # Qwen3: enable YaRN so served rope matches the calibration-time rope the
    # basis was fit under (no-op for non-Qwen3).
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
