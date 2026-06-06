"""LRoSA vLLM-backend Qasper F1 — parity check vs pca monkey_patch.

Runs LongBench-v1 Qasper through vLLM with the LRoSA attention backend +
the pca per-kv-head cs_h=32 basis, scores with pca's own qa_f1 metric, and
prints the F1. Target: match pca eval/longbench.py (0.4483 @ n_fac=256,
cs_h=32, 200 samples, head+tail truncation to max_input_len).

Run:
  VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 \
  /home/jiwonsong/.conda/envs/gaep_vllm/bin/python lrosa_qasper_eval.py \
      --num_samples 200 [--mode fkv] [--max_input_len 127500]
"""
import os, sys, json, argparse
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from _bench_common import PCA_REPO, lrosa_basis_path, fasa_idom_path, yarn_overrides

# pca repo for prompt template + metric (vendored LongBench-v1 utils)
sys.path.insert(0, PCA_REPO)
from eval.longbench_v1_utils import (
    TASK_PROMPTS, TASK_MAX_NEW_TOKENS, score_single,
    build_prompt as pca_build_prompt, tokenize_and_truncate,
)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

QASPER = os.path.join(PCA_REPO, "data", "data", "qasper.jsonl")
MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def build_token_ids(row, tok, max_input_len):
    # Exactly mirror pca eval/longbench.py: build_prompt → chat template +
    # head/tail truncate (tokenize_and_truncate). Returns a token-id list
    # for vLLM's prompt_token_ids (skips vLLM re-tokenization).
    p = pca_build_prompt(row["context"], row["input"], "qasper")
    ids = tokenize_and_truncate(p, tok, max_input_len, "qasper")  # [1, L] tensor
    return ids[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--mode", choices=["lrosa", "fkv", "quest", "fasa"], default="lrosa")
    ap.add_argument("--max_input_len", type=int, default=127500)
    ap.add_argument("--n_fac", type=int, default=256)
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_tip", type=int, default=16,
                    help="FASA-fc only: # dominant FCs per kv-head (2*n_tip channels).")
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--contig_projk", action="store_true",
                    help="LRoSA: separate contiguous proj_K cache (coalesced score scan).")
    ap.add_argument("--fp8_projk", action="store_true",
                    help="LRoSA: FP8 e4m3 proj_K cache (implies contig).")
    ap.add_argument("--basis", default=None,
                    help="LRoSA basis .pt; default = pca bases/<tag>/pca_d1_cs<N>_kv_head.")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    a = ap.parse_args()
    if a.basis is None and a.mode == "lrosa":
        a.basis = lrosa_basis_path(a.model, cs_h=a.cs_h)

    rows = []
    with open(QASPER) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= a.num_samples:
                break

    tok = AutoTokenizer.from_pretrained(a.model)
    token_ids = [build_token_ids(r, tok, a.max_input_len) for r in rows]

    kw = dict(model=a.model, max_model_len=a.max_input_len + 256,
              gpu_memory_utilization=a.gpu_mem, enforce_eager=a.eager)
    # Qwen3: enable YaRN so served rope matches the basis's calibration rope.
    ov = yarn_overrides(a.model)
    if ov:
        kw["hf_overrides"] = ov
    if a.mode == "lrosa":
        kw["kv_cache_dtype"] = "lrosa"
        ac = {"backend": "LROSA", "lrosa_basis_path": a.basis,
              "lrosa_n_fac": a.n_fac, "lrosa_cs_h": a.cs_h,
              "lrosa_use_radix_topk": True}
        if a.per_layer:
            ac["lrosa_per_layer_concat"] = True
        if a.contig_projk:
            ac["lrosa_contig_projk"] = True
        if a.fp8_projk:
            ac["lrosa_fp8_projk"] = True
        kw["attention_config"] = ac
    elif a.mode == "quest":
        kw["kv_cache_dtype"] = "quest"
        kw["attention_config"] = {"backend": "QUEST",
                                  "quest_token_budget": a.n_fac}
    elif a.mode == "fasa":
        # Paper-faithful FASA-fc: [K|V] slot, I_dom FC channels read from full K
        # each step (no proj_K). basis = fasa_idom_*.pt.
        idom = a.basis if a.basis else fasa_idom_path(a.model)
        kw["kv_cache_dtype"] = "fasa"
        kw["attention_config"] = {"backend": "LROSA", "lrosa_basis_path": idom,
                                  "lrosa_n_fac": a.n_fac, "lrosa_n_tip": a.n_tip,
                                  "lrosa_use_radix_topk": True}
    llm = LLM(**kw)

    sp = SamplingParams(temperature=0.0, max_tokens=TASK_MAX_NEW_TOKENS["qasper"])
    tp = [TokensPrompt(prompt_token_ids=t) for t in token_ids]
    outs = llm.generate(tp, sampling_params=sp)

    scores = []
    for row, o in zip(rows, outs):
        pred = o.outputs[0].text
        s = score_single(pred, row["answers"], "qasper", row.get("all_classes"))
        scores.append(s)
    f1 = sum(scores) / len(scores)
    print(f"[QASPER-F1] model={a.model} mode={a.mode} n={len(scores)} "
          f"n_fac={a.n_fac} max_input_len={a.max_input_len} F1={f1:.4f}")
    if "llama-3.1-8b" in a.model.lower():
        print(f"  (pca monkey_patch reference, Llama-3.1-8B per-kv-head cs_h=32 = 0.4483)")


if __name__ == "__main__":
    main()
