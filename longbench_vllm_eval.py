"""LongBench v1 eval on the vLLM backends (FKV / LRoSA / FASA / Quest).

The vLLM counterpart of pca's eval/longbench.py (which runs the transformers
monkey-patch stack). Builds ONE vLLM engine for the chosen backend, then runs
every requested LongBench v1 EN task through it, scoring with pca's own
longbench_v1_utils metrics (qa_f1 / rouge-L / classification / retrieval /
count / code_sim). Per-task prompts, chat-template handling, head+tail
truncation, and per-task max-new-tokens all mirror eval/longbench.py exactly,
so the numbers are comparable to the transformers-stack results.

Backends (one vLLM engine each):
  fkv   — dense full attention (kv_cache_dtype=auto)
  lrosa — learned-rotation selection (kv_cache_dtype=lrosa, D1 basis)
  fasa  — FASA-fc (kv_cache_dtype=fasa, idom basis)
  quest — page min/max (kv_cache_dtype=quest)

Run:
  TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH \
  HF_HOME=/NHNHOME/jiwonsong/hf_cache VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  CUDA_VISIBLE_DEVICES=0 python longbench_vllm_eval.py \
      --mode lrosa --tasks all --num_samples 200 --n_fac 256 --cs_h 32

  # Gemma 4 (head_dim>256, multimodal): add --cs_h 64; model cached in the
  # default HF cache so run with HF_HUB_OFFLINE=1 and DON'T set HF_HOME.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from _bench_common import PCA_REPO, lrosa_basis_path, fasa_idom_path, yarn_overrides

# pca repo for prompts + metrics (vendored LongBench v1 utils).
sys.path.insert(0, PCA_REPO)
from eval.longbench_v1_utils import (  # noqa: E402
    ENGLISH_TASKS,
    TASK_CATEGORIES,
    TASK_MAX_NEW_TOKENS,
    build_prompt as pca_build_prompt,
    score_single,
    tokenize_and_truncate,
)

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def build_llm(a):
    """One vLLM engine for the chosen backend, sized to cover the longest task
    (max_input_len prompt + the 512-token summarization generations)."""
    max_gen = max(TASK_MAX_NEW_TOKENS.values())
    kw = dict(
        model=a.model,
        max_model_len=a.max_input_len + max_gen + 16,
        gpu_memory_utilization=a.gpu_mem,
        enforce_eager=a.eager,
        enable_prefix_caching=False,
    )
    # LRoSA/FASA decode scratch (score + radix) is sized by max_num_seqs *
    # max_kv. The vLLM default max_num_seqs=1024 makes that explode at 127500
    # ctx (both in CUDA-graph capture and at inference) even though only ~5
    # seqs fit. Cap it so graphs capture and decode fits. F1 is unaffected.
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    ov = yarn_overrides(a.model)
    if ov:
        kw["hf_overrides"] = ov
    if a.mode == "lrosa":
        basis = a.basis or lrosa_basis_path(a.model, cs_h=a.cs_h)
        kw["kv_cache_dtype"] = "lrosa"
        ac = {
            "backend": "LROSA", "lrosa_basis_path": basis,
            "lrosa_n_fac": a.n_fac, "lrosa_cs_h": a.cs_h,
            "lrosa_use_radix_topk": True,
        }
        if a.per_layer:
            ac["lrosa_per_layer_concat"] = True
        if a.contig_projk:
            ac["lrosa_contig_projk"] = True
        if a.fp8_projk:
            ac["lrosa_fp8_projk"] = True
        kw["attention_config"] = ac
    elif a.mode == "fasa":
        idom = a.basis or fasa_idom_path(a.model)
        kw["kv_cache_dtype"] = "fasa"
        kw["attention_config"] = {
            "backend": "LROSA", "lrosa_basis_path": idom,
            "lrosa_n_fac": a.n_fac, "lrosa_n_tip": a.n_tip,
            "lrosa_use_radix_topk": True,
        }
    elif a.mode == "quest":
        kw["kv_cache_dtype"] = "quest"
        kw["attention_config"] = {
            "backend": "QUEST", "quest_token_budget": a.n_fac,
        }
    # fkv: dense default, no special kv_cache_dtype.
    return LLM(**kw)


def load_rows(task, num_samples):
    path = os.path.join(PCA_REPO, "data", "data", f"{task}.jsonl")
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if num_samples and len(rows) >= num_samples:
                break
    return rows


def eval_task(llm, tok, task, a):
    rows = load_rows(task, a.num_samples)
    token_ids = [
        tokenize_and_truncate(
            pca_build_prompt(r["context"], r["input"], task),
            tok, a.max_input_len, task,
        )[0].tolist()
        for r in rows
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=TASK_MAX_NEW_TOKENS[task])
    outs = llm.generate(
        [TokensPrompt(prompt_token_ids=t) for t in token_ids], sp, use_tqdm=False
    )
    scores = [
        score_single(o.outputs[0].text, r["answers"], task, r.get("all_classes"))
        for r, o in zip(rows, outs)
    ]
    return 100.0 * sum(scores) / len(scores), len(scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fkv", "lrosa", "fasa", "quest"], default="lrosa")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tasks", default="all",
                    help="comma-separated task list, or 'all' (16 EN tasks).")
    ap.add_argument("--num_samples", type=int, default=200, help="per task; 0 = all.")
    ap.add_argument("--max_input_len", type=int, default=127500)
    ap.add_argument("--n_fac", type=int, default=256)
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_tip", type=int, default=16)
    ap.add_argument("--basis", default=None)
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--contig_projk", action="store_true")
    ap.add_argument("--fp8_projk", action="store_true")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max_num_seqs", type=int, default=0,
                    help="cap concurrent seqs (0=vllm default 1024). Bounds the "
                         "LRoSA/FASA max_kv-sized decode scratch at long context.")
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--output", default=None, help="optional results .json path")
    a = ap.parse_args()

    tasks = ENGLISH_TASKS if a.tasks == "all" else a.tasks.split(",")
    if a.basis is None and a.mode == "lrosa":
        a.basis = lrosa_basis_path(a.model, cs_h=a.cs_h)

    tok = AutoTokenizer.from_pretrained(a.model)
    llm = build_llm(a)

    results = {}
    for task in tasks:
        score, n = eval_task(llm, tok, task, a)
        results[task] = score
        print(f"[LB] {a.mode:5s} {task:20s} n={n:<4d} score={score:.2f}", flush=True)

    # Category + overall averages (over the tasks actually run).
    print(f"\n=== LongBench v1  mode={a.mode}  model={a.model}  "
          f"n_fac={a.n_fac} cs_h={a.cs_h} max_input={a.max_input_len} ===")
    for cat, cat_tasks in TASK_CATEGORIES.items():
        ts = [t for t in cat_tasks if t in results]
        if ts:
            print(f"  {cat:16s}: {sum(results[t] for t in ts)/len(ts):.2f}"
                  f"   ({', '.join(f'{t}={results[t]:.1f}' for t in ts)})")
    overall = sum(results.values()) / len(results)
    print(f"  {'OVERALL':16s}: {overall:.2f}  ({len(results)} tasks)")

    if a.output:
        with open(a.output, "w") as f:
            json.dump({"mode": a.mode, "model": a.model, "n_fac": a.n_fac,
                       "cs_h": a.cs_h, "max_input_len": a.max_input_len,
                       "tasks": results, "overall": overall}, f, indent=2)
        print(f"  saved -> {a.output}")


if __name__ == "__main__":
    main()
