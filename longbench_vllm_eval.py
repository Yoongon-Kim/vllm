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
    TASK_METRIC,
    build_prompt as pca_build_prompt,
    score_single,
    tokenize_and_truncate,
)

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

MODEL = "meta-llama/Llama-3.1-8B-Instruct"


_SEER_GATE_REPOS = {
    "qwen3-8b": "SeerAttention/SeerAttention-Decode-Qwen3-8B-AttnGates",
    "qwen3-4b": "SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates",
    "qwen3-14b": "SeerAttention/SeerAttention-Decode-Qwen3-14B-AttnGates",
}


def seer_gate_path(model):
    key = model.split("/")[-1].lower()
    for k, repo in _SEER_GATE_REPOS.items():
        if k in key:
            return repo
    raise ValueError(
        f"No default SeerAttention AttnGate adapter for {model!r}; "
        f"pass --seer_gate_path explicitly.")


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
    if getattr(a, "yarn", True):
        ov = yarn_overrides(a.model)
        if ov:
            kw["hf_overrides"] = ov
    if a.mode in ("lrosa", "loki"):
        # loki = same LRoSA backend (proj_K = M@K) but a PCA-only basis (no
        # q-aware Stiefel): pca_loki_cs{N} instead of pca_d1_cs{N}.
        variant = "loki" if a.mode == "loki" else "d1"
        basis = a.basis or lrosa_basis_path(a.model, cs_h=a.cs_h, variant=variant)
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
    elif a.mode == "seer":
        kw["kv_cache_dtype"] = "seer"
        kw["attention_config"] = {
            "backend": "SEER",
            "seer_gate_path": a.seer_gate_path or seer_gate_path(a.model),
            "seer_token_budget": a.n_fac,
        }
    # fkv: dense default, no special kv_cache_dtype.
    if kw.get("kv_cache_dtype") in ("lrosa", "fasa", "quest", "seer"):
        # Hybrid models (Gemma 4, Ministral): window-bound the sliding layers'
        # KV cache instead of the combined-slot full-length cache. Output-
        # invariant (sliding attention is windowed either way); only cuts KV
        # memory / raises max batch. No-op for full-attention models.
        kw["kv_cache_dtype_skip_layers"] = ["sliding_window"]
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
    recs = []
    for r, o in zip(rows, outs):
        # Decode with the HF tokenizer, NOT vLLM's incremental detokenizer
        # (o.outputs[0].text): the latter drops inter-word spaces for
        # Mistral/tekken tokenizers, collapsing the prediction into one run-on
        # token and tanking word-level F1/ROUGE (Ministral overall 29 -> ~49).
        # Matches the transformers-stack eval (eval/longbench.py).
        pred = tok.decode(o.outputs[0].token_ids, skip_special_tokens=True)
        s = score_single(pred, r["answers"], task, r.get("all_classes"))
        recs.append({"prediction": pred, "answers": r["answers"], "score": s})
    avg = sum(x["score"] for x in recs) / len(recs)
    return 100.0 * avg, len(recs), recs


def _write_lrosa_summary(run_dir, a, per_task, status):
    """Write a pca/eval-longbench-compatible summary.json into the LRoSA
    results tree, so vLLM-measured runs land beside the transformers-stack
    runs in the same format (per_task score is 0-1, like the HF eval).
    Called after every task (status='partial') and once at the end
    ('complete'). Scoring is identical (same score_single); only the engine
    differs, recorded as engine='vllm'."""
    os.makedirs(run_dir, exist_ok=True)
    cat_avgs = {}
    for cat, ctasks in TASK_CATEGORIES.items():
        sc = [per_task[t]["score"] for t in ctasks if t in per_task]
        if sc:
            cat_avgs[cat] = sum(sc) / len(sc)
    overall = (sum(r["score"] for r in per_task.values()) / len(per_task)
               if per_task else 0.0)
    summary = {
        "run_name": a.run_name, "base_model": a.model, "mode": a.mode,
        "n_tip": a.n_tip, "n_fac": a.n_fac, "cs_h": a.cs_h,
        "max_input_len": a.max_input_len, "engine": "vllm",
        "fp8_projk": bool(a.fp8_projk), "status": status,
        "per_task": per_task, "category_averages": cat_avgs,
        "overall_average": overall,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["fkv", "lrosa", "loki", "fasa", "quest", "seer"],
                    default="lrosa")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tasks", default="all",
                    help="comma-separated task list, or 'all' (16 EN tasks).")
    ap.add_argument("--num_samples", type=int, default=200, help="per task; 0 = all.")
    ap.add_argument("--max_input_len", type=int, default=127500)
    ap.add_argument("--n_fac", type=int, default=256)
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_tip", type=int, default=16)
    ap.add_argument("--basis", default=None)
    ap.add_argument("--seer_gate_path", default=None,
                    help="SeerAttention-R AttnGate adapter (default: resolved "
                         "from --model).")
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--contig_projk", action="store_true")
    # FP8 score is the DEFAULT for LRoSA/Loki (essentially lossless vs bf16,
    # and the deployment-relevant config). --no-fp8_projk for bf16. Ignored
    # for fkv/fasa/quest and auto-gated off for head_size>256 (e.g. Gemma 4).
    ap.add_argument("--fp8_projk", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--yarn", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-yarn disables the Qwen3 YaRN override (native rope).")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max_num_seqs", type=int, default=0,
                    help="cap concurrent seqs (0=vllm default 1024). Bounds the "
                         "LRoSA/FASA max_kv-sized decode scratch at long context.")
    ap.add_argument("--gpu_mem", type=float, default=0.90)
    ap.add_argument("--output", default=None, help="optional results .json path")
    ap.add_argument("--results_dir", default=os.path.join(PCA_REPO, "results", "longbench_v1"),
                    help="LRoSA results tree; writes <dir>/<model>/<run_name>/"
                         "{summary.json, <task>.jsonl} in the pca eval format. "
                         "Pass '' to disable.")
    ap.add_argument("--run_name", default=None,
                    help="run dir name under results_dir/<model>/ (default: "
                         "--output stem, else <mode>_cs<cs_h>/<mode>_nt<n_tip>).")
    a = ap.parse_args()
    if a.run_name is None:
        if a.output:
            a.run_name = os.path.splitext(os.path.basename(a.output))[0]
        elif a.mode == "fasa":
            a.run_name = f"fasa_nt{a.n_tip}"
        elif a.mode in ("lrosa", "loki"):
            a.run_name = f"{a.mode}_cs{a.cs_h}" + ("_fp8" if a.fp8_projk else "")
        else:
            a.run_name = a.mode

    tasks = ENGLISH_TASKS if a.tasks == "all" else a.tasks.split(",")
    if a.basis is None and a.mode == "lrosa":
        a.basis = lrosa_basis_path(a.model, cs_h=a.cs_h)

    # Mistral models: AutoTokenizer picks a broken LlamaTokenizer path when
    # mistral_common is installed (drops inter-word spaces on decode -> F1
    # collapses). PreTrainedTokenizerFast forces the correct Rust
    # TokenizersBackend (round-trips spaces), matching the transformers-stack.
    if "mistral" in a.model.lower():
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(a.model)
    else:
        tok = AutoTokenizer.from_pretrained(a.model)
    llm = build_llm(a)

    run_dir = None
    if a.results_dir:
        run_dir = os.path.join(a.results_dir, a.model.split("/")[-1], a.run_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"  LRoSA results -> {run_dir}", flush=True)

    results = {}       # 0-100 task averages (for --output json)
    per_task = {}      # pca-format per-task records (score 0-1)
    for task in tasks:
        score, n, recs = eval_task(llm, tok, task, a)
        results[task] = score
        print(f"[LB] {a.mode:5s} {task:20s} n={n:<4d} score={score:.2f}", flush=True)
        if run_dir:
            with open(os.path.join(run_dir, f"{task}.jsonl"), "w") as f:
                for rec in recs:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_task[task] = {"task": task, "metric": TASK_METRIC[task],
                              "num_samples": n, "score": score / 100.0,
                              "elapsed_sec": 0.0}
            _write_lrosa_summary(run_dir, a, per_task, "partial")
    if run_dir:
        _write_lrosa_summary(run_dir, a, per_task, "complete")
        print(f"  LRoSA summary -> {run_dir}/summary.json", flush=True)

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
