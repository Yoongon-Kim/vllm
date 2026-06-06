"""vLLM reasoning eval: AIME25 / MATH-500 / GPQA on the vLLM backends.

The reasoning counterpart of longbench_vllm_eval.py. Reasoning is short-prompt /
long-thinking-decode (the opposite of LongBench), so the sparse selection runs
over the GROWING decode trace. Backends: fkv / lrosa / loki / fasa / quest.

Scoring is byte-identical to pca's eval/{aime25,math500,gpqa}.py: it imports
their format_sample / extract_answer and pca's math_grader, so only the
generation engine differs (vLLM vs HF). Prompts use the chat template with
thinking ON (enable_thinking). pass@k via --num_runs. Writes pca-format
summaries + per-sample jsonl into the LRoSA results tree (results/<eval>/...).

Run:
  TMPDIR=/NHNHOME/jiwonsong/tmp LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH \
  HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 \
  python reasoning_vllm_eval.py --eval gpqa --mode lrosa --model Qwen/Qwen3-8B \
      --cs_h 32 --n_fac 2048 --num_runs 1 --max_num_seqs 8
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from _bench_common import PCA_REPO, lrosa_basis_path, fasa_idom_path, yarn_overrides

sys.path.insert(0, PCA_REPO)
# Pure helpers reused verbatim from the transformers evals (identical scoring).
from eval.aime25 import format_sample as aime_format_sample  # noqa: E402
from eval.math500 import format_sample as math500_format_sample  # noqa: E402
from eval.gpqa import (  # noqa: E402
    format_sample as gpqa_format_sample,
    extract_answer as gpqa_extract_answer,
)
from eval.math_grader import (  # noqa: E402
    extract_answer as math_extract_answer,
    check_is_correct as math_check_is_correct,
)

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from datasets import load_dataset  # noqa: E402

# Per-model-family sampling (mirrors eval/aime25.py _SAMPLING_PRESETS).
SAMPLING_PRESETS = {
    "qwen3":    {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "gemma":    {"temperature": 1.0, "top_p": 0.95, "top_k": 64},
    "gpt_oss":  {"temperature": 1.0, "top_p": 1.0,  "top_k": 0},
    "_default": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
}


def preset_for(model):
    n = model.lower()
    for key in ("qwen3", "gemma", "gpt_oss", "gpt-oss"):
        if key.replace("-", "_") in n.replace("-", "_"):
            return SAMPLING_PRESETS.get(key.replace("-", "_"), SAMPLING_PRESETS["_default"])
    return SAMPLING_PRESETS["_default"]


# ---- per-eval adapters: (load rows) + (row -> (prompt, gold, extra)) + grade ----

def load_rows(ev, a):
    if ev == "aime25":
        ds = load_dataset(a.aime_dataset, split=a.split or "train")
    elif ev == "math500":
        ds = load_dataset(a.math_dataset, split=a.split or "test")
    elif ev == "gpqa":
        ds = load_dataset(a.gpqa_dataset, a.gpqa_config, split=a.split or "train")
    else:
        raise ValueError(ev)
    rows = list(ds)
    if a.num_samples:
        rows = rows[: a.num_samples]
    return rows


def build_prompt_and_gold(ev, row, idx, seed):
    """Returns (prompt_text, gold) where gold is the answer / letter."""
    if ev == "aime25":
        p, gold = aime_format_sample(row)
        return p, gold
    if ev == "math500":
        p, gold = math500_format_sample(row)
        return p, gold
    if ev == "gpqa":
        # Stable per-(question, seed) option shuffle, like eval/gpqa.py.
        rng = random.Random(seed * 100000 + idx)
        p, gold_letter, _opts = gpqa_format_sample(row, rng)
        return p, gold_letter


def grade(ev, response, gold):
    if ev == "gpqa":
        return gpqa_extract_answer(response) == gold
    # aime25 / math500: boxed extraction + math_equal
    return bool(math_check_is_correct(math_extract_answer(response), gold))


def build_llm(a):
    kw = dict(
        model=a.model,
        max_model_len=a.max_input_len + a.max_new_tokens + 16,
        gpu_memory_utilization=a.gpu_mem,
        enforce_eager=a.eager,
        enable_prefix_caching=False,
    )
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    ov = yarn_overrides(a.model)
    if ov:
        kw["hf_overrides"] = ov
    if a.mode in ("lrosa", "loki"):
        variant = "loki" if a.mode == "loki" else "d1"
        basis = a.basis or lrosa_basis_path(a.model, cs_h=a.cs_h, variant=variant)
        kw["kv_cache_dtype"] = "lrosa"
        ac = {"backend": "LROSA", "lrosa_basis_path": basis,
              "lrosa_n_fac": a.n_fac, "lrosa_cs_h": a.cs_h,
              "lrosa_use_radix_topk": True}
        if a.fp8_projk:
            ac["lrosa_fp8_projk"] = True
        kw["attention_config"] = ac
    elif a.mode == "fasa":
        kw["kv_cache_dtype"] = "fasa"
        kw["attention_config"] = {"backend": "LROSA",
                                  "lrosa_basis_path": a.basis or fasa_idom_path(a.model),
                                  "lrosa_n_fac": a.n_fac, "lrosa_n_tip": a.n_tip,
                                  "lrosa_use_radix_topk": True}
    elif a.mode == "quest":
        kw["kv_cache_dtype"] = "quest"
        kw["attention_config"] = {"backend": "QUEST", "quest_token_budget": a.n_fac}
    # fkv: dense default.
    return LLM(**kw)


def tokenize(tok, prompt_text, model):
    msgs = [{"role": "user", "content": prompt_text}]
    kw = dict(tokenize=True, add_generation_prompt=True)
    if "enable_thinking" in (tok.chat_template or ""):
        kw["enable_thinking"] = True
    ids = tok.apply_chat_template(msgs, **kw)
    return ids if isinstance(ids, list) else ids["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, choices=["aime25", "math500", "gpqa"])
    ap.add_argument("--mode", default="lrosa",
                    choices=["fkv", "lrosa", "loki", "fasa", "quest"])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--num_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--num_runs", type=int, default=1, help="pass@k attempts")
    ap.add_argument("--max_new_tokens", type=int, default=38912)
    ap.add_argument("--max_input_len", type=int, default=4096)
    ap.add_argument("--n_fac", type=int, default=2048)
    ap.add_argument("--cs_h", type=int, default=32)
    ap.add_argument("--n_tip", type=int, default=16)
    ap.add_argument("--basis", default=None)
    # FP8 score is the DEFAULT for LRoSA/Loki (lossless vs bf16, deployment
    # config); --no-fp8_projk for bf16. Ignored for fkv/fasa/quest and
    # auto-gated off for head_size>256 (e.g. Gemma 4 -> effectively bf16).
    ap.add_argument("--fp8_projk", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--max_num_seqs", type=int, default=0)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_p", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default=None)
    ap.add_argument("--aime_dataset", default="yentinglin/aime_2025")
    ap.add_argument("--math_dataset", default="HuggingFaceH4/MATH-500")
    ap.add_argument("--gpqa_dataset", default="Idavidrein/gpqa")
    ap.add_argument("--gpqa_config", default="gpqa_diamond")
    ap.add_argument("--results_dir", default=os.path.join(PCA_REPO, "results"))
    ap.add_argument("--run_name", default=None)
    a = ap.parse_args()

    pr = preset_for(a.model)
    if a.temperature is None:
        a.temperature = pr["temperature"]
    if a.top_p is None:
        a.top_p = pr["top_p"]
    if a.top_k is None:
        a.top_k = pr["top_k"]
    if a.run_name is None:
        a.run_name = f"{a.mode}" + (f"_cs{a.cs_h}" if a.mode in ("lrosa", "loki") else "") \
                     + ("_fp8" if a.fp8_projk else "")

    tok = AutoTokenizer.from_pretrained(a.model)
    rows = load_rows(a.eval, a)
    llm = build_llm(a)

    run_dir = None
    if a.results_dir:
        run_dir = os.path.join(a.results_dir, a.eval, a.model.split("/")[-1], a.run_name)
        os.makedirs(run_dir, exist_ok=True)

    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p,
                        top_k=(a.top_k if a.top_k and a.top_k > 0 else -1),
                        max_tokens=a.max_new_tokens)

    # per-problem: list of bool over runs (for pass@k + avg accuracy).
    n = len(rows)
    solved_count = [0] * n   # per-problem: # of the num_runs attempts that were correct
    gen_lens = [[] for _ in range(n)]  # per-problem: generation token-lengths over runs
    run_accs = []
    recs = []
    for run_id in range(a.num_runs):
        sp.seed = a.seed + run_id
        prompts, golds = [], []
        for idx, row in enumerate(rows):
            ptext, gold = build_prompt_and_gold(a.eval, row, idx, a.seed + run_id)
            golds.append(gold)
            prompts.append(TokensPrompt(prompt_token_ids=tokenize(tok, ptext, a.model)))
        outs = llm.generate(prompts, sp, use_tqdm=True)
        correct = 0
        for idx, (o, gold) in enumerate(zip(outs, golds)):
            resp = o.outputs[0].text
            glen = len(o.outputs[0].token_ids)
            ok = grade(a.eval, resp, gold)
            correct += int(ok)
            solved_count[idx] += int(ok)
            gen_lens[idx].append(glen)
            # per-SAMPLE record: one row per (problem, attempt).
            recs.append({"run": run_id, "idx": idx, "gold": gold,
                         "gen_len": glen, "correct": ok,
                         "truncated": glen >= a.max_new_tokens})
        acc = correct / n
        run_accs.append(acc)
        print(f"[REASON] {a.eval} {a.mode} run{run_id} acc={acc:.4f} ({correct}/{n})",
              flush=True)

    # pass@1 = expected single-sample accuracy, estimated by averaging over all
    # num_runs attempts per problem (= mean over problems of solved_count/num_runs
    # = mean of the per-run accuracies). This is "solve each problem K times,
    # report pass@1". pass@K = fraction solved by at least one of the K attempts.
    total_attempts = n * a.num_runs
    pass_at_1 = sum(solved_count) / total_attempts
    pass_at_k = sum(1 for c in solved_count if c > 0) / n
    # Generation-length stats: per-sample lengths live in predictions.jsonl;
    # here we aggregate per-problem (mean over runs) + overall.
    all_lens = [l for sub in gen_lens for l in sub]
    mean_gen_len = sum(all_lens) / len(all_lens) if all_lens else 0.0
    max_gen_len = max(all_lens) if all_lens else 0
    frac_truncated = (sum(1 for l in all_lens if l >= a.max_new_tokens)
                      / len(all_lens) if all_lens else 0.0)
    per_problem = [
        {"idx": i, "solved_count": solved_count[i], "n_runs": len(gen_lens[i]),
         "pass_at_1": (solved_count[i] / len(gen_lens[i]) if gen_lens[i] else 0.0),
         "gen_lens": gen_lens[i],
         "mean_gen_len": (sum(gen_lens[i]) / len(gen_lens[i]) if gen_lens[i] else 0.0)}
        for i in range(n)
    ]
    print(f"\n=== {a.eval} mode={a.mode} model={a.model} cs_h={a.cs_h} n_fac={a.n_fac} "
          f"runs={a.num_runs} fp8={bool(a.fp8_projk)} ===")
    print(f"  pass@1 = {pass_at_1:.4f}  (avg over {a.num_runs} attempts/problem, n={n})")
    print(f"  pass@{a.num_runs} = {pass_at_k:.4f}  (solved by >=1 attempt)")
    print(f"  per-run acc: {['%.3f' % x for x in run_accs]}")
    print(f"  gen_len: mean={mean_gen_len:.0f} max={max_gen_len} "
          f"truncated@{a.max_new_tokens}={frac_truncated:.3f}")

    if run_dir:
        with open(os.path.join(run_dir, "predictions.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = {
            "run_name": a.run_name, "eval": a.eval, "base_model": a.model,
            "mode": a.mode, "engine": "vllm", "cs_h": a.cs_h, "n_fac": a.n_fac,
            "n_tip": a.n_tip, "fp8_projk": bool(a.fp8_projk),
            "max_new_tokens": a.max_new_tokens, "num_samples": n,
            "num_runs": a.num_runs, "temperature": a.temperature,
            "top_p": a.top_p, "top_k": a.top_k, "status": "complete",
            "run_accuracies": run_accs, "pass_at_1": pass_at_1,
            "pass_at_k": pass_at_k, "avg_accuracy": pass_at_1,
            "solved_count": solved_count,
            "mean_gen_len": mean_gen_len, "max_gen_len": max_gen_len,
            "frac_truncated": frac_truncated, "per_problem": per_problem,
        }
        with open(os.path.join(run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"  saved -> {run_dir}")


if __name__ == "__main__":
    main()
